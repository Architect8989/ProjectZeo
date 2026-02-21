from __future__ import annotations

import base64
import json
import logging
import re
import sys
import threading
import asyncio
import copy
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional, Any, Dict

import httpx
import ollama

from operate.config import Config
from operate.models.prompts import (
    get_system_prompt,
    get_user_first_message_prompt,
    get_user_prompt,
)
from operate.utils.ocr import get_text_coordinates, get_text_element
from operate.utils.screenshot import capture_screen_with_cursor, compress_screenshot

logger = logging.getLogger(__name__)
config = Config()


# ==========================================================
# THREAD-SAFE OCR READER WITH WARMUP TIMEOUT
# ==========================================================

_OCR_READER = None
_OCR_LOCK = threading.Lock()
_OCR_WARMUP_TIMEOUT_SECONDS = 120
_OCR_UNAVAILABLE = False


def _get_ocr_reader():
    """
    EasyOCR initialisation is non-fatal (FIX-5).
    On a raw OS with no network, the model download (~150 MB) may fail.
    Falls back to coordinate-only mode when OCR is unavailable.
    """
    global _OCR_READER, _OCR_UNAVAILABLE

    if _OCR_READER is not None:
        return _OCR_READER
    if _OCR_UNAVAILABLE:
        return None

    with _OCR_LOCK:
        if _OCR_READER is not None:
            return _OCR_READER
        if _OCR_UNAVAILABLE:
            return None

        logger.warning(
            "[QwenOllamaAdapter] Initialising EasyOCR reader. "
            "This may take up to 90 seconds on CPU-only hardware …"
        )
        print(
            "[QwenOllamaAdapter] Initialising EasyOCR (first-time, may be slow) …",
            file=sys.stderr,
            flush=True,
        )

        try:
            import easyocr  # noqa: PLC0415

            result_holder: dict = {}
            error_holder: dict = {}

            def _init():
                try:
                    result_holder["reader"] = easyocr.Reader(["en"])
                except Exception as exc:
                    error_holder["err"] = exc

            t = threading.Thread(target=_init, daemon=True)
            t.start()
            t.join(timeout=_OCR_WARMUP_TIMEOUT_SECONDS)

            if t.is_alive():
                raise RuntimeError(
                    f"EasyOCR initialisation timed out after {_OCR_WARMUP_TIMEOUT_SECONDS}s"
                )

            if "err" in error_holder:
                raise RuntimeError(
                    f"EasyOCR initialisation failed: {error_holder['err']}"
                ) from error_holder["err"]

            _OCR_READER = result_holder["reader"]
            logger.info("[QwenOllamaAdapter] EasyOCR ready.")
            return _OCR_READER

        except Exception as exc:
            _OCR_UNAVAILABLE = True
            logger.warning(
                f"[QwenOllamaAdapter] EasyOCR unavailable: {exc}. "
                "Falling back to coordinate-only click resolution."
            )
            print(
                f"[QwenOllamaAdapter] WARNING: EasyOCR unavailable ({exc}). "
                "Coordinate-only mode active.",
                file=sys.stderr,
                flush=True,
            )
            return None


# ==========================================================
# OLLAMA RESPONSE SHIM
# ==========================================================

def _extract_response_content(response: Any) -> str:
    """
    Handles BOTH ollama ≥0.2 object shape (response.message.content)
    and legacy dict shape (response['message']['content']).
    """
    if hasattr(response, "message"):
        message = response.message
        if hasattr(message, "content"):
            content = message.content
            if isinstance(content, str):
                return content
            raise RuntimeError(
                f"Unexpected ollama response.message.content type: {type(content)}"
            )

    if isinstance(response, dict):
        content = response.get("message", {}).get("content")
        if isinstance(content, str):
            return content
        raise RuntimeError(
            f"Unexpected ollama dict response shape: {response!r}"
        )

    raise RuntimeError(
        f"Cannot extract content from ollama response type: {type(response)}"
    )


# ==========================================================
# MESSAGE HISTORY UTILITIES
# ==========================================================

def _build_text_summary_of_message(msg: dict) -> Optional[dict]:
    """
    Convert an older message that may contain image data into a text-only
    summary suitable for inclusion in the multi-turn history sent to Ollama.

    Keeps the role and extracts text content. Images from older turns are
    dropped to keep the context window bounded — the current turn always
    carries the live screenshot.

    Returns None if the message should be skipped entirely.
    """
    role = msg.get("role")
    if role not in ("user", "assistant", "system"):
        return None

    content = msg.get("content")

    if isinstance(content, str):
        # Plain text — include as-is
        return {"role": role, "content": content}

    if isinstance(content, list):
        # Multimodal content — extract only text parts
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text", "")
                if text.strip():
                    text_parts.append(text.strip())
        combined = "\n".join(text_parts)
        if combined:
            return {"role": role, "content": combined}
        return None

    return None


# ==========================================================
# ADAPTER
# ==========================================================

class QwenOllamaAdapter:
    """
    Local-only vision LLM adapter via Ollama.

    Provider-agnostic design: the only Ollama-specific call is self._client.chat().
    The adapter interface (get_next_action) is shared across all adapters.

    To support a different local model (llama3.2-vision, llava, etc.):
      - Create a sibling adapter file
      - Register it in adapters/factory.py _LOCAL_REGISTRY
      - No changes needed here
    """

    # Maximum number of prior conversation turns to include in each call.
    # Each turn is text-only (images stripped from older turns).
    # Keeps context window bounded on long tasks.
    MAX_HISTORY_TURNS = 10

    def __init__(self, model_name: str = "qwen2.5-vl:7b-instruct"):
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")

        self.model_name = model_name.strip()

        # §R4: read timeout raised to 120s for CPU inference compatibility.
        # CPU inference on Qwen2.5-VL 7B: 40–90s on consumer hardware.
        self._client = ollama.Client(
            timeout=httpx.Timeout(
                connect=10.0,
                read=120.0,
                write=5.0,
                pool=2.0,
            )
        )

        # Bounded executor — prevents unbounded thread spawn under async callers
        self._executor = ThreadPoolExecutor(max_workers=1)

    # ==========================================================
    # PUBLIC ENTRY — adapter interface contract
    # ==========================================================

    async def get_next_action(
        self,
        messages: List[dict],
        objective: str,
        session_id: Optional[str] = None,
    ) -> Tuple[Optional[List[dict]], Optional[Exception]]:
        """
        Execute a single action decision cycle.
        Returns (operations_list, None) on success, (None, exception) on error.
        """
        try:
            ops = await self._call_qwen_with_history(messages, objective)
            return ops, None
        except Exception as exc:
            return None, exc

    # ==========================================================
    # CORE VISION INFERENCE — FIX-3: full history forwarded
    # ==========================================================

    async def _call_qwen_with_history(
        self,
        messages: List[dict],
        objective: str,
    ) -> List[dict]:
        """
        Capture current screen, build multi-turn message list, call Ollama,
        parse JSON operations.

        FIX-3: Prior conversation turns are included as text-only history
        so the LLM knows what has already been done. Only the current
        (latest) user turn carries the live screenshot image.
        """

        # --- Build system prompt ---
        system_content = get_system_prompt(self.model_name, objective)

        # --- Build historical context (text-only, no old images) ---
        history_messages: List[dict] = []

        # Walk all prior messages, convert to text summaries
        for msg in messages:
            role = msg.get("role")

            # Skip the system message — we'll inject our own below
            if role == "system":
                continue

            summary = _build_text_summary_of_message(msg)
            if summary is not None:
                history_messages.append(summary)

        # Trim to MAX_HISTORY_TURNS (keep the most recent turns)
        if len(history_messages) > self.MAX_HISTORY_TURNS:
            history_messages = history_messages[-self.MAX_HISTORY_TURNS:]

        # --- Capture live screenshot ---
        with (
            tempfile.NamedTemporaryFile(suffix=".png", delete=True) as raw_tmp,
            tempfile.NamedTemporaryFile(suffix=".jpeg", delete=True) as jpeg_tmp,
        ):
            capture_screen_with_cursor(raw_tmp.name)
            compress_screenshot(raw_tmp.name, jpeg_tmp.name)

            with open(jpeg_tmp.name, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode("utf-8")

            # --- Build per-action user prompt ---
            # §R9: inject current objective to prevent LLM drift on long tasks
            is_first_message = len(history_messages) == 0
            base_prompt = (
                get_user_first_message_prompt()
                if is_first_message
                else get_user_prompt()
            )

            if objective and objective.strip():
                user_prompt_text = (
                    f"Current objective: {objective.strip()}\n\n{base_prompt}"
                )
            else:
                user_prompt_text = base_prompt

            # --- Assemble full message list for ollama.chat() ---
            # Structure: [system] + [text-only history] + [current user turn with screenshot]
            ollama_messages: List[Dict[str, Any]] = [
                {"role": "system", "content": system_content}
            ]
            ollama_messages.extend(history_messages)
            ollama_messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt_text + "\nReturn JSON list of operations.",
                    },
                    {
                        "type": "image",
                        "image": img_base64,
                    },
                ],
            })

            loop = asyncio.get_running_loop()

            def _blocking_call():
                return self._client.chat(
                    model=self.model_name,
                    messages=ollama_messages,
                    options={"temperature": 0},
                )

            response = await loop.run_in_executor(self._executor, _blocking_call)

            # FIX-01 (RTB-01): All post-processing that references jpeg_tmp.name
            # MUST happen inside this with block. The NamedTemporaryFile is deleted
            # by the OS when the context manager exits (delete=True). Moving these
            # calls outside caused FileNotFoundError in OCR, silently dropping all
            # text-anchored click operations.
            content = _extract_response_content(response)
            operations = self._parse_and_normalize_json(content)

            if not isinstance(operations, list):
                raise RuntimeError("LLM output must be a JSON array")

            # Filter to valid operation dicts only
            operations = [
                op for op in operations
                if isinstance(op, dict) and "operation" in op
            ]

            self._resolve_click_coordinates(operations, jpeg_tmp.name)

        return operations

    # ==========================================================
    # OCR RESOLUTION (FAIL-CLOSED)
    # ==========================================================

    def _resolve_click_coordinates(
        self,
        operations: List[dict],
        screenshot_path: str,
    ) -> None:
        """
        Resolve text-based clicks to pixel coordinates via OCR.
        Falls back to coordinate-only mode if EasyOCR is unavailable.
        Clicks with neither text nor coordinates are dropped (fail-closed).
        """
        reader = _get_ocr_reader()

        if reader is None:
            # Coordinate-only fallback
            filtered: List[dict] = []
            for op in operations:
                if op.get("operation") != "click":
                    filtered.append(op)
                    continue
                x = op.get("x")
                y = op.get("y")
                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    op["x"] = float(x)
                    op["y"] = float(y)
                    filtered.append(op)
                # else: no coords and no OCR → drop (fail-closed)
            operations.clear()
            operations.extend(filtered)
            return

        try:
            ocr_result = reader.readtext(screenshot_path)
        except Exception as exc:
            logger.warning(f"[QwenOllamaAdapter] OCR readtext failed: {exc}")
            ocr_result = []

        filtered = []
        for op in operations:
            if op.get("operation") != "click":
                filtered.append(op)
                continue

            if "text" not in op:
                # DEF-2 FIX: honour explicit x/y coordinates
                x = op.get("x")
                y = op.get("y")
                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    op["x"] = float(x)
                    op["y"] = float(y)
                    filtered.append(op)
                # else: no text, no coords → fail-closed, drop
                continue

            try:
                idx = get_text_element(ocr_result, op["text"], screenshot_path)
                coords = get_text_coordinates(ocr_result, idx, screenshot_path)

                if isinstance(coords, dict):
                    x = coords.get("x")
                    y = coords.get("y")
                    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                        op["x"] = float(x)
                        op["y"] = float(y)
                        filtered.append(op)
            except Exception:
                # Unresolvable text click → drop silently (fail-closed)
                continue

        operations.clear()
        operations.extend(filtered)

    # ==========================================================
    # HELPERS
    # ==========================================================

    def _parse_and_normalize_json(self, text: str) -> List[dict]:
        """
        Parse LLM text output into a list of operation dicts.
        GAP-1 FIX: greedy regex captures full JSON arrays (was non-greedy,
        truncated multi-operation arrays to first element).
        """
        text = text.strip()
        # Strip markdown fences
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE).strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return [parsed]
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

        # Greedy fallback — try outermost array, then outermost object
        for pattern in (r"(\[.*\])", r"(\{.*\})"):
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(1))
                    if isinstance(parsed, dict):
                        return [parsed]
                    if isinstance(parsed, list):
                        return parsed
                except Exception:
                    pass

        raise RuntimeError(
            f"No valid JSON structure found in ollama response: {text[:200]!r}"
        )
