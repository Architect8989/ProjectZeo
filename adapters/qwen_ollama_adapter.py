from __future__ import annotations

import base64
import json
import logging
import re
import sys
import threading
import asyncio
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional, Any

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


_OCR_UNAVAILABLE = False  # FIX-5: set True if EasyOCR cannot initialise


def _get_ocr_reader():
    """
    EasyOCR initialisation can block 30–90s on CPU-only machines.
    Double-checked locking with hard 120s timeout and user-visible progress.

    FIX-5 (Audit): EasyOCR initialisation is now non-fatal.
    On a raw OS with no internet access, the EasyOCR model download (~150 MB)
    will fail.  Previously this raised RuntimeError, which propagated out of
    _resolve_click_coordinates() and crashed the execution loop on every single
    action cycle.

    Fix: catch all initialisation failures, set _OCR_UNAVAILABLE=True, and
    log a warning.  _resolve_click_coordinates() checks this flag and falls
    back to coordinate-only mode — clicks with explicit x/y still work,
    text-based clicks are silently dropped (fail-closed, unchanged behaviour).
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
            import easyocr  # local import — do not hoist to module level

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
                    f"EasyOCR initialisation timed out after "
                    f"{_OCR_WARMUP_TIMEOUT_SECONDS}s — check model download or GPU availability."
                )

            if "err" in error_holder:
                raise RuntimeError(
                    f"EasyOCR initialisation failed: {error_holder['err']}"
                ) from error_holder["err"]

            _OCR_READER = result_holder["reader"]
            logger.info("[QwenOllamaAdapter] EasyOCR ready.")
            return _OCR_READER

        except Exception as exc:
            # FIX-5: non-fatal — fall back to coordinate-only mode
            _OCR_UNAVAILABLE = True
            logger.warning(
                f"[QwenOllamaAdapter] EasyOCR unavailable: {exc}. "
                "Falling back to coordinate-only click resolution. "
                "Text-based clicks will be dropped (fail-closed)."
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
    # Modern ollama ≥0.2: object with attribute access
    if hasattr(response, "message"):
        message = response.message
        if hasattr(message, "content"):
            content = message.content
            if isinstance(content, str):
                return content
            raise RuntimeError(
                f"Unexpected ollama response.message.content type: {type(content)}"
            )

    # Legacy ollama <0.2: dict-style response
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
# ADAPTER
# ==========================================================

class QwenOllamaAdapter:
    """
    Local-only Qwen-VL adapter via Ollama.
    Fully in-memory, deterministic, bounded execution.
    """

    def __init__(self, model_name: str = "qwen2.5-vl:7b-instruct"):
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("Invalid model_name")

        self.model_name = model_name.strip()

        # PATCH §R4: read timeout raised to 120s for CPU inference compatibility.
        # CPU inference on Qwen2.5-VL 7B takes 40–90s on consumer hardware.
        # The prior 25s timeout caused spurious replan cascades on every cycle.
        self._client = ollama.Client(
            timeout=httpx.Timeout(
                connect=10.0,
                read=120.0,
                write=5.0,
                pool=2.0,
            )
        )

        # Bounded executor — prevents unbounded thread spawn
        self._executor = ThreadPoolExecutor(max_workers=1)

    # ==========================================================
    # PUBLIC ENTRY
    # ==========================================================

    async def get_next_action(
        self,
        messages: List[dict],
        objective: str,
        session_id: Optional[str] = None,
    ) -> Tuple[Optional[List[dict]], Optional[Exception]]:
        try:
            ops = await self._call_qwen_with_ocr(messages, objective)
            return ops, None
        except Exception as exc:
            return None, exc

    # ==========================================================
    # CORE EXECUTION (NO PERSISTENT FILES)
    # ==========================================================

    async def _call_qwen_with_ocr(
        self,
        messages: List[dict],
        objective: str,
    ) -> List[dict]:

        local_msgs = self._confirm_system_prompt(messages, objective)

        # Secure temporary files — auto-deleted on context exit
        with (
            tempfile.NamedTemporaryFile(suffix=".png", delete=True) as raw_tmp,
            tempfile.NamedTemporaryFile(suffix=".jpeg", delete=True) as jpeg_tmp,
        ):
            capture_screen_with_cursor(raw_tmp.name)
            compress_screenshot(raw_tmp.name, jpeg_tmp.name)

            with open(jpeg_tmp.name, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode("utf-8")

            base_prompt = (
                get_user_first_message_prompt()
                if len(local_msgs) == 1
                else get_user_prompt()
            )

            # PATCH §R9: inject current objective into every per-action prompt.
            # Without this the LLM retains goal context only from the system prompt
            # and drifts on long multi-step tasks. The objective reminder anchors
            # the LLM to the current goal at every action decision point.
            if objective and objective.strip():
                user_prompt = (
                    f"Current objective: {objective.strip()}\n\n{base_prompt}"
                )
            else:
                user_prompt = base_prompt

            loop = asyncio.get_running_loop()

            def _blocking_call():
                return self._client.chat(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"{user_prompt}\n"
                                        "Return JSON list of operations."
                                    ),
                                },
                                {
                                    "type": "image",
                                    "image": img_base64,
                                },
                            ],
                        }
                    ],
                    options={"temperature": 0},
                )

            response = await loop.run_in_executor(
                self._executor,
                _blocking_call,
            )

            content = _extract_response_content(response)

            operations = self._parse_and_normalize_json(content)

            if not isinstance(operations, list):
                raise RuntimeError("LLM output must be a list")

            operations = [
                op
                for op in operations
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
        # FIX-5: reader may be None if EasyOCR is unavailable (raw OS / no internet).
        # In coordinate-only mode: clicks with x/y pass through, text-only clicks dropped.
        reader = _get_ocr_reader()

        if reader is None:
            # Coordinate-only fallback — no OCR available
            filtered_ops: List[dict] = []
            for op in operations:
                if op.get("operation") != "click":
                    filtered_ops.append(op)
                    continue
                x = op.get("x")
                y = op.get("y")
                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    op["x"] = float(x)
                    op["y"] = float(y)
                    filtered_ops.append(op)
                # else: no coords and no OCR → fail-closed, drop
            operations.clear()
            operations.extend(filtered_ops)
            return

        ocr_result = reader.readtext(screenshot_path)

        filtered_ops = []

        for op in operations:

            if op.get("operation") != "click":
                filtered_ops.append(op)
                continue

            if "text" not in op:
                # DEF-2 FIX: If the LLM provided explicit x/y coordinates,
                # the click is visually grounded and should be honoured.
                # Only drop clicks that have neither text nor coordinates.
                x = op.get("x")
                y = op.get("y")
                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    op["x"] = float(x)
                    op["y"] = float(y)
                    filtered_ops.append(op)
                # else: no text, no coords → fail-closed, drop silently
                continue

            try:
                idx = get_text_element(
                    ocr_result,
                    op["text"],
                    screenshot_path,
                )
                coords = get_text_coordinates(
                    ocr_result,
                    idx,
                    screenshot_path,
                )

                if isinstance(coords, dict):
                    x = coords.get("x")
                    y = coords.get("y")
                    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                        op["x"] = float(x)
                        op["y"] = float(y)
                        filtered_ops.append(op)

            except Exception:
                # fail-closed: unresolvable click → dropped
                continue

        operations.clear()
        operations.extend(filtered_ops)

    # ==========================================================
    # HELPERS
    # ==========================================================

    def _confirm_system_prompt(
        self,
        messages: List[dict],
        objective: str,
    ) -> List[dict]:

        local = list(messages)
        system_content = get_system_prompt(self.model_name, objective)
        system_message = {"role": "system", "content": system_content}

        if local and local[0].get("role") == "system":
            local[0]["content"] = system_content
        else:
            local.insert(0, system_message)

        return local

    def _parse_and_normalize_json(self, text: str) -> List[dict]:
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

        # GAP-1 FIX: Use greedy regex so the full JSON array/object is captured.
        # The prior `.*?` (non-greedy) with re.DOTALL stopped at the FIRST closing
        # `}` or `]`, silently truncating multi-operation arrays like
        # `[{...}, {...}, {...}]` to only the first element.
        # Strategy: try the outermost JSON array first (greedy `[…]`), then
        # outermost object (greedy `{…}`).  json.loads will reject partial matches,
        # so false positives are safe — we just continue to the next candidate.
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
