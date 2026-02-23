from __future__ import annotations

import base64
import json
import logging
import os
import re
import sys
import threading
import time
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

# RTB-07 FIX: Replace permanent _OCR_UNAVAILABLE bool with a cooldown-based
# retry mechanism. Previously, one EasyOCR failure at cold-start permanently
# disabled OCR for the process lifetime. Any subsequent task that required
# text-label click resolution failed silently because the flag could never
# recover within a session.
#
# Fix: record the timestamp of the last failure. After _OCR_RETRY_COOLDOWN_SECONDS
# (300s = 5 min), allow one retry attempt. This lets OCR recover automatically
# after transient failures (network timeout during model weight download, GPU
# warm-up delay) without requiring a process restart.
#
# FIX RB-A6: Both _OCR_UNAVAILABLE and _OCR_LAST_FAILURE_TS are now ONLY read
# and written while _OCR_LOCK is held. The previous code read both variables
# OUTSIDE the lock as a "fast path" optimization:
#
#   if _OCR_UNAVAILABLE:                              # ← read outside lock
#       if time.monotonic() - _OCR_LAST_FAILURE_TS   # ← read outside lock
#           < _OCR_RETRY_COOLDOWN_SECONDS:
#           return None
#
# This was a data race. Another thread could be inside the lock writing
# _OCR_LAST_FAILURE_TS = time.monotonic() (after a failure) while this thread
# read a stale value of 0.0. The comparison `time.monotonic() - 0.0` would be
# in the millions of seconds, causing the cooldown check to evaluate to False
# and two threads to simultaneously attempt OCR re-initialization.
#
# In the other direction: _OCR_UNAVAILABLE could be read as True by thread A
# while thread B was concurrently clearing it (setting it to False inside the
# lock after a successful re-init). Thread A would then return None and skip
# the perfectly functional OCR reader.
#
# Fix: remove the lock-free fast path entirely. All reads and writes of
# _OCR_UNAVAILABLE and _OCR_LAST_FAILURE_TS go through _OCR_LOCK.
# The performance cost is negligible: _get_ocr_reader() is called once per
# action resolution (not in a hot inner loop), and the lock is uncontested
# once OCR is initialized (_OCR_READER is not None → early return).
_OCR_UNAVAILABLE = False
_OCR_LAST_FAILURE_TS: float = 0.0
_OCR_RETRY_COOLDOWN_SECONDS = 300.0


def _get_ocr_reader():
    """
    EasyOCR initialisation is non-fatal (FIX-5).
    On a raw OS with no network, the model download (~150 MB) may fail.
    Falls back to coordinate-only mode when OCR is unavailable.

    RTB-07: After _OCR_RETRY_COOLDOWN_SECONDS from the last failure,
    clears the unavailable flag and retries initialisation once, allowing
    recovery from transient cold-start failures.

    FIX RB-A6: All reads/writes of _OCR_UNAVAILABLE and _OCR_LAST_FAILURE_TS
    are now exclusively inside _OCR_LOCK to eliminate the data race described
    above. The lock-free fast path has been removed.
    """
    global _OCR_READER, _OCR_UNAVAILABLE, _OCR_LAST_FAILURE_TS

    # Fast path: reader already initialized. Read _OCR_READER without the lock
    # because it transitions from None → object exactly once (inside the lock),
    # and object references are written atomically in CPython. Once non-None,
    # it is never set back to None, so this read is safe.
    if _OCR_READER is not None:
        return _OCR_READER

    # All remaining logic — including reading _OCR_UNAVAILABLE and
    # _OCR_LAST_FAILURE_TS — happens exclusively inside the lock.
    with _OCR_LOCK:
        # Re-check under lock (double-checked locking for _OCR_READER).
        if _OCR_READER is not None:
            return _OCR_READER

        # FIX RB-A6: Cooldown check is now fully inside the lock.
        # Both reads of _OCR_UNAVAILABLE and _OCR_LAST_FAILURE_TS are
        # serialized with writes — no torn reads possible.
        if _OCR_UNAVAILABLE:
            if time.monotonic() - _OCR_LAST_FAILURE_TS < _OCR_RETRY_COOLDOWN_SECONDS:
                return None
            # Cooldown elapsed — reset flag and attempt re-initialization.
            _OCR_UNAVAILABLE = False
            logger.info("[QwenOllamaAdapter] OCR cooldown elapsed — retrying EasyOCR init.")

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
            # FIX RB-A6: Write _OCR_UNAVAILABLE and _OCR_LAST_FAILURE_TS
            # inside the lock — consistent with the reads above.
            _OCR_UNAVAILABLE = True
            _OCR_LAST_FAILURE_TS = time.monotonic()
            logger.warning(
                f"[QwenOllamaAdapter] EasyOCR unavailable: {exc}. "
                f"Will retry after {_OCR_RETRY_COOLDOWN_SECONDS}s. "
                "Falling back to coordinate-only click resolution."
            )
            print(
                f"[QwenOllamaAdapter] WARNING: EasyOCR unavailable ({exc}). "
                f"Coordinate-only mode active. Retry in {_OCR_RETRY_COOLDOWN_SECONDS}s.",
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
        # RB-03 FIX: Windows NamedTemporaryFile PermissionError.
        # ─────────────────────────────────────────────────────────────────
        # On Windows, NamedTemporaryFile(delete=True) keeps the OS file
        # handle open until the `with` block exits. A second open() call
        # on the same path raises PermissionError because Windows does not
        # allow a second handle while the first is open with exclusive access.
        #
        # Fix: create both temp files with delete=False, close each handle
        # immediately, perform all I/O, then unlink in a finally block.
        # This is safe on all platforms (Linux/macOS unlink semantics
        # differ but the explicit cleanup is harmless).
        # ─────────────────────────────────────────────────────────────────
        raw_tmp_name = None
        jpeg_tmp_name = None
        try:
            _rtf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            raw_tmp_name = _rtf.name
            _rtf.close()  # release handle so capture can write it

            _jtf = tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False)
            jpeg_tmp_name = _jtf.name
            _jtf.close()  # release handle before compress writes it

            capture_screen_with_cursor(raw_tmp_name)
            compress_screenshot(raw_tmp_name, jpeg_tmp_name)

            with open(jpeg_tmp_name, "rb") as f:
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

            # All post-processing that references jpeg_tmp_name MUST happen
            # inside this try block while the file still exists on disk.
            # The finally clause below deletes it.
            content = _extract_response_content(response)
            operations = self._parse_and_normalize_json(content)

            if not isinstance(operations, list):
                raise RuntimeError("LLM output must be a JSON array")

            # Filter to valid operation dicts only
            operations = [
                op for op in operations
                if isinstance(op, dict) and "operation" in op
            ]

            self._resolve_click_coordinates(operations, jpeg_tmp_name)

            return operations

        finally:
            # RB-03 FIX: Explicit cleanup — safe on all platforms.
            for _tmp_path in (raw_tmp_name, jpeg_tmp_name):
                if _tmp_path is not None:
                    try:
                        os.unlink(_tmp_path)
                    except OSError:
                        pass

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
