from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
import re
import sys
import threading
import time
import asyncio
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

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

# BUG-3 FIX: Import the shared inference lock from vision_runtime.
# VisionRuntime background loop and QwenOllamaAdapter both call the same
# Qwen2.5-VL vision model. On VRAM-limited GPUs this causes OOM; on CPU
# both requests queue in Ollama causing timeout cascades. The lock ensures
# only one caller is in the model at any time.
# Import is lazy-safe: if vision_runtime is not yet initialized, the lock
# is still the module-level INFERENCE_LOCK (initialized at import time).
try:
    from core.vision.vision_runtime import get_inference_lock as _get_inference_lock
    _INFERENCE_LOCK = _get_inference_lock()
except ImportError:
    import threading as _threading
    _INFERENCE_LOCK = _threading.Lock()

# BUG-8 FIX: Module-level reference to the active VisionRuntime instance.
# Populated by set_shared_vision_runtime() called from main.py after
# VisionRuntime is initialised.  When set, _call_qwen_with_history() reuses
# the latest frame that VisionRuntime already captured instead of taking a
# second independent screenshot.  This eliminates the race condition where
# the world graph and the action decision see different screen states
# (two captures separated by hundreds of ms on slow hardware).
_SHARED_VISION_RUNTIME = None


def set_shared_vision_runtime(runtime) -> None:
    """
    BUG-8 FIX: Register the active VisionRuntime so QwenOllamaAdapter can
    reuse its cached frame instead of capturing the screen independently.

    Called from main.py immediately after VisionRuntime.start():
        from adapters.qwen_ollama_adapter import set_shared_vision_runtime
        set_shared_vision_runtime(vision_runtime)
    """
    global _SHARED_VISION_RUNTIME
    _SHARED_VISION_RUNTIME = runtime

logger = logging.getLogger(__name__)
config = Config()


# ==========================================================
# THREAD-SAFE OCR READER WITH WARMUP TIMEOUT
# ==========================================================

_OCR_READER = None
_OCR_LOCK = threading.Lock()

# BUG-4 FIX: _OCR_WARMUP_TIMEOUT_SECONDS raised from 120s to 300s and made
# env-configurable.
# Root cause: First-time EasyOCR initialisation downloads ~500 MB of model
# weights. On a 3 MB/s connection that's ~167 seconds — exceeding the old 120s
# limit. EasyOCR would be marked unavailable, falling back to coordinate-only
# mode. Text-based clicks would then be silently dropped, causing stagnation
# loops (up to MAX_STAGNANT_ITERS_UI=12 wasted LLM cycles) before replanning.
# FIX: Default 300s handles <2 MB/s connections. On GPU systems with cached
# models, init takes <5s so the higher limit has no practical cost.
# Override: PROJECTZEO_OCR_WARMUP_TIMEOUT_SECONDS=<seconds>
_OCR_WARMUP_TIMEOUT_SECONDS: int = int(
    os.environ.get("PROJECTZEO_OCR_WARMUP_TIMEOUT_SECONDS", "300")
)

# H-02 / D-02 FIX: Replace bare bool _OCR_UNAVAILABLE with threading.Event.
#
# Root cause of H-02:
#   The previous code used a module-level boolean:
#       _OCR_UNAVAILABLE = False
#   _get_ocr_reader() wrote this flag under _OCR_LOCK:
#       _OCR_UNAVAILABLE = True
#   But _call_qwen_with_history() READ it OUTSIDE any lock:
#       if _OCR_UNAVAILABLE:
#   This is a data race: CPython's GIL makes bool reads atomic, but the
#   read-check-act sequence is not atomic.  On non-CPython runtimes (PyPy,
#   GraalPy, future no-GIL CPython) a torn read is possible.
#
# Fix: replace the bare bool with threading.Event.
#   - _OCR_UNAVAILABLE_EVENT.set()   replaces  _OCR_UNAVAILABLE = True
#   - _OCR_UNAVAILABLE_EVENT.clear() replaces  _OCR_UNAVAILABLE = False
#   - _OCR_UNAVAILABLE_EVENT.is_set() replaces  if _OCR_UNAVAILABLE:
#
# threading.Event.is_set() is internally synchronized (uses a Condition /
# RLock) and is safe to call from any thread without holding an external
# lock.  This eliminates the data race without requiring callers to acquire
# _OCR_LOCK for every read.
#
# A separate _OCR_LAST_FAILURE_TS float is still needed for the cooldown
# logic.  It is only written under _OCR_LOCK (inside _get_ocr_reader()), so
# reads outside the lock are safe on CPython (float assignment is atomic via
# GIL) and acceptable on other runtimes (worst case: stale read causes one
# extra retry attempt, not a correctness failure).
_OCR_UNAVAILABLE_EVENT = threading.Event()    # H-02 FIX: was `_OCR_UNAVAILABLE = False`
_OCR_LAST_FAILURE_TS: float = 0.0
_OCR_RETRY_COOLDOWN_SECONDS = 300.0


def _get_ocr_reader():
    global _OCR_READER, _OCR_LAST_FAILURE_TS

    with _OCR_LOCK:
        if _OCR_READER is not None:
            return _OCR_READER

        if _OCR_UNAVAILABLE_EVENT.is_set():          # H-02 FIX: was  if _OCR_UNAVAILABLE:
            if time.monotonic() - _OCR_LAST_FAILURE_TS < _OCR_RETRY_COOLDOWN_SECONDS:
                return None
            _OCR_UNAVAILABLE_EVENT.clear()           # H-02 FIX: was  _OCR_UNAVAILABLE = False
            logger.info("[QwenOllamaAdapter] OCR cooldown elapsed -- retrying EasyOCR init.")

        logger.warning(
            "[QwenOllamaAdapter] Initialising EasyOCR reader. "
            "This may take up to 90 seconds on CPU-only hardware."
        )
        print(
            "[QwenOllamaAdapter] Initialising EasyOCR (first-time, may be slow)...",
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
                    "EasyOCR initialisation timed out after "
                    + str(_OCR_WARMUP_TIMEOUT_SECONDS) + "s"
                )

            if "err" in error_holder:
                raise RuntimeError(
                    "EasyOCR initialisation failed: " + str(error_holder["err"])
                ) from error_holder["err"]

            _OCR_READER = result_holder["reader"]
            logger.info("[QwenOllamaAdapter] EasyOCR ready.")
            return _OCR_READER

        except Exception as exc:
            _OCR_UNAVAILABLE_EVENT.set()             # H-02 FIX: was  _OCR_UNAVAILABLE = True
            _OCR_LAST_FAILURE_TS = time.monotonic()
            logger.warning(
                "[QwenOllamaAdapter] EasyOCR unavailable: %s. "
                "Will retry after %.0fs. "
                "Falling back to coordinate-only click resolution.",
                exc,
                _OCR_RETRY_COOLDOWN_SECONDS,
            )
            print(
                "[QwenOllamaAdapter] WARNING: EasyOCR unavailable (" + str(exc) + "). "
                "Coordinate-only mode active. "
                "Retry in " + str(int(_OCR_RETRY_COOLDOWN_SECONDS)) + "s.",
                file=sys.stderr,
                flush=True,
            )
            return None


# ==========================================================
# OLLAMA RESPONSE SHIM
# ==========================================================

def _extract_response_content(response: Any) -> str:
    """
    Handles BOTH ollama >=0.2 object shape (response.message.content)
    and legacy dict shape (response['message']['content']).
    """
    if hasattr(response, "message"):
        message = response.message
        if hasattr(message, "content"):
            content = message.content
            if isinstance(content, str):
                return content
            raise RuntimeError(
                "Unexpected ollama response.message.content type: "
                + str(type(content))
            )

    if isinstance(response, dict):
        content = response.get("message", {}).get("content")
        if isinstance(content, str):
            return content
        raise RuntimeError(
            "Unexpected ollama dict response shape: " + repr(response)
        )

    raise RuntimeError(
        "Cannot extract content from ollama response type: " + str(type(response))
    )


# ==========================================================
# MESSAGE HISTORY UTILITIES
# ==========================================================

def _build_text_summary_of_message(msg: dict) -> Optional[dict]:
    """
    Convert an older message that may contain image data into a text-only
    summary suitable for inclusion in the multi-turn history sent to Ollama.

    Images from older turns are dropped to keep the context window bounded.
    The current turn always carries the live screenshot.

    Returns None if the message should be skipped entirely.
    """
    role = msg.get("role")
    if role not in ("user", "assistant", "system"):
        return None

    content = msg.get("content")

    if isinstance(content, str):
        return {"role": role, "content": content}

    if isinstance(content, list):
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
    """

    MAX_HISTORY_TURNS = 10

    _COORD_MANDATE = (
        "\n\nCRITICAL CONSTRAINT: OCR text-recognition is unavailable on this system. "
        "You MUST use coordinate-based clicks ONLY. "
        "NEVER emit a click operation with a 'text' field. "
        "EVERY click MUST include numeric x and y values in the 0.0 to 1.0 range. "
        "Correct format: {\"operation\": \"click\", \"x\": \"0.50\", \"y\": \"0.30\"}"
    )

    def __init__(self, model_name: str = "qwen2.5-vl:7b-instruct"):
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")

        self.model_name = model_name.strip()

        self._client = ollama.Client(
            timeout=httpx.Timeout(
                connect=10.0,
                read=120.0,
                write=5.0,
                pool=2.0,
            )
        )

        self._executor = ThreadPoolExecutor(max_workers=1)

    # ==========================================================
    # PUBLIC ENTRY -- adapter interface contract
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
    # CORE VISION INFERENCE
    # ==========================================================

    async def _call_qwen_with_history(
        self,
        messages: List[dict],
        objective: str,
    ) -> List[dict]:
        """
        Capture current screen, build multi-turn message list, call Ollama,
        parse JSON operations.

        H-02 FIX: _OCR_UNAVAILABLE_EVENT.is_set() is used instead of the bare
        bool read.  threading.Event.is_set() is internally synchronized and safe
        to call from any thread without an external lock.
        """
        system_content = get_system_prompt(self.model_name, objective)

        # H-02 FIX: Use .is_set() on the Event rather than reading the bare bool.
        if _OCR_UNAVAILABLE_EVENT.is_set():           # H-02 FIX: was  if _OCR_UNAVAILABLE:
            system_content = system_content + self._COORD_MANDATE
        for msg in messages:
            role = msg.get("role")
            if role == "system":
                continue
            summary = _build_text_summary_of_message(msg)
            if summary is not None:
                history_messages.append(summary)

        if len(history_messages) > self.MAX_HISTORY_TURNS:
            history_messages = history_messages[-self.MAX_HISTORY_TURNS:]

        # MF-04 FIX: Use ExitStack to guarantee cleanup of both temp files even
        # if the second NamedTemporaryFile() call itself raises (e.g. /tmp full).
        # Previously: if _jtf creation raised after _rtf was already created,
        # raw_tmp_name was set but jpeg_tmp_name remained None, so the finally
        # block could not clean up raw_tmp_name (it would have been cleaned, but
        # the symmetric risk exists for disk-full mid-creation scenarios).
        # ExitStack registers each cleanup the moment the file is created.
        raw_tmp_name = None
        jpeg_tmp_name = None
        _cleanup_stack = contextlib.ExitStack()
        try:
            # BUG-8 FIX: Attempt to reuse the VisionRuntime's already-captured
            # frame before falling back to an independent screen capture.
            #
            # Root cause of BUG-8:
            #   VisionRuntime._loop() calls mss.grab() every cycle.
            #   _call_qwen_with_history() calls capture_screen_with_cursor()
            #   independently.  On slow hardware these two captures are separated
            #   by hundreds of milliseconds — the world graph and the action
            #   decision see DIFFERENT screen states.  This is a race condition
            #   that causes the LLM to plan against stale world state.
            #
            # Fix: request the latest frame from VisionRuntime via
            #   get_latest_frame_jpeg_b64(max_age_seconds=5.0).
            # If the frame is fresh enough, skip the independent capture entirely.
            # If VisionRuntime is not registered, paused, or the frame is stale,
            # fall back to the original independent capture so we never block
            # on a missing runtime reference.
            img_base64: Optional[str] = None
            _using_vr_frame: bool = False
            _vr = _SHARED_VISION_RUNTIME
            if _vr is not None:
                try:
                    img_base64 = _vr.get_latest_frame_jpeg_b64(max_age_seconds=5.0)
                    if img_base64 is not None:
                        _using_vr_frame = True
                except Exception:
                    img_base64 = None

            # H2 FIX: When the VisionRuntime frame is used, no JPEG is written to
            # disk so screenshot_path=None and _resolve_click_coordinates() will
            # DROP every text-only click ({"operation":"click","text":"..."}).
            # Qwen2.5-VL regularly emits text clicks — each dropped click
            # increments the stagnation counter and eventually triggers REPLAN.
            #
            # Fix: inject _COORD_MANDATE into the system prompt when we know
            # we are taking the VR-frame path.  This tells the LLM to always
            # emit {"x": ..., "y": ...} coordinate clicks instead of text clicks,
            # eliminating the drop entirely.  Only injected when OCR isn't
            # already flagged unavailable (to avoid double-appending the mandate).
            if _using_vr_frame and not _OCR_UNAVAILABLE_EVENT.is_set():
                system_content = system_content + self._COORD_MANDATE

            if img_base64 is None:
                # VisionRuntime frame unavailable or stale — fall back to
                # independent capture path (original behaviour).
                _rtf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                raw_tmp_name = _rtf.name
                _rtf.close()
                _cleanup_stack.callback(
                    lambda p=raw_tmp_name: os.unlink(p) if os.path.exists(p) else None
                )

                _jtf = tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False)
                jpeg_tmp_name = _jtf.name
                _jtf.close()
                _cleanup_stack.callback(
                    lambda p=jpeg_tmp_name: os.unlink(p) if os.path.exists(p) else None
                )

                capture_screen_with_cursor(raw_tmp_name)
                compress_screenshot(raw_tmp_name, jpeg_tmp_name)

                with open(jpeg_tmp_name, "rb") as f:
                    img_base64 = base64.b64encode(f.read()).decode("utf-8")

            is_first_message = len(history_messages) == 0
            base_prompt = (
                get_user_first_message_prompt()
                if is_first_message
                else get_user_prompt()
            )

            if objective and objective.strip():
                user_prompt_text = (
                    "Current objective: " + objective.strip() + "\n\n" + base_prompt
                )
            else:
                user_prompt_text = base_prompt

            ollama_messages: List[Dict[str, Any]] = [
                {"role": "system", "content": system_content}
            ]
            ollama_messages.extend(history_messages)
            # FIX-IMG: The ollama Python library uses {"images": [base64_str]}
            # NOT the OpenAI-style content array {"content": [{"type":"image",...}]}.
            # The content-array format causes ollama to silently discard the image.
            # This was the root cause of every vision inference returning empty/wrong
            # results — the model was answering without seeing the screen.
            ollama_messages.append({
                "role": "user",
                "content": user_prompt_text + "\nReturn JSON list of operations.",
                "images": [img_base64],
            })

            loop = asyncio.get_running_loop()

            def _blocking_call():
                # BUG-3 FIX: Acquire the shared INFERENCE_LOCK before calling Ollama.
                # VisionRuntime's background _loop() also holds this lock during
                # its model calls. This prevents both from running the same vision
                # model concurrently (GPU OOM / CPU timeout cascade).
                with _INFERENCE_LOCK:
                    return self._client.chat(
                        model=self.model_name,
                        messages=ollama_messages,
                        options={"temperature": 0},
                    )

            response = await loop.run_in_executor(self._executor, _blocking_call)

            content = _extract_response_content(response)
            operations = self._parse_and_normalize_json(content)

            if not isinstance(operations, list):
                raise RuntimeError("LLM output must be a JSON array")

            operations = [
                op for op in operations
                if isinstance(op, dict) and "operation" in op
            ]

            self._resolve_click_coordinates(operations, jpeg_tmp_name)

            return operations

        finally:
            # ExitStack callbacks handle cleanup of both temp files in LIFO order.
            # Each callback was registered immediately after the corresponding
            # NamedTemporaryFile was created, so cleanup fires even if the second
            # file was never created.  When the shared VisionRuntime frame was used,
            # no temp files were created and the ExitStack is a no-op.
            _cleanup_stack.close()

    # ==========================================================
    # OCR RESOLUTION (FAIL-CLOSED)
    # ==========================================================

    def _resolve_click_coordinates(
        self,
        operations: List[dict],
        screenshot_path: Optional[str],
    ) -> None:
        """
        Resolve text-based clicks to pixel coordinates via OCR.
        Falls back to coordinate-only mode if EasyOCR is unavailable.
        Clicks with neither text nor coordinates are dropped (fail-closed).

        BUG-8 FIX: screenshot_path may be None when the shared VisionRuntime
        frame was used (no temp JPEG was written to disk).  In that case, OCR
        text resolution is unavailable and coordinate-only mode is used.
        """
        if screenshot_path is None:
            # No JPEG on disk — strip text-only clicks; keep coordinate clicks.
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
                else:
                    # BUG-4 FIX: Explicit warning when text-only click is dropped.
                    # Previously silent — stagnation counter incremented with no
                    # log entry, making the root cause invisible in logs.
                    # Now: operator sees exactly which label was unresolvable and why.
                    _label = op.get("text") or op.get("label") or "(no label)"
                    print(
                        f"[QwenOllamaAdapter] WARNING BUG-4: text-only click DROPPED "
                        f"(label={_label!r}) — shared VisionRuntime frame was used "
                        "(no disk JPEG available for OCR). This will increment the "
                        "stagnation counter. To fix: ensure OCR is available OR "
                        "LLM must emit coordinate clicks only (x/y 0.0-1.0 range).",
                        file=sys.stderr,
                        flush=True,
                    )
                    logger.warning(
                        "[QwenOllamaAdapter] BUG-4: text-only click DROPPED "
                        "(no screenshot path for OCR). label=%r. "
                        "Stagnation counter will increment.",
                        _label,
                    )
            operations.clear()
            operations.extend(filtered)
            return

        reader = _get_ocr_reader()

        if reader is None:
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
            operations.clear()
            operations.extend(filtered)
            return

        try:
            ocr_result = reader.readtext(screenshot_path)
        except Exception as exc:
            logger.warning("[QwenOllamaAdapter] OCR readtext failed: %s", exc)
            ocr_result = []

        filtered = []
        for op in operations:
            if op.get("operation") != "click":
                filtered.append(op)
                continue

            if "text" not in op:
                x = op.get("x")
                y = op.get("y")
                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    op["x"] = float(x)
                    op["y"] = float(y)
                    filtered.append(op)
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
                    else:
                        logger.warning(
                            "[QwenOllamaAdapter] OCR click DROPPED — text=%r: "
                            "resolved coords missing x/y (got %r). "
                            "Stagnation counter will increment.",
                            op.get("text", ""),
                            coords,
                        )
                else:
                    logger.warning(
                        "[QwenOllamaAdapter] OCR click DROPPED — text=%r: "
                        "get_text_coordinates returned non-dict (%r). "
                        "Stagnation counter will increment.",
                        op.get("text", ""),
                        coords,
                    )
            except Exception as _ocr_exc:
                logger.warning(
                    "[QwenOllamaAdapter] OCR click DROPPED — text=%r: "
                    "exception during coordinate resolution: %s. "
                    "Stagnation counter will increment.",
                    op.get("text", ""),
                    _ocr_exc,
                )
                continue

        operations.clear()
        operations.extend(filtered)

    # ==========================================================
    # HELPERS
    # ==========================================================

    def _parse_and_normalize_json(self, text: str) -> List[dict]:
        """
        Parse LLM text output into a list of operation dicts.

        Greedy regex captures full JSON arrays (a non-greedy pattern would
        truncate multi-operation arrays to only the first element).
        """
        text = text.strip()
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
            "No valid JSON structure found in ollama response: "
            + repr(text[:200])
        )
