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


try:
    from core.vision.vision_runtime import get_inference_lock as _get_inference_lock
    _INFERENCE_LOCK = _get_inference_lock()
except ImportError:
    import threading as _threading
    _INFERENCE_LOCK = _threading.Lock()


_SHARED_VISION_RUNTIME = None


def set_shared_vision_runtime(runtime) -> None:
    
    global _SHARED_VISION_RUNTIME
    _SHARED_VISION_RUNTIME = runtime

logger = logging.getLogger(__name__)
config = Config()




_OCR_READER = None
_OCR_LOCK = threading.Lock()

_OCR_WARMUP_TIMEOUT_SECONDS: int = int(
    os.environ.get("PROJECTZEO_OCR_WARMUP_TIMEOUT_SECONDS", "300")
)

_OCR_UNAVAILABLE_EVENT = threading.Event()
_OCR_LAST_FAILURE_TS: float = 0.0
_OCR_RETRY_COOLDOWN_SECONDS = 300.0


def _ocr_prewarm_background() -> None:
    """Background thread: initialise EasyOCR so it is ready before first click."""
    if os.environ.get("PROJECTZEO_OCR_PREWARM", "1").strip() == "0":
        return
    try:
        import easyocr as _easyocr_check  # noqa: F401
    except ImportError:
        return  # easyocr not installed — skip pre-warm silently
    try:
        print(
            "[QwenOllamaAdapter] BUG-S2: Pre-warming EasyOCR in background "
            "(first-time may take 5-10 min on CPU to download ~500 MB weights)...",
            file=sys.stderr,
            flush=True,
        )
        _get_ocr_reader()
        print(
            "[QwenOllamaAdapter] BUG-S2: EasyOCR pre-warm complete — ready for label clicks.",
            file=sys.stderr,
            flush=True,
        )
    except Exception as _prewarm_err:
        print(
            f"[QwenOllamaAdapter] BUG-S2: EasyOCR pre-warm failed: {_prewarm_err}. "
            "Coordinate-only mode will be used.",
            file=sys.stderr,
            flush=True,
        )

_OCR_PREWARM_THREAD = threading.Thread(
    target=_ocr_prewarm_background,
    name="ocr-prewarm",
    daemon=True,
)
_OCR_PREWARM_THREAD.start()


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
        
        system_content = get_system_prompt(self.model_name, objective)

        
        history_messages: List[Dict[str, Any]] = []

        

        for msg in messages:
            role = msg.get("role")
            if role == "system":
                continue
            summary = _build_text_summary_of_message(msg)
            if summary is not None:
                history_messages.append(summary)

        if len(history_messages) > self.MAX_HISTORY_TURNS:
            history_messages = history_messages[-self.MAX_HISTORY_TURNS:]

        
        raw_tmp_name = None
        jpeg_tmp_name = None
        _cleanup_stack = contextlib.ExitStack()
        try:
            
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

            
            _needs_coord_mandate = (
                _OCR_UNAVAILABLE_EVENT.is_set()  # OCR explicitly unavailable
                or _using_vr_frame               # VR frame path: no disk JPEG for OCR
            )
            if _needs_coord_mandate:
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
            
            _cleanup_stack.close()

    # ==========================================================
    # OCR RESOLUTION (FAIL-CLOSED)
    # ==========================================================

    def _resolve_click_coordinates(
        self,
        operations: List[dict],
        screenshot_path: Optional[str],
    ) -> None:
        
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
            # BUG-S6 FIX: removed duplicate type annotation (shadowed first declaration)
            filtered = []
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
