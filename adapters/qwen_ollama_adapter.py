"""
adapters/qwen_ollama_adapter.py
================================
PATCH AUDIT FIXES:

  ⚠️  §1.3: ollama.Client response accessed via .get('message',{}).get('content')
            — ollama Python ≥0.2 returns an object (response.message.content),
            not a dict.  This raised AttributeError on modern ollama versions.
            FIX: Use attribute access with dict fallback shim.

  ⚠️  §1.3: EasyOCR reader initialised lazily with no timeout or progress signal.
            On a cold GPU-less machine first init can take 30-90 seconds,
            silently blocking the first task cycle.
            FIX: Emit a startup log, enforce a 120s init timeout via threading.

  ✅  All existing correct behaviours preserved:
        - Temperature=0 (deterministic)
        - NamedTemporaryFile auto-delete (no disk artefacts)
        - _parse_and_normalize_json markdown fence stripping + regex fallback
        - _resolve_click_coordinates fail-closed (drops unresolvable clicks)
        - Single-worker ThreadPoolExecutor (bounded)
"""

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


def _get_ocr_reader():
    """
    PATCH §1.3: EasyOCR initialisation can block 30-90s on CPU-only machines.
    This function now:
      - Logs a warning before blocking so users know what's happening.
      - Enforces a hard 120s timeout via a sentinel thread.
      - Double-checked locking for thread safety.
    """
    global _OCR_READER
    if _OCR_READER is not None:
        return _OCR_READER

    with _OCR_LOCK:
        if _OCR_READER is not None:
            return _OCR_READER

        logger.warning(
            "[QwenOllamaAdapter] Initialising EasyOCR reader. "
            "This may take up to 90 seconds on CPU-only hardware …"
        )
        print(
            "[QwenOllamaAdapter] Initialising EasyOCR (first-time, may be slow) …",
            file=sys.stderr,
            flush=True,
        )

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


# ==========================================================
# OLLAMA RESPONSE SHIM
# ==========================================================

def _extract_response_content(response: Any) -> str:
    """
    PATCH §1.3 CRITICAL: ollama Python library ≥0.2 returns a typed object
    (response.message.content), NOT a dict.  Earlier code used:
        response.get('message', {}).get('content')
    which raises AttributeError on modern ollama.

    This shim handles BOTH the legacy dict shape and the modern object shape,
    so the adapter works across ollama library versions.
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

        self._client = ollama.Client(
            timeout=httpx.Timeout(
                connect=5.0,
                read=25.0,
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

            user_prompt = (
                get_user_first_message_prompt()
                if len(local_msgs) == 1
                else get_user_prompt()
            )

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

            # PATCH §1.3: use shim that handles both object and dict response shapes
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

        reader = _get_ocr_reader()
        ocr_result = reader.readtext(screenshot_path)

        filtered_ops: List[dict] = []

        for op in operations:

            if op.get("operation") != "click":
                filtered_ops.append(op)
                continue

            if "text" not in op:
                # click with no text target → drop (fail-closed)
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

        # Fallback: extract first JSON structure via regex
        match = re.search(r"(\{.*?\}|\[.*?\])", text, re.DOTALL)
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
