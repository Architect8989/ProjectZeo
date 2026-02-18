import base64
import json
import re
import threading
import asyncio
import tempfile
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

import ollama
import httpx

from operate.config import Config
from operate.models.prompts import (
    get_system_prompt,
    get_user_first_message_prompt,
    get_user_prompt,
)
from operate.utils.ocr import get_text_coordinates, get_text_element
from operate.utils.screenshot import capture_screen_with_cursor, compress_screenshot


config = Config()

# ==========================================================
# THREAD-SAFE OCR READER
# ==========================================================

_OCR_READER = None
_OCR_LOCK = threading.Lock()


def _get_ocr_reader():
    global _OCR_READER
    if _OCR_READER is None:
        with _OCR_LOCK:
            if _OCR_READER is None:
                import easyocr
                _OCR_READER = easyocr.Reader(["en"])
    return _OCR_READER


# ==========================================================
# ADAPTER
# ==========================================================

class QwenOllamaAdapter:
    """
    Local-only Qwen-VL adapter.
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

        # Bounded executor (fixes unbounded thread spawn)
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
        except Exception as e:
            return None, e

    # ==========================================================
    # CORE EXECUTION (NO PERSISTENT FILES)
    # ==========================================================

    async def _call_qwen_with_ocr(
        self,
        messages: List[dict],
        objective: str,
    ) -> List[dict]:

        local_msgs = self._confirm_system_prompt(messages, objective)

        # Use secure temporary files auto-deleted
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as raw_tmp, \
             tempfile.NamedTemporaryFile(suffix=".jpeg", delete=True) as jpeg_tmp:

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
                                    "text": f"{user_prompt}\nReturn JSON list of operations.",
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

            content = response.get("message", {}).get("content")
            if not isinstance(content, str):
                raise RuntimeError("Unexpected Ollama response shape")

            operations = self._parse_and_normalize_json(content)

            if not isinstance(operations, list):
                raise RuntimeError("LLM output must be list")

            operations = [
                op for op in operations
                if isinstance(op, dict) and "operation" in op
            ]

            self._resolve_click_coordinates(
                operations,
                jpeg_tmp.name,
            )

            return operations

    # ==========================================================
    # OCR RESOLUTION (FAIL-CLOSED)
    # ==========================================================

    def _resolve_click_coordinates(
        self,
        operations: List[dict],
        screenshot_path: str,
    ):

        reader = _get_ocr_reader()
        ocr_result = reader.readtext(screenshot_path)

        filtered_ops = []

        for op in operations:

            if op.get("operation") != "click":
                filtered_ops.append(op)
                continue

            if "text" not in op:
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

        system_message = {
            "role": "system",
            "content": get_system_prompt(self.model_name, objective),
        }

        if local and local[0].get("role") == "system":
            local[0]["content"] = system_message["content"]
        else:
            local.insert(0, system_message)

        return local

    def _parse_and_normalize_json(self, text: str) -> List[dict]:

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

        raise RuntimeError("No valid JSON structure found")
