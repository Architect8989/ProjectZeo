import base64
import json
import os
import re
import uuid
import threading
import asyncio
from typing import List, Tuple, Optional

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
    Non-blocking + fail-closed hardened version.
    """

    MAX_SCREENSHOT_FILES = 200

    def __init__(self, model_name: str = "qwen2.5-vl:7b-instruct"):
        self.model_name = model_name

        self._client = ollama.Client(
            timeout=httpx.Timeout(
                connect=5.0,
                read=25.0,
                write=5.0,
                pool=2.0,
            )
        )

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
    # CORE EXECUTION
    # ==========================================================

    async def _call_qwen_with_ocr(
        self,
        messages: List[dict],
        objective: str,
    ) -> List[dict]:

        local_msgs = self._confirm_system_prompt(messages, objective)

        raw_screenshot = None
        jpeg_screenshot = None

        try:
            raw_screenshot = self._prepare_unique_screenshot()
            jpeg_screenshot = raw_screenshot.replace(".png", ".jpeg")

            compress_screenshot(raw_screenshot, jpeg_screenshot)

            with open(jpeg_screenshot, "rb") as f:
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

            response = await loop.run_in_executor(None, _blocking_call)

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
                jpeg_screenshot,
            )

            return operations

        finally:
            for p in (raw_screenshot, jpeg_screenshot):
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except OSError:
                    pass

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
                        op["x"] = x
                        op["y"] = y
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

    def _prepare_unique_screenshot(self) -> str:

        screenshots_dir = os.path.abspath("screenshots")
        os.makedirs(screenshots_dir, exist_ok=True)

        try:
            existing = [
                f for f in os.listdir(screenshots_dir)
                if os.path.isfile(os.path.join(screenshots_dir, f))
            ]

            if len(existing) > self.MAX_SCREENSHOT_FILES:
                # Deterministic oldest-first eviction
                existing_sorted = sorted(
                    existing,
                    key=lambda f: os.path.getmtime(
                        os.path.join(screenshots_dir, f)
                    )
                )

                to_delete = existing_sorted[
                    : len(existing_sorted) - self.MAX_SCREENSHOT_FILES
                ]

                for f in to_delete:
                    try:
                        os.remove(os.path.join(screenshots_dir, f))
                    except Exception:
                        pass

        except Exception:
            # Fail-closed: never block screenshot capture
            pass

        filename = f"screenshot_{uuid.uuid4().hex}.png"
        path = os.path.join(screenshots_dir, filename)

        capture_screen_with_cursor(path)
        return path

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
