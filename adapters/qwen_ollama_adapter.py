import base64
import json
import os
import re
import uuid
from typing import List, Tuple, Optional

import ollama
from PIL import Image

from operate.config import Config
from operate.models.prompts import (
    get_system_prompt,
    get_user_first_message_prompt,
    get_user_prompt,
)
from operate.utils.ocr import get_text_coordinates, get_text_element
from operate.utils.screenshot import capture_screen_with_cursor, compress_screenshot


config = Config()

# Persistent OCR reader (avoids reload overhead)
_OCR_READER = None


class QwenOllamaAdapter:
    """
    Local-only Qwen-VL adapter.
    Replaces multi-model apis.py usage.
    """

    def __init__(self, model_name: str = "qwen2.5-vl:7b-instruct"):
        self.model_name = model_name
        self._client = ollama.Client()

    # ==========================================================
    # PUBLIC ENTRY (matches apis.py contract)
    # ==========================================================

    async def get_next_action(
        self,
        messages: List[dict],
        objective: str,
        session_id: Optional[str] = None,
    ) -> Tuple[Optional[List[dict]], Optional[Exception]]:
        """
        Returns: (operation_list, error_object)
        """

        local_messages = list(messages)

        try:
            operations = await self._call_qwen_with_ocr(
                local_messages,
                objective,
            )
            return operations, None
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

        # Inject/confirm system prompt (pure copy)
        local_msgs = self._confirm_system_prompt(messages, objective)

        raw_screenshot = self._prepare_unique_screenshot()
        jpeg_screenshot = raw_screenshot.replace(".png", ".jpeg")

        try:
            compress_screenshot(raw_screenshot, jpeg_screenshot)

            with open(jpeg_screenshot, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode("utf-8")

            user_prompt = (
                get_user_first_message_prompt()
                if len(local_msgs) == 1
                else get_user_prompt()
            )

            response = self._client.chat(
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

            operations = self._parse_and_normalize_json(
                response["message"]["content"]
            )

            # OCR coordinate resolution
            self._resolve_click_coordinates(
                operations,
                jpeg_screenshot,
            )

            return operations

        finally:
            for p in (raw_screenshot, jpeg_screenshot):
                if os.path.exists(p):
                    os.remove(p)

    # ==========================================================
    # OCR RESOLUTION
    # ==========================================================

    def _resolve_click_coordinates(
        self,
        operations: List[dict],
        screenshot_path: str,
    ):
        global _OCR_READER

        import easyocr

        if _OCR_READER is None:
            _OCR_READER = easyocr.Reader(["en"])

        ocr_result = _OCR_READER.readtext(screenshot_path)

        for op in operations:
            if op.get("operation") == "click" and "text" in op:
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
                op["x"] = coords["x"]
                op["y"] = coords["y"]

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
            local[0] = system_message
        else:
            local.insert(0, system_message)

        return local

    def _prepare_unique_screenshot(self) -> str:
        screenshots_dir = "screenshots"
        os.makedirs(screenshots_dir, exist_ok=True)

        filename = f"screenshot_{uuid.uuid4().hex}.png"
        path = os.path.join(screenshots_dir, filename)

        capture_screen_with_cursor(path)
        return path

    def _parse_and_normalize_json(self, text: str) -> List[dict]:
        text = text.strip()

        # Strip markdown fences
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE).strip()

        decoder = json.JSONDecoder()

        for start_char in ("{", "["):
            idx = text.find(start_char)
            if idx != -1:
                try:
                    data, _ = decoder.raw_decode(text[idx:])
                    if isinstance(data, dict):
                        return [data]
                    if isinstance(data, list):
                        return data
                    raise RuntimeError(
                        f"Unexpected JSON structure: {type(data)}"
                    )
                except json.JSONDecodeError:
                    continue

        raise RuntimeError(
            f"No valid JSON structure found. Sample: {text[:80]}"
)
