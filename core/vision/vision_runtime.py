from __future__ import annotations

import time
import threading
from typing import Dict, Any, Optional, List
import base64
import io

from PIL import Image, ImageGrab
import ollama

# -------------------------------------------------
# ERRORS
# -------------------------------------------------


class VisionUnavailableError(RuntimeError):
    """Vision runtime cannot observe screen."""


class VisionDegradedError(RuntimeError):
    """Vision runtime observing unstable or invalid frames."""


# -------------------------------------------------
# CONFIG
# -------------------------------------------------


QWEN_MODEL = "qwen2.5-vl:7b-instruct"

MAX_LATENCY_SECONDS = 1.5
MAX_FRAME_BYTES = 4 * 1024 * 1024  # 4MB hard ceiling
MAX_ELEMENTS = 128

# -------------------------------------------------
# VISION RUNTIME
# -------------------------------------------------


class VisionRuntime:
    """
    Continuous local vision processor.

    DESIGN PRINCIPLES:
    - Observer-safe
    - Planner-agnostic
    - Executor-blind
    - Deterministic schema
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._last_frame_ts: Optional[float] = None
        self._consecutive_failures: int = 0
        self._healthy: bool = True

    # -------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------

    def process_frame(self) -> Dict[str, Any]:
        """
        Capture framebuffer and return structured perception.

        This is the ONLY entry point for vision.
        """
        start = time.monotonic()

        with self._lock:
            if not self._healthy:
                raise VisionUnavailableError("Vision runtime unhealthy")

            frame_ts = time.monotonic()

            try:
                image = self._capture_frame()
                encoded = self._encode_image(image)

                perception = self._call_qwen(encoded)

                output = self._normalize_output(
                    perception=perception,
                    frame_ts=frame_ts,
                )

                self._last_frame_ts = frame_ts
                self._consecutive_failures = 0

            except Exception as e:
                self._consecutive_failures += 1
                if self._consecutive_failures >= 5:
                    self._healthy = False
                raise VisionDegradedError(f"Vision failure: {e}") from e

            latency = time.monotonic() - start
            if latency > MAX_LATENCY_SECONDS:
                raise VisionDegradedError(
                    f"Vision latency exceeded: {latency:.2f}s"
                )

            return output

    def is_healthy(self) -> bool:
        return self._healthy

    # -------------------------------------------------
    # FRAME CAPTURE
    # -------------------------------------------------

    def _capture_frame(self) -> Image.Image:
        """
        Capture full screen including cursor.
        """
        try:
            img = ImageGrab.grab(all_screens=True)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        except Exception as e:
            raise VisionUnavailableError(
                f"Framebuffer capture failed: {e}"
            )

    def _encode_image(self, img: Image.Image) -> str:
        """
        Encode image to base64 JPEG with size enforcement.
        """
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        data = buf.getvalue()

        if len(data) > MAX_FRAME_BYTES:
            raise VisionDegradedError(
                f"Frame too large: {len(data)} bytes"
            )

        return base64.b64encode(data).decode("utf-8")

    # -------------------------------------------------
    # MODEL CALL
    # -------------------------------------------------

    def _call_qwen(self, image_b64: str) -> Dict[str, Any]:
        """
        Call local Qwen vision model.
        """
        prompt = (
            "You are a vision perception system.\n"
            "Return ONLY valid JSON.\n\n"
            "Schema:\n"
            "{\n"
            '  "elements": [\n'
            "    {\"type\": \"button|text|input|menu|icon\", "
            "\"text\": string, "
            "\"x\": number, "
            "\"y\": number}\n"
            "  ],\n"
            '  "dialogs": [],\n'
            '  "apps": [],\n'
            '  "focused_app": string|null\n'
            "}\n\n"
            "Rules:\n"
            "- Coordinates must be normalized (0..1)\n"
            "- Max 128 elements\n"
            "- No hallucinated text\n"
        )

        response = ollama.chat(
            model=QWEN_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": image_b64},
                    ],
                }
            ],
        )

        content = response["message"]["content"]

        return self._parse_json(content)

    # -------------------------------------------------
    # NORMALIZATION
    # -------------------------------------------------

    def _normalize_output(
        self,
        *,
        perception: Dict[str, Any],
        frame_ts: float,
    ) -> Dict[str, Any]:
        """
        Enforce strict schema and bounds.
        """
        elements = perception.get("elements") or []
        if not isinstance(elements, list):
            elements = []

        normalized_elements: List[Dict[str, Any]] = []

        for el in elements[:MAX_ELEMENTS]:
            if not isinstance(el, dict):
                continue

            x = el.get("x")
            y = el.get("y")

            if not self._valid_coord(x) or not self._valid_coord(y):
                continue

            normalized_elements.append(
                {
                    "type": str(el.get("type", "unknown")),
                    "text": str(el.get("text", "")).strip(),
                    "x": float(x),
                    "y": float(y),
                }
            )

        return {
            "available": True,
            "frame_ts": frame_ts,
            "elements": normalized_elements,
            "dialogs": perception.get("dialogs") or [],
            "apps": perception.get("apps") or [],
            "focused_app": perception.get("focused_app"),
        }

    # -------------------------------------------------
    # UTILS
    # -------------------------------------------------

    def _valid_coord(self, v: Any) -> bool:
        return isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0

    def _parse_json(self, raw: str) -> Dict[str, Any]:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
        try:
            import json
            return json.loads(raw)
        except Exception as e:
            raise VisionDegradedError(
                f"Invalid JSON from vision model: {e}"
  )
