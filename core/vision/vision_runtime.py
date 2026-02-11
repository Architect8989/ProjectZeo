from __future__ import annotations

import time
import threading
from typing import Dict, Any, Optional, List
import base64
import io
import json
import copy

from PIL import Image, ImageGrab
import ollama


class VisionUnavailableError(RuntimeError):
    pass


class VisionDegradedError(RuntimeError):
    pass


QWEN_MODEL = "qwen2.5-vl:7b-instruct"

MAX_LATENCY_SECONDS = 1.5
MAX_FRAME_BYTES = 4 * 1024 * 1024
MAX_ELEMENTS = 128
MAX_CONSECUTIVE_FAILURES = 5
CAPTURE_INTERVAL_SECONDS = 0.5


class VisionRuntime:

    def __init__(self):
        self._lock = threading.RLock()

        self._last_output: Optional[Dict[str, Any]] = None
        self._last_frame_ts: Optional[float] = None

        self._consecutive_failures: int = 0
        self._healthy: bool = True
        self._running: bool = False

        self._thread: Optional[threading.Thread] = None

    # -------------------------------------------------
    # LIFECYCLE
    # -------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._running:
                return

            self._running = True
            self._healthy = True
            self._consecutive_failures = 0

            self._thread = threading.Thread(
                target=self._loop,
                name="VisionRuntime",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._running = False

    def is_healthy(self) -> bool:
        with self._lock:
            return self._healthy

    def get_latest(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._last_output is None:
                return None
            return copy.deepcopy(self._last_output)

    # -------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------

    def _loop(self) -> None:
        while True:
            with self._lock:
                if not self._running:
                    return

            try:
                output = self._process_frame_internal()

                with self._lock:
                    if not self._running:
                        return

                    self._last_output = output
                    self._last_frame_ts = output["frame_ts"]
                    self._consecutive_failures = 0
                    self._healthy = True  # ✅ RECOVERY ON SUCCESS

            except Exception:
                with self._lock:
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        self._healthy = False

            time.sleep(CAPTURE_INTERVAL_SECONDS)

    # -------------------------------------------------
    # FRAME PROCESSING
    # -------------------------------------------------

    def _process_frame_internal(self) -> Dict[str, Any]:
        start = time.monotonic()

        frame_ts = time.monotonic()

        image = self._capture_frame()
        encoded = self._encode_image(image)

        perception = self._call_qwen(encoded)

        output = self._normalize_output(
            perception=perception,
            frame_ts=frame_ts,
        )

        latency = time.monotonic() - start
        if latency > MAX_LATENCY_SECONDS:
            raise VisionDegradedError(
                f"Vision latency exceeded: {latency:.2f}s"
            )

        return output

    # -------------------------------------------------
    # FRAME CAPTURE
    # -------------------------------------------------

    def _capture_frame(self) -> Image.Image:
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
        prompt = (
            "You are a visual perception system.\n"
            "You do NOT infer intent.\n"
            "You do NOT suggest actions.\n"
            "Return ONLY valid JSON.\n"
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

        elements = perception.get("elements")
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
            parts = raw.split("```")
            if len(parts) >= 3:
                raw = parts[1]
            else:
                raw = parts[-1]

        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise VisionDegradedError("Vision output not JSON object")
            return parsed
        except Exception as e:
            raise VisionDegradedError(
                f"Invalid JSON from vision model: {e}"
    )
