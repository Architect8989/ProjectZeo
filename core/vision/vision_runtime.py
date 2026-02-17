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
import httpx


class VisionUnavailableError(RuntimeError):
    pass


class VisionDegradedError(RuntimeError):
    pass


# ==================================================
# CONFIG
# ==================================================

MAX_ALLOWED_LATENCY_SECONDS = 3.0          # health threshold
NETWORK_CONNECT_TIMEOUT = 5.0
NETWORK_READ_TIMEOUT = 25.0

MAX_FRAME_BYTES = 4 * 1024 * 1024
MAX_ELEMENTS = 128
MAX_CONSECUTIVE_FAILURES = 5
CAPTURE_INTERVAL_SECONDS = 0.5


# ==================================================
# VISION RUNTIME
# ==================================================

class VisionRuntime:

    def __init__(self, model_name: str):
        if not isinstance(model_name, str) or not model_name.strip():
            raise VisionUnavailableError("Vision model_name must be non-empty")

        self._model_name = model_name.strip()

        self._lock = threading.Lock()
        self._last_output: Optional[Dict[str, Any]] = None
        self._last_frame_ts: Optional[float] = None
        self._consecutive_failures: int = 0
        self._healthy: bool = True
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None

        # Real granular network timeout
        self._ollama_client = ollama.Client(
            timeout=httpx.Timeout(
                connect=NETWORK_CONNECT_TIMEOUT,
                read=NETWORK_READ_TIMEOUT,
                write=5.0,
                pool=5.0,
            )
        )

    # ==================================================
    # LIFECYCLE
    # ==================================================

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

        if self._thread:
            self._thread.join(timeout=2.0)

    def is_healthy(self) -> bool:
        with self._lock:
            return self._healthy

    def get_latest(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._last_output)

    # ==================================================
    # MAIN LOOP
    # ==================================================

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
                    self._healthy = True

            except Exception:
                with self._lock:
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        self._healthy = False

            time.sleep(CAPTURE_INTERVAL_SECONDS)

    # ==================================================
    # FRAME PROCESSING
    # ==================================================

    def _process_frame_internal(self) -> Dict[str, Any]:
        start = time.monotonic()
        frame_ts = time.monotonic()

        image = self._capture_frame()
        encoded = self._encode_image(image)
        perception = self._call_model(encoded)

        latency = time.monotonic() - start

        # Health threshold (NOT network timeout)
        if latency > MAX_ALLOWED_LATENCY_SECONDS:
            raise VisionDegradedError(
                f"Vision latency exceeded: {latency:.2f}s"
            )

        return self._normalize_output(
            perception=perception,
            frame_ts=frame_ts,
        )

    # ==================================================
    # FRAME CAPTURE
    # ==================================================

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

    # ==================================================
    # MODEL CALL (NO HARDCODE)
    # ==================================================

    def _call_model(self, image_b64: str) -> Dict[str, Any]:

        prompt = (
            "Return ONLY valid JSON in this schema:\n"
            "{\n"
            '  "elements": [{ "type": string, "text": string, '
            '"x": 0.0-1.0, "y": 0.0-1.0, '
            '"state": string|null }],\n'
            '  "dialogs": [],\n'
            '  "apps": [],\n'
            '  "focused_app": string|null\n'
            "}\n"
            "No explanation. No markdown."
        )

        try:
            response = self._ollama_client.chat(
                model=self._model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image", "image": image_b64},
                        ],
                    }
                ],
                options={
                    "temperature": 0,
                },
            )
        except Exception as e:
            raise VisionUnavailableError(
                f"Vision model call failed: {e}"
            )

        if not isinstance(response, dict):
            raise VisionDegradedError("Invalid vision response type")

        content = (
            response.get("message", {})
            .get("content")
        )

        if not isinstance(content, str):
            raise VisionDegradedError(
                "Unexpected response structure"
            )

        return self._parse_json(content)

    # ==================================================
    # NORMALIZATION
    # ==================================================

    def _normalize_output(
        self,
        *,
        perception: Dict[str, Any],
        frame_ts: float,
    ) -> Dict[str, Any]:

        if not isinstance(perception, dict):
            raise VisionDegradedError("Perception not object")

        elements = perception.get("elements", [])
        if not isinstance(elements, list):
            raise VisionDegradedError("Invalid elements")

        normalized_elements: List[Dict[str, Any]] = []

        for el in elements[:MAX_ELEMENTS]:
            if not isinstance(el, dict):
                continue

            x = el.get("x")
            y = el.get("y")

            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                continue

            x = float(min(max(float(x), 0.0), 1.0))
            y = float(min(max(float(y), 0.0), 1.0))

            normalized_elements.append(
                {
                    "type": str(el.get("type", "unknown")).strip(),
                    "text": str(el.get("text", "")).strip(),
                    "x": x,
                    "y": y,
                    "interactable": self._is_interactable(el),
                    "state": el.get("state"),
                }
            )

        focused_app = perception.get("focused_app")
        if focused_app is not None and not isinstance(focused_app, str):
            focused_app = None

        with self._lock:
            if self._last_frame_ts is not None:
                if frame_ts <= self._last_frame_ts:
                    frame_ts = self._last_frame_ts + 1e-6

        return {
            "available": True,
            "frame_ts": frame_ts,
            "elements": normalized_elements,
            "dialogs": perception.get("dialogs", []) if isinstance(perception.get("dialogs"), list) else [],
            "apps": perception.get("apps", []) if isinstance(perception.get("apps"), list) else [],
            "focused_app": focused_app,
        }

    # ==================================================
    # UTILITIES
    # ==================================================

    def _is_interactable(self, element: Dict[str, Any]) -> bool:
        element_type = str(element.get("type", "")).lower()

        interactive_types = {
            "button", "link", "input", "checkbox",
            "radio", "select", "textarea",
            "slider", "tab", "menu",
            "menuitem", "switch", "combobox",
        }

        if element_type in interactive_types:
            return True

        if element.get("state") is not None:
            return True

        return False

    def _parse_json(self, raw: str) -> Dict[str, Any]:
        raw = raw.strip()

        # Remove fenced blocks safely
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) >= 3 else parts[-1]

        try:
            parsed = json.loads(raw)
        except Exception as e:
            raise VisionDegradedError(
                f"Invalid JSON from vision model: {e}"
            )

        if not isinstance(parsed, dict):
            raise VisionDegradedError(
                "Vision output must be JSON object"
            )

        return parsed
