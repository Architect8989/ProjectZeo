from __future__ import annotations

import sys
import time
import threading
import re
from typing import Dict, Any, Optional, List
import base64
import io
import json
import copy
import os
import concurrent.futures

from PIL import Image
import ollama
import httpx

from core.mode_controller import VisionUnavailableError


class VisionDegradedError(RuntimeError):
    pass


def _shutdown_executor_compat(executor, wait: bool = False) -> None:
    if sys.version_info >= (3, 9):
        executor.shutdown(wait=wait, cancel_futures=True)
    else:
        executor.shutdown(wait=wait)


MAX_ALLOWED_LATENCY_SECONDS = 180.0
NETWORK_CONNECT_TIMEOUT = 5.0
NETWORK_READ_TIMEOUT = 180.0
MODEL_CALL_TIMEOUT_SECONDS = 200.0

MAX_FRAME_BYTES = 4 * 1024 * 1024
MAX_ELEMENTS = 128
MAX_CONSECUTIVE_FAILURES = 5

CAPTURE_INTERVAL_SECONDS = 0.5
FRAME_SKIP_THRESHOLD_MULTIPLIER = 5.0

INFERENCE_LOCK = threading.Lock()


def get_inference_lock() -> threading.Lock:
    return INFERENCE_LOCK


_INJECTION_TYPE_RE = re.compile(
    r"injection[_\-]?attempt|prompt[_\-]?injection|ignore[_\-]?previous",
    re.IGNORECASE,
)

_INJECTION_TEXT_MARKERS: List[str] = [
    "ignore previous instructions",
    "ignore all previous",
    "disregard instructions",
    "new instruction",
    "system prompt",
    "you are now",
    "act as",
    "jailbreak",
]


def _element_is_injection(element: Dict[str, Any]) -> bool:
    if not isinstance(element, dict):
        return False

    elem_type = str(element.get("type") or "")
    if _INJECTION_TYPE_RE.search(elem_type):
        return True

    elem_text = str(element.get("text") or "").lower()
    for marker in _INJECTION_TEXT_MARKERS:
        if marker in elem_text:
            return True

    for val in element.values():
        if isinstance(val, str) and "injection" in val.lower():
            return True

    return False


def _filter_injection_elements(
    elements: List[Dict[str, Any]],
    *,
    source: str = "unknown",
) -> List[Dict[str, Any]]:
    clean: List[Dict[str, Any]] = []
    blocked = 0
    for elem in elements:
        if _element_is_injection(elem):
            blocked += 1
            print(
                f"[VisionRuntime] M6 INJECTION FILTER: blocked element "
                f"type={elem.get('type')!r} text={str(elem.get('text',''))[:60]!r} "
                f"source={source}",
                file=sys.stderr,
            )
        else:
            clean.append(elem)
    if blocked:
        print(
            f"[VisionRuntime] M6: {blocked} injection element(s) removed "
            f"from VL output before WorldGraph ingestion.",
            file=sys.stderr,
        )
    return clean


def _extract_vision_content(response: Any) -> str:
    if hasattr(response, "message"):
        message = response.message
        if hasattr(message, "content"):
            content = message.content
            if isinstance(content, str):
                return content
            raise VisionDegradedError(
                f"Unexpected ollama response.message.content type: {type(content)}"
            )

    if isinstance(response, dict):
        content = response.get("message", {}).get("content")
        if isinstance(content, str):
            return content
        raise VisionDegradedError(
            f"Unexpected ollama dict response shape: {response!r}"
        )

    raise VisionDegradedError(
        f"Cannot extract content from ollama response type: {type(response)}"
    )


class VisionRuntime:

    def __init__(self, model_name: str):
        if not isinstance(model_name, str) or not model_name.strip():
            raise VisionUnavailableError("Vision model_name must be non-empty")

        self._model_name = model_name.strip()

        self._lock = threading.Lock()
        self._last_output: Optional[Dict[str, Any]] = None
        self._last_good_output: Optional[Dict[str, Any]] = None
        self._last_frame_ts: Optional[float] = None
        self._last_raw_image: Optional[Any] = None
        self._consecutive_failures: int = 0
        self._healthy: bool = True
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None

        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        self._ollama_client = ollama.Client(
            timeout=httpx.Timeout(
                connect=NETWORK_CONNECT_TIMEOUT,
                read=NETWORK_READ_TIMEOUT,
                write=5.0,
                pool=5.0,
            )
        )

        self._validate_display_environment()
        self._check_multi_monitor()

    def _validate_display_environment(self) -> None:
        if os.name != "nt":
            if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
                raise VisionUnavailableError(
                    "No display environment detected (headless mode unsupported)"
                )

    def _check_multi_monitor(self) -> None:
        try:
            import mss as _mss
            with _mss.mss() as sct:
                n_monitors = len(sct.monitors) - 1
                if n_monitors > 1:
                    print(
                        f"[VisionRuntime] WARNING: {n_monitors} monitors detected. "
                        "Only PRIMARY monitor (mss.monitors[1]) is captured.",
                        file=sys.stderr,
                    )
                self._primary_monitor = dict(sct.monitors[1]) if len(sct.monitors) > 1 else None
        except Exception:
            self._primary_monitor = None

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._healthy = True
            self._consecutive_failures = 0
            self._thread = threading.Thread(
                target=self._loop, name="VisionRuntime", daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._last_raw_image = None
        if self._thread:
            self._thread.join(timeout=3.0)
        _shutdown_executor_compat(self._executor, wait=False)

    def is_healthy(self) -> bool:
        with self._lock:
            return self._healthy

    def reset_health(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                self._consecutive_failures = 0
                self._healthy = True

    def get_latest(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._last_output)

    def get_latest_frame_jpeg_b64(self, max_age_seconds: float = 5.0) -> Optional[str]:
        with self._lock:
            frame_ts = self._last_frame_ts
            raw = self._last_raw_image

        if frame_ts is None or raw is None:
            return None

        age = time.monotonic() - frame_ts
        if age > max_age_seconds:
            return None

        try:
            buf = io.BytesIO()
            if hasattr(raw, "save"):
                raw.save(buf, format="JPEG", quality=70)
            elif isinstance(raw, (bytes, bytearray)):
                return base64.b64encode(raw).decode()
            else:
                return None
            return base64.b64encode(buf.getvalue()).decode()
        except Exception:
            return None

    def _loop(self) -> None:
        while True:
            with self._lock:
                if not self._running:
                    break

            loop_start = time.monotonic()

            try:
                frame_b64, raw_image = self._capture_frame()

                if frame_b64 is None:
                    time.sleep(CAPTURE_INTERVAL_SECONDS)
                    continue

                with self._lock:
                    self._last_raw_image = raw_image

                inference_start = time.monotonic()

                with INFERENCE_LOCK:
                    raw_output = self._call_model(frame_b64)

                inference_elapsed = time.monotonic() - inference_start

                if inference_elapsed > MAX_ALLOWED_LATENCY_SECONDS:
                    print(
                        f"[VisionRuntime] WARNING: inference latency "
                        f"{inference_elapsed:.1f}s exceeds limit {MAX_ALLOWED_LATENCY_SECONDS}s.",
                        file=sys.stderr,
                    )

                normalized = self._normalize_output(raw_output)

                if isinstance(normalized, dict):
                    elements_raw = normalized.get("elements") or normalized.get("entities") or []
                    if isinstance(elements_raw, list):
                        frame_label = str(normalized.get("frame_ts", "unknown"))
                        filtered = _filter_injection_elements(
                            elements_raw, source=frame_label
                        )
                        normalized["elements"] = filtered
                        normalized["entities"] = filtered

                    final_entities = normalized.get("entities") or []
                    if not final_entities and self._last_good_output is not None:
                        print(
                            "[VisionRuntime] M6: VL parse returned 0 entities after "
                            "injection filter — retaining last known-good output.",
                            file=sys.stderr,
                        )
                        merged = dict(self._last_good_output)
                        merged["frame_ts"] = normalized.get("frame_ts", merged.get("frame_ts"))
                        merged["focused_app"] = normalized.get("focused_app", merged.get("focused_app"))
                        normalized = merged
                    elif final_entities:
                        self._last_good_output = copy.deepcopy(normalized)

                with self._lock:
                    self._last_output = normalized
                    self._last_frame_ts = time.monotonic()
                    self._consecutive_failures = 0
                    self._healthy = True

            except Exception as exc:
                with self._lock:
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        self._healthy = False
                print(
                    f"[VisionRuntime] Error in loop: {exc} "
                    f"(consecutive_failures={self._consecutive_failures})",
                    file=sys.stderr,
                )

            elapsed = time.monotonic() - loop_start
            sleep_time = CAPTURE_INTERVAL_SECONDS - elapsed
            if sleep_time / CAPTURE_INTERVAL_SECONDS > (1 / FRAME_SKIP_THRESHOLD_MULTIPLIER):
                time.sleep(max(0.0, sleep_time))

    def _capture_frame(self):
        try:
            import mss as _mss
            with _mss.mss() as sct:
                monitor = (
                    sct.monitors[1]
                    if len(sct.monitors) > 1
                    else sct.monitors[0]
                )
                screenshot = sct.grab(monitor)
                img = Image.frombytes(
                    "RGB",
                    (screenshot.width, screenshot.height),
                    screenshot.rgb,
                )
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=70)
                raw_bytes = buf.getvalue()

                if len(raw_bytes) > MAX_FRAME_BYTES:
                    return None, None

                return base64.b64encode(raw_bytes).decode(), img

        except Exception as exc:
            print(f"[VisionRuntime] Screenshot capture failed: {exc}", file=sys.stderr)
            return None, None

    def _call_model(self, frame_b64: str) -> str:
        prompt = (
            "Analyze this screenshot. Return ONLY a JSON object with these fields: "
            "elements (list), focused_app (string), dialogs (list), frame_ts (number). "
            "Each element: {type, text, x, y, interactable, state, confidence}. "
            "x/y normalized 0.0-1.0. No prose. No markdown."
        )

        future = self._executor.submit(
            self._ollama_client.chat,
            model=self._model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [frame_b64],
                }
            ],
            options={"temperature": 0},
        )
        try:
            response = future.result(timeout=MODEL_CALL_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            raise VisionDegradedError(
                f"Model call timed out after {MODEL_CALL_TIMEOUT_SECONDS}s"
            )
        return _extract_vision_content(response)

    def _normalize_output(self, raw: str) -> Dict[str, Any]:
        if not isinstance(raw, str) or not raw.strip():
            return {"elements": [], "entities": [], "focused_app": None, "frame_ts": time.time()}

        text = re.sub(r"```(?:json)?", "", raw).strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except json.JSONDecodeError:
                    parsed = {}
            else:
                parsed = {}

        if not isinstance(parsed, dict):
            parsed = {}

        elements = parsed.get("elements") or parsed.get("entities") or []
        if not isinstance(elements, list):
            elements = []

        clean_elements = []
        for el in elements[:MAX_ELEMENTS]:
            if not isinstance(el, dict):
                continue
            clean_el = {
                "type": str(el.get("type") or "unknown").lower().strip(),
                "text": str(el.get("text") or "").strip(),
                "x": self._clamp_coord(el.get("x")),
                "y": self._clamp_coord(el.get("y")),
                "interactable": bool(el.get("interactable", True)),
                "state": el.get("state"),
                "confidence": max(0.0, min(1.0, float(el.get("confidence", 0.8)))),
            }
            clean_elements.append(clean_el)

        dialogs = parsed.get("dialogs", [])
        if not isinstance(dialogs, list):
            dialogs = []

        return {
            "elements": clean_elements,
            "entities": clean_elements,
            "focused_app": str(parsed.get("focused_app") or ""),
            "dialogs": dialogs,
            "frame_ts": time.time(),
            "available": True,
        }

    @staticmethod
    def _clamp_coord(value: Any) -> float:
        try:
            v = float(value)
            import math
            if math.isnan(v) or math.isinf(v):
                return 0.0
            return max(0.0, min(1.0, v))
        except Exception:
            return 0.0
