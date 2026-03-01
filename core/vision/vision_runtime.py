from __future__ import annotations

import sys
import time
import threading
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

# FIX-4: Import the canonical VisionUnavailableError from mode_controller.
from core.mode_controller import VisionUnavailableError


class VisionDegradedError(RuntimeError):
    pass




def _shutdown_executor_compat(executor, wait: bool = False) -> None:
    
    if sys.version_info >= (3, 9):
        executor.shutdown(wait=wait, cancel_futures=True)
    else:
        executor.shutdown(wait=wait)


MAX_ALLOWED_LATENCY_SECONDS  = 180.0   # CPU inference (Qwen2.5-VL 7B) takes 40-90s; allow headroom
NETWORK_CONNECT_TIMEOUT      = 5.0
NETWORK_READ_TIMEOUT         = 180.0   # raised to match new latency ceiling
MODEL_CALL_TIMEOUT_SECONDS   = 200.0   # outer hard-kill > latency ceiling so latency check fires first

MAX_FRAME_BYTES = 4 * 1024 * 1024
MAX_ELEMENTS = 128
MAX_CONSECUTIVE_FAILURES = 5

# BUG-4 FIX: CAPTURE_INTERVAL_SECONDS was 0.5s but CPU inference takes 40-90s.
# The old loop did: inference(45s blocking) → sleep(0.5s) → repeat.
# World graph updated every ~45s, never every 0.5s as documented.
# During multi-step tasks, the planner saw state that was minutes stale.
#
# Fix: CAPTURE_INTERVAL_SECONDS is now the *minimum* sleep between frames only
# when inference is fast. We track actual inference duration and apply
# frame-skip logic: if inference took longer than CAPTURE_INTERVAL_SECONDS,
# skip the sleep entirely (the inference itself was the delay).
# This means on CPU: we capture as fast as possible without wasting
# an extra 0.5s sleep on top of a 45s inference.
# On GPU: we still sleep 0.5s between fast 1-2s inferences (5 Hz target).
CAPTURE_INTERVAL_SECONDS = 0.5

# BUG-4 FIX: If inference takes longer than this multiplier × CAPTURE_INTERVAL,
# skip the post-inference sleep (the inference already introduced enough delay).
FRAME_SKIP_THRESHOLD_MULTIPLIER = 5.0  # skip sleep if inference > 2.5s

# BUG-3 FIX: Shared inference lock so VisionRuntime background loop and
# QwenOllamaAdapter (action execution) cannot call the same Ollama model
# concurrently. On VRAM-limited GPUs this causes OOM; on CPU it doubles
# already-slow inference time. Both callers acquire this lock before calling
# the model. The lock is module-level so it is shared across all instances.
INFERENCE_LOCK = threading.Lock()


def get_inference_lock() -> threading.Lock:
    """
    BUG-3 FIX: Public accessor for the module-level INFERENCE_LOCK.

    QwenOllamaAdapter imports and acquires this lock before calling
    ollama.Client.chat() so that the background VisionRuntime loop and
    the synchronous action-decision path never call the same Ollama vision
    model concurrently.

    Usage in qwen_ollama_adapter.py:
        from core.vision.vision_runtime import get_inference_lock
        with get_inference_lock():
            response = self._client.chat(...)
    """
    return INFERENCE_LOCK


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
        self._last_frame_ts: Optional[float] = None
        self._last_raw_image: Optional[Any] = None  # BUG-8: shared raw frame
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
        self._check_multi_monitor()  # GAP-3: warn on multi-monitor setups

    # ==================================================
    # DISPLAY VALIDATION
    # ==================================================

    def _validate_display_environment(self) -> None:
        if os.name != "nt":
            if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
                raise VisionUnavailableError(
                    "No display environment detected (headless mode unsupported)"
                )

    # GAP-3 FIX: Detect multi-monitor setups and emit a startup warning.
    # mss.monitors[0] is the full combined virtual desktop; mss.monitors[1] is
    # the primary monitor.  If a user has two or more monitors and we capture
    # monitors[0], normalized coordinates (0.0–1.0) span the FULL virtual
    # desktop width.  pyautogui.click() however uses absolute pixel coordinates
    # within the primary monitor's coordinate space.  This mismatch causes click
    # targets to land at the wrong position on multi-monitor setups.
    #
    # Fix: capture mss.monitors[1] (primary only) so that 0.0–1.0 always maps
    # to primary-monitor pixels, matching pyautogui's default coordinate space.
    # If only one monitor is present, monitors[1] == monitors[0] — no change.
    #
    # _primary_monitor is set at init time and cached; if the user rearranges
    # monitors mid-session, a restart is required (acceptable; same as Xrandr).
    def _check_multi_monitor(self) -> None:
        """GAP-3 FIX: Detect multi-monitor setup and select primary monitor."""
        try:
            import mss as _mss
            with _mss.mss() as sct:
                n_monitors = len(sct.monitors) - 1  # monitors[0] is virtual desktop
                if n_monitors > 1:
                    print(
                        f"[VisionRuntime] GAP-3 WARNING: {n_monitors} monitors detected. "
                        "ProjectZeo captures the PRIMARY monitor only (mss.monitors[1]). "
                        "Click coordinates are normalised to primary monitor dimensions. "
                        "Applications must be on the PRIMARY monitor for correct operation. "
                        "Secondary monitor content is NOT visible to the vision model.",
                        file=sys.stderr,
                    )
                # Store primary monitor dimensions for coordinate normalisation
                self._primary_monitor = dict(sct.monitors[1]) if len(sct.monitors) > 1 else None
        except Exception:
            self._primary_monitor = None

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
            self._thread.join(timeout=3.0)

        # FIX-C2: Python 3.8-compatible executor shutdown.
        _shutdown_executor_compat(self._executor, wait=False)

    def is_healthy(self) -> bool:
        with self._lock:
            return self._healthy

    def get_latest(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._last_output)

    # BUG-8 FIX: Expose the latest captured frame as a JPEG base64 string so
    # QwenOllamaAdapter can reuse it instead of capturing the screen a second
    # time independently.  Two separate mss/PIL grabs separated by hundreds of
    # milliseconds create a race condition where the world graph and the action
    # decision see different screen states.
    #
    # max_age_seconds: if the stored frame is older than this, return None so
    # the caller falls back to its own capture (avoids serving a very stale frame
    # when the background loop is paused or running slowly on CPU).
    def get_latest_frame_jpeg_b64(
        self, max_age_seconds: float = 5.0
    ) -> Optional[str]:
        """
        BUG-8 FIX: Return the most recent captured frame as a base64 JPEG string.

        Returns None when:
          - No frame has been captured yet.
          - The most recent frame is older than max_age_seconds (stale on CPU).
          - Encoding fails for any reason.

        Callers (QwenOllamaAdapter) should fall back to an independent capture
        when this returns None.
        """
        with self._lock:
            frame_ts = self._last_frame_ts
            raw_image = self._last_raw_image  # type: ignore[attr-defined]

        if frame_ts is None or raw_image is None:
            return None

        age = time.time() - frame_ts
        if age > max_age_seconds:
            return None

        try:
            return self._encode_image(raw_image)
        except Exception:
            return None

    # ==================================================
    # MAIN LOOP
    # ==================================================

    def _loop(self) -> None:
        while True:

            with self._lock:
                if not self._running:
                    return

            # BUG-4 FIX: Track inference wall time so we can skip the post-inference
            # sleep when the model itself was the delay (CPU inference: 40-90s).
            _inference_start = time.monotonic()

            try:
                output = self._process_frame_internal()
                _inference_elapsed = time.monotonic() - _inference_start

                with self._lock:
                    if not self._running:
                        return

                    frame_age = time.time() - output.get("frame_ts", 0.0)
                    if frame_age > MAX_ALLOWED_LATENCY_SECONDS:
                        self._consecutive_failures += 1
                        if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                            self._healthy = False
                        # Do not update last_output with stale data.
                        # BUG-4 FIX: Skip sleep if inference already took long enough.
                        _skip_threshold = CAPTURE_INTERVAL_SECONDS * FRAME_SKIP_THRESHOLD_MULTIPLIER
                        if _inference_elapsed < _skip_threshold:
                            time.sleep(CAPTURE_INTERVAL_SECONDS)
                        else:
                            import sys as _sys
                            print(
                                f"[VisionRuntime] BUG-4: Frame-skip: inference took "
                                f"{_inference_elapsed:.1f}s > {_skip_threshold:.1f}s threshold. "
                                "Skipping post-inference sleep to avoid doubling latency.",
                                file=_sys.stderr,
                            )
                        continue

                    self._last_output = output
                    self._last_frame_ts = output["frame_ts"]
                    self._consecutive_failures = 0
                    self._healthy = True

            except Exception:
                _inference_elapsed = time.monotonic() - _inference_start
                with self._lock:
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        self._healthy = False

            # BUG-4 FIX: Frame-skip sleep — only sleep if inference was faster
            # than the skip threshold. On CPU (45s+ inference), skip the extra
            # 0.5s sleep entirely; on GPU (sub-1s inference), sleep normally.
            _skip_threshold = CAPTURE_INTERVAL_SECONDS * FRAME_SKIP_THRESHOLD_MULTIPLIER
            _inference_elapsed = time.monotonic() - _inference_start
            if _inference_elapsed < _skip_threshold:
                time.sleep(CAPTURE_INTERVAL_SECONDS)

    # ==================================================
    # FRAME PROCESSING
    # ==================================================

    def _process_frame_internal(self) -> Dict[str, Any]:
        start = time.time()

        frame_ts = time.time()

        image = self._capture_frame()

        # BUG-8 FIX: Store raw image under lock so get_latest_frame_jpeg_b64()
        # can serve it to QwenOllamaAdapter without a second screen capture.
        with self._lock:
            self._last_raw_image = image

        encoded = self._encode_image(image)
        perception = self._call_model_with_timeout(encoded)

        latency = time.time() - start

        if latency > MAX_ALLOWED_LATENCY_SECONDS:
            raise VisionDegradedError(
                f"Vision latency exceeded: {latency:.2f}s > {MAX_ALLOWED_LATENCY_SECONDS}s"
            )

        return self._normalize_output(
            perception=perception,
            frame_ts=frame_ts,
        )

    # ==================================================
    # FRAME CAPTURE
    # ==================================================

    def _capture_frame(self) -> Image.Image:
        # GAP-3 FIX: Use mss.monitors[1] (PRIMARY monitor) instead of
        # mss.monitors[0] (full virtual desktop).  On multi-monitor setups,
        # monitors[0] spans all screens, making normalised coordinates
        # 0.0–1.0 cover the ENTIRE virtual desktop width.  pyautogui clicks
        # use primary-monitor pixel space, so click targets on multi-monitor
        # setups land at wrong positions when using the combined desktop frame.
        # monitors[1] is identical to monitors[0] on single-monitor setups.
        # PRIMARY: mss — works on X11/Wayland/Windows/Mac without scrot
        try:
            import mss as _mss
            with _mss.mss() as sct:
                # Prefer stored primary monitor bounds; fall back to monitors[1]
                monitor = (
                    self._primary_monitor
                    if getattr(self, "_primary_monitor", None)
                    else (sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0])
                )
                raw = sct.grab(monitor)
                img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
                return img
        except Exception as _mss_err:
            pass  # fall through to PIL fallback

        # FALLBACK: PIL ImageGrab (requires scrot on Linux)
        try:
            from PIL import ImageGrab as _IG
            img = _IG.grab(all_screens=False)  # GAP-3: primary only
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        except Exception as e:
            raise VisionUnavailableError(
                f"Screen capture failed (tried mss + PIL.ImageGrab). "
                f"Last error: {e}. "
                f"Fix: pip install mss  OR  sudo apt-get install scrot"
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
    # MODEL CALL (BOUNDED)
    # ==================================================

    def _call_model_with_timeout(self, image_b64: str) -> Dict[str, Any]:

        def _invoke():
            return self._call_model(image_b64)

        future = self._executor.submit(_invoke)

        try:
            return future.result(timeout=MODEL_CALL_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise VisionUnavailableError(
                f"Vision model call timed out after {MODEL_CALL_TIMEOUT_SECONDS}s"
            )

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

        # BUG-3 FIX: Acquire the module-level INFERENCE_LOCK before calling Ollama.
        # QwenOllamaAdapter also acquires this lock via get_inference_lock() before
        # its own client.chat() call. This prevents both from calling the same
        # vision model simultaneously (GPU OOM / CPU timeout cascade).
        # The lock is non-reentrant; if this thread already holds it, we deadlock —
        # but _call_model is only ever called from _loop (background thread) or
        # _call_model_with_timeout (executor thread), never recursively.
        try:
            with INFERENCE_LOCK:
                response = self._ollama_client.chat(
                    model=self._model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [image_b64],
                        }
                    ],
                    options={"temperature": 0},
                )
        except Exception as e:
            raise VisionUnavailableError(
                f"Vision model call failed: {e}"
            )

        # FIX-3: Use compatibility shim.
        content = _extract_vision_content(response)

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
            if self._last_frame_ts is not None and frame_ts <= self._last_frame_ts:
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

        
        if raw.startswith("```"):
            # Remove the opening fence marker
            raw = raw[3:]
            # Strip optional language identifier (e.g. "json", "JSON")
            if "\n" in raw:
                first_line, remainder = raw.split("\n", 1)
                stripped_tag = first_line.strip()
                # Only strip if it's a plain language tag, not start of JSON
                if stripped_tag and not stripped_tag.startswith(("{", "[")):
                    raw = remainder
            # Remove closing fence if present
            if raw.rstrip().endswith("```"):
                raw = raw.rstrip()[:-3]
            raw = raw.strip()

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
