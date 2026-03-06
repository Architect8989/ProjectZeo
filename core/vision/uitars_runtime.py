from __future__ import annotations

import base64
import json
import logging
import time
import threading
from typing import Any, Dict, List, Optional

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Element type vocabulary — normalised from UI-TARS-2 output taxonomy
# ---------------------------------------------------------------------------
_UITARS_TYPE_MAP: Dict[str, str] = {
    # UI-TARS-2 native types → ProjectZeo canonical types
    "button":       "button",
    "icon":         "button",
    "tab":          "button",
    "menu_item":    "button",
    "checkbox":     "checkbox",
    "radio":        "radio",
    "input":        "input",
    "text_field":   "input",
    "textarea":     "input",
    "text":         "text",
    "label":        "text",
    "heading":      "text",
    "link":         "link",
    "image":        "image",
    "scroll":       "scroll",
    "slider":       "slider",
    "dropdown":     "dropdown",
    "combobox":     "dropdown",
    "window":       "container",
    "panel":        "container",
    "toolbar":      "container",
    "dialog":       "dialog",
    "tooltip":      "tooltip",
    "unknown":      "unknown",
}

# Maximum elements to return per frame
_MAX_ELEMENTS = 50

# Browser app names that indicate external content context
_BROWSER_APPS: frozenset = frozenset({
    "firefox", "chrome", "chromium", "brave", "edge", "safari",
    "google-chrome", "microsoft-edge", "opera", "vivaldi",
})

# UI-TARS-2 system prompt — instructs the model to output structured JSON
_UITARS_SYSTEM_PROMPT = """You are UI-TARS-2, a GUI interaction specialist.
Analyze the screenshot and return a JSON object with this exact schema:
{
  "focused_app": "<app name or 'unknown'>",
  "elements": [
    {
      "type": "<button|input|text|link|image|scroll|dropdown|checkbox|dialog|unknown>",
      "text": "<visible text or aria-label or empty string>",
      "x": <normalized 0.0-1.0 horizontal center>,
      "y": <normalized 0.0-1.0 vertical center>,
      "interactable": <true|false>,
      "state": "<enabled|disabled|focused|checked|unchecked|null>",
      "confidence": <0.0-1.0>
    }
  ],
  "dialogs": [
    {"text": "<dialog text>", "type": "<alert|confirm|prompt|modal>"}
  ]
}
Rules:
- Coordinates are normalized 0.0–1.0 from top-left
- List ALL interactable elements plus key text/label elements
- confidence reflects detection certainty (0.9+ for clearly visible elements)
- For browser content, include all visible links, inputs, and buttons
- Output ONLY valid JSON, no markdown, no explanation"""


class UITARSRuntime:
    """
    Vision runtime backed by UI-TARS-2 native GUI agent model.

    API-compatible with VisionRuntime: can replace it transparently in ObserverCore.
    Both have:
        start()                → begin background capture + analysis thread
        stop()                 → stop the thread
        get_latest() → dict    → latest perception frame
        capture_and_analyze()  → dict (immediate capture, blocking)
    """

    def __init__(
        self,
        *,
        model_id: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: float = 30.0,
        capture_interval_seconds: float = 0.5,
        fallback_to_visionruntime: bool = True,
    ) -> None:
        """
        Args:
            model_id:                  UI-TARS-2 model name on SGLang server
            base_url:                  SGLang server URL for vision endpoint
            timeout_seconds:           Per-request timeout (UI-TARS-2 is fast on GPU)
            capture_interval_seconds:  Background capture interval when AT-SPI not wired
            fallback_to_visionruntime: Fall back to VisionRuntime when UITARS unavailable
        """
        from config.model_config import get_vision_endpoint  # noqa: PLC0415
        vision_ep = get_vision_endpoint()

        self._model_id = model_id or vision_ep.model_id
        self._base_url = (base_url or vision_ep.base_url).rstrip("/")
        self._timeout = timeout_seconds
        self._capture_interval = capture_interval_seconds
        self._fallback_enabled = fallback_to_visionruntime

        self._latest_frame: Optional[Dict[str, Any]] = None
        self._latest_lock = threading.Lock()

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self._run_lock = threading.Lock()

        # Statistics
        self._total_frames: int = 0
        self._parse_errors: int = 0
        self._fallback_frames: int = 0

        # Lazy-initialised httpx client
        self._client = None
        self._client_lock = threading.Lock()

        # Fallback VisionRuntime (lazy init)
        self._vr_fallback = None
        self._vr_init_lock = threading.Lock()
        self._uitars_available: Optional[bool] = None  # None = not yet checked

        _logger.info(
            "[UITARSRuntime] Initialised. model=%s url=%s",
            self._model_id, self._base_url,
        )

    # =========================================================================
    # Public API — matches VisionRuntime
    # =========================================================================

    def start(self) -> None:
        """Start background capture thread."""
        with self._run_lock:
            if self._running:
                return
            self._stop_event.clear()
            self._running = True
            self._thread = threading.Thread(
                target=self._run, name="uitars-runtime", daemon=True
            )
            self._thread.start()
            _logger.info("[UITARSRuntime] Background thread started.")

    def stop(self) -> None:
        """Stop background capture thread."""
        with self._run_lock:
            if not self._running:
                return
            self._stop_event.set()
            thread = self._thread
            self._thread = None
            self._running = False
        if thread and thread.is_alive():
            thread.join(timeout=3.0)
        _logger.info("[UITARSRuntime] Stopped.")

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Return the most recent perception frame (non-blocking)."""
        with self._latest_lock:
            return dict(self._latest_frame) if self._latest_frame else None

    def capture_and_analyze(self) -> Dict[str, Any]:
        """
        Capture a screenshot and run UI-TARS-2 inference (blocking).
        Used directly by ObserverCore.tick() when AT-SPI event triggers perception.
        """
        if self._uitars_available is None:
            self._uitars_available = self._check_server_health()
            if not self._uitars_available:
                _logger.warning(
                    "[UITARSRuntime] Server not available at %s. "
                    "Falling back to VisionRuntime for this session.",
                    self._base_url,
                )

        if not self._uitars_available and self._fallback_enabled:
            return self._capture_via_fallback()

        screenshot_b64 = self._take_screenshot_b64()
        if screenshot_b64 is None:
            return self._empty_frame()

        return self._run_inference(screenshot_b64)

    # =========================================================================
    # Background loop
    # =========================================================================

    def _run(self) -> None:
        _logger.debug("[UITARSRuntime] Background loop started.")
        while not self._stop_event.is_set():
            start = time.monotonic()
            try:
                frame = self.capture_and_analyze()
                with self._latest_lock:
                    self._latest_frame = frame
            except Exception as exc:
                _logger.debug("[UITARSRuntime] Background frame error: %s", exc)

            elapsed = time.monotonic() - start
            sleep_for = self._capture_interval - elapsed
            if sleep_for > 0:
                self._stop_event.wait(timeout=sleep_for)

        _logger.debug("[UITARSRuntime] Background loop exited.")

    # =========================================================================
    # Screenshot capture
    # =========================================================================

    def _take_screenshot_b64(self) -> Optional[str]:
        """Capture a screenshot and return as base64-encoded JPEG."""
        try:
            import mss  # noqa: PLC0415
            import io
            from PIL import Image  # noqa: PLC0415

            with mss.mss() as sct:
                monitor = sct.monitors[0]
                sct_img = sct.grab(monitor)

            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

            # Resize to 1280×720 for consistent inference speed
            target_w, target_h = 1280, 720
            img.thumbnail((target_w, target_h), Image.LANCZOS)

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode("ascii")

        except Exception as exc:
            _logger.debug("[UITARSRuntime] Screenshot failed: %s", exc)
            return None

    # =========================================================================
    # Inference
    # =========================================================================

    def _run_inference(self, screenshot_b64: str) -> Dict[str, Any]:
        """Submit screenshot to UI-TARS-2 and parse the structured output."""
        try:
            import httpx  # noqa: PLC0415
        except ImportError:
            _logger.warning("[UITARSRuntime] httpx not available — falling back.")
            return self._capture_via_fallback()

        messages = [
            {"role": "system", "content": _UITARS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{screenshot_b64}",
                        },
                    },
                    {
                        "type": "text",
                        "text": "Analyze this screenshot and return the structured JSON.",
                    },
                ],
            },
        ]

        payload = {
            "model": self._model_id,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0.0,
            "stream": False,
        }

        try:
            client = self._get_httpx_client()
            response = client.post(
                f"{self._base_url}/v1/chat/completions",
                content=json.dumps(payload),
                timeout=self._timeout,
            )
            if response.status_code != 200:
                _logger.warning(
                    "[UITARSRuntime] Server returned %d. Falling back.",
                    response.status_code,
                )
                self._uitars_available = False
                return self._capture_via_fallback()

            data = response.json()
            content = data["choices"][0]["message"]["content"]
            frame = self._parse_uitars_output(content)
            self._total_frames += 1
            return frame

        except Exception as exc:
            _logger.debug("[UITARSRuntime] Inference request failed: %s", exc)
            self._uitars_available = False
            return self._capture_via_fallback()

    def _parse_uitars_output(self, raw_content: str) -> Dict[str, Any]:
        
        try:
            # Strip markdown fences if present
            import re  # noqa: PLC0415
            clean = re.sub(r"```(?:json)?", "", raw_content).strip()
            # Find the outermost JSON object
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            if not match:
                self._parse_errors += 1
                return self._empty_frame()

            parsed = json.loads(match.group(0))

        except (json.JSONDecodeError, AttributeError) as exc:
            self._parse_errors += 1
            _logger.debug("[UITARSRuntime] JSON parse error: %s", exc)
            return self._empty_frame()

        focused_app_raw = str(parsed.get("focused_app") or "").lower().strip()
        is_browser = any(
            focused_app_raw == b or focused_app_raw.startswith(b)
            for b in _BROWSER_APPS
        )

        raw_elements: List[Dict] = parsed.get("elements", [])
        if not isinstance(raw_elements, list):
            raw_elements = []

        clean_elements: List[Dict[str, Any]] = []
        for el in raw_elements[:_MAX_ELEMENTS]:
            if not isinstance(el, dict):
                continue

            raw_type = str(el.get("type") or "unknown").lower().strip()
            canonical_type = _UITARS_TYPE_MAP.get(raw_type, "unknown")

            try:
                x = max(0.0, min(1.0, float(el.get("x", 0.5))))
                y = max(0.0, min(1.0, float(el.get("y", 0.5))))
            except (TypeError, ValueError):
                x, y = 0.5, 0.5

            try:
                confidence = max(0.0, min(1.0, float(el.get("confidence", 0.8))))
            except (TypeError, ValueError):
                confidence = 0.8

            state = el.get("state")
            if state not in (None, "enabled", "disabled", "focused", "checked", "unchecked"):
                state = None

            clean_elements.append({
                "type": canonical_type,
                "text": str(el.get("text") or "").strip()[:200],
                "x": x,
                "y": y,
                "interactable": bool(el.get("interactable", True)),
                "state": state,
                "confidence": confidence,
                # CRIT-5: tag external source for browser elements
                "_external_content_source": (
                    is_browser or bool(el.get("_external_content_source", False))
                ),
                "_uitars_native": True,  # provenance flag
            })

        dialogs: List[Dict] = parsed.get("dialogs", [])
        if not isinstance(dialogs, list):
            dialogs = []

        return {
            "available": True,
            "elements": clean_elements,
            "entities": clean_elements,  # alias for backward compat
            "focused_app": str(parsed.get("focused_app") or ""),
            "dialogs": dialogs[:5],
            "frame_ts": time.time(),
            "_browser_context": is_browser,
            "_uitars_native": True,
        }

    # =========================================================================
    # Fallback to VisionRuntime
    # =========================================================================

    def _get_vr_fallback(self):
        """Lazy-initialise VisionRuntime as fallback."""
        if self._vr_fallback is not None:
            return self._vr_fallback
        with self._vr_init_lock:
            if self._vr_fallback is None:
                try:
                    from core.vision.vision_runtime import VisionRuntime  # noqa: PLC0415
                    self._vr_fallback = VisionRuntime()
                    _logger.info("[UITARSRuntime] VisionRuntime fallback initialised.")
                except Exception as exc:
                    _logger.warning(
                        "[UITARSRuntime] VisionRuntime fallback init failed: %s", exc
                    )
        return self._vr_fallback

    def _capture_via_fallback(self) -> Dict[str, Any]:
        """Route to VisionRuntime (Qwen2.5-VL 7B) when UI-TARS-2 unavailable."""
        self._fallback_frames += 1
        vr = self._get_vr_fallback()
        if vr is None:
            return self._empty_frame()
        try:
            frame = vr.capture_and_analyze()
            if isinstance(frame, dict):
                frame["_uitars_native"] = False
                return frame
        except Exception as exc:
            _logger.debug("[UITARSRuntime] VisionRuntime fallback also failed: %s", exc)
        return self._empty_frame()

    # =========================================================================
    # Utilities
    # =========================================================================

    def _check_server_health(self) -> bool:
        """Check if the SGLang vision server is reachable."""
        try:
            import httpx  # noqa: PLC0415
            with httpx.Client(timeout=5.0) as c:
                resp = c.get(f"{self._base_url}/health")
                return resp.status_code == 200
        except Exception:
            return False

    def _get_httpx_client(self):
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is None:
                import httpx  # noqa: PLC0415
                self._client = httpx.Client(
                    timeout=httpx.Timeout(
                        connect=10.0,
                        read=self._timeout,
                        write=10.0,
                        pool=5.0,
                    ),
                    headers={"Content-Type": "application/json"},
                )
        return self._client

    @staticmethod
    def _empty_frame() -> Dict[str, Any]:
        return {
            "available": False,
            "elements": [],
            "entities": [],
            "focused_app": "",
            "dialogs": [],
            "frame_ts": time.time(),
            "_browser_context": False,
            "_uitars_native": False,
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_frames": self._total_frames,
            "parse_errors": self._parse_errors,
            "fallback_frames": self._fallback_frames,
            "uitars_available": self._uitars_available,
            "model_id": self._model_id,
            "base_url": self._base_url,
        }
