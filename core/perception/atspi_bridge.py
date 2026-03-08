"""
core/perception/atspi_bridge.py — AT-SPI Event Bridge

PATCH HISTORY:
  GAP-2     : Fix trigger method name. create_atspi_triggered_capture() called
              vision_runtime._capture_and_analyse() (underscore-prefixed private
              method) but the actual method name in VisionRuntime is
              capture_and_analyze() (public, American spelling). This caused
              AT-SPI events to silently fail to trigger VLM inference — the
              event fired but no capture happened.
  LAYER-5   : AT-SPI window registry for popup attack detection (Research §6.3).
              Maintains a set of legitimate window IDs opened during the task.
              Windows that appear without a corresponding window:create event
              are flagged as SUSPICIOUS by the GIILoop.
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Callable, Dict, List, Optional, Set

_logger = logging.getLogger(__name__)


class ATSPIUnavailableError(RuntimeError):
    """Raised when pyatspi2 is not available on this platform."""
    pass


_TRIGGER_EVENTS = [
    "window:activate",
    "window:create",
    "window:destroy",
    "object:state-changed:focused",
    "object:state-changed:showing",
    "object:children-changed",
]

_MIN_TRIGGER_INTERVAL_SECONDS = 0.05


class ATSPIBridge:

    def __init__(
        self,
        *,
        on_change_callback: Callable[[str, dict], None],
        trigger_events: Optional[List[str]] = None,
        min_trigger_interval: float = _MIN_TRIGGER_INTERVAL_SECONDS,
    ) -> None:
        self._callback = on_change_callback
        self._trigger_events = trigger_events or _TRIGGER_EVENTS
        self._min_interval = min_trigger_interval
        self._last_trigger_ts: float = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._atspi_available = False
        self._atspi_registry = None

        # LAYER-5: Window registry for popup attack detection (Research §6.3)
        # Maps window_id → creation_timestamp. Populated from window:create events.
        self._window_registry: Dict[int, float] = {}
        self._window_registry_lock = threading.Lock()

    @property
    def window_registry(self) -> Set[int]:
        """Return the set of legitimate window IDs seen during this task."""
        with self._window_registry_lock:
            return set(self._window_registry.keys())

    def reset_window_registry(self) -> None:
        """Clear the window registry at task start."""
        with self._window_registry_lock:
            self._window_registry.clear()
        _logger.debug("[ATSPIBridge] Window registry cleared for new task.")

    def is_available(self) -> bool:
        try:
            import pyatspi  # noqa: F401
            return True
        except ImportError:
            return False

    def start(self) -> bool:
        try:
            import pyatspi
            self._atspi_available = True
        except ImportError:
            _logger.warning(
                "[ATSPIBridge] pyatspi2 not installed. "
                "Event-driven perception unavailable — falling back to polling. "
                "Install: sudo apt-get install python3-atspi"
            )
            return False

        self._running = True
        self._thread = threading.Thread(
            target=self._run_event_loop,
            name="atspi_event_loop",
            daemon=True,
        )
        self._thread.start()
        _logger.info(
            "[ATSPIBridge] Started. Subscribed to %d event types. min_interval=%.3fs",
            len(self._trigger_events), self._min_interval,
        )
        return True

    def stop(self) -> None:
        self._running = False
        if self._atspi_available and self._atspi_registry is not None:
            try:
                import pyatspi
                pyatspi.Registry.stop()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        _logger.info("[ATSPIBridge] Stopped.")

    def _run_event_loop(self) -> None:
        try:
            import pyatspi
            registry = pyatspi.Registry
            self._atspi_registry = registry

            for event_type in self._trigger_events:
                try:
                    registry.registerEventListener(self._on_event, event_type)
                    _logger.debug("[ATSPIBridge] Subscribed: %s", event_type)
                except Exception as reg_exc:
                    _logger.warning("[ATSPIBridge] Failed to subscribe to %s: %s", event_type, reg_exc)

            _logger.info("[ATSPIBridge] Event loop starting.")
            pyatspi.Registry.start(synchronous=False)

            while self._running:
                time.sleep(0.1)

        except Exception as exc:
            _logger.error("[ATSPIBridge] Event loop error: %s", exc)
        finally:
            _logger.info("[ATSPIBridge] Event loop exited.")

    def _on_event(self, event) -> None:
        now = time.time()
        with self._lock:
            if now - self._last_trigger_ts < self._min_interval:
                return
            self._last_trigger_ts = now

        event_type = str(getattr(event, "type", "unknown"))
        source_name = ""
        source_role = ""
        window_id = None
        try:
            source_name = str(event.source.name or "") if event.source else ""
            source_role = str(event.source.getRole() if event.source else "")
            # Try to get a stable window identifier
            if hasattr(event.source, "getApplication"):
                app = event.source.getApplication()
                if app is not None:
                    window_id = id(app)
        except Exception:
            pass

        # LAYER-5: Track window:create events in the registry
        if "window:create" in event_type and window_id is not None:
            with self._window_registry_lock:
                self._window_registry[window_id] = now
            _logger.debug("[ATSPIBridge] Window registry: added window_id=%d", window_id)

        event_info = {
            "event_type": event_type,
            "source_name": source_name,
            "source_role": source_role,
            "window_id": window_id,
            "ts": now,
        }

        _logger.debug(
            "[ATSPIBridge] Event: %s | source=%r role=%r",
            event_type, source_name[:40], source_role,
        )

        try:
            self._callback(event_type, event_info)
        except Exception as cb_exc:
            _logger.warning("[ATSPIBridge] Callback error: %s", cb_exc)


def create_atspi_triggered_capture(
    vision_runtime,
    world_graph,
    *,
    min_trigger_interval: float = _MIN_TRIGGER_INTERVAL_SECONDS,
) -> ATSPIBridge:
    """Wire ATSPIBridge into VisionRuntime with corrected method name."""

    def _on_ui_change(event_type: str, event_info: dict) -> None:
        """Trigger VisionRuntime capture when UI changes are detected."""
        try:
            # GAP-2 FIX: Method name corrected.
            # Original code called:   vision_runtime._capture_and_analyse()
            # Actual method name is:  vision_runtime.capture_and_analyze()  (public, US spelling)
            # Also try trigger_capture() for implementations that expose a public trigger API.
            if hasattr(vision_runtime, "trigger_capture"):
                vision_runtime.trigger_capture(reason=f"atspi:{event_type}")
            elif hasattr(vision_runtime, "capture_and_analyze"):
                vision_runtime.capture_and_analyze()
            elif hasattr(vision_runtime, "_capture_and_analyze"):
                vision_runtime._capture_and_analyze()
            else:
                _logger.warning(
                    "[ATSPIBridge] GAP-2: No capture method found on VisionRuntime. "
                    "Expected: capture_and_analyze() or trigger_capture(). "
                    "AT-SPI events will not trigger VLM inference."
                )
            _logger.debug("[ATSPIBridge] Triggered capture for: %s", event_type)
        except Exception as exc:
            _logger.warning("[ATSPIBridge] VLM trigger failed: %s", exc)

    return ATSPIBridge(
        on_change_callback=_on_ui_change,
        min_trigger_interval=min_trigger_interval,
    )



class ATSPIUnavailableError(RuntimeError):
    """Raised when pyatspi2 is not available on this platform."""
    pass


# ---------------------------------------------------------------------------
# AT-SPI event types that trigger VLM inference
# ---------------------------------------------------------------------------
_TRIGGER_EVENTS = [
    "window:activate",              # Window brought to foreground
    "window:create",                # New window appeared
    "window:destroy",               # Window closed
    "object:state-changed:focused", # Focus changed (button, field, etc.)
    "object:state-changed:showing", # Element became visible
    "object:children-changed",      # UI tree structure changed
]

# Minimum interval between trigger events to prevent inference flooding
_MIN_TRIGGER_INTERVAL_SECONDS = 0.05  # 50ms — matches human reaction time threshold
