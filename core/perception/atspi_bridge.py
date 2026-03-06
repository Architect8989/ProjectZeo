from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Callable, List, Optional

_logger = logging.getLogger(__name__)


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

    def is_available(self) -> bool:
        """Check if AT-SPI2 is available on this system."""
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
            "[ATSPIBridge] Started. Subscribed to %d event types. "
            "min_trigger_interval=%.3fs",
            len(self._trigger_events), self._min_interval,
        )
        return True

    def stop(self) -> None:
        """Stop the AT-SPI event listener."""
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
        """Background thread: subscribe to AT-SPI events and run event loop."""
        try:
            import pyatspi
            registry = pyatspi.Registry
            self._atspi_registry = registry

            for event_type in self._trigger_events:
                try:
                    registry.registerEventListener(self._on_event, event_type)
                    _logger.debug("[ATSPIBridge] Subscribed: %s", event_type)
                except Exception as reg_exc:
                    _logger.warning(
                        "[ATSPIBridge] Failed to subscribe to %s: %s", event_type, reg_exc
                    )

            _logger.info("[ATSPIBridge] Event loop starting.")
            pyatspi.Registry.start(synchronous=False)

            # Keep thread alive while running
            while self._running:
                time.sleep(0.1)

        except Exception as exc:
            _logger.error("[ATSPIBridge] Event loop error: %s", exc)
        finally:
            _logger.info("[ATSPIBridge] Event loop exited.")

    def _on_event(self, event) -> None:
        """AT-SPI event handler — called from the AT-SPI event thread."""
        now = time.time()
        with self._lock:
            if now - self._last_trigger_ts < self._min_interval:
                return  # Rate-limit: suppress rapid-fire events
            self._last_trigger_ts = now

        event_type = str(getattr(event, "type", "unknown"))
        source_name = ""
        source_role = ""
        try:
            source_name = str(event.source.name or "") if event.source else ""
            source_role = str(event.source.getRole() if event.source else "")
        except Exception:
            pass

        event_info = {
            "event_type": event_type,
            "source_name": source_name,
            "source_role": source_role,
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


# ---------------------------------------------------------------------------
# Integration helper: wire ATSPIBridge into VisionRuntime
# ---------------------------------------------------------------------------

def create_atspi_triggered_capture(
    vision_runtime,
    world_graph,
    *,
    min_trigger_interval: float = _MIN_TRIGGER_INTERVAL_SECONDS,
) -> ATSPIBridge:
    
    def _on_ui_change(event_type: str, event_info: dict) -> None:
        """Trigger VisionRuntime capture when UI changes are detected."""
        try:
            # Signal VisionRuntime to capture immediately
            if hasattr(vision_runtime, "trigger_capture"):
                vision_runtime.trigger_capture(reason=f"atspi:{event_type}")
            elif hasattr(vision_runtime, "_capture_and_analyse"):
                vision_runtime._capture_and_analyse()
            _logger.debug(
                "[ATSPIBridge] Triggered VLM capture for event: %s", event_type
            )
        except Exception as exc:
            _logger.warning("[ATSPIBridge] VLM trigger failed: %s", exc)

    return ATSPIBridge(
        on_change_callback=_on_ui_change,
        min_trigger_interval=min_trigger_interval,
    )
