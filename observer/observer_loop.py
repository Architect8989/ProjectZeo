from __future__ import annotations

import logging
import threading
import time
import traceback
from typing import Optional, Callable, Dict, Any

from observer.observer_core import ObserverCore, ObserverBlindnessError
from core.vision.vision_runtime import VisionRuntime
from core.vision.world_graph import WorldGraph

# UI-TARS-2 runtime — lazy import; available when SGLang is configured.
# VisionRuntime (Qwen2.5-VL Ollama) is kept as the fallback.
_UITARSRuntime = None
def _get_uitars_runtime_cls():
    global _UITARSRuntime
    if _UITARSRuntime is not None:
        return _UITARSRuntime
    try:
        from core.vision.uitars_runtime import UITARSRuntime as _U
        _UITARSRuntime = _U
        return _UITARSRuntime
    except ImportError:
        return None

_logger = logging.getLogger(__name__)

_REQUIRED_PERCEPTION_KEYS: frozenset = frozenset({"available"})
_ENTITY_KEY_ALIASES: tuple = ("entities", "elements", "nodes", "ui_elements")

# AT-SPI fallback polling interval when AT-SPI is unavailable or app is non-accessible
_ATSPI_FALLBACK_INTERVAL: float = float(
    __import__("os").environ.get("PROJECTZEO_ATSPI_FALLBACK_INTERVAL", "0.5")
)

# Time between AT-SPI-triggered inferences (prevents flooding on rapid events)
_ATSPI_MIN_TRIGGER_INTERVAL: float = 0.05  # 50ms minimum — matches human reaction time


def _validate_perception_schema(perception: dict) -> "dict | None":
    """Validate and normalise VisionRuntime output to the expected perception schema."""

    if not isinstance(perception, dict):
        _logger.warning(
            "[ObserverLoop] Schema validation failed: perception is %s, expected dict",
            type(perception).__name__,
        )
        return None

    if "available" not in perception:
        _logger.warning(
            "[ObserverLoop] Schema validation failed: 'available' key missing. "
            "VisionRuntime output schema may have changed. Keys present: %s",
            sorted(perception.keys()),
        )
        return None

    validated = dict(perception)
    validated["available"] = bool(validated["available"])

    # Normalise 'frame_ts'
    raw_ts = validated.get("frame_ts")
    if raw_ts is not None:
        try:
            validated["frame_ts"] = float(raw_ts)
        except (TypeError, ValueError):
            validated["frame_ts"] = None
    else:
        validated.setdefault("frame_ts", None)

    # Normalise entity list key
    if "entities" not in validated:
        for alias in _ENTITY_KEY_ALIASES[1:]:
            if alias in validated:
                _logger.debug(
                    "[ObserverLoop] Schema: '%s' → 'entities' alias normalised.", alias
                )
                validated["entities"] = validated[alias]
                break
        else:
            validated.setdefault("entities", [])

    if not isinstance(validated.get("entities"), list):
        validated["entities"] = []

    # Normalise 'focused_app'
    if "focused_app" in validated and not isinstance(validated["focused_app"], str):
        del validated["focused_app"]

    return validated


class ObserverLoop:
    

    DEFAULT_TICK_INTERVAL: float = 0.20   # 5 Hz baseline for polling fallback

    def __init__(
        self,
        *,
        observer: ObserverCore,
        vision_runtime: VisionRuntime,
        world_graph: Optional[WorldGraph] = None,
        tick_interval: float = DEFAULT_TICK_INTERVAL,
        uitars_runtime=None,
    ) -> None:
        self._observer = observer
        # VIS-1 FIX: UITARSRuntime is the primary vision source when SGLang is
        # configured (PROJECTZEO_USE_SGLANG=1).  VisionRuntime (Qwen2.5-VL via
        # Ollama) is kept as the fallback for CPU-only deployments or when
        # UITARSRuntime is unavailable.  Never delete VisionRuntime — it is the
        # production fallback that keeps the system operational on CPU hosts.
        self._vision = vision_runtime            # fallback (always present)
        self._uitars: Optional[Any] = uitars_runtime  # primary (GPU/SGLang)
        self._using_uitars: bool = uitars_runtime is not None
        self._uitars_failures: int = 0           # consecutive UITARSRuntime errors
        _MAX_UITARS_FAILURES_BEFORE_FALLBACK = 3  # stored on self below
        self._max_uitars_failures = _MAX_UITARS_FAILURES_BEFORE_FALLBACK
        if self._using_uitars:
            _logger.info(
                "[ObserverLoop] UITARSRuntime registered as PRIMARY vision source. "
                "VisionRuntime retained as fallback."
            )
        self._world_graph = world_graph

        self._tick_interval: float = max(0.05, float(tick_interval))

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running: bool = False
        self._lock = threading.Lock()

        # Telemetry counters — readable via get_telemetry()
        self._total_frames: int = 0
        self._schema_rejection_count: int = 0
        self._ingest_failure_count: int = 0
        self._atspi_trigger_count: int = 0
        self._polling_trigger_count: int = 0

        # Mode flags
        self._pause_event = threading.Event()
        self._lightweight_mode_event = threading.Event()
        self._lightweight_tick_interval: float = 1.0
        self._full_tick_interval: float = self._tick_interval

        # AT-SPI integration
        self._atspi_bridge = None
        self._atspi_available = False
        self._last_atspi_trigger_ts: float = 0.0
        self._atspi_trigger_lock = threading.Lock()

        # Timestamp of last successful frame ingest (for fallback polling interval)
        self._last_frame_ts: float = 0.0

    # =========================================================================
    # AT-SPI wiring (VIS-2 FIX)
    # =========================================================================

    def _init_atspi(self) -> bool:
        
        try:
            from core.perception.atspi_bridge import ATSPIBridge, ATSPIUnavailableError  # noqa: PLC0415

            self._atspi_bridge = ATSPIBridge(
                on_change_callback=self._on_atspi_event,
                min_trigger_interval_seconds=_ATSPI_MIN_TRIGGER_INTERVAL,
            )
            self._atspi_bridge.start()
            self._atspi_available = True
            _logger.info(
                "[ObserverLoop] AT-SPI bridge started. "
                "Event-driven perception active (fallback polling interval=%.2fs).",
                _ATSPI_FALLBACK_INTERVAL,
            )
            return True

        except ImportError:
            _logger.info(
                "[ObserverLoop] core.perception.atspi_bridge not found. Using polling only."
            )
        except Exception as exc:
            _logger.info(
                "[ObserverLoop] AT-SPI init failed (%s). Using polling fallback.", exc
            )

        return False

    def _on_atspi_event(self) -> None:
        
        now = time.monotonic()
        with self._atspi_trigger_lock:
            # Enforce minimum interval between AT-SPI-triggered inferences
            if now - self._last_atspi_trigger_ts < _ATSPI_MIN_TRIGGER_INTERVAL:
                return
            self._last_atspi_trigger_ts = now

        if self._lightweight_mode_event.is_set():
            # In lightweight mode, skip AT-SPI-triggered inference
            return

        try:
            self._atspi_trigger_count += 1
            raw_perception = self._capture_frame()
            self._ingest_frame(raw_perception)
        except Exception as exc:
            _logger.debug("[ObserverLoop] AT-SPI triggered frame error: %s", exc)

    # =========================================================================
    # Frame capture — UITARSRuntime (primary) with VisionRuntime (fallback)
    # =========================================================================

    def _capture_frame(self) -> Optional[Dict[str, Any]]:
        """
        VIS-1 FIX: Capture a perception frame.

        Priority:
          1. UITARSRuntime (UI-TARS-2 via SGLang) — when registered and healthy.
          2. VisionRuntime (Qwen2.5-VL via Ollama) — fallback, always available.

        Automatic fallback: after _max_uitars_failures consecutive UITARSRuntime
        errors, the loop demotes to VisionRuntime for the remainder of the session
        and logs a WARNING so operators know to investigate the SGLang server.
        Recovery: call reset_uitars() to re-enable UITARSRuntime.
        """
        if self._using_uitars and self._uitars is not None:
            try:
                result = self._uitars.get_latest()
                # Successful UITARSRuntime call — reset failure counter.
                self._uitars_failures = 0
                return result
            except Exception as exc:
                self._uitars_failures += 1
                _logger.warning(
                    "[ObserverLoop] UITARSRuntime error #%d/%d: %s. "
                    "%s",
                    self._uitars_failures,
                    self._max_uitars_failures,
                    exc,
                    "Falling back to VisionRuntime permanently this session."
                    if self._uitars_failures >= self._max_uitars_failures
                    else "Retrying next frame.",
                )
                if self._uitars_failures >= self._max_uitars_failures:
                    self._using_uitars = False
                    _logger.error(
                        "[ObserverLoop] UITARSRuntime DEMOTED after %d consecutive "
                        "failures. VisionRuntime (fallback) is now the active source. "
                        "Call reset_uitars() to re-enable UITARSRuntime.",
                        self._max_uitars_failures,
                    )
                # Fall through to VisionRuntime
        # VisionRuntime fallback (always present)
        return self._vision.get_latest()

    def reset_uitars(self) -> None:
        """Re-enable UITARSRuntime after it was demoted due to consecutive failures."""
        if self._uitars is not None:
            self._uitars_failures = 0
            self._using_uitars = True
            _logger.info("[ObserverLoop] UITARSRuntime re-enabled (manual reset).")

    def get_active_vision_source(self) -> str:
        """Return the name of the currently active vision source."""
        if self._using_uitars and self._uitars is not None:
            return "UITARSRuntime"
        return "VisionRuntime"

    # =========================================================================
    # Frame ingestion (shared by AT-SPI callback and polling fallback)
    # =========================================================================

    def _ingest_frame(self, raw_perception: Optional[Dict[str, Any]]) -> None:
        
        if (
            isinstance(raw_perception, dict)
            and raw_perception.get("available") is True
        ):
            validated = _validate_perception_schema(raw_perception)

            if validated is not None:
                self._total_frames += 1
                self._last_frame_ts = time.monotonic()
                self._observer.attach_perception_state(
                    {
                        "available": True,
                        "frame_ts": validated.get("frame_ts"),
                        "perception": validated,
                    }
                )
                if self._world_graph is not None:
                    try:
                        self._world_graph.ingest(validated)
                    except Exception as ingest_err:
                        self._ingest_failure_count += 1
                        _logger.debug(
                            "[ObserverLoop] world_graph.ingest() failed (total=%d): %s",
                            self._ingest_failure_count, ingest_err,
                        )
            else:
                self._schema_rejection_count += 1
                if self._schema_rejection_count % 10 == 1:
                    _logger.warning(
                        "[ObserverLoop] VisionRuntime schema rejection #%d — "
                        "perception dict does not match expected schema.",
                        self._schema_rejection_count,
                    )
                self._observer.attach_perception_state(
                    {"available": False, "frame_ts": None, "perception": None}
                )
        else:
            self._observer.attach_perception_state(
                {"available": False, "frame_ts": None, "perception": None}
            )

        self._observer.tick()

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self) -> None:
        """Start the observer loop: AT-SPI bridge (primary) + polling thread (fallback)."""
        with self._lock:
            if self._running:
                return

            # Attempt AT-SPI event-driven perception (VIS-2 FIX)
            self._init_atspi()

            self._stop_event.clear()
            self._running = True
            self._thread = threading.Thread(
                target=self._run,
                name="observer-loop",
                daemon=True,
            )
            self._thread.start()
            _logger.info(
                "[ObserverLoop] Started. atspi=%s polling_interval=%.3fs.",
                self._atspi_available, self._tick_interval,
            )

    def stop(self) -> None:
        """Stop the polling thread and AT-SPI bridge."""
        with self._lock:
            if not self._running:
                return
            self._stop_event.set()
            thread = self._thread
            self._thread = None
            self._running = False

        if thread and thread.is_alive():
            thread.join(timeout=2.0)
            if thread.is_alive():
                _logger.warning("[ObserverLoop] Thread did not exit within 2s — proceeding.")

        # Stop AT-SPI bridge
        if self._atspi_bridge is not None:
            try:
                self._atspi_bridge.stop()
            except Exception:
                pass

        _logger.info("[ObserverLoop] Stopped.")

    def pause(self) -> None:
        self.set_lightweight_mode(True)

    def resume(self) -> None:
        self.set_lightweight_mode(False)

    def set_lightweight_mode(self, enabled: bool) -> None:
        """
        Toggle lightweight mode:
            enabled=True  → 1 Hz polling, VL inference skipped, AT-SPI callbacks ignored
            enabled=False → full rate VL inference resumed
        """
        if enabled:
            self._lightweight_mode_event.set()
            self._pause_event.clear()
            _logger.info(
                "[ObserverLoop] Lightweight mode ENABLED (1 Hz; VL inference paused)."
            )
        else:
            self._lightweight_mode_event.clear()
            self._pause_event.clear()
            _logger.info(
                "[ObserverLoop] Lightweight mode DISABLED (%.1f Hz VL inference resumed).",
                self._full_tick_interval,
            )

    def is_lightweight(self) -> bool:
        return self._lightweight_mode_event.is_set()

    # =========================================================================
    # Polling loop (fallback for non-accessible apps)
    # =========================================================================

    def _run(self) -> None:
        
        _logger.info("[ObserverLoop] Polling loop started.")
        fallback_interval = _ATSPI_FALLBACK_INTERVAL if self._atspi_available else self._tick_interval

        try:
            while not self._stop_event.is_set():
                start_ts = time.monotonic()

                _in_lightweight = self._lightweight_mode_event.is_set()
                if _in_lightweight:
                    # Lightweight: emit heartbeat stub, skip VL inference
                    time.sleep(self._lightweight_tick_interval)
                    try:
                        stub = {
                            "available": True,
                            "frame_ts": time.time(),
                            "entities": [],
                            "focused_app": None,
                            "_lightweight": True,
                        }
                        self._observer.attach_perception_state({
                            "available": True,
                            "frame_ts": stub["frame_ts"],
                            "perception": stub,
                        })
                        self._observer.tick()
                    except Exception:
                        pass
                    continue

                now = time.monotonic()

                # When AT-SPI is active: only poll if stale (no AT-SPI event recently)
                if self._atspi_available:
                    time_since_last_frame = now - self._last_frame_ts
                    if time_since_last_frame < fallback_interval:
                        # AT-SPI triggered recently; skip this polling cycle
                        elapsed = time.monotonic() - start_ts
                        sleep_for = self._tick_interval - elapsed
                        if sleep_for > 0:
                            time.sleep(sleep_for)
                        continue
                    # Stale state: force a poll even in accessible apps
                    _logger.debug(
                        "[ObserverLoop] Stale fallback poll (%.2fs since last frame).",
                        time_since_last_frame,
                    )

                try:
                    raw_perception = self._capture_frame()
                    self._polling_trigger_count += 1
                    self._ingest_frame(raw_perception)

                except ObserverBlindnessError as obe:
                    health = self._observer.get_health_snapshot()
                    _logger.error(
                        "[ObserverLoop] OBSERVER BLIND: %s | "
                        "consecutive_misses=%d | schema_rejections=%d | "
                        "atspi_triggers=%d | polling_triggers=%d | "
                        "total_valid_frames=%d | uptime=%.1fs",
                        obe,
                        health.get("consecutive_misses", -1),
                        self._schema_rejection_count,
                        self._atspi_trigger_count,
                        self._polling_trigger_count,
                        self._total_frames,
                        health.get("uptime_seconds", -1.0),
                    )
                    try:
                        self._observer.attach_perception_state(
                            {"available": False, "frame_ts": None, "perception": None}
                        )
                    except Exception:
                        pass
                    raise

                except Exception:
                    traceback.print_exc()
                    try:
                        self._observer.attach_perception_state(
                            {"available": False, "frame_ts": None, "perception": None}
                        )
                    except Exception:
                        pass

                elapsed = time.monotonic() - start_ts
                sleep_for = self._tick_interval - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)

        finally:
            with self._lock:
                self._running = False
            _logger.info("[ObserverLoop] Polling loop exited.")

    # =========================================================================
    # Introspection
    # =========================================================================

    def is_running(self) -> bool:
        with self._lock:
            return (
                self._running
                and self._thread is not None
                and self._thread.is_alive()
            )

    def get_telemetry(self) -> Dict[str, Any]:
        """Return telemetry counters and health info for monitoring and diagnostics."""
        return {
            "total_valid_frames":       self._total_frames,
            "schema_rejection_count":   self._schema_rejection_count,
            "ingest_failure_count":     self._ingest_failure_count,
            "atspi_trigger_count":      self._atspi_trigger_count,
            "polling_trigger_count":    self._polling_trigger_count,
            "atspi_available":          int(self._atspi_available),
            "is_paused":                int(self._pause_event.is_set()),
            "is_lightweight_mode":      int(self._lightweight_mode_event.is_set()),
            "is_running":               int(self.is_running()),
            # VIS-1: active vision source tracking
            "active_vision_source":     self.get_active_vision_source(),
            "uitars_registered":        int(self._uitars is not None),
            "uitars_active":            int(self._using_uitars),
            "uitars_consecutive_errors": self._uitars_failures,
        }

    def frame_age_seconds(self) -> Optional[float]:
        if self._last_frame_ts == 0.0:
            return None
        return time.monotonic() - self._last_frame_ts
