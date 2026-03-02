from __future__ import annotations

import threading
import time
import traceback
import logging
from typing import Optional, Dict, Any

from observer.observer_core import ObserverCore, ObserverBlindnessError
from core.vision.vision_runtime import VisionRuntime
from core.vision.world_graph import WorldGraph

_logger = logging.getLogger(__name__)

_REQUIRED_PERCEPTION_KEYS: frozenset = frozenset({"available"})

_ENTITY_KEY_ALIASES: tuple = ("entities", "elements", "nodes", "ui_elements")


def _validate_perception_schema(perception: dict) -> "dict | None":
    
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

    # ---- Normalise 'frame_ts' ----
    raw_ts = validated.get("frame_ts")
    if raw_ts is not None:
        try:
            validated["frame_ts"] = float(raw_ts)
        except (TypeError, ValueError):
            _logger.debug(
                "[ObserverLoop] Schema: 'frame_ts' %r is not float — setting None.", raw_ts
            )
            validated["frame_ts"] = None
    else:
        validated.setdefault("frame_ts", None)

    # ---- Normalise entity list key ----
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
        _logger.debug(
            "[ObserverLoop] Schema: 'entities' is %s — resetting to [].",
            type(validated.get("entities")).__name__,
        )
        validated["entities"] = []

    # ---- Normalise 'focused_app' ----
    if "focused_app" in validated and not isinstance(validated["focused_app"], str):
        _logger.debug(
            "[ObserverLoop] Schema: 'focused_app' is %s (expected str) — removing.",
            type(validated["focused_app"]).__name__,
        )
        del validated["focused_app"]

    return validated


class ObserverLoop:
    

    DEFAULT_TICK_INTERVAL: float = 0.20  # 5 Hz baseline

    def __init__(
        self,
        *,
        observer: ObserverCore,
        vision_runtime: VisionRuntime,
        world_graph: Optional[WorldGraph] = None,
        tick_interval: float = DEFAULT_TICK_INTERVAL,
    ) -> None:
        self._observer = observer
        self._vision = vision_runtime
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

        # GPU contention: pause() sets this; resume() clears it
        self._pause_event = threading.Event()

    

    def start(self) -> None:
        """Spawn the observer thread.  Safe to call after stop() or restart."""
        with self._lock:
            if self._running:
                return

            
            self._stop_event.clear()
            self._running = True

            self._thread = threading.Thread(
                target=self._run,
                name="observer-loop",
                daemon=True,
            )
            self._thread.start()
            _logger.info("[ObserverLoop] Started (tick_interval=%.3fs).", self._tick_interval)

    def stop(self) -> None:
        """Signal the observer thread to stop and wait up to 2 seconds."""
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
                _logger.warning(
                    "[ObserverLoop] Thread did not exit within 2 s — proceeding."
                )

        _logger.info("[ObserverLoop] Stopped.")

    def pause(self) -> None:
        """Pause vision inference.  Call when entering EXECUTING to free GPU."""
        self._pause_event.set()
        _logger.debug("[ObserverLoop] Paused.")

    def resume(self) -> None:
        """Resume vision inference.  Call after RESTORING completes."""
        self._pause_event.clear()
        _logger.debug("[ObserverLoop] Resumed.")

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    def _run(self) -> None:
        _logger.info("[ObserverLoop] Thread running.")
        try:
            while not self._stop_event.is_set():
                start_ts = time.monotonic()

                # GPU contention: skip vision inference while EXECUTING
                if self._pause_event.is_set():
                    time.sleep(0.5)
                    continue

                try:
                    raw_perception: Optional[Dict[str, Any]] = self._vision.get_latest()

                    if (
                        isinstance(raw_perception, dict)
                        and raw_perception.get("available") is True
                    ):
                        validated = _validate_perception_schema(raw_perception)

                        if validated is not None:
                            self._total_frames += 1
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
                                    # World graph failures must NEVER break the loop
                                    self._ingest_failure_count += 1
                                    _logger.debug(
                                        "[ObserverLoop] world_graph.ingest() failed "
                                        "(total=%d): %s",
                                        self._ingest_failure_count,
                                        ingest_err,
                                    )
                        else:
                            self._schema_rejection_count += 1
                            # Sample: log every 1st, 11th, 21st… to avoid flooding
                            if self._schema_rejection_count % 10 == 1:
                                _logger.warning(
                                    "[ObserverLoop] VisionRuntime schema rejection #%d — "
                                    "perception dict does not match expected schema. "
                                    "Check VisionRuntime._call_model() output format.",
                                    self._schema_rejection_count,
                                )
                            self._observer.attach_perception_state(
                                {"available": False, "frame_ts": None, "perception": None}
                            )
                    else:
                        self._observer.attach_perception_state(
                            {"available": False, "frame_ts": None, "perception": None}
                        )

                    # Authoritative heartbeat tick
                    self._observer.tick()

                except ObserverBlindnessError as obe:
                    health = self._observer.get_health_snapshot()
                    _logger.error(
                        "[ObserverLoop] OBSERVER BLIND: %s | "
                        "consecutive_misses=%d | schema_rejections=%d | "
                        "total_valid_frames=%d | uptime=%.1fs",
                        obe,
                        health.get("consecutive_misses", -1),
                        self._schema_rejection_count,
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
            _logger.info("[ObserverLoop] Thread exited.")

    # =========================================================================
    # INTROSPECTION
    # =========================================================================

    def is_running(self) -> bool:
        
        with self._lock:
            return (
                self._running
                and self._thread is not None
                and self._thread.is_alive()
            )

    def get_telemetry(self) -> Dict[str, int]:
        
        return {
            "total_valid_frames": self._total_frames,
            "schema_rejection_count": self._schema_rejection_count,
            "ingest_failure_count": self._ingest_failure_count,
            "is_paused": int(self._pause_event.is_set()),
            "is_running": int(self.is_running()),
        }
