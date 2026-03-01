import threading
import time
import traceback
import logging
from typing import Optional, Dict, Any

from observer.observer_core import ObserverCore, ObserverBlindnessError
from core.vision.vision_runtime import VisionRuntime
from core.vision.world_graph import WorldGraph

_logger = logging.getLogger(__name__)

_REQUIRED_PERCEPTION_KEYS: frozenset = frozenset({
    "available",
})

# Keys that VisionRuntime may use interchangeably. If 'elements' is present
# but 'entities' is absent, we normalize 'elements' → 'entities' so
# world_graph.ingest() receives the canonical key it expects.
_ENTITY_KEY_ALIASES: tuple = ("entities", "elements", "nodes", "ui_elements")


def _validate_perception_schema(perception: dict) -> "dict | None":
    
    if not isinstance(perception, dict):
        _logger.warning(
            "[ObserverLoop] Schema validation failed: perception is %s, expected dict",
            type(perception).__name__,
        )
        return None

    # ---- 1. Validate 'available' (required) ----
    if "available" not in perception:
        _logger.warning(
            "[ObserverLoop] Schema validation failed: 'available' key missing. "
            "VisionRuntime output schema may have changed. "
            "Keys present: %s",
            sorted(perception.keys()),
        )
        return None

    validated = dict(perception)  # shallow copy — ingest() should not mutate
    validated["available"] = bool(validated["available"])

    # ---- 2. Normalize 'frame_ts' ----
    raw_ts = validated.get("frame_ts")
    if raw_ts is not None:
        try:
            validated["frame_ts"] = float(raw_ts)
        except (TypeError, ValueError):
            _logger.debug(
                "[ObserverLoop] Schema normalization: 'frame_ts' value %r is not float, "
                "setting to None.",
                raw_ts,
            )
            validated["frame_ts"] = None
    else:
        validated.setdefault("frame_ts", None)

    # ---- 3. Normalize entity list key ----
    # If the canonical 'entities' key is absent but an alias is present,
    # copy the alias value to 'entities' so world_graph.ingest() receives it.
    if "entities" not in validated:
        for alias in _ENTITY_KEY_ALIASES[1:]:  # skip 'entities' itself
            if alias in validated:
                _logger.debug(
                    "[ObserverLoop] Schema normalization: '%s' → 'entities' "
                    "(VisionRuntime used a non-canonical entity list key).",
                    alias,
                )
                validated["entities"] = validated[alias]
                break
        else:
            # No entity list found — default to empty list (not an error;
            # the first few frames from VisionRuntime may have no entities yet)
            validated.setdefault("entities", [])

    # Ensure entity list is actually a list
    if not isinstance(validated.get("entities"), list):
        _logger.debug(
            "[ObserverLoop] Schema normalization: 'entities' is %s, resetting to [].",
            type(validated.get("entities")).__name__,
        )
        validated["entities"] = []

    # ---- 4. Normalize 'focused_app' ----
    if "focused_app" in validated and not isinstance(validated["focused_app"], str):
        _logger.debug(
            "[ObserverLoop] Schema normalization: 'focused_app' is %s (expected str), removing.",
            type(validated["focused_app"]).__name__,
        )
        del validated["focused_app"]

    return validated


class ObserverLoop:
    

    DEFAULT_TICK_INTERVAL = 0.20  # 5 Hz baseline

    def __init__(
        self,
        *,
        observer: ObserverCore,
        vision_runtime: VisionRuntime,
        world_graph: Optional[WorldGraph] = None,
        tick_interval: float = DEFAULT_TICK_INTERVAL,
    ):
        self._observer = observer
        self._vision = vision_runtime
        self._world_graph = world_graph

        self._tick_interval = max(0.05, float(tick_interval))

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self._lock = threading.Lock()

        # Telemetry counters — readable via introspection
        self._total_frames: int = 0
        self._schema_rejection_count: int = 0
        self._ingest_failure_count: int = 0

        # GPU contention management: pause() during EXECUTING mode
        # so the execution LLM gets the full GPU instead of sharing
        # with the background vision model
        self._pause_event = threading.Event()  # set = paused

    # ==================================================
    # LIFECYCLE
    # ==================================================

    def start(self) -> None:
        with self._lock:
            if self._running:
                return

            # FIX-RESTART: Clear stop event before starting so a restart after
            # ObserverBlindnessError (which sets _stop_event) works correctly.
            # Without this, stop() → start() sequence left _stop_event set,
            # causing the new thread to exit immediately on first iteration.
            self._stop_event.clear()
            self._running = True

            self._thread = threading.Thread(
                target=self._run,
                name="observer-loop",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return

            self._stop_event.set()
            thread = self._thread
            self._thread = None
            self._running = False

        if thread:
            thread.join(timeout=2.0)

    def pause(self) -> None:
        """Pause vision inference — call when entering EXECUTING to free GPU."""
        self._pause_event.set()

    def resume(self) -> None:
        """Resume vision inference — call after RESTORING completes."""
        self._pause_event.clear()

    # ==================================================
    # MAIN LOOP
    # ==================================================

    def _run(self) -> None:
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
                        # P1-SCHEMA FIX: Validate and normalize before any downstream use.
                        validated_perception = _validate_perception_schema(raw_perception)

                        if validated_perception is not None:
                            self._total_frames += 1
                            self._observer.attach_perception_state(
                                {
                                    "available": True,
                                    "frame_ts": validated_perception.get("frame_ts"),
                                    "perception": validated_perception,
                                }
                            )

                            if self._world_graph is not None:
                                try:
                                    self._world_graph.ingest(validated_perception)
                                except Exception as ingest_err:
                                    # World graph failure must NEVER break observer loop
                                    self._ingest_failure_count += 1
                                    _logger.debug(
                                        "[ObserverLoop] world_graph.ingest() failed "
                                        "(total failures: %d): %s",
                                        self._ingest_failure_count,
                                        ingest_err,
                                    )
                        else:
                            # Schema validation failed — count rejection and emit
                            # unavailable state so ObserverCore increments its miss counter
                            self._schema_rejection_count += 1
                            if self._schema_rejection_count % 10 == 1:
                                # Log every 1st, 11th, 21st... rejection to avoid spam
                                _logger.warning(
                                    "[ObserverLoop] VisionRuntime schema rejection "
                                    "#%d — perception dict does not match expected "
                                    "schema. Check VisionRuntime output format. "
                                    "Observer will remain in warmup until schema is corrected.",
                                    self._schema_rejection_count,
                                )
                            self._observer.attach_perception_state(
                                {
                                    "available": False,
                                    "frame_ts": None,
                                    "perception": None,
                                }
                            )
                    else:
                        self._observer.attach_perception_state(
                            {
                                "available": False,
                                "frame_ts": None,
                                "perception": None,
                            }
                        )

                    # Authoritative tick
                    self._observer.tick()

                except ObserverBlindnessError as obe:
                    # HARDEN-OL-1: Structured telemetry on blindness so operators
                    # can distinguish genuine failure from slow CPU inference.
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
                    # FIX-BLIND: Do NOT set _stop_event on blindness — let
                    # main.py's ObserverBlindnessError handler (which restarts
                    # vision_runtime then calls observer.reset_for_new_task())
                    # propagate the recovery. Setting stop_event here meant
                    # observer_loop.start() on restart silently exited immediately
                    # because _stop_event was still set.
                    try:
                        self._observer.attach_perception_state(
                            {
                                "available": False,
                                "frame_ts": None,
                                "perception": None,
                            }
                        )
                    except Exception:
                        pass

                    # Propagate to main.py for restart handling
                    raise

                except Exception:
                    traceback.print_exc()
                    try:
                        self._observer.attach_perception_state(
                            {
                                "available": False,
                                "frame_ts": None,
                                "perception": None,
                            }
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

    # ==================================================
    # INTROSPECTION
    # ==================================================

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def get_telemetry(self) -> Dict[str, int]:
        """
        Return loop telemetry for operator dashboards and health monitoring.

        Returns
        -------
        dict with keys:
          total_valid_frames    — frames that passed schema validation
          schema_rejection_count — frames rejected by schema validator
          ingest_failure_count  — frames where world_graph.ingest() raised
        """
        return {
            "total_valid_frames": self._total_frames,
            "schema_rejection_count": self._schema_rejection_count,
            "ingest_failure_count": self._ingest_failure_count,
        }
