import time
import threading
from typing import Dict, Deque, Optional
from collections import deque
import copy


class ObserverBlindnessError(RuntimeError):
    """Observer is alive but perception input is unusable."""


class ObserverCore:
    """
    Passive witness core.

    HARD GUARANTEES:
    - Observer never mutates external state
    - Snapshot isolation (deep copies only)
    - Deterministic blindness semantics
    - No permanent blindness unless upstream fails persistently
    - Observer has ZERO execution authority
    """

    MAX_HISTORY = 1000

    STARTUP_GRACE_TICKS = 30
    STARTUP_GRACE_SECONDS = 15.0

    # GAP-4 FIX: On CPU-only hardware, Qwen2.5-VL takes 40–90s per inference.
    # The VisionRuntime runs at CAPTURE_INTERVAL_SECONDS=0.5 but the actual
    # frame delivery rate is limited by inference speed (1 frame per 40–90s).
    # The ObserverLoop ticks at 5Hz (0.20s). At 5Hz, 15 misses = 3 seconds
    # before the observer declares itself blind — far less than one inference
    # cycle on CPU hardware. This caused permanent spurious blindness on every
    # non-GPU machine, preventing any task from ever being executed.
    #
    # Fix: raise MAX_CONSECUTIVE_MISSES to 600 (= 120 seconds at 5Hz).
    # 120 seconds comfortably covers the worst-case 90s CPU inference latency
    # plus scheduling jitter, while still catching genuine vision failures
    # (e.g. Ollama crash, display disappearing) within 2 minutes.
    # STARTUP_GRACE_SECONDS is also raised to 150s for the same reason.
    STARTUP_GRACE_SECONDS = 150.0
    MAX_CONSECUTIVE_MISSES = 600   # was 15; 600 ticks @ 5Hz = 120s grace
    BLIND_RECOVERY_SECONDS = 10.0  # was 5.0; extend to survive slow frames

    # --------------------------------------------------
    # INIT
    # --------------------------------------------------

    def __init__(self):
        self._clock = time.monotonic
        self.start_time = self._clock()

        self.tick_count: int = 0
        self.last_tick_ts: Optional[float] = None

        self.last_frame_seen: Optional[float] = None
        self.first_frame_seen: Optional[float] = None

        self.observer_healthy: bool = True
        self.blind_reason: Optional[str] = None
        self._blind_timestamp: Optional[float] = None
        self._consecutive_misses: int = 0

        self._lock = threading.RLock()

        # Authoritative state surface (flat perception)
        self._state: Dict[str, object] = {
            "uptime_seconds": 0.0,
            "tick_count": 0,
            "last_tick_ts": None,
            "perception_available": False,
            "perception_frame_ts": None,
            "perception": None,  # RAW perception payload only
        }

        self._history: Deque[Dict[str, object]] = deque(
            maxlen=self.MAX_HISTORY
        )

        print("[OBSERVER] Initialized")

    # --------------------------------------------------
    # TASK BOUNDARY
    # --------------------------------------------------

    def reset_for_new_task(self) -> None:
        with self._lock:
            self._history.clear()

            self.tick_count = 0
            self.last_tick_ts = None
            self.last_frame_seen = None
            self.first_frame_seen = None

            self.observer_healthy = True
            self.blind_reason = None
            self._blind_timestamp = None
            self._consecutive_misses = 0

            self.start_time = self._clock()

            self._state = {
                "uptime_seconds": 0.0,
                "tick_count": 0,
                "last_tick_ts": None,
                "perception_available": False,
                "perception_frame_ts": None,
                "perception": None,
            }

    # --------------------------------------------------
    # BLINDNESS
    # --------------------------------------------------

    def _mark_blind(self, reason: str) -> None:
        if not self.observer_healthy:
            return

        self.observer_healthy = False
        self.blind_reason = reason
        self._blind_timestamp = self._clock()

    # --------------------------------------------------
    # MAIN TICK
    # --------------------------------------------------

    def tick(self) -> Dict[str, object]:
        with self._lock:
            now = self._clock()

            # ---- RECOVERY ----
            if (
                not self.observer_healthy
                and self.last_frame_seen is not None
                and self._blind_timestamp is not None
                and (now - self.last_frame_seen)
                <= self.BLIND_RECOVERY_SECONDS
            ):
                self.observer_healthy = True
                self.blind_reason = None
                self._blind_timestamp = None
                self._consecutive_misses = 0

            if not self.observer_healthy:
                raise ObserverBlindnessError(
                    f"Observer blind: {self.blind_reason}"
                )

            # ---- STARTUP GRACE ----
            if self.first_frame_seen is None:
                grace_ok = (
                    self.tick_count < self.STARTUP_GRACE_TICKS
                    or (now - self.start_time)
                    < self.STARTUP_GRACE_SECONDS
                )
                if not grace_ok:
                    self._mark_blind(
                        "No perception input within startup grace window"
                    )
                    raise ObserverBlindnessError(
                        "Observer blind: no initial perception"
                    )

            # ---- ADVANCE CLOCK ----
            self.tick_count += 1
            self.last_tick_ts = now

            self._state["uptime_seconds"] = round(
                now - self.start_time, 2
            )
            self._state["tick_count"] = self.tick_count
            self._state["last_tick_ts"] = now

            snapshot = copy.deepcopy(self._state)
            self._history.append(snapshot)
            return snapshot

    # --------------------------------------------------
    # PERCEPTION ATTACHMENT (FIXED)
    # --------------------------------------------------

    def attach_perception_state(
        self, perception_state: Dict[str, object]
    ) -> None:
        """
        Attach perception from ObserverLoop.

        perception_state structure:
        {
            "available": bool,
            "frame_ts": float,
            "perception": { ... raw perception payload ... }
        }
        """

        if not isinstance(perception_state, dict):
            return

        with self._lock:
            available = bool(perception_state.get("available"))
            frame_ts = perception_state.get("frame_ts")
            raw_payload = perception_state.get("perception")

            if available and isinstance(raw_payload, dict):
                now = self._clock()
                self.last_frame_seen = now

                if self.first_frame_seen is None:
                    self.first_frame_seen = now

                self._consecutive_misses = 0
            else:
                self._consecutive_misses += 1

            self._state["perception_available"] = available
            self._state["perception_frame_ts"] = frame_ts

            # CRITICAL FIX:
            # Store ONLY the raw perception payload.
            if isinstance(raw_payload, dict):
                self._state["perception"] = copy.deepcopy(raw_payload)
            else:
                self._state["perception"] = None

            if (
                not available
                and self._consecutive_misses
                >= self.MAX_CONSECUTIVE_MISSES
            ):
                self._mark_blind(
                    f"Perception unavailable for "
                    f"{self._consecutive_misses} consecutive ticks"
                )

    # --------------------------------------------------
    # AUTHORITATIVE SNAPSHOT
    # --------------------------------------------------

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            return copy.deepcopy(self._state)

    # --------------------------------------------------
    # INTROSPECTION
    # --------------------------------------------------

    def get_state(self) -> Dict[str, object]:
        with self._lock:
            return copy.deepcopy(self._state)

    def is_healthy(self) -> bool:
        with self._lock:
            return self.observer_healthy

    def get_health_snapshot(self) -> Dict[str, object]:
        with self._lock:
            now = self._clock()
            return {
                "observer_healthy": self.observer_healthy,
                "blind_reason": self.blind_reason,
                "uptime_seconds": round(now - self.start_time, 2),
                "ticks": self.tick_count,
                "first_perception_seen": self.first_frame_seen is not None,
                "consecutive_misses": self._consecutive_misses,
                "history_depth": len(self._history),
        }
