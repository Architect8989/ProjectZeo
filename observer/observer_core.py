import time
import threading
from typing import Dict, Deque, Optional
from collections import deque
import copy


class ObserverBlindnessError(RuntimeError):
    """Observer is alive without usable vision."""


class ObserverCore:
    """
    Passive witness core.

    Guarantees:
    - Snapshot isolation
    - Observer amnesia between tasks
    - Non-permanent blindness semantics
    """

    MAX_HISTORY = 1000

    STARTUP_GRACE_TICKS = 10          # relaxed
    STARTUP_GRACE_SECONDS = 5.0       # relaxed

    MAX_CONSECUTIVE_MISSES = 5        # before blindness

    def __init__(self):
        self._clock = time.monotonic
        self.start_time = self._clock()

        self.tick_count = 0
        self.last_tick_ts: Optional[float] = None
        self.last_frame_seen: Optional[float] = None
        self.first_frame_seen: Optional[float] = None

        self.observer_healthy: bool = True
        self.blind_reason: Optional[str] = None
        self._consecutive_misses = 0

        self._lock = threading.RLock()

        self._state: Dict[str, object] = {
            "uptime_seconds": 0.0,
            "tick_count": 0,
            "last_tick_ts": None,
            "screen_available": False,
            "screen_hash": None,
            "screen_frame_ts": None,
            "ui_snapshot": None,
        }

        self._history: Deque[Dict[str, object]] = deque(
            maxlen=self.MAX_HISTORY
        )

        print("[OBSERVER] Initialized")

    # --------------------------------------------------
    # TASK BOUNDARY
    # --------------------------------------------------

    def reset_for_new_task(self) -> None:
        """
        Enforces full observer amnesia between executions.
        """
        with self._lock:
            self._history.clear()

            self.tick_count = 0
            self.last_tick_ts = None
            self.last_frame_seen = None
            self.first_frame_seen = None

            self.observer_healthy = True
            self.blind_reason = None
            self._consecutive_misses = 0

            self.start_time = self._clock()

            self._state = {
                "uptime_seconds": 0.0,
                "tick_count": 0,
                "last_tick_ts": None,
                "screen_available": False,
                "screen_hash": None,
                "screen_frame_ts": None,
                "ui_snapshot": None,
            }

    # --------------------------------------------------

    def _mark_blind(self, reason: str) -> None:
        with self._lock:
            self.observer_healthy = False
            self.blind_reason = reason

    # --------------------------------------------------

    def tick(self) -> Dict[str, object]:
        with self._lock:
            if not self.observer_healthy:
                raise ObserverBlindnessError(
                    f"Observer blind: {self.blind_reason}"
                )

            now = self._clock()

            # ---- startup grace ----
            if self.last_frame_seen is None:
                grace_ok = (
                    self.tick_count < self.STARTUP_GRACE_TICKS
                    or (now - self.start_time)
                    < self.STARTUP_GRACE_SECONDS
                )

                if not grace_ok:
                    self._mark_blind(
                        "No initial frame within startup grace"
                    )
                    raise ObserverBlindnessError(
                        "Observer blind: no initial frame"
                    )

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

    def attach_screen_state(
        self, screen_state: Dict[str, object]
    ) -> None:
        with self._lock:
            available = bool(screen_state.get("available"))

            if available:
                now = self._clock()
                self.last_frame_seen = now
                if self.first_frame_seen is None:
                    self.first_frame_seen = now
                self._consecutive_misses = 0
            else:
                self._consecutive_misses += 1

            self._state["screen_available"] = available
            self._state["screen_hash"] = screen_state.get(
                "screen_hash"
            )
            self._state["screen_frame_ts"] = screen_state.get(
                "frame_ts"
            )

            # Blind only after sustained loss
            if (
                not available
                and self._consecutive_misses
                >= self.MAX_CONSECUTIVE_MISSES
            ):
                self._mark_blind(
                    f"Screen unavailable for "
                    f"{self._consecutive_misses} consecutive ticks"
                )

    def attach_ui_snapshot(self, ui_snapshot) -> None:
        with self._lock:
            self._state["ui_snapshot"] = copy.deepcopy(
                ui_snapshot
            )

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
                "uptime_seconds": round(
                    now - self.start_time, 2
                ),
                "ticks": self.tick_count,
                "first_frame_seen": self.first_frame_seen
                is not None,
                "consecutive_misses": self._consecutive_misses,
                "history_depth": len(self._history),
        }
