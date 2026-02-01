import time
import threading
from typing import Dict, Deque, Optional
from collections import deque


class ObserverBlindnessError(RuntimeError):
    """Observer is alive without vision."""


class ObserverCore:
    """
    Passive witness core.

    ROLE:
    - Owns time
    - Owns continuity
    - Owns before/after truth
    """

    MAX_HISTORY = 1000
    MAX_NO_FRAME_SECONDS = 0.5

    # ---- cold-start tolerance ----
    STARTUP_GRACE_TICKS = 5
    STARTUP_GRACE_SECONDS = 2.0

    def __init__(self):
        # -------------------------------------------------
        # CLOCK SELECTION (CRITICAL FIX)
        # -------------------------------------------------
        try:
            self._clock = time.monotonic
            self.using_monotonic = True
        except Exception:
            self._clock = time.time
            self.using_monotonic = False

        self.start_time = self._clock()
        self.tick_count = 0
        self.last_tick_ts: Optional[float] = None
        self.last_frame_seen: Optional[float] = None
        self.first_frame_seen: Optional[float] = None

        self.observer_healthy: bool = True
        self.blind_reason: Optional[str] = None

        # Re-entrant to avoid self-deadlock
        self._lock = threading.RLock()

        self.state: Dict[str, object] = {
            "uptime_seconds": 0.0,
            "tick_count": 0,
            "last_tick_ts": None,
            "screen_available": False,
            "screen_text_hash": None,
            "screen_frame_ts": None,
            "ui_snapshot": None,
        }

        self.history: Deque[Dict[str, object]] = deque(
            maxlen=self.MAX_HISTORY
        )

        print(
            "[OBSERVER] Initialized "
            f"(clock={'monotonic' if self.using_monotonic else 'wall'})"
        )

    # -------------------------------------------------

    def _mark_blind(self, reason: str) -> None:
        """
        Monotonic blindness transition.
        Once blind, never recovers.
        """
        with self._lock:
            if not self.observer_healthy:
                return
            self.observer_healthy = False
            self.blind_reason = reason

    # -------------------------------------------------

    def tick(self) -> Dict[str, object]:
        with self._lock:

            if not self.observer_healthy:
                raise ObserverBlindnessError(
                    f"Observer permanently blind: {self.blind_reason}"
                )

            now = self._clock()

            # ---- COLD START HANDLING ----
            if self.last_frame_seen is None:
                grace_ticks_ok = self.tick_count < self.STARTUP_GRACE_TICKS
                grace_time_ok = (now - self.start_time) < self.STARTUP_GRACE_SECONDS

                if grace_ticks_ok or grace_time_ok:
                    self.tick_count += 1
                    self.last_tick_ts = now

                    self.state["uptime_seconds"] = round(
                        now - self.start_time, 2
                    )
                    self.state["tick_count"] = self.tick_count
                    self.state["last_tick_ts"] = now

                    self.history.append(dict(self.state))
                    return dict(self.state)

                self._mark_blind("Observer never received initial frame")
                raise ObserverBlindnessError(
                    "Observer permanently blind: no initial frame"
                )

            # ---- POST-STARTUP BLINDNESS CHECK ----
            if now - self.last_frame_seen > self.MAX_NO_FRAME_SECONDS:
                self._mark_blind("Observer lost vision")
                raise ObserverBlindnessError("Observer lost vision")

            # ---- NORMAL TICK ----
            self.tick_count += 1
            self.last_tick_ts = now

            self.state = {
                "uptime_seconds": round(now - self.start_time, 2),
                "tick_count": self.tick_count,
                "last_tick_ts": now,
                "screen_available": self.state["screen_available"],
                "screen_text_hash": self.state["screen_text_hash"],
                "screen_frame_ts": self.state["screen_frame_ts"],
                "ui_snapshot": self.state["ui_snapshot"],
            }

            self.history.append(dict(self.state))
            return dict(self.state)

    # -------------------------------------------------

    def attach_screen_state(self, screen_state: Dict[str, object]) -> None:
        with self._lock:
            if screen_state.get("available"):
                now = self._clock()
                self.last_frame_seen = now
                if self.first_frame_seen is None:
                    self.first_frame_seen = now

            self.state["screen_available"] = bool(
                screen_state.get("available")
            )
            self.state["screen_text_hash"] = screen_state.get("screen_text_hash")
            self.state["screen_frame_ts"] = screen_state.get("frame_ts")

    def attach_ui_snapshot(self, ui_snapshot) -> None:
        with self._lock:
            self.state["ui_snapshot"] = ui_snapshot

    # -------------------------------------------------

    def get_state(self) -> Dict[str, object]:
        with self._lock:
            return dict(self.state)

    # -------------------------------------------------

    def is_healthy(self) -> bool:
        with self._lock:
            return bool(self.observer_healthy)

    def get_health_snapshot(self) -> Dict[str, object]:
        """
        Forensic-grade observer health snapshot.
        """
        with self._lock:
            now = self._clock()
            last_seen = self.last_frame_seen

            return {
                "observer_healthy": self.observer_healthy,
                "blind_reason": self.blind_reason,
                "uptime_seconds": round(now - self.start_time, 2),
                "ticks": self.tick_count,
                "last_tick_ts": self.last_tick_ts,
                "last_frame_seen_age": (
                    now - last_seen if last_seen else None
                ),
                "first_frame_seen": self.first_frame_seen is not None,
                "history_depth": len(self.history),
        }
