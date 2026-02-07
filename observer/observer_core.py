import time
import threading
from typing import Dict, Deque, Optional
from collections import deque
import copy


class ObserverBlindnessError(RuntimeError):
    """Observer is alive but vision is unusable."""


class ObserverCore:
    """
    Passive witness core.

    HARD GUARANTEES:
    - Observer never mutates external inputs
    - Snapshot isolation (deep copies only)
    - Deterministic blindness semantics
    - No permanent blindness unless upstream fails persistently
    - Observer has NO authority to act
    """

    MAX_HISTORY = 1000

    # ---- FIXED: startup grace widened ----
    STARTUP_GRACE_TICKS = 30
    STARTUP_GRACE_SECONDS = 15.0

    # ---- FIXED: transient loss tolerance ----
    MAX_CONSECUTIVE_MISSES = 15

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
        Hard observer amnesia between executions.
        """
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
                "screen_available": False,
                "screen_hash": None,
                "screen_frame_ts": None,
                "ui_snapshot": None,
            }

    # --------------------------------------------------
    # BLINDNESS
    # --------------------------------------------------

    def _mark_blind(self, reason: str) -> None:
        """
        Transition to blind state. Idempotent.
        """
        if not self.observer_healthy:
            return

        self.observer_healthy = False
        self.blind_reason = reason
        self._blind_timestamp = self._clock()

    # --------------------------------------------------
    # MAIN TICK
    # --------------------------------------------------

    def tick(self) -> Dict[str, object]:
        """
        Advances observer clock.
        Raises ObserverBlindnessError if vision is unusable.
        """
        with self._lock:
            now = self._clock()

            # ---- RECOVERY LOGIC (FIXED) ----
            if (
                not self.observer_healthy
                and self.last_frame_seen is not None
                and self._blind_timestamp is not None
                and (now - self._blind_timestamp) > 5.0
            ):
                self.observer_healthy = True
                self.blind_reason = None
                self._blind_timestamp = None
                self._consecutive_misses = 0

            if not self.observer_healthy:
                raise ObserverBlindnessError(
                    f"Observer blind: {self.blind_reason}"
                )

            # ---- startup grace enforcement ----
            if self.last_frame_seen is None:
                grace_ok = (
                    self.tick_count < self.STARTUP_GRACE_TICKS
                    or (now - self.start_time)
                    < self.STARTUP_GRACE_SECONDS
                )

                if not grace_ok:
                    self._mark_blind(
                        "No initial frame within startup grace window"
                    )
                    raise ObserverBlindnessError(
                        "Observer blind: no initial frame"
                    )

            # ---- advance clock ----
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
    # SCREEN ATTACHMENT
    # --------------------------------------------------

    def attach_screen_state(
        self, screen_state: Dict[str, object]
    ) -> None:
        """
        Attach raw screen availability and hash metadata.
        """
        if not isinstance(screen_state, dict):
            return

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
                "screen_text_hash"
            )
            self._state["screen_frame_ts"] = screen_state.get(
                "frame_ts"
            )

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
        """
        Attach structured UI perception (read-only).
        """
        with self._lock:
            self._state["ui_snapshot"] = (
                copy.deepcopy(ui_snapshot)
                if ui_snapshot is not None
                else None
            )

    # --------------------------------------------------
    # UI QUERY
    # --------------------------------------------------

    def find_click_target(
        self,
        *,
        contains: Optional[str] = None,
        exact: Optional[str] = None,
    ) -> Optional[Dict[str, float]]:
        if not contains and not exact:
            return None

        with self._lock:
            ui_snapshot = self._state.get("ui_snapshot")
            if not isinstance(ui_snapshot, dict):
                return None

            elements = ui_snapshot.get("elements")
            if not isinstance(elements, list):
                return None

            contains_l = contains.lower() if contains else None
            exact_l = exact.lower() if exact else None

            for el in elements:
                if not isinstance(el, dict):
                    continue

                text = str(el.get("text", "")).strip().lower()

                if exact_l is not None:
                    if text != exact_l:
                        continue
                elif contains_l is not None:
                    if contains_l not in text:
                        continue

                x = el.get("x")
                y = el.get("y")

                if (
                    isinstance(x, (int, float))
                    and isinstance(y, (int, float))
                    and 0.0 <= float(x) <= 1.0
                    and 0.0 <= float(y) <= 1.0
                ):
                    return {"x": float(x), "y": float(y)}

            return None

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
                "uptime_seconds": round(
                    now - self.start_time, 2
                ),
                "ticks": self.tick_count,
                "first_frame_seen": self.first_frame_seen is not None,
                "consecutive_misses": self._consecutive_misses,
                "history_depth": len(self._history),
        }
