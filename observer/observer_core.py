import time
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

    def __init__(self):
        self.start_time = time.monotonic()
        self.tick_count = 0
        self.last_tick_ts: Optional[float] = None
        self.last_frame_seen_mono: Optional[float] = None

        self.state: Dict[str, object] = {
            "uptime_seconds": 0.0,
            "tick_count": 0,
            "last_tick_ts": None,
            "screen_available": False,
            "screen_text_hash": None,
            "screen_frame_ts": None,
            "ui_snapshot": None,
        }

        self.history: Deque[Dict[str, object]] = deque(maxlen=self.MAX_HISTORY)

        print("[OBSERVER] Initialized (witness mode)")

    def tick(self) -> Dict[str, object]:
        now = time.monotonic()

        if self.last_frame_seen_mono is None:
            raise ObserverBlindnessError("Observer has never seen a frame")

        if now - self.last_frame_seen_mono > self.MAX_NO_FRAME_SECONDS:
            raise ObserverBlindnessError("Observer lost vision")

        self.tick_count += 1
        self.last_tick_ts = now

        self.state.update(
            {
                "uptime_seconds": round(now - self.start_time, 2),
                "tick_count": self.tick_count,
                "last_tick_ts": now,
            }
        )

        self.history.append(dict(self.state))
        return dict(self.state)

    def attach_screen_state(self, screen_state: Dict[str, object]) -> None:
        if screen_state.get("available"):
            self.last_frame_seen_mono = time.monotonic()

        self.state.update(
            {
                "screen_available": screen_state.get("available"),
                "screen_text_hash": screen_state.get("screen_text_hash"),
                "screen_frame_ts": screen_state.get("frame_ts"),
            }
        )

    def attach_ui_snapshot(self, ui_snapshot) -> None:
        self.state["ui_snapshot"] = ui_snapshot

    def get_state(self) -> Dict[str, object]:
        return dict(self.state)
