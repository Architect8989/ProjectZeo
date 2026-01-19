import time
from typing import Dict


class ObserverCore:
    """
    Passive observer core.
    This class must NEVER interact with the system.
    It only maintains internal state and heartbeat.
    """

    def __init__(self):
        self.start_time = time.time()
        self.tick_count = 0
        self.last_tick_ts = None

        # Placeholder for future observed state
        self.state: Dict[str, object] = {
            "uptime_seconds": 0.0,
            "tick_count": 0,
            "last_tick_ts": None,
        }

        print("[OBSERVER] Initialized")

    def tick(self) -> Dict[str, object]:
        """
        Single observer pulse.
        No side effects. No IO. No perception yet.
        """
        now = time.time()
        self.tick_count += 1
        self.last_tick_ts = now

        self.state["uptime_seconds"] = round(now - self.start_time, 2)
        self.state["tick_count"] = self.tick_count
        self.state["last_tick_ts"] = self.last_tick_ts

        return self.state

    def get_state(self) -> Dict[str, object]:
        """
        Read-only access to observer state.
        """
        return dict(self.state)
