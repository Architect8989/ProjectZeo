import time
from typing import Optional


class PerceptionHealth:
    """
    Tracks perception quality and recovers understanding,
    never the environment.
    """

    STALE_LIMIT = 5.0
    UNSTABLE_LIMIT = 3

    def __init__(self):
        self.last_good_ts: Optional[float] = None
        self.unstable_count = 0

    def update(self, frame_ts: Optional[float], available: bool) -> bool:
        """
        Returns True if perception is considered stable.
        """
        now = time.time()

        if not available or frame_ts is None:
            self.unstable_count += 1
            return False

        if self.last_good_ts and (now - frame_ts) > self.STALE_LIMIT:
            self.unstable_count += 1
            return False

        self.last_good_ts = now
        self.unstable_count = 0
        return True

    def degraded(self) -> bool:
        return self.unstable_count >= self.UNSTABLE_LIMIT
