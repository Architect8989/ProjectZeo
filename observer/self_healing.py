import time
from typing import Optional


class PerceptionHealth:
    """
    Tracks perception quality over time.

    Guarantees:
    - Monotonic-only timing
    - No oscillation under jitter
    - Explicit degradation + recovery thresholds
    """

    # seconds since last fresh frame before considering stale
    STALE_LIMIT_SECONDS = 5.0

    # consecutive failures before degraded
    DEGRADE_AFTER_MISSES = 3

    # consecutive successes required to recover
    RECOVER_AFTER_HITS = 2

    def __init__(self):
        self._clock = time.monotonic

        self._last_frame_mono: Optional[float] = None
        self._misses = 0
        self._hits = 0
        self._degraded = False

    def update(self, frame_ts: Optional[float], available: bool) -> bool:
        """
        Update health based on latest perception input.

        Returns True if perception is currently stable.
        """
        now = self._clock()

        # --- failure cases ---
        if not available or frame_ts is None:
            self._record_miss()
            return False

        # frame_ts is wall-clock; compare staleness against now safely
        if self._last_frame_mono is not None:
            if (now - self._last_frame_mono) > self.STALE_LIMIT_SECONDS:
                self._record_miss()
                return False

        # --- success case ---
        self._last_frame_mono = now
        self._record_hit()
        return not self._degraded

    def degraded(self) -> bool:
        return self._degraded

    def reset(self) -> None:
        """
        Full reset of health state.
        """
        self._last_frame_mono = None
        self._misses = 0
        self._hits = 0
        self._degraded = False

    # --------------------------------------------------
    # INTERNAL
    # --------------------------------------------------

    def _record_miss(self) -> None:
        self._misses += 1
        self._hits = 0

        if self._misses >= self.DEGRADE_AFTER_MISSES:
            self._degraded = True

    def _record_hit(self) -> None:
        self._hits += 1
        self._misses = 0

        if self._degraded and self._hits >= self.RECOVER_AFTER_HITS:
            self._degraded = False
