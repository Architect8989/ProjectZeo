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

    # seconds since last healthy frame before considering stale
    STALE_LIMIT_SECONDS = 5.0

    # consecutive failures before degraded
    DEGRADE_AFTER_MISSES = 3

    # consecutive healthy frames required to recover
    RECOVER_AFTER_HITS = 2

    def __init__(self):
        self._clock = time.monotonic

        self._last_frame_mono: Optional[float] = None
        self._misses: int = 0
        self._hits: int = 0
        self._degraded: bool = False

    def update(self, frame_ts: Optional[float], available: bool) -> bool:
        """
        Update health based on latest perception input.

        Returns True iff perception is currently stable.
        """
        now = self._clock()

        # --------------------------------------------------
        # HARD FAILURE CASES
        # --------------------------------------------------
        if not available or frame_ts is None:
            self._record_miss()
            return False

        # --------------------------------------------------
        # STALENESS CHECK (MONOTONIC ONLY)
        # --------------------------------------------------
        if self._last_frame_mono is not None:
            elapsed = now - self._last_frame_mono
            if elapsed > self.STALE_LIMIT_SECONDS:
                self._record_miss()
                return False

        # --------------------------------------------------
        # SUCCESS CASE
        # --------------------------------------------------
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
    # INTERNAL (STRICT STATE MACHINE)
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
