import time
import threading
from typing import Optional


class InputSource:
    SOC = "SOC"
    HUMAN = "HUMAN"


class InputTracker:
    """
    Tracks input origin.

    This module NEVER blocks input.
    It only classifies it.

    Contract:
    - Uses monotonic clock
    - Thread-safe
    - Deterministic thresholds
    """

    SOC_ACTION_WINDOW_SECONDS = 0.25

    def __init__(self):
        self._last_soc_action_ts: Optional[float] = None
        self._lock = threading.Lock()

    # -------------------------------------------------

    def mark_soc_action(self) -> None:
        with self._lock:
            self._last_soc_action_ts = time.monotonic()

    def classify_input(self, event_ts: float) -> str:
        """
        event_ts MUST be monotonic time.
        """

        with self._lock:
            last = self._last_soc_action_ts

        if last is None:
            return InputSource.HUMAN

        delta = event_ts - last

        # Guard against clock anomalies
        if delta < 0:
            return InputSource.HUMAN

        if delta > self.SOC_ACTION_WINDOW_SECONDS:
            return InputSource.HUMAN

        return InputSource.SOC
