import time
from typing import Optional


class InputSource:
    SOC = "SOC"
    HUMAN = "HUMAN"


class InputTracker:
    """
    Tracks input origin.

    This module NEVER blocks input.
    It only classifies it.
    """

    def __init__(self):
        self._last_soc_action_ts: Optional[float] = None

    def mark_soc_action(self):
        self._last_soc_action_ts = time.monotonic()

    def classify_input(self, event_ts: float) -> str:
        """
        If input occurred outside SOC action window,
        classify as HUMAN.
        """
        if self._last_soc_action_ts is None:
            return InputSource.HUMAN

        if (event_ts - self._last_soc_action_ts) > 0.2:
            return InputSource.HUMAN

        return InputSource.SOC
