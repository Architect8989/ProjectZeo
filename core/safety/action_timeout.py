import threading
import time
from contextlib import contextmanager


class ActionTimeout(Exception):
    pass


# -------------------------------------------------
# Internal Timer
# -------------------------------------------------

class _TimeoutGuard:
    def __init__(self, seconds: float):
        self.seconds = float(seconds)
        self._expired = False
        self._timer: threading.Timer | None = None

    def _trip(self):
        self._expired = True

    def start(self):
        self._timer = threading.Timer(self.seconds, self._trip)
        self._timer.daemon = True
        self._timer.start()

    def cancel(self):
        if self._timer:
            self._timer.cancel()

    def check(self):
        if self._expired:
            raise ActionTimeout("Action timeout exceeded")


# -------------------------------------------------
# Public Context Manager
# -------------------------------------------------

@contextmanager
def action_timeout(seconds: int):
    """
    Cross-platform action timeout.

    - No signals
    - Thread safe
    - Supports nesting
    """

    guard = _TimeoutGuard(seconds)
    guard.start()

    try:
        yield
        guard.check()
    finally:
        guard.cancel()
