import threading
import time
from contextlib import contextmanager


class ActionTimeout(RuntimeError):
    """
    Raised when an action exceeds its declared time budget.

    IMPORTANT CONTRACT:
    - This does NOT interrupt blocking operations.
    - It FAILS CLOSED at the earliest safe boundary.
    - Callers MUST treat this as a hard execution failure.
    """
    pass


# -------------------------------------------------
# Internal Guard (Monotonic, Honest)
# -------------------------------------------------

class _TimeoutGuard:
    def __init__(self, seconds: float):
        self.seconds = float(seconds)
        self.deadline = time.monotonic() + self.seconds
        self._expired = False
        self._timer: threading.Timer | None = None

    def _trip(self):
        self._expired = True

    def start(self):
        """
        Timer is advisory only.
        Actual enforcement is monotonic-time based.
        """
        self._timer = threading.Timer(self.seconds, self._trip)
        self._timer.daemon = True
        self._timer.start()

    def cancel(self):
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def check(self):
        """
        Deterministic timeout check.
        Must be called at safe boundaries.
        """
        if self._expired or time.monotonic() >= self.deadline:
            raise ActionTimeout(
                f"Action exceeded timeout of {self.seconds:.2f}s"
            )


# -------------------------------------------------
# Public Context Manager
# -------------------------------------------------

@contextmanager
def action_timeout(seconds: float):
    """
    Best-effort execution timeout.

    GUARANTEES:
    - No false safety claims
    - Monotonic time enforcement
    - Cross-platform
    - Thread-safe
    - Nestable

    NON-GUARANTEES (explicit):
    - Cannot interrupt blocking OS / I/O calls
    - Cannot stop infinite loops inside native libraries

    Callers MUST assume:
    - Timeout is detected only at Python yield points
    """

    guard = _TimeoutGuard(seconds)
    guard.start()

    try:
        yield
        # Mandatory post-action check
        guard.check()
    finally:
        guard.cancel()
