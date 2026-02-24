import threading
import time
import concurrent.futures
from contextlib import contextmanager
from typing import Optional


class ActionTimeout(RuntimeError):
    """
    Raised when an action exceeds its declared time budget.

    HARD CONTRACTS (unchanged):
    - This does NOT interrupt blocking OS / I/O calls.
    - Timeout is detected only when the guarded code yields back to Python,
      OR when run_with_timeout() unblocks the calling thread after the deadline.
    - Callers MUST treat this as a hard execution failure.

    FIX RB-3: The original implementation used a background threading.Timer
    that set _expired=True, then checked that flag only AFTER the guarded code
    returned. This meant:
      1. A blocking call (pyautogui waiting on display queue, type_text stalling
         on a frozen window) would run indefinitely — the timer fired but had no
         mechanism to interrupt the blocking C/OS call or unblock the caller.
      2. The 30-second bound in action_timeout() was advisory only; stuck UI
         operations could hang the main loop for up to MAX_TASK_SECONDS (90 min).

    New design: run_with_timeout() submits the guarded callable to a
    caller-supplied (or implicitly created) ThreadPoolExecutor and calls
    future.result(timeout=seconds). This UNBLOCKS THE CALLING THREAD after the
    deadline (raising ActionTimeout) while the background thread continues
    running until the blocking call naturally returns or the process exits.

    Limitation (unchanged, now explicitly documented):
    True preemptive interruption of blocking native calls is impossible in
    CPython without SIGALRM (Linux only, not thread-safe with threading) or
    process isolation. The ThreadPoolExecutor approach is the strongest guarantee
    achievable within CPython's threading model.
    """
    pass


# -------------------------------------------------
# FIX RB-A3: Per-task executor — NO module-level singleton
# -------------------------------------------------
#
# REMOVED: module-level `_UI_EXECUTOR = ThreadPoolExecutor(max_workers=1)`
#
# Root cause of the bug:
#   A module-level singleton executor has exactly ONE worker thread shared
#   across all tasks in the process. When task A submits a UI action that
#   blocks (e.g. pyautogui waiting on a frozen display queue), that single
#   worker is occupied. Task B's action submission succeeds (the future is
#   enqueued) but the worker never picks it up until task A's blocking call
#   returns. Task B's 30-second timeout therefore starts running while the
#   worker is still processing task A — the timeout guarantee is violated.
#   In the worst case, a 1-hour-long blocking call in task A means task B's
#   timeout fires after 30 seconds of waiting, even though task B's UI action
#   never actually executed.
#
# Fix:
#   The executor is now created per-task inside operate_main() and passed
#   explicitly to run_with_timeout() via the `executor` parameter. Each task
#   gets its own single-worker executor, so its timeout guarantee is isolated.
#
#   operate_main() shuts down the executor in a finally block after the task
#   completes (success, failure, or exception), ensuring no thread pool leaks.
#
#   For callers that don't need task isolation (one-off uses, tests), passing
#   executor=None creates a temporary executor that is shut down immediately
#   after the call. This is slightly less efficient but always correct.
#
# Usage in operate.py:
#   # At the start of operate_main():
#   _task_executor = concurrent.futures.ThreadPoolExecutor(
#       max_workers=1, thread_name_prefix="ui_timeout_worker"
#   )
#   try:
#       ...
#       result = run_with_timeout(fn, seconds=30, executor=_task_executor)
#       ...
#   finally:
#       _task_executor.shutdown(wait=False)


def run_with_timeout(
    fn,
    *,
    seconds: float,
    operation_hint: str = "",
    executor: Optional[concurrent.futures.ThreadPoolExecutor] = None,
):
    """
    Execute fn() in a background thread. Unblock the calling thread and raise
    ActionTimeout if fn() does not complete within `seconds`.

    FIX RB-A3: The `executor` parameter replaces the removed module-level
    singleton. Callers MUST pass a per-task executor created in operate_main()
    to ensure timeout isolation between concurrent tasks.

    Parameters
    ----------
    fn : callable
        Zero-argument callable to execute with a timeout.
    seconds : float
        Maximum wall-clock seconds to wait for fn() to complete.
    operation_hint : str
        Human-readable label for error messages and logging.
    executor : ThreadPoolExecutor, optional
        Per-task executor to submit fn() to. If None, a temporary
        single-use executor is created and shut down after the call.
        Callers that care about timeout isolation MUST pass an explicit
        executor created per task.

    Returns
    -------
    Any
        Return value of fn() if it completes within `seconds`.

    Raises
    ------
    ActionTimeout
        If fn() does not complete within `seconds`. The background thread
        continues running — callers should treat this as a hard failure and
        allow the background thread to drain naturally.
    Exception
        Any exception raised by fn() is re-raised unchanged (not wrapped).

    Notes
    -----
    NON-GUARANTEES (explicit):
    - Cannot kill or interrupt the background thread.
    - Cannot interrupt blocking system calls in native libraries (X11, Win32).
    - Background thread persists until it unblocks or process exits.
    - Not safe to call with fn() that mutates shared state without locking —
      the background thread may still be running after ActionTimeout is raised.
    """
    seconds = float(seconds)
    if seconds <= 0:
        raise ValueError(f"timeout must be positive, got {seconds}")

    _own_executor = False
    if executor is None:
        # Fallback: create a temporary executor for this single call.
        # Less efficient than a per-task executor but always correct.
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="ui_timeout_worker_adhoc",
        )
        _own_executor = True

    try:
        future = executor.submit(fn)

        try:
            return future.result(timeout=seconds)
        except concurrent.futures.TimeoutError:
            hint = f" [{operation_hint}]" if operation_hint else ""
            raise ActionTimeout(
                f"Action exceeded timeout of {seconds:.2f}s{hint}. "
                "The background UI thread is still running and will continue "
                "until the blocking OS call returns or the process exits."
            )
        except concurrent.futures.CancelledError:
            raise ActionTimeout(
                f"Action was cancelled before execution"
                f"{': ' + operation_hint if operation_hint else ''}."
            )
    finally:
        if _own_executor:
            # Shut down the temporary executor without waiting for pending work.
            # The background thread (if still running) will complete naturally.
            executor.shutdown(wait=False)


# -------------------------------------------------
# Legacy Advisory Guard (retained for yield-point checks)
# -------------------------------------------------

class _TimeoutGuard:
    """
    Monotonic time guard for yield-point checking within Python loops.

    DOES NOT interrupt blocking calls. Use run_with_timeout() instead for
    any code that calls into OS/native libraries.

    Retained for:
    - Secondary safety check inside the action_timeout() context manager.
    - Timeout enforcement in Python-only loops that yield frequently.
    """
    def __init__(self, seconds: float):
        self.seconds = float(seconds)
        self.deadline = time.monotonic() + self.seconds
        self._expired = False
        self._timer: threading.Timer | None = None

    def _trip(self):
        self._expired = True

    def start(self):
        """
        Start a background timer. Advisory only — sets _expired flag.
        Actual enforcement is via check() at Python yield points.
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
        Yield-point timeout check.

        Raises ActionTimeout if the deadline has passed.

        NON-GUARANTEE: Only fires if Python execution reaches this call.
        Cannot detect or interrupt blocking OS/native-library calls.
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
    Best-effort execution timeout context manager.

    FIX RB-3: Updated to add run_with_timeout() as the recommended strong
    enforcement mechanism for blocking I/O. The context manager itself remains
    a lightweight yield-point guard via _TimeoutGuard — callers that use
    `with action_timeout(N):` as a block wrapper get the legacy advisory check.

    For true calling-thread unblocking on blocking OS calls, use
    run_with_timeout() directly with a per-task executor:

        result = run_with_timeout(
            lambda: os_backend.mouse(coord),
            seconds=30,
            executor=_task_executor,
        )

    GUARANTEES:
    - No false safety claims.
    - Monotonic time enforcement at Python yield points.
    - Cross-platform, thread-safe, nestable.

    NON-GUARANTEES (explicit, audit RB-3):
    - The context manager alone cannot interrupt blocking OS / I/O calls.
    - Use run_with_timeout() with an explicit executor for blocking native calls.
    - Background threads (from run_with_timeout) persist until naturally unblocked.

    Callers MUST assume:
    - Without run_with_timeout(), timeout is detected only at Python yield points.
    """

    guard = _TimeoutGuard(seconds)
    guard.start()

    try:
        yield guard  # caller can call guard.check() or use run_with_timeout()
        # Mandatory post-block check for any Python-level overruns
        guard.check()
    finally:
        guard.cancel()
