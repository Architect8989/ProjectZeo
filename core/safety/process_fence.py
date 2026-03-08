"""
core/safety/process_fence.py — Process Fence (Layer 7, Research §8.1)

Implements the process spawn fence described in the research report. Any
child processes spawned inside the `process_fence()` context manager are
terminated on failure or exception, preventing process leakage across task
boundaries.

Usage:
    from core.safety.process_fence import process_fence
    with process_fence():
        some_action_that_may_spawn_processes()

The fence works by:
  1. Snapshotting alive PIDs before the block
  2. On any exception, diffing PIDs and terminating newcomers
  3. On success, letting processes live (user-visible apps should persist)

Research reference: §8.1 — "processes started during task execution survive
failure events and contaminate subsequent tasks."
"""
from __future__ import annotations

import contextlib
import logging
import os

_logger = logging.getLogger(__name__)


@contextlib.contextmanager
def process_fence(*, terminate_on_success: bool = False):
    """
    Context manager that tracks spawned processes and terminates them on failure.

    Args:
        terminate_on_success: If True, also terminate spawned processes on
                              clean exit. Default False — user apps should
                              persist after a successful action.
    """
    try:
        import psutil  # type: ignore
        _psutil_available = True
    except ImportError:
        _psutil_available = False
        _logger.debug("[ProcessFence] psutil not available — fence disabled.")

    if not _psutil_available:
        yield
        return

    try:
        import psutil  # type: ignore

        # Snapshot PIDs before the block
        before_pids: set = set()
        for p in psutil.process_iter(["pid"]):
            try:
                before_pids.add(p.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        yielded_ok = False
        try:
            yield
            yielded_ok = True
        finally:
            if not yielded_ok or terminate_on_success:
                # Find processes that appeared inside the block
                after_pids: set = set()
                for p in psutil.process_iter(["pid"]):
                    try:
                        after_pids.add(p.pid)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                spawned = after_pids - before_pids
                if spawned:
                    _logger.info(
                        "[ProcessFence] Fence triggered — terminating %d spawned process(es): %s",
                        len(spawned), sorted(spawned),
                    )
                    for pid in sorted(spawned):
                        try:
                            proc = psutil.Process(pid)
                            proc.terminate()
                            _logger.debug("[ProcessFence] Terminated PID %d (%s)", pid, proc.name())
                        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                            pass
                        except Exception as e:
                            _logger.debug("[ProcessFence] Could not terminate PID %d: %s", pid, e)
    except Exception as fence_exc:
        _logger.warning("[ProcessFence] Fence error (non-fatal, proceeding): %s", fence_exc)
        yield
