import time
import threading
import json
import os
import sys
import pathlib
from enum import Enum
from typing import Optional, Deque, Dict, Callable
from collections import deque


class SystemMode(str, Enum):
    OBSERVER = "OBSERVER"
    ARMED = "ARMED"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    RESTORING = "RESTORING"


class ModeTransitionError(Exception):
    pass


class VisionUnavailableError(ModeTransitionError):
    pass


class ObserverUnavailableError(ModeTransitionError):
    pass


class PlanningTimeoutError(ModeTransitionError):
    pass


class ArmedTimeoutError(ModeTransitionError):
    pass


class ModeController:

    MAX_TRANSITION_HISTORY = 2000
    MAX_PLANNING_SECONDS = 180.0
    MAX_PLAN_ID_LENGTH = 128

    # §R8: ARMED mode timeout — prevents infinite ARMED stall.
    # FIX RB-5 / H-6: Raised from 30.0 to LLM_CALL_TIMEOUT_SECONDS + 60.0 (210.0s).
    # CPU-only Ollama inference takes 40–90s. With MAX_ARMED_SECONDS=30s the
    # ARMED timeout fired before planning completed, triggering ArmedTimeoutError
    # → OBSERVER revert → replan count not incremented → infinite OBSERVER loop.
    # Formula: LLM_CALL_TIMEOUT_SECONDS (150s) + 60s safety margin = 210s.
    MAX_ARMED_SECONDS: float = 210.0

    _MODULE_ROOT = pathlib.Path(__file__).resolve().parents[1]
    TRANSITION_LOG_PATH = _MODULE_ROOT / "logs" / "mode_transitions.jsonl"

    def __init__(self):
        self._lock = threading.RLock()

        self._mode: SystemMode = SystemMode.OBSERVER
        self._mode_entered_at: float = time.time()
        self._last_transition_reason: Optional[str] = None

        self._planning_started_at: Optional[float] = None
        self._replan_in_progress: bool = False

        self._snapshot_id: Optional[str] = None
        self._snapshot_consumed: bool = False

        self._intent: Optional[str] = None
        self._intent_frozen: bool = False

        self._planning_completed: bool = False
        self._execution_plan_attached: bool = False
        self._execution_plan_id: Optional[str] = None

        self._vision_ok: bool = False
        self._observer_healthy: bool = True
        self._failure_reason: Optional[str] = None

        self._input_locked: bool = False
        self._llm_callable: Optional[Callable] = None

        self._transition_history: Deque[Dict[str, object]] = deque(
            maxlen=self.MAX_TRANSITION_HISTORY
        )

        # FIX SI-1 / RB-A1: Guard the log directory creation against read-only
        # filesystems (containers, NFS mounts, CI environments).
        #
        # Previous code: self.TRANSITION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        # This raised PermissionError unconditionally on any filesystem where the
        # process does not have write permission to create <project_root>/logs/.
        # The exception propagated out of __init__, crashing the process before
        # any task could ever start — observer_loop, vision_runtime, and
        # intent_listener were never reached.
        #
        # Fix: wrap mkdir in try/except. On failure, set _transition_log_available=False
        # so _commit_transition() skips all file I/O gracefully. In-memory
        # _transition_history still accumulates entries for forensic inspection
        # within the session. Operational correctness is unaffected.
        self._transition_log_available: bool = True
        try:
            self.TRANSITION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as _mkdir_err:
            self._transition_log_available = False
            print(
                f"[ModeController] WARNING: Cannot create transition log directory "
                f"({self.TRANSITION_LOG_PATH.parent!r}): {_mkdir_err}. "
                "Transition logging is DISABLED for this session. "
                "This is expected on read-only filesystems (containers, NFS, CI). "
                "In-memory transition history remains available via get_authority_snapshot().",
                file=sys.stderr,
            )

    # ==================================================
    # READS
    # ==================================================

    @property
    def mode(self) -> SystemMode:
        with self._lock:
            return self._mode

    def is_armed(self) -> bool:
        with self._lock:
            return self._mode is SystemMode.ARMED

    def get_intent(self) -> Optional[str]:
        with self._lock:
            return self._intent

    # ==================================================
    # REPLAN CONTROL
    # ==================================================

    def begin_replan_sequence(self) -> None:
        with self._lock:
            if self._replan_in_progress:
                raise ModeTransitionError("Replan already in progress")

            if self._mode not in (
                SystemMode.EXECUTING,
                SystemMode.PLANNING,
            ):
                raise ModeTransitionError(
                    f"Replan not allowed in mode: {self._mode.value}"
                )

            self._replan_in_progress = True

    def end_replan_sequence(self) -> None:
        with self._lock:
            self._replan_in_progress = False

    # ==================================================
    # SNAPSHOT CONTRACT
    # ==================================================

    def attach_snapshot(self, snapshot_id: str) -> None:
        with self._lock:
            if self._mode is not SystemMode.OBSERVER:
                raise ModeTransitionError("Snapshot attach only in OBSERVER")
            if not snapshot_id or not snapshot_id.strip():
                raise ModeTransitionError("Invalid snapshot_id")
            if self._snapshot_id is not None:
                raise ModeTransitionError("Snapshot already attached")

            self._snapshot_id = snapshot_id.strip()
            self._snapshot_consumed = False

    def consume_snapshot(self) -> str:
        with self._lock:
            if self._mode is not SystemMode.ARMED:
                raise ModeTransitionError("Snapshot consume only in ARMED")
            if not self._snapshot_id:
                raise ModeTransitionError("No snapshot attached")
            if self._snapshot_consumed:
                raise ModeTransitionError("Snapshot already consumed")

            self._snapshot_consumed = True
            return self._snapshot_id

    # ==================================================
    # LLM
    # ==================================================

    def inject_llm_callable(self, llm_call: Callable) -> None:
        if not callable(llm_call):
            raise TypeError("llm_call must be callable")

        with self._lock:
            if self._llm_callable is not None:
                raise RuntimeError("LLM already injected")
            self._llm_callable = llm_call

    def get_llm_callable(self) -> Callable:
        with self._lock:
            if self._mode not in (SystemMode.PLANNING, SystemMode.ARMED):
                raise RuntimeError("LLM only accessible in PLANNING or ARMED")
            if self._llm_callable is None:
                raise RuntimeError("No LLM injected")
            return self._llm_callable

    # ==================================================
    # HEALTH
    # ==================================================

    def update_observer_health(
        self, healthy: bool, *, reason: Optional[str] = None
    ) -> None:
        with self._lock:
            self._observer_healthy = bool(healthy)
            if not healthy:
                self._failure_reason = reason or "observer unhealthy"

    def update_vision_status(self, ok: bool) -> None:
        with self._lock:
            self._vision_ok = bool(ok)

    # ==================================================
    # TRANSITIONS
    # ==================================================

    def arm(self, intent: str) -> None:
        # FIX-M1 (RB-1 / SI-1): The _replan_in_progress guard was REMOVED from
        # arm(). Previously, begin_replan_sequence() set _replan_in_progress=True
        # and then the replan path called arm(), which immediately raised
        # ModeTransitionError("Replan in progress"). The flag was only cleared in
        # a finally block AFTER the exception propagated — meaning every replan
        # attempt triggered a shutdown. MAX_REPLANS=3 was unreachable dead code.
        #
        # The concurrent-replan guard is still correctly enforced by
        # begin_replan_sequence(): if _replan_in_progress is already True it
        # raises ModeTransitionError("Replan already in progress"). arm() itself
        # does NOT need to reject the replan path — it must be callable from it.
        #
        # Use arm_for_replan() when called from an active replan sequence (it
        # bypasses the OBSERVER mode requirement since the mode controller will
        # have been reset to OBSERVER by force_observer() before arm() is needed).
        with self._lock:
            if self._mode is not SystemMode.OBSERVER:
                raise ModeTransitionError("Cannot arm unless OBSERVER")

            if not self._snapshot_id:
                raise ModeTransitionError("Snapshot required before arm")

            if not intent or not intent.strip():
                raise ModeTransitionError("Intent must be non-empty")

            self._intent = intent.strip()
            self._intent_frozen = False
            self._planning_completed = False
            self._execution_plan_attached = False
            self._execution_plan_id = None
            self._failure_reason = None

            self._commit_transition(SystemMode.ARMED, "intent armed", False)

    def arm_for_replan(self, intent: str) -> None:
        """
        FIX-M1: Arm the system during an active replan sequence.

        Identical to arm() but does NOT require _replan_in_progress=False.
        Called exclusively from the replan path in main.py after
        begin_replan_sequence() has set _replan_in_progress=True and
        force_observer() has reset the mode to OBSERVER.

        Why a separate method rather than modifying arm():
        - arm() is the normal arming path; it must fail-closed for all other
          callers.
        - arm_for_replan() is the replanning arming path; it is explicitly
          permitted to run while _replan_in_progress=True.
        - Keeping them separate makes the call site in main.py self-documenting
          and avoids adding a boolean 'bypass' parameter to arm() which would be
          invisible at call sites.
        """
        with self._lock:
            # Replan can only arm from OBSERVER (set by force_observer() earlier
            # in the replan sequence).
            if self._mode is not SystemMode.OBSERVER:
                raise ModeTransitionError(
                    "arm_for_replan requires OBSERVER mode — "
                    "call force_observer() before arm_for_replan()"
                )

            if not intent or not intent.strip():
                raise ModeTransitionError("Intent must be non-empty")

            self._intent = intent.strip()
            self._intent_frozen = False
            self._planning_completed = False
            self._execution_plan_attached = False
            self._execution_plan_id = None
            self._failure_reason = None

            self._commit_transition(SystemMode.ARMED, "intent armed (replan)", False)

    def begin_planning(self) -> None:
        with self._lock:
            if self._mode is not SystemMode.ARMED:
                raise ModeTransitionError("Planning requires ARMED")

            if not self._intent:
                raise ModeTransitionError("Intent missing")

            if not self._observer_healthy:
                raise ObserverUnavailableError(self._failure_reason)

            if not self._vision_ok:
                raise VisionUnavailableError("vision unavailable")

            self._intent_frozen = True
            self._planning_started_at = time.time()

            self._commit_transition(
                SystemMode.PLANNING,
                "planning started",
                False,
            )

    def check_planning_timeout(self) -> None:
        with self._lock:
            if self._mode is not SystemMode.PLANNING:
                return

            if (
                self._planning_started_at
                and (time.time() - self._planning_started_at)
                > self.MAX_PLANNING_SECONDS
            ):
                raise PlanningTimeoutError("Planning timeout exceeded")

    def check_armed_timeout(self) -> None:
        """
        Detect and recover from infinite ARMED stall.

        DEF-5 FIX preserved: raise decision captured inside the lock so no
        thread can race between the lock release and the raise.
        """
        _should_raise = False
        _elapsed = 0.0

        with self._lock:
            if self._mode is not SystemMode.ARMED:
                return

            _elapsed = time.time() - self._mode_entered_at
            if _elapsed > self.MAX_ARMED_SECONDS:
                self._input_locked = False
                self._snapshot_id = None
                self._snapshot_consumed = False
                self._intent = None
                self._intent_frozen = False
                self._planning_completed = False
                self._execution_plan_attached = False
                self._execution_plan_id = None
                self._planning_started_at = None
                self._replan_in_progress = False
                self._failure_reason = "armed_timeout"

                self._commit_transition(
                    SystemMode.OBSERVER,
                    f"armed_timeout_recovery (elapsed={_elapsed:.1f}s)",
                    True,
                )

                _should_raise = True

        if _should_raise:
            raise ArmedTimeoutError(
                f"ARMED mode timed out after {self.MAX_ARMED_SECONDS}s — "
                "reverted to OBSERVER. Check LLM callable availability."
            )

    def attach_execution_plan(self, plan_id: str) -> None:
        with self._lock:
            if self._mode is not SystemMode.PLANNING:
                raise ModeTransitionError("Plan attach requires PLANNING")

            if not plan_id or not plan_id.strip():
                raise ModeTransitionError("Invalid plan_id")

            pid = plan_id.strip()
            if len(pid) > self.MAX_PLAN_ID_LENGTH:
                raise ModeTransitionError("Plan ID too long")

            self._execution_plan_id = pid
            self._execution_plan_attached = True

    def mark_planning_complete(self) -> None:
        with self._lock:
            if not self._execution_plan_attached:
                raise ModeTransitionError("No execution plan attached")
            self._planning_completed = True

    def execute(self) -> None:
        with self._lock:
            if self._mode is not SystemMode.PLANNING:
                raise ModeTransitionError("Execute requires PLANNING")
            if not self._planning_completed:
                raise ModeTransitionError("Planning incomplete")
            if not self._vision_ok:
                raise VisionUnavailableError("vision unavailable")
            if not self._observer_healthy:
                raise ObserverUnavailableError(self._failure_reason)

            self._input_locked = True

            self._commit_transition(
                SystemMode.EXECUTING,
                f"execution started (plan={self._execution_plan_id})",
                False,
            )

    def begin_restoration(self) -> None:
        with self._lock:
            if self._mode is not SystemMode.EXECUTING:
                raise ModeTransitionError("Restoration requires EXECUTING")

            self._commit_transition(
                SystemMode.RESTORING,
                "restoration started",
                False,
            )

    def complete_execution(self) -> None:
        with self._lock:
            self._input_locked = False
            self._snapshot_id = None
            self._snapshot_consumed = False
            self._intent = None
            self._intent_frozen = False
            self._planning_completed = False
            self._execution_plan_attached = False
            self._execution_plan_id = None
            self._planning_started_at = None
            self._replan_in_progress = False
            self._failure_reason = None

            self._commit_transition(
                SystemMode.OBSERVER,
                "execution completed",
                False,
            )

    # ==================================================
    # COMMIT TRANSITION
    # ==================================================

    def _commit_transition(
        self,
        target: SystemMode,
        reason: str,
        forced: bool,
    ) -> None:
        now = time.time()
        prev = self._mode

        self._mode = target
        self._mode_entered_at = now
        self._last_transition_reason = reason

        entry = {
            "ts": now,
            "from": prev.value,
            "to": target.value,
            "reason": reason,
            "forced": forced,
            "vision_ok": self._vision_ok,
            "observer_healthy": self._observer_healthy,
            "plan_attached": self._execution_plan_attached,
            "plan_id": self._execution_plan_id,
        }

        self._transition_history.append(entry)

        # FIX SI-1 / RB-A1: Skip all file I/O when the log directory could not
        # be created at startup (read-only filesystem). The in-memory history
        # above still accumulates entries; only the durable JSONL log is absent.
        if not self._transition_log_available:
            return

        # FIX F-03 / RB-08: Always close the file descriptor in a finally block.
        # Previously, if os.write() or os.fsync() raised (e.g. disk full), the
        # fd was never closed, leaking a file descriptor on every transition.
        # Under adversarial disk-full conditions this would exhaust the system fd
        # limit, silently breaking all further transition logging.
        fd = None
        dir_fd = None
        try:
            fd = os.open(
                self.TRANSITION_LOG_PATH,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o644,
            )
            os.write(fd, (json.dumps(entry, sort_keys=True) + "\n").encode())
            os.fsync(fd)
        except Exception:
            pass
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except Exception:
                    pass

        try:
            dir_fd = os.open(
                str(self.TRANSITION_LOG_PATH.parent),
                os.O_RDONLY,
            )
            os.fsync(dir_fd)
        except Exception:
            pass
        finally:
            if dir_fd is not None:
                try:
                    os.close(dir_fd)
                except Exception:
                    pass

    # ==================================================
    # EMERGENCY RECOVERY
    # ==================================================

    def force_observer(self) -> None:
        """
        Unconditionally reset to OBSERVER mode.

        Bypass all transition guards. Called during:
          - crash recovery (dirty auth state at startup)
          - replan sequences that must restart from a clean slate
          - ARMED timeout recovery (check_armed_timeout)

        This method MUST NOT raise.
        """
        with self._lock:
            self._input_locked = False
            self._snapshot_id = None
            self._snapshot_consumed = False
            self._intent = None
            self._intent_frozen = False
            self._planning_completed = False
            self._execution_plan_attached = False
            self._execution_plan_id = None
            self._planning_started_at = None
            self._replan_in_progress = False
            self._failure_reason = None

            self._commit_transition(
                SystemMode.OBSERVER,
                "forced_observer_recovery",
                True,
            )

    # ==================================================
    # FORENSICS
    # ==================================================

    def get_authority_snapshot(self) -> Dict[str, object]:
        with self._lock:
            return {
                "mode": self._mode.value,
                "observer_healthy": self._observer_healthy,
                "vision_ok": self._vision_ok,
                "failure_reason": self._failure_reason,
                "input_locked": self._input_locked,
                "intent_frozen": self._intent_frozen,
                "planning_completed": self._planning_completed,
                "execution_plan_attached": self._execution_plan_attached,
                "execution_plan_id": self._execution_plan_id,
                "transition_history_depth": len(self._transition_history),
                "transition_log_available": self._transition_log_available,
            }
