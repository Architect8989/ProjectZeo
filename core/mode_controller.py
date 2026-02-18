import time
import threading
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


class ModeController:

    MAX_TRANSITION_HISTORY = 2000
    MAX_PLANNING_SECONDS = 60.0
    MAX_PLAN_ID_LENGTH = 128

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
    # LLM INJECTION
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
            if self._mode is not SystemMode.PLANNING:
                raise RuntimeError("LLM only accessible in PLANNING")
            if self._llm_callable is None:
                raise RuntimeError("No LLM injected")
            return self._llm_callable

    # ==================================================
    # HEALTH SIGNALS
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
        with self._lock:
            if self._replan_in_progress:
                raise ModeTransitionError("Replan in progress")

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

            self._commit_transition(SystemMode.PLANNING, "planning started", False)

    def check_planning_timeout(self) -> None:
        with self._lock:
            if self._mode is not SystemMode.PLANNING:
                return

            if not self._planning_started_at:
                raise PlanningTimeoutError("Planning timestamp missing")

            if time.time() - self._planning_started_at > self.MAX_PLANNING_SECONDS:
                self._reset_internal_state()
                self._commit_transition(
                    SystemMode.OBSERVER,
                    "planning timeout",
                    True,
                )
                raise PlanningTimeoutError("Planning timeout exceeded")

    def attach_execution_plan(self, plan_id: str) -> None:
        with self._lock:
            if self._mode is not SystemMode.PLANNING:
                raise ModeTransitionError("Plan attach only in PLANNING")

            if not plan_id or not plan_id.strip():
                raise ModeTransitionError("Invalid plan_id")

            plan_id = plan_id.strip()

            if len(plan_id) > self.MAX_PLAN_ID_LENGTH:
                raise ModeTransitionError("plan_id too long")

            if self._execution_plan_attached:
                raise ModeTransitionError("Plan already attached")

            self._execution_plan_attached = True
            self._execution_plan_id = plan_id

    def mark_planning_complete(self) -> None:
        with self._lock:
            if self._mode is not SystemMode.PLANNING:
                raise ModeTransitionError("Planning not active")
            if not self._execution_plan_attached:
                raise ModeTransitionError("Plan not attached")

            self._planning_completed = True
            self._planning_started_at = None

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

    # ==================================================
    # COMPLETION
    # ==================================================

    def complete_execution(self, reason: str = "execution complete") -> None:
        with self._lock:
            if self._mode is not SystemMode.RESTORING:
                raise ModeTransitionError("Completion requires RESTORING")

            self._reset_internal_state()

            self._commit_transition(
                SystemMode.OBSERVER,
                reason,
                False,
            )

    def force_observer(self) -> None:
        with self._lock:
            self._reset_internal_state()
            self._failure_reason = None
            self._commit_transition(
                SystemMode.OBSERVER,
                "forced reset",
                True,
            )

    def _reset_internal_state(self) -> None:
        self._intent = None
        self._intent_frozen = False
        self._planning_completed = False
        self._execution_plan_attached = False
        self._execution_plan_id = None
        self._input_locked = False
        self._snapshot_id = None
        self._snapshot_consumed = False
        self._planning_started_at = None
        self._replan_in_progress = False

    # ==================================================
    # TRANSITION COMMIT
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

        self._transition_history.append(
            {
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
        }
