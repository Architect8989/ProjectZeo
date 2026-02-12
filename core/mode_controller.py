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
    MAX_PLANNING_SECONDS = 60.0  # HARD FAIL-CLOSED LIMIT

    def __init__(self):
        self._lock = threading.RLock()

        self._mode: SystemMode = SystemMode.OBSERVER
        self._mode_entered_at: float = time.time()
        self._last_transition_reason: Optional[str] = None

        self._planning_started_at: Optional[float] = None

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

        self._llm_callable: Optional[Callable[[str], str]] = None

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
                raise ModeTransitionError(
                    "Snapshot can only attach in OBSERVER mode"
                )
            if not snapshot_id:
                raise ModeTransitionError("Invalid snapshot_id")
            if self._snapshot_id is not None:
                raise ModeTransitionError(
                    "Snapshot already attached for this cycle"
                )

            self._snapshot_id = snapshot_id
            self._snapshot_consumed = False

    def consume_snapshot(self) -> str:
        with self._lock:
            if self._mode is not SystemMode.ARMED:
                raise ModeTransitionError(
                    "Snapshot can only be consumed in ARMED state"
                )
            if not self._snapshot_id:
                raise ModeTransitionError("No snapshot attached")
            if self._snapshot_consumed:
                raise ModeTransitionError("Snapshot already consumed")

            self._snapshot_consumed = True
            return self._snapshot_id

    # ==================================================
    # LLM INJECTION
    # ==================================================

    def inject_llm_callable(self, llm_call: Callable[[str], str]) -> None:
        if not callable(llm_call):
            raise TypeError("llm_call must be callable")

        with self._lock:
            if self._llm_callable is not None:
                raise RuntimeError("LLM callable already injected")
            self._llm_callable = llm_call

    def get_llm_callable(self) -> Callable[[str], str]:
        with self._lock:
            if self._mode is not SystemMode.PLANNING:
                raise RuntimeError(
                    f"LLM callable only accessible during PLANNING (current: {self._mode.value})"
                )
            if self._llm_callable is None:
                raise RuntimeError(
                    "No LLM callable injected into ModeController"
                )
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
            if self._mode is not SystemMode.OBSERVER:
                raise ModeTransitionError("Cannot arm unless in OBSERVER")
            if not self._snapshot_id:
                raise ModeTransitionError(
                    "Cannot arm without snapshot boundary"
                )
            if not intent or not intent.strip():
                raise ModeTransitionError("Intent must be non-empty")

            self._intent = intent.strip()
            self._intent_frozen = False
            self._planning_completed = False
            self._execution_plan_attached = False
            self._execution_plan_id = None
            self._failure_reason = None

            self._commit_transition(
                SystemMode.ARMED,
                reason="intent armed",
                forced=False,
            )

    def begin_planning(self) -> None:
        with self._lock:
            if self._mode is not SystemMode.ARMED:
                raise ModeTransitionError(
                    "Planning requires ARMED state"
                )
            if not self._intent:
                raise ModeTransitionError("No intent available")
            if not self._observer_healthy:
                raise ObserverUnavailableError(
                    self._failure_reason
                )
            if not self._vision_ok:
                raise VisionUnavailableError("vision unavailable")

            self._intent_frozen = True
            self._planning_started_at = time.time()

            self._commit_transition(
                SystemMode.PLANNING,
                reason="planning started",
                forced=False,
            )

    def check_planning_timeout(self) -> None:
        with self._lock:
            if self._mode is not SystemMode.PLANNING:
                return

            if self._planning_started_at is None:
                raise PlanningTimeoutError("Planning start timestamp missing")

            elapsed = time.time() - self._planning_started_at
            if elapsed > self.MAX_PLANNING_SECONDS:
                self._reset_internal_state()
                self._commit_transition(
                    SystemMode.OBSERVER,
                    reason="planning timeout",
                    forced=True,
                )
                raise PlanningTimeoutError(
                    f"Planning exceeded {self.MAX_PLANNING_SECONDS}s"
                )

    def attach_execution_plan(self, plan_id: str) -> None:
        with self._lock:
            if self._mode is not SystemMode.PLANNING:
                raise ModeTransitionError(
                    "Execution plan can only attach during PLANNING"
                )
            if not plan_id or not plan_id.strip():
                raise ModeTransitionError("Invalid plan_id")
            if self._execution_plan_attached:
                raise ModeTransitionError(
                    "Execution plan already attached"
                )

            self._execution_plan_attached = True
            self._execution_plan_id = plan_id.strip()

    def mark_planning_complete(self) -> None:
        with self._lock:
            if self._mode is not SystemMode.PLANNING:
                raise ModeTransitionError("Planning not active")
            if not self._execution_plan_attached:
                raise ModeTransitionError(
                    "Cannot complete planning without plan"
                )

            self._planning_completed = True
            self._planning_started_at = None

    def execute(self) -> None:
        with self._lock:
            if self._mode is not SystemMode.PLANNING:
                raise ModeTransitionError(
                    "Execute requires PLANNING state"
                )
            if not self._planning_completed:
                raise ModeTransitionError("Plan not completed")
            if not self._vision_ok:
                raise VisionUnavailableError("vision unavailable")
            if not self._observer_healthy:
                raise ObserverUnavailableError(
                    self._failure_reason
                )

            self._input_locked = True

            self._commit_transition(
                SystemMode.EXECUTING,
                reason=f"execution started (plan={self._execution_plan_id})",
                forced=False,
            )

    def consume_intent(self) -> str:
        with self._lock:
            if self._mode is not SystemMode.EXECUTING:
                raise ModeTransitionError(
                    "Intent consumed outside EXECUTING"
                )
            if not self._intent:
                raise ModeTransitionError("No intent available")
            return self._intent

    def begin_restoration(self) -> None:
        with self._lock:
            if self._mode is not SystemMode.EXECUTING:
                raise ModeTransitionError(
                    "Restoration requires EXECUTING state"
                )

            self._commit_transition(
                SystemMode.RESTORING,
                reason="restoration started",
                forced=False,
            )

    # ==================================================
    # COMPLETION
    # ==================================================

    def complete_execution(self, reason: str = "execution complete") -> None:
        with self._lock:
            if self._mode is not SystemMode.RESTORING:
                raise ModeTransitionError(
                    "Completion requires RESTORING state"
                )

            self._reset_internal_state()

            self._commit_transition(
                SystemMode.OBSERVER,
                reason=reason,
                forced=False,
            )

    def force_observer(self) -> None:
        with self._lock:
            self._reset_internal_state()
            self._failure_reason = None

            self._commit_transition(
                SystemMode.OBSERVER,
                reason="forced reset",
                forced=True,
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
