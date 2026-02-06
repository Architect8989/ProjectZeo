import time
import threading
import os
from enum import Enum
from typing import Optional, Deque, Dict, Callable
from collections import deque


class SystemMode(str, Enum):
    OBSERVER = "OBSERVER"
    ARMED = "ARMED"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"


class ModeTransitionError(Exception):
    pass


class VisionUnavailableError(ModeTransitionError):
    pass


class ModeController:
    """
    Single authoritative lifecycle controller.

    HARD GUARANTEES:
    - Linear lifecycle: OBSERVER → ARMED → PLANNING → EXECUTING → OBSERVER
    - Snapshot must exist before planning (enforced externally)
    - Planning MUST attach a plan artifact
    - Execution impossible without completed + attached plan
    - Abort always returns to OBSERVER

    IMPORTANT:
    - This class does NOT think
    - This class does NOT plan
    - This class only enforces authority + lifecycle
    """

    MAX_TRANSITION_HISTORY = 2000

    def __init__(self):
        self._lock = threading.RLock()

        self._mode: SystemMode = SystemMode.OBSERVER
        self._mode_entered_at: float = time.time()
        self._last_transition_reason: Optional[str] = None

        # ---- HUMAN AUTHORITY ----
        self._intent: Optional[str] = None
        self._intent_frozen: bool = False

        # ---- PLANNING CONTRACT ----
        self._planning_completed: bool = False
        self._execution_plan_attached: bool = False
        self._execution_plan_id: Optional[str] = None

        # ---- HEALTH ----
        self._vision_ok: bool = False
        self._observer_healthy: bool = True
        self._vision_failed_permanently: bool = False
        self._failure_reason: Optional[str] = None

        self._input_locked: bool = False

        self._transition_history: Deque[Dict[str, object]] = deque(
            maxlen=self.MAX_TRANSITION_HISTORY
        )

    # --------------------------------------------------
    # READS
    # --------------------------------------------------

    @property
    def mode(self) -> SystemMode:
        with self._lock:
            return self._mode

    def is_armed(self) -> bool:
        with self._lock:
            return self._mode == SystemMode.ARMED

    # --------------------------------------------------
    # BRAIN–BODY INTERFACE (FIXED)
    # --------------------------------------------------

    def get_intent(self) -> Optional[str]:
        """
        Read-only access to the raw human prompt.

        - No mutation
        - No interpretation
        - No preprocessing
        """
        with self._lock:
            return self._intent

    def get_llm_callable(self) -> Callable[[str], str]:
        """
        Returns the external brain callable.

        ModeController does NOT reason.
        It only exposes a configured brain entrypoint.
        """
        from anthropic import Anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")

        client = Anthropic(api_key=api_key)

        def llm_call(prompt: str) -> str:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        return llm_call

    # --------------------------------------------------
    # HEALTH
    # --------------------------------------------------

    def update_observer_health(
        self, healthy: bool, *, reason: Optional[str] = None
    ) -> None:
        with self._lock:
            if healthy:
                return

            if not self._observer_healthy:
                return

            self._observer_healthy = False
            self._vision_failed_permanently = True
            self._failure_reason = reason or "observer failure"

            if self._mode in (
                SystemMode.PLANNING,
                SystemMode.EXECUTING,
            ):
                self._abort_locked(
                    "observer health lost during active task"
                )

    def update_vision_status(self, ok: bool) -> None:
        with self._lock:
            self._vision_ok = bool(ok)

    # --------------------------------------------------
    # TRANSITIONS
    # --------------------------------------------------

    def arm(self, intent: str) -> None:
        with self._lock:
            if self._mode != SystemMode.OBSERVER:
                raise ModeTransitionError(
                    "Cannot arm unless in OBSERVER"
                )

            if not intent or not intent.strip():
                raise ModeTransitionError(
                    "Intent must be non-empty"
                )

            self._intent = intent.strip()
            self._intent_frozen = False

            self._planning_completed = False
            self._execution_plan_attached = False
            self._execution_plan_id = None

            self._commit_transition(
                SystemMode.ARMED,
                reason="intent armed",
                forced=False,
            )

    def begin_planning(self) -> None:
        with self._lock:
            if self._mode != SystemMode.ARMED:
                raise ModeTransitionError(
                    "Planning requires ARMED state"
                )

            if not self._intent:
                raise ModeTransitionError("No intent available")

            if not self._observer_healthy:
                raise VisionUnavailableError(self._failure_reason)

            self._intent_frozen = True

            self._commit_transition(
                SystemMode.PLANNING,
                reason="planning started",
                forced=False,
            )

    # --------------------------------------------------
    # PLANNING CONTRACT
    # --------------------------------------------------

    def attach_execution_plan(self, plan_id: str) -> None:
        with self._lock:
            if self._mode != SystemMode.PLANNING:
                raise ModeTransitionError(
                    "Execution plan can only be attached during PLANNING"
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
            if self._mode != SystemMode.PLANNING:
                raise ModeTransitionError(
                    "Planning not active"
                )

            if not self._execution_plan_attached:
                raise ModeTransitionError(
                    "Cannot complete planning without attached ExecutionPlan"
                )

            self._planning_completed = True

    def execute(self) -> None:
        with self._lock:
            if self._mode != SystemMode.PLANNING:
                raise ModeTransitionError(
                    "Execute requires PLANNING state"
                )

            if not self._planning_completed:
                raise ModeTransitionError(
                    "Cannot execute without completed plan"
                )

            if not self._execution_plan_attached:
                raise ModeTransitionError(
                    "No ExecutionPlan attached"
                )

            if not self._observer_healthy or not self._vision_ok:
                raise VisionUnavailableError(
                    self._failure_reason or "vision unavailable"
                )

            self._input_locked = True

            self._commit_transition(
                SystemMode.EXECUTING,
                reason=f"execution started (plan={self._execution_plan_id})",
                forced=False,
            )

    def consume_intent(self) -> Optional[str]:
        with self._lock:
            if self._mode != SystemMode.EXECUTING:
                raise ModeTransitionError(
                    "Intent consumed outside EXECUTING"
                )

            intent = self._intent
            self._intent = None
            return intent

    # --------------------------------------------------
    # COMPLETION / ABORT
    # --------------------------------------------------

    def complete_execution(self, reason: str = "execution complete") -> None:
        with self._lock:
            if self._mode != SystemMode.EXECUTING:
                return

            self._reset_internal_state()

            self._commit_transition(
                SystemMode.OBSERVER,
                reason=reason,
                forced=False,
            )

    def force_observer(self) -> None:
        with self._lock:
            self._reset_internal_state()
            self._commit_transition(
                SystemMode.OBSERVER,
                reason="forced reset",
                forced=True,
            )

    def _abort_locked(self, reason: str) -> None:
        self._failure_reason = reason
        self._reset_internal_state()
        self._commit_transition(
            SystemMode.OBSERVER,
            reason=reason,
            forced=True,
        )

    def _reset_internal_state(self) -> None:
        self._intent = None
        self._intent_frozen = False

        self._planning_completed = False
        self._execution_plan_attached = False
        self._execution_plan_id = None

        self._input_locked = False

    # --------------------------------------------------
    # SINGLE COMMIT POINT
    # --------------------------------------------------

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

    # --------------------------------------------------
    # FORENSICS
    # --------------------------------------------------

    def get_authority_snapshot(self) -> Dict[str, object]:
        with self._lock:
            return {
                "mode": self._mode.value,
                "observer_healthy": self._observer_healthy,
                "vision_ok": self._vision_ok,
                "vision_failed_permanently": self._vision_failed_permanently,
                "failure_reason": self._failure_reason,
                "input_locked": self._input_locked,
                "intent_frozen": self._intent_frozen,
                "planning_completed": self._planning_completed,
                "execution_plan_attached": self._execution_plan_attached,
                "execution_plan_id": self._execution_plan_id,
                "transition_history_depth": len(
                    self._transition_history
                ),
    }
