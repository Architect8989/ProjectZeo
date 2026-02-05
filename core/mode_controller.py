import time
import threading
from enum import Enum
from typing import Optional, Deque, Dict
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

    Enforces:
    - OBSERVER → ARMED → PLANNING → EXECUTING → OBSERVER
    - Intent frozen before planning
    - Execution only after planning
    - Abort always returns to OBSERVER safely
    """

    MAX_TRANSITION_HISTORY = 2000

    def __init__(self):
        self._lock = threading.RLock()

        self._mode: SystemMode = SystemMode.OBSERVER
        self._mode_entered_at: float = time.time()
        self._last_transition_reason: Optional[str] = None

        self._intent: Optional[str] = None
        self._intent_frozen: bool = False
        self._planning_completed: bool = False

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
    # HEALTH
    # --------------------------------------------------

    def update_observer_health(
        self, healthy: bool, *, reason: Optional[str] = None
    ) -> None:
        with self._lock:
            if healthy:
                return

            if self._observer_healthy:
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

    def mark_planning_complete(self) -> None:
        """
        Explicit signal from planner that a plan exists.
        """
        with self._lock:
            if self._mode != SystemMode.PLANNING:
                raise ModeTransitionError(
                    "Planning not active"
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

            if not self._observer_healthy or not self._vision_ok:
                raise VisionUnavailableError(
                    self._failure_reason or "vision unavailable"
                )

            self._input_locked = True

            self._commit_transition(
                SystemMode.EXECUTING,
                reason="execution started",
                forced=False,
            )

    def consume_intent(self) -> Optional[str]:
        """
        Idempotent-safe intent consumption.
        """
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
                return  # already safe

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
                "transition_history_depth": len(
                    self._transition_history
                ),
        }
