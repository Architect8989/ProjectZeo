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
    Single authoritative execution-mode state machine.

    Guarantees:
    - Explicit think → act separation
    - Frozen intent before execution
    - OS access only in EXECUTING
    - Atomic, auditable transitions
    """

    MAX_TRANSITION_HISTORY = 2000

    def __init__(self):
        self._lock = threading.RLock()

        # ---- authority ----
        self._mode: SystemMode = SystemMode.OBSERVER
        self._mode_entered_at: float = time.time()
        self._last_transition_reason: Optional[str] = None

        # ---- intent ----
        self._intent: Optional[str] = None
        self._intent_frozen: bool = False

        # ---- health ----
        self._vision_ok: bool = False
        self._observer_healthy: bool = True
        self._vision_failed_permanently: bool = False
        self._failure_reason: Optional[str] = None

        # ---- input ----
        self._input_locked: bool = False

        # ---- audit ----
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

    @property
    def mode_uptime_seconds(self) -> float:
        with self._lock:
            return time.time() - self._mode_entered_at

    def is_armed(self) -> bool:
        with self._lock:
            return self._mode == SystemMode.ARMED

    # --------------------------------------------------
    # HEALTH SIGNALS
    # --------------------------------------------------

    def update_observer_health(
        self, healthy: bool, *, reason: Optional[str] = None
    ) -> None:
        """
        Observer blindness is permanent.
        Any loss during PLANNING or EXECUTING aborts immediately.
        """
        with self._lock:
            if healthy:
                return

            if self._observer_healthy:
                self._observer_healthy = False
                self._vision_failed_permanently = True
                self._failure_reason = reason or "observer failure"

                if self._mode in (SystemMode.PLANNING, SystemMode.EXECUTING):
                    self._abort_locked(
                        "observer health lost during active task"
                    )

    def update_vision_status(self, ok: bool) -> None:
        with self._lock:
            self._vision_ok = bool(ok)

    # --------------------------------------------------
    # TRANSITIONS
    # --------------------------------------------------

    def arm(self, reason: str) -> None:
        """
        OBSERVER → ARMED
        Accept intent exactly once.
        """
        with self._lock:
            if self._mode != SystemMode.OBSERVER:
                raise ModeTransitionError(
                    "Cannot arm unless in OBSERVER"
                )

            if not reason or not reason.strip():
                raise ModeTransitionError(
                    "Arm requires a non-empty intent"
                )

            self._intent = reason
            self._intent_frozen = False

            self._commit_transition(
                SystemMode.ARMED, reason, forced=False
            )

    def begin_planning(self, reason: str = "begin planning") -> None:
        """
        ARMED → PLANNING
        Intent becomes frozen here.
        """
        with self._lock:
            if self._mode != SystemMode.ARMED:
                raise ModeTransitionError(
                    "Planning requires ARMED state"
                )

            if not self._intent:
                raise ModeTransitionError("No intent to plan")

            if not self._observer_healthy:
                raise VisionUnavailableError(self._failure_reason)

            self._intent_frozen = True

            self._commit_transition(
                SystemMode.PLANNING, reason, forced=False
            )

    def execute(self, reason: str) -> None:
        """
        PLANNING → EXECUTING
        OS access becomes legal here.
        """
        with self._lock:
            if self._mode != SystemMode.PLANNING:
                raise ModeTransitionError(
                    "Execute requires PLANNING state"
                )

            if not self._observer_healthy:
                raise VisionUnavailableError(self._failure_reason)

            if not self._vision_ok:
                raise VisionUnavailableError("Vision not available")

            self._input_locked = True

            self._commit_transition(
                SystemMode.EXECUTING, reason, forced=False
            )

    def consume_intent(self) -> str:
        """
        Intent can be consumed exactly once, only in EXECUTING.
        """
        with self._lock:
            if self._mode != SystemMode.EXECUTING:
                raise ModeTransitionError(
                    "Intent consumed outside EXECUTING"
                )

            if not self._intent:
                raise ModeTransitionError("No intent available")

            intent = self._intent
            self._intent = None
            return intent

    # --------------------------------------------------
    # COMPLETION
    # --------------------------------------------------

    def complete_execution(
        self, reason: str = "execution complete"
    ) -> None:
        """
        EXECUTING → OBSERVER
        """
        with self._lock:
            if self._mode != SystemMode.EXECUTING:
                raise ModeTransitionError(
                    "Cannot complete unless EXECUTING"
                )

            self._input_locked = False
            self._intent = None
            self._intent_frozen = False

            self._commit_transition(
                SystemMode.OBSERVER, reason, forced=False
            )

    # --------------------------------------------------
    # EMERGENCY
    # --------------------------------------------------

    def force_observer(self) -> None:
        """
        Emergency reset.
        """
        with self._lock:
            self._intent = None
            self._intent_frozen = False
            self._input_locked = False

            self._commit_transition(
                SystemMode.OBSERVER,
                reason="forced reset",
                forced=True,
            )

    def _abort_locked(self, reason: str) -> None:
        self._intent = None
        self._intent_frozen = False
        self._input_locked = False

        self._commit_transition(
            SystemMode.OBSERVER,
            reason=reason,
            forced=True,
        )

    # --------------------------------------------------
    # SINGLE COMMIT POINT
    # --------------------------------------------------

    def _commit_transition(
        self,
        target: SystemMode,
        reason: str,
        forced: bool,
    ) -> None:
        prev = self._mode
        now = time.time()

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
        """
        Single authoritative snapshot.
        """
        with self._lock:
            return {
                "mode": self._mode.value,
                "mode_uptime_seconds": self.mode_uptime_seconds,
                "observer_healthy": self._observer_healthy,
                "vision_ok": self._vision_ok,
                "vision_failed_permanently": self._vision_failed_permanently,
                "failure_reason": self._failure_reason,
                "input_locked": self._input_locked,
                "intent_frozen": self._intent_frozen,
                "transition_history_depth": len(
                    self._transition_history
                ),
        }
