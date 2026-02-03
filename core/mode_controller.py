import time
import threading
from enum import Enum
from typing import Optional, Deque, Dict
from collections import deque


class SystemMode(str, Enum):
    OBSERVER = "OBSERVER"
    ARMED = "ARMED"
    EXECUTING = "EXECUTING"


class ModeTransitionError(Exception):
    pass


class VisionUnavailableError(ModeTransitionError):
    pass


class ModeController:
    """
    Single authority state machine.
    Atomic. Deterministic. Auditable.
    """

    MAX_TRANSITION_HISTORY = 2000

    def __init__(self):
        self._lock = threading.RLock()

        self._mode: SystemMode = SystemMode.OBSERVER
        self._intent: Optional[str] = None

        self._mode_entered_at = time.time()
        self._last_transition_reason: Optional[str] = None

        self._vision_ok = False
        self._observer_healthy = True
        self._vision_failed_permanently = False
        self._failure_reason: Optional[str] = None

        self._input_locked = False

        self._transition_history: Deque[Dict[str, object]] = deque(
            maxlen=self.MAX_TRANSITION_HISTORY
        )

        self._allowed_transitions = {
            SystemMode.OBSERVER: {SystemMode.ARMED},
            SystemMode.ARMED: {SystemMode.EXECUTING, SystemMode.OBSERVER},
            SystemMode.EXECUTING: {SystemMode.OBSERVER},
        }

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
        with self._lock:
            if healthy:
                return

            if self._observer_healthy:
                self._observer_healthy = False
                self._vision_failed_permanently = True
                self._failure_reason = reason or "observer failure"

                if self._mode == SystemMode.EXECUTING:
                    self._abort_locked(
                        "observer health lost mid-execution"
                    )

    def update_vision_status(self, ok: bool) -> None:
        with self._lock:
            self._vision_ok = bool(ok)

    # --------------------------------------------------
    # ATOMIC TRANSITIONS
    # --------------------------------------------------

    def arm(self, reason: str) -> None:
        """
        Atomically capture intent + arm.
        """
        with self._lock:
            if self._mode != SystemMode.OBSERVER:
                raise ModeTransitionError(
                    "Cannot arm unless in OBSERVER"
                )

            self._intent = reason
            self._commit_transition(
                SystemMode.ARMED, reason, forced=False
            )

    def execute(self, reason: str) -> None:
        with self._lock:
            if self._mode != SystemMode.ARMED:
                raise ModeTransitionError(
                    "Execute requested while not ARMED"
                )

            if not self._observer_healthy:
                raise VisionUnavailableError(self._failure_reason)

            if not self._vision_ok:
                raise VisionUnavailableError(
                    "Vision not available"
                )

            self._input_locked = True
            self._commit_transition(
                SystemMode.EXECUTING, reason, forced=False
            )

    def consume_intent(self) -> str:
        with self._lock:
            if self._mode != SystemMode.EXECUTING:
                raise ModeTransitionError(
                    "Intent consumed outside EXECUTING"
                )

            if self._intent is None:
                raise ModeTransitionError("No intent present")

            intent = self._intent
            self._intent = None
            return intent

    # --------------------------------------------------
    # EMERGENCY / RECOVERY
    # --------------------------------------------------

    def force_observer(self) -> None:
        """
        Emergency reset.
        Still atomic. Still audited.
        """
        with self._lock:
            self._intent = None
            self._input_locked = False
            self._commit_transition(
                SystemMode.OBSERVER,
                reason="forced reset",
                forced=True,
            )

    def _abort_locked(self, reason: str) -> None:
        self._intent = None
        self._input_locked = False
        self._commit_transition(
            SystemMode.OBSERVER, reason, forced=True
        )

    # --------------------------------------------------
    # INTERNAL COMMIT (SINGLE POINT)
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
                "from": prev,
                "to": target,
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
                "mode_uptime_seconds": self.mode_uptime_seconds,
                "observer_healthy": self._observer_healthy,
                "vision_ok": self._vision_ok,
                "vision_failed_permanently": self._vision_failed_permanently,
                "failure_reason": self._failure_reason,
                "input_locked": self._input_locked,
                "transition_history_depth": len(
                    self._transition_history
                ),
        }
