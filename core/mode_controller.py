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
    """Execution requested without verified vision."""


class ModeController:
    """
    Authority firewall for the system.
    """

    MAX_TRANSITION_HISTORY = 2000

    def __init__(self):
        self._mode: SystemMode = SystemMode.OBSERVER
        self._mode_entered_at: float = time.time()
        self._last_transition_reason: Optional[str] = None
        self._vision_ok: bool = False
        self._input_locked: bool = False

        # === ADDITIONS ===
        self._observer_healthy: bool = True
        self._vision_failed_permanently: bool = False
        self._failure_reason: Optional[str] = None
        self._lock = threading.Lock()

        self._transition_history: Deque[Dict[str, object]] = deque(
            maxlen=self.MAX_TRANSITION_HISTORY
        )

        self._allowed_transitions = {
            SystemMode.OBSERVER: {SystemMode.ARMED},
            SystemMode.ARMED: {SystemMode.EXECUTING, SystemMode.OBSERVER},
            SystemMode.EXECUTING: {SystemMode.OBSERVER},
        }

        self._log_state("[MODE] Initialized")

    @property
    def mode(self) -> SystemMode:
        return self._mode

    @property
    def mode_uptime_seconds(self) -> float:
        return round(time.time() - self._mode_entered_at, 2)

    @property
    def last_transition_reason(self) -> Optional[str]:
        return self._last_transition_reason

    # === ADDITIONS ===
    def update_observer_health(self, healthy: bool, *, reason: Optional[str] = None) -> None:
        """
        Observer health is monotonic: once false, it cannot recover.
        """
        with self._lock:
            if not healthy and self._observer_healthy:
                self._observer_healthy = False
                self._vision_failed_permanently = True
                self._failure_reason = reason or "observer reported blindness"

                # Abort immediately if executing
                if self._mode == SystemMode.EXECUTING:
                    self._force_abort("Observer health lost mid-execution")

    def update_vision_status(self, ok: bool) -> None:
        with self._lock:
            if not ok:
                self._vision_failed_permanently = True
            self._vision_ok = bool(ok) and not self._vision_failed_permanently

    def lock_input(self) -> None:
        """Lock mouse and keyboard input to prevent interference."""
        self._input_locked = True

    def release_input(self) -> None:
        """Release input lock once execution is complete."""
        self._input_locked = False

    def request_transition(
        self,
        target_mode: SystemMode,
        reason: str,
        *,
        force: bool = False,
    ) -> None:
        with self._lock:
            if not reason or not reason.strip():
                raise ModeTransitionError(
                    "Mode transition requires an explicit reason."
                )

            current = self._mode
            if target_mode == current:
                return

            allowed = self._allowed_transitions.get(current, set())
            if not force and target_mode not in allowed:
                raise ModeTransitionError(
                    f"Illegal mode transition: {current} -> {target_mode}"
                )

            # === ADDITIONS: hard authority gates ===
            if not self._observer_healthy:
                raise VisionUnavailableError(
                    f"Observer unhealthy: {self._failure_reason}"
                )

            if target_mode == SystemMode.EXECUTING and not self._vision_ok:
                raise VisionUnavailableError(
                    "Cannot enter EXECUTING without live screen vision"
                )

            if target_mode == SystemMode.EXECUTING:
                self.lock_input()

            self._mode = target_mode
            self._mode_entered_at = time.time()
            self._last_transition_reason = reason

            self._record_transition(current, target_mode, reason, force)

            self._log_state(
                f"[MODE] {current} -> {target_mode} | reason='{reason}'"
                + (" | FORCED" if force else "")
            )

    def arm(self, reason: str) -> None:
        self.request_transition(SystemMode.ARMED, reason)

    def execute(self, reason: str) -> None:
        self.request_transition(SystemMode.EXECUTING, reason)

    def disarm(self, reason: str) -> None:
        self.request_transition(SystemMode.OBSERVER, reason, force=True)
        self.release_input()

    # === ADDITIONS ===
    def _force_abort(self, reason: str) -> None:
        """
        Non-recoverable abort path.
        """
        prev = self._mode
        self._mode = SystemMode.OBSERVER
        self._mode_entered_at = time.time()
        self._last_transition_reason = reason
        self.release_input()

        self._record_transition(prev, SystemMode.OBSERVER, reason, force=True)
        self._log_state(f"[MODE] ABORTED {prev} -> OBSERVER | reason='{reason}'")

    def _record_transition(
        self,
        from_mode: SystemMode,
        to_mode: SystemMode,
        reason: str,
        forced: bool,
    ) -> None:
        self._transition_history.append(
            {
                "ts": time.time(),
                "from": from_mode,
                "to": to_mode,
                "reason": reason,
                "forced": forced,
                "vision_ok": self._vision_ok,
                "observer_healthy": self._observer_healthy,
            }
        )

    def get_authority_snapshot(self) -> Dict[str, object]:
        """
        Forensic-grade authority state.
        """
        return {
            "mode": self._mode,
            "mode_uptime_seconds": self.mode_uptime_seconds,
            "observer_healthy": self._observer_healthy,
            "vision_ok": self._vision_ok,
            "vision_failed_permanently": self._vision_failed_permanently,
            "failure_reason": self._failure_reason,
            "input_locked": self._input_locked,
            "transition_history_depth": len(self._transition_history),
        }

    def _log_state(self, message: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"{ts} {message}")
