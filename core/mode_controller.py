import time
from enum import Enum
from typing import Optional


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

    def __init__(self):
        self._mode: SystemMode = SystemMode.OBSERVER
        self._mode_entered_at: float = time.time()
        self._last_transition_reason: Optional[str] = None
        self._vision_ok: bool = False
        self._input_locked: bool = False

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

    def update_vision_status(self, ok: bool) -> None:
        self._vision_ok = bool(ok)

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
        if not reason or not reason.strip():
            raise ModeTransitionError("Mode transition requires an explicit reason.")

        current = self._mode
        if target_mode == current:
            return

        allowed = self._allowed_transitions.get(current, set())
        if not force and target_mode not in allowed:
            raise ModeTransitionError(f"Illegal mode transition: {current} -> {target_mode}")

        if target_mode == SystemMode.EXECUTING and not self._vision_ok:
            raise VisionUnavailableError(
                "Cannot enter EXECUTING without live screen vision"
            )

        if target_mode == SystemMode.EXECUTING:
            self.lock_input()  # Lock input during execution

        self._mode = target_mode
        self._mode_entered_at = time.time()
        self._last_transition_reason = reason

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
        self.release_input()  # Release input when disarmed

    def _log_state(self, message: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"{ts} {message}")
