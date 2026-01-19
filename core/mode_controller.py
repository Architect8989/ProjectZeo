import time
from enum import Enum
from typing import Optional


class SystemMode(str, Enum):
    """
    Explicit system authority modes.

    OBSERVER   : Passive, read-only. No execution possible.
    ARMED      : Intent acknowledged, execution still blocked.
    EXECUTING  : Execution explicitly authorized.
    """

    OBSERVER = "OBSERVER"
    ARMED = "ARMED"
    EXECUTING = "EXECUTING"


class ModeTransitionError(Exception):
    """Raised when an invalid or unsafe mode transition is attempted."""
    pass


class ModeController:
    """
    Authority firewall for the system.

    This class is the ONLY place where execution authority can be escalated.
    If this class is bypassed, the system design is broken.
    """

    def __init__(self):
        self._mode: SystemMode = SystemMode.OBSERVER
        self._mode_entered_at: float = time.time()
        self._last_transition_reason: Optional[str] = None

        # Transition graph (hard rules)
        self._allowed_transitions = {
            SystemMode.OBSERVER: {SystemMode.ARMED},
            SystemMode.ARMED: {SystemMode.EXECUTING, SystemMode.OBSERVER},
            SystemMode.EXECUTING: {SystemMode.OBSERVER},
        }

        self._log_state("[MODE] Initialized")

    # ---------------------------------------------------------------------
    # Read-only properties
    # ---------------------------------------------------------------------

    @property
    def mode(self) -> SystemMode:
        """Current system mode (read-only)."""
        return self._mode

    @property
    def mode_uptime_seconds(self) -> float:
        """How long the system has been in the current mode."""
        return round(time.time() - self._mode_entered_at, 2)

    @property
    def last_transition_reason(self) -> Optional[str]:
        """Human-readable reason for the last mode change."""
        return self._last_transition_reason

    # ---------------------------------------------------------------------
    # Transition logic
    # ---------------------------------------------------------------------

    def request_transition(
        self,
        target_mode: SystemMode,
        reason: str,
        *,
        force: bool = False,
    ) -> None:
        """
        Request a mode transition.

        Parameters:
        - target_mode: desired SystemMode
        - reason: explicit justification (required)
        - force: emergency override (discouraged, logged)

        Raises:
        - ModeTransitionError if transition is invalid
        """

        if not reason or not reason.strip():
            raise ModeTransitionError(
                "Mode transition requires an explicit reason."
            )

        current = self._mode

        if target_mode == current:
            return  # no-op, safe

        allowed = self._allowed_transitions.get(current, set())

        if not force and target_mode not in allowed:
            raise ModeTransitionError(
                f"Illegal mode transition: {current} -> {target_mode}"
            )

        # Commit transition
        self._mode = target_mode
        self._mode_entered_at = time.time()
        self._last_transition_reason = reason

        self._log_state(
            f"[MODE] {current} -> {target_mode} | reason='{reason}'"
            + (" | FORCED" if force else "")
        )

    # ---------------------------------------------------------------------
    # Convenience helpers (explicit intent)
    # ---------------------------------------------------------------------

    def arm(self, reason: str) -> None:
        """Move from OBSERVER to ARMED."""
        self.request_transition(SystemMode.ARMED, reason)

    def execute(self, reason: str) -> None:
        """Move from ARMED to EXECUTING."""
        self.request_transition(SystemMode.EXECUTING, reason)

    def disarm(self, reason: str) -> None:
        """Return safely to OBSERVER from any state."""
        self.request_transition(SystemMode.OBSERVER, reason, force=True)

    # ---------------------------------------------------------------------
    # Internal logging
    # ---------------------------------------------------------------------

    def _log_state(self, message: str) -> None:
        """
        Internal logging hook.

        IMPORTANT:
        - No external logging frameworks here
        - No IO side effects beyond stdout
        - Audit systems can hook later
        """
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"{ts} {message}")
