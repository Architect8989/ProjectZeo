import time
import threading
from typing import Optional, Dict

from authority.input_tracker import InputTracker, InputSource
from authority.authority_policy import AuthorityPolicy, AuthorityDecision


class InputArbitrator:
    """
    Arbitrates control between SOC and human.

    HARD GUARANTEES:
    - Human input always dominates
    - Forced release is monotonic until explicit clear
    - Watchdog cannot be bypassed by SOC activity
    """

    # MUST exceed action_timeout (5s)
    EMERGENCY_RECLAIM_TIMEOUT_SECONDS = 8.0

    def __init__(self):
        self.tracker = InputTracker()
        self.policy = AuthorityPolicy()

        self._clock = time.monotonic
        self._last_soc_action_mono: Optional[float] = None

        self._forced_release: bool = False
        self._lock = threading.Lock()

        self._start_watchdog()

    # -------------------------------------------------
    # SOC LIVENESS
    # -------------------------------------------------

    def soc_action_started(self) -> None:
        """
        Marks SOC liveness.

        IMPORTANT:
        - Does NOT clear forced release
        - Forced release can only be cleared explicitly
        """
        self.tracker.mark_soc_action()
        with self._lock:
            self._last_soc_action_mono = self._clock()

    # -------------------------------------------------
    # DECISION
    # -------------------------------------------------

    def evaluate(
        self,
        *,
        input_event_ts: float,
        high_risk: bool,
        soc_confident: bool,
    ) -> AuthorityDecision:
        """
        Decide whether SOC continues or releases control.

        ORDER OF PRECEDENCE:
        1. Forced release (absolute)
        2. Human input (policy)
        3. High-risk escalation (policy)
        4. Continue
        """
        source = self.tracker.classify_input(input_event_ts)

        with self._lock:
            if self._forced_release:
                return AuthorityDecision.RELEASE

        # Human input → policy
        if source == InputSource.HUMAN:
            return self.policy.decide(
                human_intervened=True,
                high_risk=high_risk,
                soc_confident=soc_confident,
            )

        # High-risk even without human → policy
        if high_risk:
            return self.policy.decide(
                human_intervened=False,
                high_risk=True,
                soc_confident=soc_confident,
            )

        return AuthorityDecision.CONTINUE

    # -------------------------------------------------
    # FAILSAFE MECHANISMS
    # -------------------------------------------------

    def emergency_reclaim(self) -> None:
        """
        Immediate, monotonic reclaim.
        """
        with self._lock:
            self._forced_release = True

    def clear_emergency_reclaim(self) -> None:
        """
        Explicit manual clear.

        MUST be called only after human confirmation.
        """
        with self._lock:
            self._forced_release = False
            self._last_soc_action_mono = None

    # -------------------------------------------------
    # WATCHDOG
    # -------------------------------------------------

    def _start_watchdog(self) -> None:
        threading.Thread(
            target=self._watchdog_loop,
            daemon=True,
        ).start()

    def _watchdog_loop(self) -> None:
        """
        Deadman switch.

        If SOC goes silent, control is reclaimed
        and CANNOT be auto-cleared by SOC.
        """
        while True:
            time.sleep(0.5)

            with self._lock:
                if self._last_soc_action_mono is None:
                    continue

                idle = self._clock() - self._last_soc_action_mono
                if idle > self.EMERGENCY_RECLAIM_TIMEOUT_SECONDS:
                    self._forced_release = True

    # -------------------------------------------------
    # FORENSICS
    # -------------------------------------------------

    def get_authority_snapshot(self) -> Dict[str, object]:
        with self._lock:
            return {
                "forced_release": self._forced_release,
                "last_soc_action_mono": self._last_soc_action_mono,
                "timeout_seconds": self.EMERGENCY_RECLAIM_TIMEOUT_SECONDS,
        }
