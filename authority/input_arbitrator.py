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
    - Watchdog thread can be cleanly stopped
    """

    EMERGENCY_RECLAIM_TIMEOUT_SECONDS = 8.0
    WATCHDOG_INTERVAL_SECONDS = 0.5

    def __init__(self):
        self.tracker = InputTracker()
        self.policy = AuthorityPolicy()

        self._clock = time.monotonic
        self._last_soc_action_mono: Optional[float] = None

        self._forced_release: bool = False
        self._lock = threading.Lock()

        self._stop_event = threading.Event()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            daemon=True,
        )
        self._watchdog_thread.start()

    # -------------------------------------------------
    # SOC LIVENESS
    # -------------------------------------------------

    def soc_action_started(self) -> None:
        """
        Marks SOC liveness.

        DOES NOT clear forced release.
        """
        self.tracker.mark_soc_action()
        now = self._clock()
        with self._lock:
            self._last_soc_action_mono = now

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

        if source == InputSource.HUMAN:
            return self.policy.decide(
                human_intervened=True,
                high_risk=high_risk,
                soc_confident=soc_confident,
            )

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
        with self._lock:
            self._forced_release = True

    def clear_emergency_reclaim(self) -> None:
        with self._lock:
            self._forced_release = False
            self._last_soc_action_mono = None

    # -------------------------------------------------
    # WATCHDOG
    # -------------------------------------------------

    def shutdown(self) -> None:
        """
        Cleanly stop watchdog thread.
        """
        self._stop_event.set()
        if self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=2)

    def _watchdog_loop(self) -> None:
        """
        Deadman switch.
        """
        while not self._stop_event.is_set():
            time.sleep(self.WATCHDOG_INTERVAL_SECONDS)

            with self._lock:
                if self._forced_release:
                    continue

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
