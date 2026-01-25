import time
import threading
from typing import Optional, Dict

from authority.input_tracker import InputTracker, InputSource
from authority.authority_policy import AuthorityPolicy, AuthorityDecision


class InputArbitrator:
    """
    Arbitrates control between SOC and human.
    SOC NEVER fights the human.
    """

    EMERGENCY_RECLAIM_TIMEOUT_SECONDS = 3.0

    def __init__(self):
        self.tracker = InputTracker()
        self.policy = AuthorityPolicy()

        # === ADDITIONS ===
        self._last_soc_action_ts: Optional[float] = None
        self._forced_release: bool = False
        self._lock = threading.Lock()

        self._start_watchdog()

    # -------------------------------------------------
    # Existing API
    # -------------------------------------------------

    def soc_action_started(self):
        self.tracker.mark_soc_action()

        # ADDITION
        with self._lock:
            self._last_soc_action_ts = time.time()

    def evaluate(
        self,
        *,
        input_event_ts: float,
        high_risk: bool,
        soc_confident: bool,
    ) -> AuthorityDecision:

        source = self.tracker.classify_input(input_event_ts)

        # ADDITION: forced reclaim
        if self._forced_release:
            return AuthorityDecision.RELEASE

        if source == InputSource.HUMAN:
            return self.policy.decide(
                human_intervened=True,
                high_risk=high_risk,
                soc_confident=soc_confident,
            )

        return AuthorityDecision.CONTINUE

    # -------------------------------------------------
    # ADDITIONS — FAILSAFE MECHANISMS
    # -------------------------------------------------

    def emergency_reclaim(self) -> None:
        """
        Can be bound to OS-level hotkey.
        Forces immediate input release.
        """
        with self._lock:
            self._forced_release = True

    def clear_emergency_reclaim(self) -> None:
        with self._lock:
            self._forced_release = False

    def _start_watchdog(self) -> None:
        t = threading.Thread(
            target=self._watchdog_loop,
            daemon=True,
        )
        t.start()

    def _watchdog_loop(self) -> None:
        """
        Deadman switch:
        If SOC stops emitting action heartbeats,
        human control is restored.
        """
        while True:
            time.sleep(0.5)

            with self._lock:
                if self._last_soc_action_ts is None:
                    continue

                idle = time.time() - self._last_soc_action_ts
                if idle > self.EMERGENCY_RECLAIM_TIMEOUT_SECONDS:
                    self._forced_release = True

    def get_authority_snapshot(self) -> Dict[str, object]:
        """
        Forensic snapshot of input arbitration state.
        """
        with self._lock:
            return {
                "forced_release": self._forced_release,
                "last_soc_action_ts": self._last_soc_action_ts,
                "timeout_seconds": self.EMERGENCY_RECLAIM_TIMEOUT_SECONDS,
    }
