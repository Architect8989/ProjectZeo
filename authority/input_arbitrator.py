import time
from authority.input_tracker import InputTracker, InputSource
from authority.authority_policy import AuthorityPolicy, AuthorityDecision


class InputArbitrator:
    """
    Arbitrates control between SOC and human.
    SOC NEVER fights the human.
    """

    def __init__(self):
        self.tracker = InputTracker()
        self.policy = AuthorityPolicy()

    def soc_action_started(self):
        self.tracker.mark_soc_action()

    def evaluate(
        self,
        *,
        input_event_ts: float,
        high_risk: bool,
        soc_confident: bool,
    ) -> AuthorityDecision:

        source = self.tracker.classify_input(input_event_ts)

        if source == InputSource.HUMAN:
            return self.policy.decide(
                human_intervened=True,
                high_risk=high_risk,
                soc_confident=soc_confident,
            )

        return AuthorityDecision.CONTINUE
