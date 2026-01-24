from enum import Enum


class AuthorityDecision(str, Enum):
    CONTINUE = "CONTINUE"
    YIELD = "YIELD"
    ABORT = "ABORT"


class AuthorityPolicy:
    """
    Decides what to do when human input is detected.
    """

    def decide(
        self,
        *,
        human_intervened: bool,
        high_risk: bool,
        soc_confident: bool,
    ) -> AuthorityDecision:

        if not human_intervened:
            return AuthorityDecision.CONTINUE

        # Human always wins
        if high_risk:
            return AuthorityDecision.ABORT

        # Allow co-pilot style intervention
        if not soc_confident:
            return AuthorityDecision.YIELD

        # Default: yield safely
        return AuthorityDecision.YIELD
