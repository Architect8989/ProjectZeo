from enum import Enum


class AuthorityDecision(str, Enum):
    CONTINUE = "CONTINUE"
    YIELD = "YIELD"
    ABORT = "ABORT"
    RELEASE = "RELEASE"   # compatibility alias (treated as YIELD)


class AuthorityPolicy:
    """
    Authority decision policy.

    HARD RULES:
    - Human input always dominates automation
    - High-risk situations abort immediately
    - No silent overrides
    - Deterministic outcomes (no randomness)
    """

    def decide(
        self,
        *,
        human_intervened: bool,
        high_risk: bool,
        soc_confident: bool,
    ) -> AuthorityDecision:

        # -------------------------------------------------
        # NO HUMAN → SOC MAY CONTINUE
        # -------------------------------------------------
        if not human_intervened:
            return AuthorityDecision.CONTINUE

        # -------------------------------------------------
        # HUMAN INPUT DETECTED → HUMAN WINS
        # -------------------------------------------------

        # High-risk + human = immediate abort
        if high_risk:
            return AuthorityDecision.ABORT

        # Human intervened, SOC not confident → yield
        if not soc_confident:
            return AuthorityDecision.YIELD

        # Human intervened, SOC confident → still yield
        # (no contested authority allowed)
        return AuthorityDecision.YIELD
