"""
authority/authority_policy.py
==============================
PATCHES APPLIED (Audit Fixes):

  ✅  FIX-1 (CRITICAL): Added AuthorityDecision.WAIT to the enum.
           operate/operate.py references AuthorityDecision.WAIT in its
           authority evaluation guard, but the value was never defined here.
           At runtime this raised AttributeError on the very first action of
           every task — making the execution engine completely non-functional.

           WAIT is semantically distinct from YIELD:
             - YIELD: human intervened, kernel should pause and give way.
             - WAIT:  kernel should pause and retry after a brief window
                      (e.g. system still loading, action pre-condition not met).
           Both result in a retry loop in operate.py before REPLAN is triggered.

All existing correct behaviours preserved:
  - CONTINUE: no human input — SOC may proceed
  - YIELD: human intervened — pause and give way
  - ABORT: high-risk + human — immediate hard stop
  - RELEASE: compatibility alias for YIELD
"""

from enum import Enum


class AuthorityDecision(str, Enum):
    CONTINUE = "CONTINUE"
    YIELD    = "YIELD"
    WAIT     = "WAIT"     # FIX-1: was missing — caused AttributeError in operate.py
    ABORT    = "ABORT"
    RELEASE  = "RELEASE"  # compatibility alias (treated as YIELD/WAIT in operate.py)


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
