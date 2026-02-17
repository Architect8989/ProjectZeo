# core/cognition/action_ranker.py

from typing import List, Dict, Any
import math
import random


class ActionRanker:
    """
    Decision-theoretic action selector.

    Combines:
    - Risk-adjusted Expected Utility
    - UCB1 exploration bonus
    - Thompson sampling
    - Entropy-aware softmax mixed strategy

    IMPORTANT:
    - No regret updates occur here.
    - Regret must be updated AFTER real reward is observed.
    """

    MIN_TAU = 0.15
    MAX_TAU = 1.5

    # ==================================================
    # ACTION SELECTION
    # ==================================================

    def select(
        self,
        actions: List[Dict[str, Any]],
        belief_state,
    ) -> Dict[str, Any]:

        if not actions:
            raise RuntimeError("No candidate actions")

        action_ids = [self._action_key(a) for a in actions]

        scores = []

        for action, key in zip(actions, action_ids):

            # Exploitation
            eu = belief_state.expected_utility(key)

            # Optimism under uncertainty
            ucb = belief_state.ucb_score(key)

            # Posterior sampling
            thompson = belief_state.thompson_sample(key)

            # Combined score
            combined = 0.5 * eu + 0.3 * ucb + 0.2 * thompson

            scores.append(combined)

        # --------------------------------------------------
        # Entropy-aware temperature
        # --------------------------------------------------

        entropy = belief_state.entropy()
        tau = self._entropy_temperature(entropy)

        # --------------------------------------------------
        # Numerically stable softmax (log-sum-exp trick)
        # --------------------------------------------------

        max_score = max(scores)
        shifted = [(s - max_score) / tau for s in scores]

        exp_scores = [math.exp(s) for s in shifted]
        total = sum(exp_scores)

        if total <= 0.0:
            # fallback deterministic
            return actions[scores.index(max_score)]

        probabilities = [s / total for s in exp_scores]

        selected_index = random.choices(
            range(len(actions)),
            weights=probabilities,
            k=1,
        )[0]

        return actions[selected_index]

    # ==================================================
    # ENTROPY-ADAPTIVE TEMPERATURE
    # ==================================================

    def _entropy_temperature(self, entropy: float) -> float:
        """
        High entropy → exploration (higher tau)
        Low entropy → exploitation (lower tau)
        """

        tau = entropy
        tau = max(self.MIN_TAU, min(self.MAX_TAU, tau))
        return tau

    # ==================================================
    # ACTION KEYING
    # ==================================================

    def _action_key(self, action: Dict[str, Any]) -> str:
        """
        Deterministic action identity for reward tracking.
        """

        op = action.get("operation", "")
        target = str(action.get("target", ""))
        text = str(action.get("text", ""))
        keys = str(action.get("keys", ""))

        return f"{op}|{target}|{text}|{keys}"
