# core/cognition/action_ranker.py

from typing import List, Dict, Any
import math
import random
import hashlib
import json


class ActionRanker:
    """
    Decision-theoretic action selector.

    Combines:
    - Risk-adjusted Expected Utility
    - Pure exploration bonus (UCB minus mean)
    - Thompson sampling
    - Entropy-aware softmax

    No regret updates occur here.
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

            # ---------------- Exploitation ----------------
            eu = belief_state.expected_utility(key)

            # ---------------- Exploration ----------------
            # UCB = mean + exploration
            ucb_full = belief_state.ucb_score(key)

            rewards = belief_state.action_rewards.get(key)
            mean_reward = (
                sum(rewards) / len(rewards)
                if rewards else 0.0
            )

            exploration_bonus = ucb_full - mean_reward

            # ---------------- Posterior Sampling ----------------
            thompson = belief_state.thompson_sample(key)

            # ---------------- Balanced Combination ----------------
            # Do NOT double-count mean reward.
            combined = (
                0.6 * eu
                + 0.2 * exploration_bonus
                + 0.2 * thompson
            )

            scores.append(combined)

        # --------------------------------------------------
        # Entropy-aware temperature
        # --------------------------------------------------

        entropy = belief_state.entropy()
        tau = self._entropy_temperature(entropy)

        # --------------------------------------------------
        # Numerically stable softmax (overflow safe)
        # --------------------------------------------------

        max_score = max(scores)
        shifted = [(s - max_score) / tau for s in scores]

        # Prevent overflow
        exp_scores = [math.exp(min(50.0, s)) for s in shifted]
        total = sum(exp_scores)

        if total <= 0.0:
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
        Low entropy → exploitation (low tau)
        High entropy → exploration (high tau)
        """

        # Smooth mapping instead of direct identity
        tau = math.tanh(entropy) * self.MAX_TAU

        return max(self.MIN_TAU, min(self.MAX_TAU, tau))

    # ==================================================
    # ACTION KEYING (COLLISION-SAFE)
    # ==================================================

    def _action_key(self, action: Dict[str, Any]) -> str:
        """
        Canonical deterministic identity.
        Prevents collisions caused by string concatenation.
        """

        canonical = json.dumps(
            action,
            sort_keys=True,
            default=str,
        )

        return hashlib.sha256(
            canonical.encode()
        ).hexdigest()[:16]
