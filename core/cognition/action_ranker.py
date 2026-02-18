# core/cognition/action_ranker.py

from typing import List, Dict, Any
import math
import hashlib
import json


class ActionRanker:
    """
    Deterministic action selector.

    Guarantees:
    - No randomness
    - Stable action identity
    - Single Thompson source (delegated to BeliefState)
    - Bounded exploration bonus
    """

    MIN_TAU = 0.15
    MAX_TAU = 1.5
    MAX_EXPLORATION_BONUS = 1.0  # prevent UCB domination

    # ==================================================
    # ACTION SELECTION (DETERMINISTIC)
    # ==================================================

    def select(
        self,
        actions: List[Dict[str, Any]],
        belief_state,
    ) -> Dict[str, Any]:

        if not actions:
            raise RuntimeError("No candidate actions")

        action_ids = [self.action_key(a) for a in actions]
        scores = []

        for action, key in zip(actions, action_ids):

            eu = belief_state.expected_utility(key)

            ucb_full = belief_state.ucb_score(key)

            rewards = belief_state.action_rewards.get(key)
            mean_reward = (
                sum(rewards) / len(rewards)
                if rewards else 0.0
            )

            exploration_bonus = ucb_full - mean_reward
            exploration_bonus = min(
                exploration_bonus,
                self.MAX_EXPLORATION_BONUS,
            )

            # Delegated Thompson (single coherent implementation)
            thompson = belief_state.thompson_sample(key)

            combined = (
                0.6 * eu
                + 0.2 * exploration_bonus
                + 0.2 * thompson
            )

            scores.append(combined)

        # --------------------------------------------------
        # Entropy-adaptive temperature
        # --------------------------------------------------

        entropy = belief_state.entropy()
        tau = self._entropy_temperature(entropy)

        # --------------------------------------------------
        # Deterministic softmax (argmax of probabilities)
        # --------------------------------------------------

        max_score = max(scores)
        shifted = [(s - max_score) / tau for s in scores]

        exp_scores = [
            math.exp(min(50.0, s))
            for s in shifted
        ]
        total = sum(exp_scores)

        if total <= 0.0:
            return actions[scores.index(max_score)]

        probabilities = [s / total for s in exp_scores]

        # Deterministic: choose highest probability
        selected_index = max(
            range(len(actions)),
            key=lambda i: probabilities[i],
        )

        return actions[selected_index]

    # ==================================================
    # ENTROPY TEMPERATURE
    # ==================================================

    def _entropy_temperature(self, entropy: float) -> float:
        tau = math.tanh(entropy) * self.MAX_TAU
        return max(self.MIN_TAU, min(self.MAX_TAU, tau))

    # ==================================================
    # STABLE ACTION KEY
    # ==================================================

    def action_key(self, action: Dict[str, Any]) -> str:
        """
        Stable identity based only on logical operation
        and canonical essential fields.

        Prevents history invalidation if extra metadata
        fields are added to the action dict.
        """

        operation = str(action.get("operation", "")).strip()

        # canonical minimal identity subset
        canonical_subset = {
            "operation": operation,
            "target": action.get("target"),
            "text": action.get("text"),
            "keys": action.get("keys"),
        }

        canonical = json.dumps(
            canonical_subset,
            sort_keys=True,
            default=str,
        )

        return hashlib.sha256(
            canonical.encode()
        ).hexdigest()[:16]
