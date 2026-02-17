# core/cognition/action_ranker.py

from typing import List, Dict, Any
import math
import random


class ActionRanker:
    """
    Decision-theoretic action selector.

    Combines:
    - Risk-adjusted Expected Utility
    - UCB1 exploration
    - Thompson sampling
    - Regret minimization
    - Softmax mixed strategy equilibrium
    - Entropy-aware temperature control
    """

    MIN_TAU = 0.15
    MAX_TAU = 1.5

    def select(
        self,
        actions: List[Dict[str, Any]],
        belief_state,
    ) -> Dict[str, Any]:

        if not actions:
            raise RuntimeError("No candidate actions")

        action_ids = [self._action_key(a) for a in actions]

        # Score each action
        scores = []
        best_reward_estimate = float("-inf")

        for action, key in zip(actions, action_ids):

            eu = belief_state.expected_utility(key)
            ucb = belief_state.ucb_score(key)
            thompson = belief_state.thompson_sample(key)

            # Combine exploitation + exploration
            combined = 0.5 * eu + 0.3 * ucb + 0.2 * thompson

            # Regret penalty
            regret_penalty = belief_state.regret.get(key, 0.0)
            combined -= regret_penalty * 0.1

            scores.append(combined)

            if combined > best_reward_estimate:
                best_reward_estimate = combined

        # Update regrets
        for key, score in zip(action_ids, scores):
            belief_state.update_regret(
                key,
                reward=score,
                best_reward=best_reward_estimate,
            )

        # Entropy-aware temperature
        entropy = belief_state.entropy()
        tau = self._entropy_temperature(entropy)

        # Softmax mixed strategy equilibrium
        exp_scores = [math.exp(s / tau) for s in scores]
        total = sum(exp_scores)
        probabilities = [s / total for s in exp_scores]

        selected_index = random.choices(
            range(len(actions)),
            weights=probabilities,
            k=1,
        )[0]

        selected_action = actions[selected_index]

        return selected_action

    # ==================================================
    # ENTROPY-ADAPTIVE TEMPERATURE
    # ==================================================

    def _entropy_temperature(self, entropy: float) -> float:
        """
        High entropy → high exploration (higher tau)
        Low entropy → exploitation (lower tau)
        """

        # Normalize entropy roughly
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
