# core/cognition/action_ranker.py

from typing import List, Dict, Any
import math
import random
import hashlib
import json


class ActionRanker:
    MIN_TAU = 0.15
    MAX_TAU = 1.5

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

            thompson = self._local_thompson_sample(
                rewards if rewards else []
            )

            combined = (
                0.6 * eu
                + 0.2 * exploration_bonus
                + 0.2 * thompson
            )

            scores.append(combined)

        entropy = belief_state.entropy()
        tau = self._entropy_temperature(entropy)

        max_score = max(scores)
        shifted = [(s - max_score) / tau for s in scores]

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

    def _local_thompson_sample(self, rewards) -> float:
        successes = sum(1 for r in rewards if r > 0)
        failures = sum(1 for r in rewards if r < 0)
        return random.betavariate(successes + 1, failures + 1)

    def _entropy_temperature(self, entropy: float) -> float:
        tau = math.tanh(entropy) * self.MAX_TAU
        return max(self.MIN_TAU, min(self.MAX_TAU, tau))

    def action_key(self, action: Dict[str, Any]) -> str:
        canonical = json.dumps(
            action,
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(
            canonical.encode()
        ).hexdigest()[:16]
