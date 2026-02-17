from typing import Dict, Any, List
import time
import math
import random
import hashlib
import json


class BeliefState:
    """
    Decision-theoretic cognitive belief engine.
    """

    EXPLORATION_C = 1.4
    RISK_LAMBDA = 0.3
    SOFTMAX_TAU = 0.5

    def __init__(self):
        self.created_at = time.time()

        self.state_probabilities: Dict[str, float] = {
            "neutral": 1.0
        }

        self.action_counts: Dict[str, int] = {}
        self.action_rewards: Dict[str, List[float]] = {}
        self.regret: Dict[str, float] = {}

        self.progress_score: float = 0.0
        self.environment_stability: float = 1.0

        self.commitment_hash: str = "GENESIS"

    # ==================================================
    # BAYESIAN UPDATE (UNION-SAFE)
    # ==================================================

    def bayesian_update(self, likelihoods: Dict[str, float]) -> None:
        all_states = set(self.state_probabilities) | set(likelihoods)

        new_belief: Dict[str, float] = {}
        total = 0.0

        for state in all_states:
            prior = self.state_probabilities.get(state, 0.01)
            likelihood = likelihoods.get(state, 0.01)
            posterior = prior * likelihood
            new_belief[state] = posterior
            total += posterior

        if total == 0:
            return

        self.state_probabilities = {
            s: v / total for s, v in new_belief.items()
        }

    # ==================================================
    # ENTROPY
    # ==================================================

    def entropy(self) -> float:
        return -sum(
            p * math.log(p + 1e-12)
            for p in self.state_probabilities.values()
        )

    # ==================================================
    # EXPECTED UTILITY
    # ==================================================

    def expected_utility(self, action: str) -> float:
        rewards = self.action_rewards.get(action, [])
        if not rewards:
            return 0.0

        mean_reward = sum(rewards) / len(rewards)
        variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)

        return mean_reward - self.RISK_LAMBDA * variance

    # ==================================================
    # UCB EXPLORATION BONUS
    # ==================================================

    def ucb_score(self, action: str) -> float:
        total_actions = sum(self.action_counts.values()) + 1
        count = self.action_counts.get(action, 0) + 1

        mean_reward = self.expected_utility(action)

        exploration = self.EXPLORATION_C * math.sqrt(
            math.log(total_actions) / count
        )

        return mean_reward + exploration

    # ==================================================
    # THOMPSON SAMPLING (BOUNDED)
    # ==================================================

    def thompson_sample(self, action: str) -> float:
        rewards = self.action_rewards.get(action, [])
        if not rewards:
            return random.random()

        mean = sum(rewards) / len(rewards)
        variance = (
            sum((r - mean) ** 2 for r in rewards) / len(rewards)
        ) + 1e-6

        sample = random.gauss(mean, math.sqrt(variance))

        min_r = min(rewards)
        max_r = max(rewards)

        lower_bound = min_r - 1.0
        upper_bound = max_r + 1.0

        return max(lower_bound, min(upper_bound, sample))

    # ==================================================
    # COUNTERFACTUAL REGRET UPDATE
    # ==================================================

    def update_regret(self, action: str, reward: float, best_reward: float):
        regret_value = best_reward - reward
        self.regret[action] = self.regret.get(action, 0.0) + regret_value

    # ==================================================
    # SOFTMAX EQUILIBRIUM SELECTION (STABLE)
    # ==================================================

    def softmax_select(self, actions: List[str]) -> str:
        if not actions:
            raise ValueError("No actions provided to softmax_select")

        scores = [self.ucb_score(a) for a in actions]

        max_score = max(scores)
        shifted = [
            (s - max_score) / self.SOFTMAX_TAU
            for s in scores
        ]

        exp_scores = [math.exp(s) for s in shifted]
        total = sum(exp_scores)

        if total <= 0.0:
            return actions[scores.index(max_score)]

        probabilities = [s / total for s in exp_scores]

        return random.choices(actions, weights=probabilities, k=1)[0]

    # ==================================================
    # RECORD ACTION OUTCOME
    # ==================================================

    def record_action(self, action: str, reward: float):
        self.action_counts[action] = self.action_counts.get(action, 0) + 1
        self.action_rewards.setdefault(action, []).append(reward)

    # ==================================================
    # ENVIRONMENT STABILITY
    # ==================================================

    def compute_environment_stability(self, delta: Dict[str, Any]) -> None:
        if not delta:
            return

        significant = delta.get("significant_change", False)

        if significant:
            self.environment_stability *= 0.8
        else:
            self.environment_stability = min(
                1.0, self.environment_stability + 0.05
            )

    # ==================================================
    # COMMITMENT HASH CHAIN (DETERMINISTIC)
    # ==================================================

    def commit(self, action: str, observation: Dict[str, Any]) -> None:
        payload = (
            self.commitment_hash
            + action
            + json.dumps(observation, sort_keys=True, default=str)
            + json.dumps(self.state_probabilities, sort_keys=True)
        )

        self.commitment_hash = hashlib.sha256(
            payload.encode()
        ).hexdigest()

    # ==================================================
    # CONVERGENCE DETECTION (FIXED SIGNATURE)
    # ==================================================

    def converged(
        self,
        min_iterations: int = 0,
        current_iteration: int = 0,
    ) -> bool:

        if current_iteration < min_iterations:
            return False

        low_entropy = self.entropy() < 0.1
        stable_env = self.environment_stability > 0.9
        high_progress = self.progress_score > 0.5

        return low_entropy and stable_env and high_progress

    # ==================================================
    # SUMMARY
    # ==================================================

    def summary(self) -> Dict[str, Any]:
        return {
            "entropy": self.entropy(),
            "state_distribution": self.state_probabilities,
            "action_counts": self.action_counts,
            "regret": self.regret,
            "progress_score": self.progress_score,
            "environment_stability": self.environment_stability,
            "commitment_hash": self.commitment_hash,
        }
