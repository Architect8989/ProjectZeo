# core/cognition/belief_state.py

from typing import Dict, Any, List
import time
import math
import random
import hashlib


class BeliefState:
    """
    Decision-theoretic cognitive belief engine.

    Implements:
    - Bayesian belief updates
    - Entropy measurement
    - Thompson Sampling
    - UCB1 exploration
    - Risk-adjusted expected utility
    - Counterfactual regret minimization
    - Softmax equilibrium selection
    - Cryptographic commitment chain
    """

    EXPLORATION_C = 1.4
    RISK_LAMBDA = 0.3
    SOFTMAX_TAU = 0.5

    def __init__(self):
        self.created_at = time.time()

        # Probabilistic state hypothesis
        self.state_probabilities: Dict[str, float] = {
            "neutral": 1.0
        }

        # Action statistics
        self.action_counts: Dict[str, int] = {}
        self.action_rewards: Dict[str, List[float]] = {}
        self.regret: Dict[str, float] = {}

        # Progress
        self.progress_score: float = 0.0

        # Stability metric
        self.environment_stability: float = 1.0

        # Commitment chain
        self.commitment_hash: str = "GENESIS"

    # ==================================================
    # BAYESIAN UPDATE
    # ==================================================

    def bayesian_update(self, likelihoods: Dict[str, float]) -> None:
        """
        Update belief using Bayes rule.

        likelihoods: P(O | S)
        """

        new_belief = {}
        total = 0.0

        for state, prior in self.state_probabilities.items():
            likelihood = likelihoods.get(state, 0.01)
            posterior = prior * likelihood
            new_belief[state] = posterior
            total += posterior

        if total == 0:
            return

        for state in new_belief:
            new_belief[state] /= total

        self.state_probabilities = new_belief

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
        """
        Risk-adjusted expected utility.
        """

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
    # THOMPSON SAMPLING
    # ==================================================

    def thompson_sample(self, action: str) -> float:
        rewards = self.action_rewards.get(action, [])
        if not rewards:
            return random.random()

        mean = sum(rewards) / len(rewards)
        variance = (
            sum((r - mean) ** 2 for r in rewards) / len(rewards)
        ) + 1e-6

        return random.gauss(mean, math.sqrt(variance))

    # ==================================================
    # COUNTERFACTUAL REGRET UPDATE
    # ==================================================

    def update_regret(self, action: str, reward: float, best_reward: float):
        regret_value = best_reward - reward
        self.regret[action] = self.regret.get(action, 0.0) + regret_value

    # ==================================================
    # SOFTMAX EQUILIBRIUM SELECTION
    # ==================================================

    def softmax_select(self, actions: List[str]) -> str:
        scores = []

        for action in actions:
            score = self.ucb_score(action)
            scores.append(score)

        exp_scores = [
            math.exp(score / self.SOFTMAX_TAU)
            for score in scores
        ]

        total = sum(exp_scores)
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
    # COMMITMENT HASH CHAIN
    # ==================================================

    def commit(self, action: str, observation: Dict[str, Any]) -> None:
        payload = (
            self.commitment_hash
            + action
            + str(observation)
            + str(self.state_probabilities)
        )

        self.commitment_hash = hashlib.sha256(
            payload.encode()
        ).hexdigest()

    # ==================================================
    # CONVERGENCE DETECTION
    # ==================================================

    def converged(self) -> bool:
        low_entropy = self.entropy() < 0.1
        stable_env = self.environment_stability > 0.9
        return low_entropy and stable_env

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
