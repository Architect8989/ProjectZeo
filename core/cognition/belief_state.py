from typing import Dict, Any, List
from collections import deque
import time
import math
import random
import hashlib
import json
import struct


class BeliefState:
    """
    Decision-theoretic cognitive belief engine.
    Production-hardened and convergence-safe.
    """

    EXPLORATION_C = 1.4
    RISK_LAMBDA = 0.3
    SOFTMAX_TAU = 0.5
    REWARD_WINDOW = 100
    REGRET_SCALE = 0.05
    PRIOR_ALPHA = 0.001  # Dirichlet prior floor
    REGRET_DECAY = 0.995  # prevents unbounded regret growth

    def __init__(self):
        self.created_at = time.time()

        self.state_probabilities: Dict[str, float] = {
            "neutral": 1.0
        }

        self.action_counts: Dict[str, int] = {}
        self.action_rewards: Dict[str, deque] = {}
        self.regret: Dict[str, float] = {}

        self.progress_score: float = 0.0
        self.environment_stability: float = 1.0
        self.commitment_hash: str = "GENESIS"

    # ==================================================
    # BAYESIAN UPDATE (DIRICHLET SAFE)
    # ==================================================

    def bayesian_update(self, likelihoods: Dict[str, float]) -> None:
        all_states = set(self.state_probabilities) | set(likelihoods)

        new_belief: Dict[str, float] = {}
        total = 0.0

        for state in all_states:
            prior = self.state_probabilities.get(state, self.PRIOR_ALPHA)
            likelihood = likelihoods.get(state, self.PRIOR_ALPHA)

            posterior = prior * max(likelihood, self.PRIOR_ALPHA)
            new_belief[state] = posterior
            total += posterior

        if total <= 0:
            return

        self.state_probabilities = {
            s: v / total for s, v in new_belief.items()
        }

    # ==================================================
    # ENTROPY
    # ==================================================

    def entropy(self) -> float:
        total = 0.0
        for p in self.state_probabilities.values():
            if p > 0:
                total -= p * math.log(p)
        return total

    # ==================================================
    # EXPECTED UTILITY
    # ==================================================

    def expected_utility(self, action: str) -> float:
        rewards = self.action_rewards.get(action)
        if not rewards:
            return 0.0

        n = len(rewards)
        mean = sum(rewards) / n
        variance = sum((r - mean) ** 2 for r in rewards) / n

        return mean - self.RISK_LAMBDA * variance

    # ==================================================
    # UCB
    # ==================================================

    def ucb_score(self, action: str) -> float:
        total_actions = sum(self.action_counts.values()) + 1
        count = self.action_counts.get(action, 0) + 1

        rewards = self.action_rewards.get(action)
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0

        exploration = self.EXPLORATION_C * math.sqrt(
            math.log(total_actions) / count
        )

        return mean_reward + exploration

    # ==================================================
    # THOMPSON SAMPLING (BETA STABLE)
    # ==================================================

    def thompson_sample(self, action: str) -> float:
        rewards = self.action_rewards.get(action)
        if not rewards:
            return random.random()

        # Map rewards to success/failure
        successes = sum(1 for r in rewards if r > 0) + 1
        failures = sum(1 for r in rewards if r <= 0) + 1

        return random.betavariate(successes, failures)

    # ==================================================
    # REGRET UPDATE (DECAYED)
    # ==================================================

    def update_regret(self, action: str, reward: float, best_reward: float):
        # Apply decay to all regret values
        for k in list(self.regret.keys()):
            self.regret[k] *= self.REGRET_DECAY

        regret_value = best_reward - reward
        if regret_value > 0:
            self.regret[action] = (
                self.regret.get(action, 0.0) + regret_value
            )

    # ==================================================
    # SOFTMAX SELECTION (NUMERICALLY SAFE)
    # ==================================================

    def softmax_select(self, actions: List[str]) -> str:
        if not actions:
            raise ValueError("No actions provided")

        scores = []
        for a in actions:
            base_score = self.ucb_score(a)
            regret_penalty = min(
                self.regret.get(a, 0.0) * self.REGRET_SCALE,
                0.5,
            )
            scores.append(base_score - regret_penalty)

        max_score = max(scores)
        tau = max(0.15, self.SOFTMAX_TAU)

        shifted = [(s - max_score) / tau for s in scores]

        # Overflow protection
        exp_scores = [math.exp(min(50, s)) for s in shifted]
        total = sum(exp_scores)

        if total <= 0:
            return actions[scores.index(max_score)]

        probabilities = [s / total for s in exp_scores]

        return random.choices(actions, weights=probabilities, k=1)[0]

    # ==================================================
    # RECORD ACTION
    # ==================================================

    def record_action(self, action: str, reward: float):
        self.action_counts[action] = self.action_counts.get(action, 0) + 1

        if action not in self.action_rewards:
            self.action_rewards[action] = deque(maxlen=self.REWARD_WINDOW)

        self.action_rewards[action].append(reward)

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
                1.0,
                self.environment_stability + 0.05,
            )

        # Clamp to [0,1]
        self.environment_stability = max(
            0.0,
            min(1.0, self.environment_stability),
        )

    # ==================================================
    # COMMITMENT HASH (DETERMINISTIC)
    # ==================================================

    def _stable_float_bytes(self, d: Dict[str, float]) -> bytes:
        parts = []
        for k in sorted(d):
            parts.append(k.encode())
            parts.append(struct.pack("!d", float(d[k])))
        return b"".join(parts)

    def commit(self, action: str, observation: Dict[str, Any]) -> None:
        obs_bytes = json.dumps(
            observation,
            sort_keys=True,
            default=str,
        ).encode()

        prob_bytes = self._stable_float_bytes(
            self.state_probabilities
        )

        payload = (
            self.commitment_hash.encode()
            + action.encode()
            + obs_bytes
            + prob_bytes
        )

        self.commitment_hash = hashlib.sha256(
            payload
        ).hexdigest()

    # ==================================================
    # CONVERGENCE (PLAN-BOUND)
    # ==================================================

    def converged(
        self,
        *,
        min_iterations: int = 0,
        current_iteration: int = 0,
        plan_steps_total: int = 0,
        steps_completed: int = 0,
    ) -> bool:

        if current_iteration < min_iterations:
            return False

        # Require full plan completion if plan provided
        if plan_steps_total > 0:
            if steps_completed < plan_steps_total:
                return False

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
