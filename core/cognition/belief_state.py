from typing import Dict, Any, Tuple
from collections import deque
import time
import math
import hashlib
import struct
import numpy as np


class BeliefState:

    EXPLORATION_C = 1.4
    RISK_LAMBDA = 0.3
    REWARD_WINDOW = 100
    PRIOR_ALPHA = 0.01
    REGRET_DECAY = 0.995
    MAX_STATES = 64
    MAX_REGRET = 100.0
    MIN_ENTROPY_FLOOR = 0.1
    NORMALIZE_EPS = 1e-8
    REWARD_CLAMP = 3.0

    def __init__(self):
        self.created_at = time.time()
        self.state_probabilities: Dict[str, float] = {"neutral": 1.0}
        self.action_counts: Dict[str, int] = {}
        self.action_rewards: Dict[str, deque] = {}
        self.regret: Dict[str, Tuple[float, int]] = {}
        self.progress_score: float = 0.0
        self.environment_stability: float = 1.0
        self.commitment_hash: str = "GENESIS"
        self._iteration_counter: int = 0

    # =========================================================
    # BELIEF UPDATE
    # =========================================================

    def bayesian_update(self, likelihoods: Dict[str, float]) -> None:
        if not likelihoods:
            return

        all_states = set(self.state_probabilities) | set(likelihoods)
        new_belief: Dict[str, float] = {}
        total = 0.0

        for state in all_states:
            prior = self.state_probabilities.get(state, self.PRIOR_ALPHA)
            likelihood = max(likelihoods.get(state, self.PRIOR_ALPHA), self.PRIOR_ALPHA)
            posterior = prior * likelihood
            new_belief[state] = posterior
            total += posterior

        if total <= 0:
            return

        normalized = {s: v / total for s, v in new_belief.items()}

        pruned = {
            s: p for s, p in normalized.items()
            if p >= self.PRIOR_ALPHA * 0.1
        }

        if len(pruned) > self.MAX_STATES:
            pruned = dict(
                sorted(pruned.items(), key=lambda x: x[1], reverse=True)[: self.MAX_STATES]
            )

        total = sum(pruned.values())
        if total <= 0:
            return

        self.state_probabilities = {s: v / total for s, v in pruned.items()}

        if self.entropy() < self.MIN_ENTROPY_FLOOR:
            uniform = 1.0 / len(self.state_probabilities)
            blended = {
                k: 0.95 * v + 0.05 * uniform
                for k, v in self.state_probabilities.items()
            }
            total = sum(blended.values())
            self.state_probabilities = {k: v / total for k, v in blended.items()}

    def entropy(self) -> float:
        return -sum(
            p * math.log(p)
            for p in self.state_probabilities.values()
            if p > 0
        )

    # =========================================================
    # ACTION SCORING
    # =========================================================

    def expected_utility(self, action: str) -> float:
        rewards = self.action_rewards.get(action)
        if not rewards:
            return 0.0

        n = len(rewards)
        mean = sum(rewards) / n
        variance = sum((r - mean) ** 2 for r in rewards) / n
        return mean - self.RISK_LAMBDA * variance

    def ucb_score(self, action: str) -> float:
        total_actions = sum(self.action_counts.values()) + 1
        count = self.action_counts.get(action, 0) + 1
        rewards = self.action_rewards.get(action)

        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0

        exploration = self.EXPLORATION_C * math.sqrt(
            math.log(total_actions) / count
        )

        return mean_reward + exploration

    # =========================================================
    # TRUE DETERMINISTIC THOMPSON SAMPLING
    # =========================================================

    def thompson_sample(self, action: str) -> float:
        rewards = self.action_rewards.get(action)
        if not rewards:
            return 0.5

        successes = sum(1 for r in rewards if r > 0)
        failures = sum(1 for r in rewards if r < 0)

        alpha = successes + 1.0
        beta = failures + 1.0

        seed_material = (
            f"{self.commitment_hash}:{action}:{self._iteration_counter}"
        ).encode("utf-8")

        digest = hashlib.sha256(seed_material).digest()
        seed = int.from_bytes(digest[:8], byteorder="big", signed=False)

        rng = np.random.default_rng(seed)

        sample = rng.beta(alpha, beta)

        return float(sample)

    # =========================================================
    # REGRET TRACKING
    # =========================================================

    def _get_effective_regret(self, action: str) -> float:
        entry = self.regret.get(action)
        if not entry:
            return 0.0

        raw_value, last_iter = entry
        delta_iter = self._iteration_counter - last_iter
        decayed = raw_value * (self.REGRET_DECAY ** delta_iter)
        return min(decayed, self.MAX_REGRET)

    def update_regret(self, action: str, reward: float, best_reward: float):
        self._iteration_counter += 1

        regret_value = best_reward - reward
        if regret_value <= 0:
            return

        current = self._get_effective_regret(action)
        updated = min(current + regret_value, self.MAX_REGRET)

        self.regret[action] = (updated, self._iteration_counter)

    # =========================================================
    # RECORDING WITH NORMALIZATION
    # =========================================================

    def record_action(self, action: str, reward: float):

        if action not in self.action_rewards:
            self.action_rewards[action] = deque(maxlen=self.REWARD_WINDOW)

        history = self.action_rewards[action]

        if history:
            mean = sum(history) / len(history)
            variance = sum((r - mean) ** 2 for r in history) / len(history)
            std = math.sqrt(max(variance, self.NORMALIZE_EPS))
            normalized = (reward - mean) / std
            normalized = max(-self.REWARD_CLAMP, min(self.REWARD_CLAMP, normalized))
        else:
            normalized = reward

        history.append(normalized)
        self.action_counts[action] = self.action_counts.get(action, 0) + 1

    # =========================================================
    # ENVIRONMENT MODEL
    # =========================================================

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

        self.environment_stability = max(0.0, min(1.0, self.environment_stability))

    # =========================================================
    # STABLE COMMITMENT HASH
    # =========================================================

    def _stable_value_bytes(self, value: Any) -> bytes:
        if isinstance(value, float):
            return struct.pack("!d", value)
        if isinstance(value, (int, bool)):
            return str(value).encode()
        if isinstance(value, str):
            return value.encode()
        if isinstance(value, dict):
            parts = []
            for k in sorted(value):
                parts.append(k.encode())
                parts.append(self._stable_value_bytes(value[k]))
            return b"".join(parts)
        if isinstance(value, list):
            parts = []
            for item in value:
                parts.append(self._stable_value_bytes(item))
            return b"".join(parts)
        return str(value).encode()

    def commit(self, action: str, observation: Dict[str, Any]) -> None:

        obs_bytes = self._stable_value_bytes(observation)

        prob_bytes = b"".join(
            k.encode() + struct.pack("!d", v)
            for k, v in sorted(self.state_probabilities.items())
        )

        payload = (
            self.commitment_hash.encode()
            + action.encode()
            + obs_bytes
            + prob_bytes
        )

        self.commitment_hash = hashlib.sha256(payload).hexdigest()

    # =========================================================
    # CONVERGENCE
    # =========================================================

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

        if plan_steps_total <= 0:
            return False

        if steps_completed < plan_steps_total:
            return False

        return True

    # =========================================================
    # SUMMARY
    # =========================================================

    def summary(self) -> Dict[str, Any]:
        return {
            "entropy": self.entropy(),
            "state_distribution": self.state_probabilities,
            "action_counts": self.action_counts,
            "regret": {
                k: self._get_effective_regret(k)
                for k in self.regret
            },
            "progress_score": self.progress_score,
            "environment_stability": self.environment_stability,
            "commitment_hash": self.commitment_hash,
        }
