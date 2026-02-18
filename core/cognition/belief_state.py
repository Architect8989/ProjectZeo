from typing import Dict, Any, List, Tuple
from collections import deque
import time
import math
import hashlib
import json
import struct


class BeliefState:

    EXPLORATION_C = 1.4
    RISK_LAMBDA = 0.3
    REWARD_WINDOW = 100
    REGRET_SCALE = 0.05
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
            likelihood = likelihoods.get(state, self.PRIOR_ALPHA)
            posterior = prior * max(likelihood, self.PRIOR_ALPHA)
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
            sorted_states = sorted(
                pruned.items(),
                key=lambda x: x[1],
                reverse=True,
            )[: self.MAX_STATES]
            pruned = dict(sorted_states)

        total = sum(pruned.values())
        if total > 0:
            self.state_probabilities = {
                s: v / total for s, v in pruned.items()
            }

        # entropy floor injection
        if self.entropy() < self.MIN_ENTROPY_FLOOR:
            uniform = 1.0 / len(self.state_probabilities)
            for k in self.state_probabilities:
                self.state_probabilities[k] = (
                    0.95 * self.state_probabilities[k]
                    + 0.05 * uniform
                )

    def entropy(self) -> float:
        total = 0.0
        for p in self.state_probabilities.values():
            if p > 0:
                total -= p * math.log(p)
        return total

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
    # DETERMINISTIC THOMPSON
    # =========================================================

    def thompson_sample(self, action: str) -> float:
        rewards = self.action_rewards.get(action)
        if not rewards:
            return 0.5

        successes = sum(1 for r in rewards if r > 0)
        failures = sum(1 for r in rewards if r < 0)

        # deterministic pseudo-random from commitment hash
        seed_material = (
            self.commitment_hash + action
        ).encode()

        digest = hashlib.sha256(seed_material).digest()
        deterministic_uniform = struct.unpack("!Q", digest[:8])[0] / 2**64

        alpha = successes + 1
        beta = failures + 1

        # Beta mean blended with deterministic uniform
        beta_mean = alpha / (alpha + beta)

        return 0.7 * beta_mean + 0.3 * deterministic_uniform

    # =========================================================
    # REGRET TRACKING
    # =========================================================

    def _get_effective_regret(self, action: str) -> float:
        entry = self.regret.get(action)
        if not entry:
            return 0.0

        raw_value, last_iter = entry
        delta_iter = self._iteration_counter - last_iter
        value = raw_value * (self.REGRET_DECAY ** delta_iter)
        return min(value, self.MAX_REGRET)

    def update_regret(self, action: str, reward: float, best_reward: float):
        self._iteration_counter += 1

        regret_value = best_reward - reward

        current = self._get_effective_regret(action)
        updated = current + regret_value
        updated = max(0.0, min(updated, self.MAX_REGRET))

        self.regret[action] = (updated, self._iteration_counter)

    # =========================================================
    # RECORDING WITH NORMALIZATION
    # =========================================================

    def record_action(self, action: str, reward: float):

        if action not in self.action_rewards:
            self.action_rewards[action] = deque(maxlen=self.REWARD_WINDOW)

        history = self.action_rewards[action]

        # running mean/std normalization
        if history:
            mean = sum(history) / len(history)
            variance = sum((r - mean) ** 2 for r in history) / len(history)
            std = math.sqrt(max(variance, self.NORMALIZE_EPS))
            normalized = (reward - mean) / std
            normalized = max(
                -self.REWARD_CLAMP,
                min(self.REWARD_CLAMP, normalized),
            )
        else:
            normalized = reward

        history.append(normalized)

        self.action_counts[action] = (
            self.action_counts.get(action, 0) + 1
        )

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

        self.environment_stability = max(
            0.0,
            min(1.0, self.environment_stability),
        )

    # =========================================================
    # STABLE COMMITMENT HASH
    # =========================================================

    def _stable_float_bytes(self, d: Dict[str, float]) -> bytes:
        parts = []
        for k in sorted(d):
            parts.append(k.encode())
            parts.append(struct.pack("!d", float(d[k])))
        return b"".join(parts)

    def _stable_observation_bytes(self, obs: Dict[str, Any]) -> bytes:
        parts = []
        for k in sorted(obs):
            v = obs[k]
            parts.append(k.encode())
            if isinstance(v, float):
                parts.append(struct.pack("!d", v))
            else:
                parts.append(str(v).encode())
        return b"".join(parts)

    def commit(self, action: str, observation: Dict[str, Any]) -> None:

        obs_bytes = self._stable_observation_bytes(observation)
        prob_bytes = self._stable_float_bytes(
            self.state_probabilities
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
