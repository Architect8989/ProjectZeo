from __future__ import annotations

from typing import Dict, Any, Tuple
from collections import deque
import time
import math
import hashlib
import numpy as np


class BeliefState:

    EXPLORATION_C = 1.4
    RISK_LAMBDA = 0.3
    REWARD_WINDOW = 100
    PRIOR_ALPHA = 0.01
    REGRET_DECAY = 0.995
    MAX_STATES = 64
    MAX_REGRET = 100.0

    MIN_ENTROPY_FLOOR = 0.3

    BOOTSTRAP_REWARD_SCALE: float = 3.0

    REWARD_CLAMP: float = 3.0
    NORMALIZE_EPS: float = 1e-8

    RAW_REWARD_MIN: float = -1.0
    RAW_REWARD_MAX: float = 1.0

    MAX_ENTROPY_CONVERGENCE: float = 2.0

    THOMPSON_WINDOW = 20

    _FALLBACK_PRUNE_THRESHOLD = PRIOR_ALPHA * 2.0

    def __init__(self, intent_hash: str = "") -> None:
        self.created_at: float = time.time()

        self.state_probabilities: Dict[str, float] = {"neutral": 1.0}

        self.action_counts: Dict[str, int] = {}
        self.action_rewards: Dict[str, deque] = {}
        self._raw_action_rewards: Dict[str, deque] = {}

        self.regret: Dict[str, Tuple[float, int]] = {}

        self.progress_score: float = 0.0
        self.environment_stability: float = 1.0
        self.consecutive_high_stability_count: int = 0

        self.task_identity_hash: str = (
            hashlib.sha256(intent_hash.encode("utf-8")).hexdigest()
            if intent_hash
            else "GENESIS"
        )
        self.commitment_chain_hash: str = self.task_identity_hash

        self._iteration_counter: int = 0
        self._sample_counter: int = 0
        self._regret_decay: float = self.REGRET_DECAY

        self._welford_n: Dict[str, int] = {}
        self._welford_mean: Dict[str, float] = {}
        self._welford_M2: Dict[str, float] = {}

    @property
    def commitment_hash(self) -> str:
        return self.task_identity_hash

    @commitment_hash.setter
    def commitment_hash(self, value: str) -> None:
        self.task_identity_hash = value
        if self.commitment_chain_hash in ("GENESIS", ""):
            self.commitment_chain_hash = value

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

        if total <= 0.0:
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
        if total <= 0.0:
            return

        if len(pruned) == 1:
            (sole_state,) = pruned.keys()
            pruned["__prior_fallback__"] = self.PRIOR_ALPHA
            total = sum(pruned.values())

        self.state_probabilities = {s: v / total for s, v in pruned.items()}

        _current_entropy = self.entropy()
        if _current_entropy < self.MIN_ENTROPY_FLOOR:
            _MAX_BLEND_WEIGHT = 0.30
            for _ in range(20):
                if _current_entropy >= self.MIN_ENTROPY_FLOOR:
                    break
                _deficit = self.MIN_ENTROPY_FLOOR - _current_entropy
                _w = min(_deficit / self.MIN_ENTROPY_FLOOR, _MAX_BLEND_WEIGHT)
                _n = len(self.state_probabilities)
                if _n == 0:
                    break
                blended = {
                    k: (1.0 - _w) * v + _w / _n
                    for k, v in self.state_probabilities.items()
                }
                _total = sum(blended.values())
                if _total > 0.0:
                    self.state_probabilities = {k: v / _total for k, v in blended.items()}
                _current_entropy = self.entropy()

        fallback_prob = self.state_probabilities.get("__prior_fallback__", 0.0)
        if 0.0 < fallback_prob <= self._FALLBACK_PRUNE_THRESHOLD:
            if self.entropy() >= self.MIN_ENTROPY_FLOOR:
                pruned_dist = {
                    k: v for k, v in self.state_probabilities.items()
                    if k != "__prior_fallback__"
                }
                if pruned_dist:
                    total = sum(pruned_dist.values())
                    if total > 0.0:
                        self.state_probabilities = {
                            k: v / total for k, v in pruned_dist.items()
                        }

    def entropy(self) -> float:
        return -sum(
            p * math.log(p)
            for p in self.state_probabilities.values()
            if p > 0.0
        )

    def expected_utility(self, action: str) -> float:
        rewards = self.action_rewards.get(action)
        if not rewards:
            return 0.0

        n = len(rewards)
        mean = sum(rewards) / n
        variance = sum((r - mean) ** 2 for r in rewards) / n
        raw_penalty = self.RISK_LAMBDA * variance
        bounded_penalty = min(raw_penalty, abs(mean))
        return mean - bounded_penalty

    def ucb_score(self, action: str) -> float:
        count = self.action_counts.get(action, 0)
        if count == 0:
            return float("inf")

        total_actions = max(sum(self.action_counts.values()) + 1, 2)
        rewards = self.action_rewards.get(action)
        mean_reward_raw = sum(rewards) / len(rewards) if rewards else 0.0

        mean_reward_01 = (mean_reward_raw + self.REWARD_CLAMP) / (2.0 * self.REWARD_CLAMP)
        mean_reward_01 = max(0.0, min(1.0, mean_reward_01))

        exploration = self.EXPLORATION_C * math.sqrt(
            math.log(total_actions) / count
        )
        return mean_reward_01 + exploration

    def thompson_sample(self, action: str) -> float:
        rewards = self.action_rewards.get(action)
        if not rewards:
            return 0.0

        recent = list(rewards)[-self.THOMPSON_WINDOW:]
        n = len(recent)

        _mu0: float = 0.0
        _sigma0_sq: float = self.REWARD_CLAMP ** 2

        _welford_n = self._welford_n.get(action, 0)
        if _welford_n >= 3:
            _obs_variance = max(
                self._welford_M2.get(action, 0.0) / max(_welford_n - 1, 1),
                self.NORMALIZE_EPS,
            )
        else:
            _obs_variance = _sigma0_sq

        _prior_prec = 1.0 / _sigma0_sq
        _obs_prec = n / _obs_variance
        _post_prec = _prior_prec + _obs_prec
        _post_variance = 1.0 / _post_prec

        _sum_rewards = sum(recent)
        _post_mean = _post_variance * (_prior_prec * _mu0 + _sum_rewards / _obs_variance)

        _post_mean = max(-self.REWARD_CLAMP, min(self.REWARD_CLAMP, _post_mean))

        self._sample_counter += 1
        seed_material = (
            f"{self.commitment_chain_hash}:{action}:"
            f"{self._iteration_counter}:{self._sample_counter}"
        ).encode("utf-8")
        digest = hashlib.sha256(seed_material).digest()
        seed = int.from_bytes(digest[:8], byteorder="big", signed=False)

        _rng = np.random.default_rng(seed)

        _lo = -self.REWARD_CLAMP
        _hi = self.REWARD_CLAMP
        _std = math.sqrt(_post_variance)

        sample: float = _post_mean

        for _ in range(128):
            _candidate = float(_rng.normal(loc=_post_mean, scale=_std))
            if _lo <= _candidate <= _hi:
                sample = _candidate
                break

        return float(sample)

    def _get_effective_regret(self, action: str) -> float:
        entry = self.regret.get(action)
        if not entry:
            return 0.0

        raw_value, last_iter = entry
        delta_iter = self._iteration_counter - last_iter
        decayed = raw_value * (self._regret_decay ** delta_iter)
        decayed = min(decayed, self.MAX_REGRET)

        if delta_iter > 0:
            self.regret[action] = (decayed, self._iteration_counter)

        return decayed

    def set_plan_horizon(self, total_steps: int, iters_per_step: int = 13) -> None:
        total_iters = max(total_steps * iters_per_step, 1)
        target_fraction = 0.05
        self._regret_decay = target_fraction ** (1.0 / total_iters)

    def update_regret(self, action: str, reward: float, best_reward: float) -> None:
        self._iteration_counter += 1

        regret_value = best_reward - reward
        if regret_value <= 0.0:
            return

        current = self._get_effective_regret(action)
        updated = min(current + regret_value, self.MAX_REGRET)
        self.regret[action] = (updated, self._iteration_counter)

    def record_action(self, action: str, reward: float) -> None:
        _chain_input = f"{self.commitment_chain_hash}:{action}".encode("utf-8")
        self.commitment_chain_hash = hashlib.sha256(_chain_input).hexdigest()

        if action not in self.action_rewards:
            self.action_rewards[action] = deque(maxlen=self.REWARD_WINDOW)
        if action not in self._raw_action_rewards:
            self._raw_action_rewards[action] = deque(maxlen=self.REWARD_WINDOW)

        reward = max(self.RAW_REWARD_MIN, min(self.RAW_REWARD_MAX, float(reward)))
        self._raw_action_rewards[action].append(reward)

        n = self._welford_n.get(action, 0) + 1
        self._welford_n[action] = n

        prev_mean = self._welford_mean.get(action, 0.0)
        delta = reward - prev_mean
        new_mean = prev_mean + delta / n
        self._welford_mean[action] = new_mean

        prev_M2 = self._welford_M2.get(action, 0.0)
        delta2 = reward - new_mean
        new_M2 = prev_M2 + delta * delta2
        self._welford_M2[action] = new_M2

        if n >= 3:
            variance = new_M2 / max(n - 1, 1)
            std = math.sqrt(max(variance, self.NORMALIZE_EPS))
            normalised = (reward - new_mean) / std
            normalised = max(-self.REWARD_CLAMP, min(self.REWARD_CLAMP, normalised))

            if n == 3 and action in self._raw_action_rewards:
                raw_deque = self._raw_action_rewards[action]
                if len(raw_deque) >= 2:
                    old_entries = list(raw_deque)[:2]
                    _two_sample_mean = (old_entries[0] + old_entries[1]) / 2.0

                    renormalized = []
                    for r in old_entries:
                        z = (r - _two_sample_mean) / std
                        renormalized.append(
                            max(-self.REWARD_CLAMP, min(self.REWARD_CLAMP, z))
                        )
                    existing = self.action_rewards.get(action)
                    if existing is not None and len(existing) == 2:
                        existing.clear()
                        existing.extend(renormalized)
        else:
            normalised = (reward / self.BOOTSTRAP_REWARD_SCALE) * self.REWARD_CLAMP
            normalised = max(-self.REWARD_CLAMP, min(self.REWARD_CLAMP, normalised))

        self.action_rewards[action].append(normalised)
        self.action_counts[action] = self.action_counts.get(action, 0) + 1

    def flush_regret_on_success(self) -> None:
        self.regret.clear()

    def reset_sample_counter(self) -> None:
        self._sample_counter = 0

    def reset_for_new_task(self, intent_hash: str = "") -> None:
        import hashlib as _hashlib

        self._iteration_counter = 0
        self._sample_counter = 0
        self._regret_decay = self.REGRET_DECAY
        self.regret.clear()
        self.progress_score = 0.0

        new_identity = (
            _hashlib.sha256(intent_hash.encode("utf-8")).hexdigest()
            if intent_hash
            else "GENESIS"
        )
        self.task_identity_hash = new_identity
        self.commitment_chain_hash = new_identity

    def global_best_reward(self) -> "float | None":
        best: "float | None" = None
        for history in self._raw_action_rewards.values():
            if history:
                local_max = max(history)
                if best is None or local_max > best:
                    best = local_max
        return best

    def compute_environment_stability(self, delta: Dict[str, Any]) -> None:
        if not delta:
            return

        significant = delta.get("significant_change", False)
        if significant:
            self.environment_stability = max(0.0, self.environment_stability - 0.2)
            self.consecutive_high_stability_count = 0
        else:
            self.environment_stability = min(1.0, self.environment_stability + 0.05)
            if self.environment_stability >= 0.8:
                self.consecutive_high_stability_count += 1
            else:
                self.consecutive_high_stability_count = 0

    def summary(self) -> dict:
        top_states = dict(
            sorted(self.state_probabilities.items(), key=lambda x: x[1], reverse=True)[:5]
        )
        return {
            "entropy": round(self.entropy(), 4),
            "state_probabilities": {k: round(v, 4) for k, v in top_states.items()},
            "progress_score": round(self.progress_score, 4),
            "environment_stability": round(self.environment_stability, 4),
            "consecutive_high_stability_count": self.consecutive_high_stability_count,
            "iteration": self._iteration_counter,
            "total_actions_tried": len(self.action_counts),
        }

    def to_dict(self) -> dict:
        return {
            "state_probabilities": dict(self.state_probabilities),
            "progress_score": self.progress_score,
            "environment_stability": self.environment_stability,
            "consecutive_high_stability_count": self.consecutive_high_stability_count,
            "commitment_hash": self.commitment_hash,
            "commitment_chain_hash": self.commitment_chain_hash,
            "_iteration_counter": self._iteration_counter,
            "_sample_counter": self._sample_counter,
            "_regret_decay": self._regret_decay,
            "action_counts": dict(self.action_counts),
            "action_rewards": {k: list(v) for k, v in self.action_rewards.items()},
            "_raw_action_rewards": {
                k: list(v) for k, v in self._raw_action_rewards.items()
            },
            "regret": {k: list(v) for k, v in self.regret.items()},
            "_welford_n": dict(self._welford_n),
            "_welford_mean": dict(self._welford_mean),
            "_welford_M2": dict(self._welford_M2),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict, *, intent_hash: str = "") -> "BeliefState":
        if not isinstance(data, dict):
            return cls(intent_hash=intent_hash)

        try:
            instance = cls.__new__(cls)

            instance.created_at = float(data.get("created_at", 0.0) or 0.0)

            raw_probs = data.get("state_probabilities", {"neutral": 1.0})
            instance.state_probabilities = (
                dict(raw_probs) if isinstance(raw_probs, dict) else {"neutral": 1.0}
            )
            instance.progress_score = float(data.get("progress_score", 0.0))
            instance.environment_stability = float(
                data.get("environment_stability", 1.0)
            )
            instance.consecutive_high_stability_count = int(
                data.get("consecutive_high_stability_count", 0)
            )

            instance.commitment_hash = str(data.get("commitment_hash", "GENESIS"))
            instance.commitment_chain_hash = str(
                data.get("commitment_chain_hash", instance.task_identity_hash)
            )
            instance._iteration_counter = int(data.get("_iteration_counter", 0))
            instance._sample_counter = int(data.get("_sample_counter", 0))
            instance._regret_decay = float(
                data.get("_regret_decay", cls.REGRET_DECAY)
            )

            instance.action_counts = dict(data.get("action_counts", {}))

            instance.action_rewards = {}
            for k, v in data.get("action_rewards", {}).items():
                instance.action_rewards[k] = deque(
                    (float(x) for x in v), maxlen=cls.REWARD_WINDOW
                )

            instance._raw_action_rewards = {}
            for k, v in data.get("_raw_action_rewards", {}).items():
                instance._raw_action_rewards[k] = deque(
                    (float(x) for x in v), maxlen=cls.REWARD_WINDOW
                )

            instance.regret = {}
            for k, v in data.get("regret", {}).items():
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    instance.regret[k] = (float(v[0]), int(v[1]))

            instance._welford_n = {
                k: int(v) for k, v in data.get("_welford_n", {}).items()
            }
            instance._welford_mean = {
                k: float(v) for k, v in data.get("_welford_mean", {}).items()
            }
            instance._welford_M2 = {
                k: float(v) for k, v in data.get("_welford_M2", {}).items()
            }

            return instance

        except Exception:
            return cls(intent_hash=intent_hash)
