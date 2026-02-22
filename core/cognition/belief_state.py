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

    # MATH-02 FIX: Raise MIN_ENTROPY_FLOOR from 0.1 to 0.3.
    # With __prior_fallback__ at PRIOR_ALPHA=0.01 and a dominant state at
    # p≈0.99, entropy ≈ 0.08 nats — below 0.1. But the fallback injection
    # lifts entropy to ≈0.14 nats which is above 0.1, making the floor
    # inoperative in all realistic belief collapse scenarios. 0.3 nats
    # corresponds roughly to a 2-state distribution at p=[0.93, 0.07] —
    # the floor now actively blends any near-collapse belief, providing the
    # exploration protection it was designed to give.
    MIN_ENTROPY_FLOOR = 0.3

    NORMALIZE_EPS = 1e-8
    REWARD_CLAMP = 3.0

    # MATH-08 FIX: Use only the most recent THOMPSON_WINDOW samples for Beta
    # parameter estimation. With REWARD_WINDOW=100, Alpha+Beta≈102 and the
    # Beta distribution variance≈0.0024 — effectively degenerate (near-mean
    # for all mature actions). THOMPSON_WINDOW=20 keeps variance≈0.012,
    # preserving meaningful exploration spread even for well-explored actions.
    THOMPSON_WINDOW = 20

    _FALLBACK_PRUNE_THRESHOLD = PRIOR_ALPHA * 2.0

    def __init__(self):
        self.created_at = time.time()
        self.state_probabilities: Dict[str, float] = {"neutral": 1.0}
        self.action_counts: Dict[str, int] = {}
        self.action_rewards: Dict[str, deque] = {}
        # MATH-04 FIX: Track raw (pre-normalisation) rewards in a parallel
        # structure. The normalised action_rewards deque feeds UCB/Thompson/
        # expected_utility. The raw deque feeds regret and global_best_reward()
        # so regret is measured in interpretable units (confidence delta ∈ [-0.5, 0.5])
        # rather than session-relative z-scores.
        self._raw_action_rewards: Dict[str, deque] = {}
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

        if len(pruned) == 1:
            (sole_state,) = pruned.keys()
            pruned["__prior_fallback__"] = self.PRIOR_ALPHA
            total = sum(pruned.values())

        self.state_probabilities = {s: v / total for s, v in pruned.items()}

        # MATH-02 FIX: Entropy floor at 0.3 nats now reliably catches
        # near-collapse distributions. Additionally, explicitly check whether
        # any single state dominates at ≥0.9 probability and force blending
        # even when entropy is marginally above the floor due to fallback
        # injection. This makes the floor robust to the prior_fallback sentinel.
        dominant_p = max(self.state_probabilities.values())
        if self.entropy() < self.MIN_ENTROPY_FLOOR or dominant_p >= 0.90:
            uniform = 1.0 / len(self.state_probabilities)
            blended = {
                k: 0.95 * v + 0.05 * uniform
                for k, v in self.state_probabilities.items()
            }
            total = sum(blended.values())
            self.state_probabilities = {k: v / total for k, v in blended.items()}

        fallback_prob = self.state_probabilities.get("__prior_fallback__", 0.0)
        if 0.0 < fallback_prob <= self._FALLBACK_PRUNE_THRESHOLD:
            pruned_dist = {
                k: v for k, v in self.state_probabilities.items()
                if k != "__prior_fallback__"
            }
            if pruned_dist:
                total = sum(pruned_dist.values())
                if total > 0:
                    self.state_probabilities = {k: v / total for k, v in pruned_dist.items()}

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
        total_actions = max(sum(self.action_counts.values()) + 1, 2)
        count = self.action_counts.get(action, 0) + 1
        rewards = self.action_rewards.get(action)

        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0

        exploration = self.EXPLORATION_C * math.sqrt(
            math.log(total_actions) / count
        )

        return mean_reward + exploration

    # =========================================================
    # THOMPSON SAMPLING — MATH-08 FIX
    # =========================================================

    def thompson_sample(self, action: str) -> float:
        rewards = self.action_rewards.get(action)
        if not rewards:
            return 0.5

        # MATH-08 FIX: Use only the most recent THOMPSON_WINDOW samples for
        # Beta parameter estimation. Using all REWARD_WINDOW=100 samples drives
        # Alpha+Beta≈102 (variance≈0.0024), effectively degenerating Thompson
        # sampling to the mean for any mature action. With THOMPSON_WINDOW=20,
        # variance≈0.012 — enough spread to maintain exploration diversity.
        recent = list(rewards)[-self.THOMPSON_WINDOW:]

        scaled = [(r + self.REWARD_CLAMP) / (2 * self.REWARD_CLAMP) for r in recent]
        scaled = [min(1.0, max(0.0, v)) for v in scaled]

        alpha = 1.0 + sum(scaled)
        beta = 1.0 + sum(1.0 - v for v in scaled)

        seed_material = (
            f"{self.commitment_hash}:{action}:{self._iteration_counter}"
        ).encode("utf-8")

        digest = hashlib.sha256(seed_material).digest()
        seed = int.from_bytes(digest[:8], byteorder="big", signed=False)

        rng = np.random.default_rng(seed)

        sample = rng.beta(alpha, beta)

        return float(sample)

    # =========================================================
    # REGRET TRACKING — MATH-07 FIX
    # =========================================================

    def _get_effective_regret(self, action: str) -> float:
        entry = self.regret.get(action)
        if not entry:
            return 0.0

        raw_value, last_iter = entry
        delta_iter = self._iteration_counter - last_iter
        decayed = raw_value * (self.REGRET_DECAY ** delta_iter)
        decayed = min(decayed, self.MAX_REGRET)

        # MATH-07 FIX: Persist the decayed value back to storage so that
        # summary() and any future calls see the current decayed value, not
        # the stale peak. Without this, the stored raw_value accumulates at
        # the historical peak forever — queries use the correct decayed value
        # but storage remains misleading for debugging and any code that reads
        # self.regret directly.
        if delta_iter > 0:
            self.regret[action] = (decayed, self._iteration_counter)

        return decayed

    def update_regret(self, action: str, reward: float, best_reward: float):
        """
        Update regret for this action.

        Parameters
        ----------
        action : str
        reward : float
            MATH-04 FIX: Must be a RAW reward (confidence delta, not z-score).
            Callers (operate.py) should pass `raw_reward` directly, not the
            normalised value from `action_rewards`. This makes regret meaningful
            across sessions and comparable between actions with different
            sample-count histories.
        best_reward : float
            The best RAW reward observed across all actions this session.
            Use `global_best_reward()` which now reads from `_raw_action_rewards`.
        """
        self._iteration_counter += 1

        regret_value = best_reward - reward
        if regret_value <= 0:
            return

        current = self._get_effective_regret(action)
        updated = min(current + regret_value, self.MAX_REGRET)

        self.regret[action] = (updated, self._iteration_counter)

    # =========================================================
    # RECORDING WITH NORMALISATION — MATH-03 FIX
    # =========================================================

    def record_action(self, action: str, reward: float):

        if action not in self.action_rewards:
            self.action_rewards[action] = deque(maxlen=self.REWARD_WINDOW)
        if action not in self._raw_action_rewards:
            # MATH-04 FIX: parallel raw reward window
            self._raw_action_rewards[action] = deque(maxlen=self.REWARD_WINDOW)

        history = self.action_rewards[action]

        # MATH-04 FIX: store raw reward before normalisation
        self._raw_action_rewards[action].append(reward)

        # MATH-03 FIX: Compute z-score statistics AFTER appending the current
        # raw reward to the history so that the current observation is included
        # in the mean/variance. The previous code computed stats from the
        # existing window (N samples), then appended the normalised value —
        # introducing a one-step lag where the current reward was normalised by
        # statistics that did not include it. We now append the raw value,
        # compute stats over N+1 (including the new point), then replace the
        # last entry with the normalised value.
        if len(history) >= 3:
            # Temporarily append raw reward to compute inclusive statistics
            history.append(reward)
            mean = sum(history) / len(history)
            variance = sum((r - mean) ** 2 for r in history) / len(history)
            std = math.sqrt(max(variance, self.NORMALIZE_EPS))
            normalized = (reward - mean) / std
            normalized = max(-self.REWARD_CLAMP, min(self.REWARD_CLAMP, normalized))
            # Replace the just-appended raw value with the normalised value
            history[-1] = normalized
        elif history:
            # First 2 samples: identity scaling (avoid extreme z-scores from
            # near-zero variance). Include current raw value in the deque as-is.
            normalized = max(-self.REWARD_CLAMP, min(self.REWARD_CLAMP, reward))
            history.append(normalized)
        else:
            history.append(reward)

        self.action_counts[action] = self.action_counts.get(action, 0) + 1

    # =========================================================
    # GLOBAL BEST REWARD — MATH-04 FIX
    # =========================================================

    def global_best_reward(self) -> float:
        """
        MATH-04 FIX: Return the best RAW reward seen across all actions this
        session. Uses _raw_action_rewards (pre-normalisation) so regret is
        computed in interpretable units (confidence delta ∈ [-0.5, 0.5])
        rather than session-relative z-scores.

        Returns 0.0 if no actions have been recorded yet.
        """
        best = 0.0
        for history in self._raw_action_rewards.values():
            if history:
                best = max(best, max(history))
        return best

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
            "global_best_reward": self.global_best_reward(),
        }
