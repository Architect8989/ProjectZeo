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

    
    MIN_ENTROPY_FLOOR = 0.3

    NORMALIZE_EPS = 1e-8
    REWARD_CLAMP = 3.0

    
    THOMPSON_WINDOW = 20

    _FALLBACK_PRUNE_THRESHOLD = PRIOR_ALPHA * 2.0

    def __init__(self, intent_hash: str = ""):
        self.created_at = time.time()
        self.state_probabilities: Dict[str, float] = {"neutral": 1.0}
        self.action_counts: Dict[str, int] = {}
        self.action_rewards: Dict[str, deque] = {}
        
        
        self._raw_action_rewards: Dict[str, deque] = {}
        self.regret: Dict[str, Tuple[float, int]] = {}
        self.progress_score: float = 0.0
        self.environment_stability: float = 1.0

        
        self.commitment_hash: str = (
            hashlib.sha256(intent_hash.encode("utf-8")).hexdigest()
            if intent_hash
            else "GENESIS"
        )
        self._iteration_counter: int = 0

        
        self.consecutive_high_stability_count: int = 0

        
        self._sample_counter: int = 0

        
        self._regret_decay: float = self.REGRET_DECAY

        
        
        self._welford_n: Dict[str, int] = {}          # count of rewards seen
        self._welford_mean: Dict[str, float] = {}     # running mean
        self._welford_M2: Dict[str, float] = {}       # running sum of squared deviations

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

     
        if self.entropy() < self.MIN_ENTROPY_FLOOR:
            
            _MAX_BLEND_WEIGHT = 0.30
            for _ in range(20):  # safety cap — should converge in < 5
                _H = self.entropy()
                if _H >= self.MIN_ENTROPY_FLOOR:
                    break
                _deficit = self.MIN_ENTROPY_FLOOR - _H
                _w = min(_deficit / self.MIN_ENTROPY_FLOOR, _MAX_BLEND_WEIGHT)
                _n = len(self.state_probabilities)
                blended = {
                    k: (1.0 - _w) * v + _w / _n
                    for k, v in self.state_probabilities.items()
                }
                _total = sum(blended.values())
                if _total > 0:
                    self.state_probabilities = {k: v / _total for k, v in blended.items()}

        fallback_prob = self.state_probabilities.get("__prior_fallback__", 0.0)
        if 0.0 < fallback_prob <= self._FALLBACK_PRUNE_THRESHOLD:
            
            if self.entropy() >= self.MIN_ENTROPY_FLOOR:
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
        """
        Risk-adjusted expected utility with bounded penalty.

        FIX B-MATH-3: Without a penalty cap, RISK_LAMBDA * variance can reach
        0.3 * 9.0 = 2.7 (when rewards alternate between ±3.0), causing a
        high-mean-high-variance action (mean=+2.0) to score EU = 2.0 - 2.7 = -0.7
        — worse than an untried action (EU = 0.0).  The risk penalty should
        reduce reward, not invert the sign of a clearly positive action.

        Fix: cap the penalty at |mean| so the worst-case EU for a positive-mean
        action is 0.0 (neutral), never negative.
        """
        rewards = self.action_rewards.get(action)
        if not rewards:
            return 0.0

        n = len(rewards)
        mean = sum(rewards) / n
        variance = sum((r - mean) ** 2 for r in rewards) / n
        raw_penalty = self.RISK_LAMBDA * variance
        # Penalty capped at |mean| to prevent sign inversion.
        bounded_penalty = min(raw_penalty, abs(mean))
        return mean - bounded_penalty

    def ucb_score(self, action: str) -> float:
        """
        UCB1 score with scale-corrected exploitation term.

        FIX B-MATH-1: action_rewards stores Welford z-scores on [-REWARD_CLAMP,
        REWARD_CLAMP] = [-3, 3], while the UCB1 exploration bonus is unscaled
        (bounded by EXPLORATION_C * sqrt(log T)).  Mixing these two incompatible
        scales invalidates UCB1's O(sqrt(K T log T)) regret bound and can cause
        visited failing actions (mean ≈ -3) to dominate unvisited actions.

        Fix: map mean_reward from [-REWARD_CLAMP, REWARD_CLAMP] → [0, 1] before
        adding the exploration bonus.  This restores the [0,1] assumption that
        UCB1 requires for both exploitation and exploration terms.
        """
        count = self.action_counts.get(action, 0)
        if count == 0:
            return float('inf')  # Must explore: standard UCB1 guarantee

        total_actions = max(sum(self.action_counts.values()) + 1, 2)
        rewards = self.action_rewards.get(action)

        mean_reward_raw = sum(rewards) / len(rewards) if rewards else 0.0

        # Normalize to [0, 1] so both terms share the same scale.
        mean_reward_01 = (mean_reward_raw + self.REWARD_CLAMP) / (2.0 * self.REWARD_CLAMP)
        mean_reward_01 = max(0.0, min(1.0, mean_reward_01))

        exploration = self.EXPLORATION_C * math.sqrt(
            math.log(total_actions) / count
        )

        return mean_reward_01 + exploration

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

        # HAR-06 (MS-03) + MR-04 FIX: Seed incorporates both _iteration_counter
        # (advances per regret update — unique across rounds) and _sample_counter
        # (advances per thompson_sample() call — unique within a round).
        # Previously all samples in the same selection round shared the same
        # _iteration_counter, making seeds identical when action strings are
        # similar. _sample_counter eliminates this within-round collision.
        self._sample_counter += 1
        seed_material = (
            f"{self.commitment_hash}:{action}:{self._iteration_counter}:{self._sample_counter}"
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
        # MR-05 FIX: Use _regret_decay (instance, tuned to plan horizon)
        # instead of the class-level REGRET_DECAY constant.
        decayed = raw_value * (self._regret_decay ** delta_iter)
        decayed = min(decayed, self.MAX_REGRET)

        if delta_iter > 0:
            self.regret[action] = (decayed, self._iteration_counter)

        return decayed

    def set_plan_horizon(self, total_steps: int, iters_per_step: int = 13) -> None:
        
        total_iters = max(total_steps * iters_per_step, 1)
        target_fraction = 0.05  # regret decays to 5% by plan end
        self._regret_decay = target_fraction ** (1.0 / total_iters)


    def update_regret(self, action: str, reward: float, best_reward: float):
       
        self._iteration_counter += 1

        regret_value = best_reward - reward
        if regret_value <= 0:
            return

        current = self._get_effective_regret(action)
        updated = min(current + regret_value, self.MAX_REGRET)

        self.regret[action] = (updated, self._iteration_counter)

    # =========================================================
    # RECORDING WITH NORMALISATION — MATH-03 / FIX-06 / FIX H6/M-2
    # =========================================================

    def record_action(self, action: str, reward: float):
        
        if action not in self.action_rewards:
            self.action_rewards[action] = deque(maxlen=self.REWARD_WINDOW)
        if action not in self._raw_action_rewards:
            self._raw_action_rewards[action] = deque(maxlen=self.REWARD_WINDOW)

        # Always store the unmodified raw reward for regret / global best.
        self._raw_action_rewards[action].append(reward)

        # ---- Welford incremental update ----
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
            # Sufficient history for z-score normalisation.
            # Population variance = M2 / n; add NORMALIZE_EPS for stability.
            variance = new_M2 / n
            std = math.sqrt(max(variance, self.NORMALIZE_EPS))
            normalised = (reward - new_mean) / std
            normalised = max(-self.REWARD_CLAMP, min(self.REWARD_CLAMP, normalised))

            
            if n == 3 and action in self._raw_action_rewards:
                raw_deque = self._raw_action_rewards[action]
                if len(raw_deque) >= 2:
                    # Renormalize the first 2 raw rewards using the current
                    # Welford mean/std (which now include all 3 samples).
                    old_entries = list(raw_deque)[:2]
                    renormalized = []
                    for r in old_entries:
                        z = (r - new_mean) / std
                        renormalized.append(
                            max(-self.REWARD_CLAMP, min(self.REWARD_CLAMP, z))
                        )
                    # Replace first 2 entries in the normalised deque.
                    # action_rewards[action] has exactly 2 entries at this point.
                    existing = self.action_rewards.get(action)
                    if existing is not None and len(existing) == 2:
                        existing.clear()
                        existing.extend(renormalized)

        else:
            
            _raw_scale = 0.5  # max absolute raw reward (confidence - 0.5)
            if _raw_scale > 0:
                normalised = reward / _raw_scale * self.REWARD_CLAMP
            else:
                normalised = 0.0
            normalised = max(-self.REWARD_CLAMP, min(self.REWARD_CLAMP, normalised))

        self.action_rewards[action].append(normalised)
        self.action_counts[action] = self.action_counts.get(action, 0) + 1

    # =========================================================
    # GLOBAL BEST REWARD — MATH-04 FIX
    # =========================================================

    def global_best_reward(self):
        
        best = None
        for history in self._raw_action_rewards.values():
            if history:
                local_max = max(history)
                if best is None or local_max > best:
                    best = local_max
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
            # MR-01a FIX: Reset the consecutive high-stability counter on
            # any significant change so the 3-obs gate cannot be satisfied
            # across a stability boundary.
            self.consecutive_high_stability_count = 0
        else:
            self.environment_stability = min(
                1.0,
                self.environment_stability + 0.05,
            )
            # MR-01a FIX: Increment consecutive counter when stable.
            if self.environment_stability > 0.7:
                self.consecutive_high_stability_count += 1
            else:
                self.consecutive_high_stability_count = 0

        self.environment_stability = max(0.0, min(1.0, self.environment_stability))

    # =========================================================
    # STABLE COMMITMENT HASH
    # =========================================================

    def _stable_value_bytes(self, value: Any) -> bytes:
        
        if value is None:
            # \x00 tag: unambiguously None — cannot be produced by any str value.
            return b"\x00"
        if isinstance(value, float):
            
            return b"\x03" + struct.pack("!d", round(value, 6))
        if isinstance(value, (int, bool)):
            return b"\x02" + str(value).encode()
        if isinstance(value, str):
            return b"\x01" + value.encode()
        if isinstance(value, dict):
            parts = []
            for k in sorted(value):
                parts.append(k.encode())
                parts.append(self._stable_value_bytes(value[k]))
            return b"".join(parts)
        if isinstance(value, list):
            
            result = b""
            for item in value:
                item_bytes = self._stable_value_bytes(item)
                result += struct.pack("!I", len(item_bytes)) + item_bytes
            return result
        # Unknown type: tag \xff + type name + stringified value.
        # This ensures distinct Python types that happen to have the same str()
        # representation (e.g. a custom object whose __str__ returns "None")
        # still produce distinct byte sequences.
        type_name = type(value).__name__.encode()
        value_bytes = str(value).encode()
        return b"\xff" + struct.pack("!H", len(type_name)) + type_name + value_bytes

    def commit(self, action: str, observation: Dict[str, Any]) -> None:

        obs_bytes = self._stable_value_bytes(observation)

        # FIX SI-5: Quantize probability floats to 6 decimal places before
        # packing. Mirrors the _stable_value_bytes float fix — state_probabilities
        # are normalised floats susceptible to the same platform drift. Without
        # quantization, commit() produces platform-divergent hashes for identical
        # belief distributions.
        prob_bytes = b"".join(
            k.encode() + struct.pack("!d", round(v, 6))
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
    # CONVERGENCE — FIX M-5 / H4
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
    # SERIALIZATION — Fix 12 (Thompson counter persistence)
    # =========================================================

    def to_dict(self) -> dict:
        
        return {
            "commitment_hash": self.commitment_hash,
            "iteration_counter": self._iteration_counter,
            "sample_counter": self._sample_counter,
            "state_probabilities": dict(self.state_probabilities),
            "action_counts": dict(self.action_counts),
            "action_rewards": {
                k: list(v) for k, v in self.action_rewards.items()
            },
            "raw_action_rewards": {
                k: list(v) for k, v in self._raw_action_rewards.items()
            },
            "regret": {
                k: list(v) for k, v in self.regret.items()
            },
            "progress_score": self.progress_score,
            "environment_stability": self.environment_stability,
            "consecutive_high_stability_count": self.consecutive_high_stability_count,
            "regret_decay": self._regret_decay,
            "welford_n": dict(self._welford_n),
            "welford_mean": dict(self._welford_mean),
            "welford_M2": dict(self._welford_M2),
        }

    @classmethod
    def from_dict(cls, data: dict, intent_hash: str = "") -> "BeliefState":
        """
        Fix 12: Reconstruct a BeliefState from a previously serialised dict.

        The reconstructed instance has the same commitment_hash,
        _iteration_counter, and _sample_counter as when to_dict() was called,
        so Thompson sampling seeds continue the same sequence as before the
        restart — making post-restart replay reproducible.
        """
        obj = cls(intent_hash=intent_hash)

        # Restore commitment chain state (overrides the intent-derived genesis)
        if isinstance(data.get("commitment_hash"), str):
            obj.commitment_hash = data["commitment_hash"]
        if isinstance(data.get("iteration_counter"), int):
            obj._iteration_counter = data["iteration_counter"]
        if isinstance(data.get("sample_counter"), int):
            obj._sample_counter = data["sample_counter"]

        if isinstance(data.get("state_probabilities"), dict):
            obj.state_probabilities = dict(data["state_probabilities"])
        if isinstance(data.get("action_counts"), dict):
            obj.action_counts = dict(data["action_counts"])

        for key, vals in (data.get("action_rewards") or {}).items():
            if isinstance(vals, list):
                obj.action_rewards[key] = deque(vals, maxlen=cls.REWARD_WINDOW)
        for key, vals in (data.get("raw_action_rewards") or {}).items():
            if isinstance(vals, list):
                obj._raw_action_rewards[key] = deque(vals, maxlen=cls.REWARD_WINDOW)

        for key, val in (data.get("regret") or {}).items():
            if isinstance(val, list) and len(val) == 2:
                obj.regret[key] = tuple(val)

        if isinstance(data.get("progress_score"), (int, float)):
            obj.progress_score = float(data["progress_score"])
        if isinstance(data.get("environment_stability"), (int, float)):
            obj.environment_stability = float(data["environment_stability"])
        if isinstance(data.get("consecutive_high_stability_count"), int):
            obj.consecutive_high_stability_count = data["consecutive_high_stability_count"]
        if isinstance(data.get("regret_decay"), (int, float)):
            obj._regret_decay = float(data["regret_decay"])

        # Restore Welford running statistics
        if isinstance(data.get("welford_n"), dict):
            obj._welford_n = {k: int(v) for k, v in data["welford_n"].items()}
        if isinstance(data.get("welford_mean"), dict):
            obj._welford_mean = {k: float(v) for k, v in data["welford_mean"].items()}
        if isinstance(data.get("welford_M2"), dict):
            obj._welford_M2 = {k: float(v) for k, v in data["welford_M2"].items()}

        return obj

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
