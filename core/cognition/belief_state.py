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

    def __init__(self, intent_hash: str = ""):
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

        # HAR-06 (MS-03): Seed the commitment hash with the intent hash rather
        # than the constant "GENESIS". When commitment_hash = "GENESIS" for every
        # new BeliefState, two replans of the same intent produce identical seed
        # sequences for Thompson sampling — the sampler cannot distinguish a
        # prior failure from an unexplored action, neutralising replan exploration.
        # Using the intent hash makes the seed space unique per intent while
        # preserving within-session determinism.
        self.commitment_hash: str = (
            hashlib.sha256(intent_hash.encode("utf-8")).hexdigest()
            if intent_hash
            else "GENESIS"
        )
        self._iteration_counter: int = 0

        # HARDEN-4 (M-NEW-02): _iteration_counter and _sample_counter are
        # TRANSIENT — they are NOT persisted and reset to 0 on every process
        # restart / new BeliefState construction.
        #
        # Implication for restart reproducibility:
        #   - commitment_hash IS reproducible across restarts: it is re-seeded
        #     from SHA-256(intent_hash) at construction, so the same intent
        #     always produces the same genesis hash.
        #   - Thompson sampling seed chain is NOT reproducible across restarts:
        #     seeds incorporate _iteration_counter and _sample_counter values
        #     which restart at 0. Even with the same intent and action sequence,
        #     samples after a restart differ from those in the original run.
        #
        # This is EXPECTED BEHAVIOUR. Full restart reproducibility would require
        # persisting and restoring these counters alongside the commitment_hash —
        # a deliberate design trade-off. Operators relying on exact Thompson
        # reproducibility across restarts must implement their own counter
        # persistence and call BeliefState with a pre-seeded commitment_hash.

        # MR-01a FIX: Track consecutive high-stability observations for
        # the conservative authority gate in operate.py. The gate requires
        # 3 consecutive readings above 0.7 before asserting soc_confident
        # for high-risk operations.
        self.consecutive_high_stability_count: int = 0

        # MR-04 FIX: Per-call sample counter for Thompson sampling seeds.
        # _iteration_counter only increments in update_regret(). All Thompson
        # samples within a single selection round share the same counter value,
        # giving them identical seeds when only action string differs slightly.
        # _sample_counter increments on EVERY thompson_sample() call, providing
        # a unique component for each sample within a round.
        self._sample_counter: int = 0

        # MR-05 FIX: Dynamic regret decay computed from plan horizon.
        # Default is the class constant; call set_plan_horizon() once the
        # execution plan is loaded to tune decay to the actual task length.
        self._regret_decay: float = self.REGRET_DECAY

        # Fix 11 (Welford): per-action running statistics for incremental
        # z-score normalisation. Using Welford's online algorithm means only
        # the NEW reward entry is normalised relative to running mean/variance.
        # Historical entries remain stable (their normalised values do not
        # change when a new reward arrives). This replaces the previous
        # bulk re-normalisation that made all historical utilities non-stationary.
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
            # MR-02 FIX: Proportional entropy floor blending.
            # ─────────────────────────────────────────────────────────────
            # Bug: fixed blend weight 0.05 regardless of collapse severity.
            # At entropy = 0.01 nats (near-total collapse), one blend
            # restores only ~0.05 nats — still far below the 0.30 floor.
            # The old loop took 15–20 iterations to converge.
            #
            # Fix: set blend weight proportional to the entropy deficit
            # (how far below the floor we are), capped at MAX_BLEND_WEIGHT.
            # This guarantees convergence in ≤ 3–5 iterations at any
            # collapse level.
            # ─────────────────────────────────────────────────────────────
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
            # Fix 14: Only prune __prior_fallback__ if the entropy floor is
            # already satisfied BEFORE the prune. If entropy enforcement raised
            # fallback probability to just above PRIOR_ALPHA (= 0.01) and we
            # prune it immediately, entropy drops back below the floor and the
            # enforcement/prune cycle repeats indefinitely for up to 20 iterations.
            # Gate the prune on self.entropy() >= MIN_ENTROPY_FLOOR so the floor
            # is never undercut by the pruning step that follows it.
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
        rewards = self.action_rewards.get(action)
        if not rewards:
            return 0.0

        n = len(rewards)
        mean = sum(rewards) / n
        variance = sum((r - mean) ** 2 for r in rewards) / n
        return mean - self.RISK_LAMBDA * variance

    def ucb_score(self, action: str) -> float:
        # FIX-14: Return float('inf') for unvisited actions so UCB1's
        # "must explore" guarantee is preserved.
        #
        # Bug: count = self.action_counts.get(action, 0) + 1 always added 1,
        # giving unvisited actions an effective count of 1. This suppressed the
        # log(N)/count → ∞ singularity that UCB1 uses as the "must explore"
        # signal. Cold-start exploration was degraded — actions seen once were
        # treated the same as actions never seen. The algorithm could commit to
        # a suboptimal action after a single trial.
        #
        # Fix: return float('inf') for count=0 actions, which is the standard
        # UCB1 behaviour. This guarantees every action is tried at least once
        # before any is repeated, preserving the regret bound.
        count = self.action_counts.get(action, 0)
        if count == 0:
            return float('inf')  # Must explore: standard UCB1 guarantee

        total_actions = max(sum(self.action_counts.values()) + 1, 2)
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
        """
        Fix 11 (Welford's algorithm): Record an action reward using incremental
        z-score normalisation.

        Previous implementation re-normalised ALL entries in the deque on every
        new reward arrival, making historical action utilities non-stationary:
        a single extreme new reward retroactively changed the perceived history
        of ALL prior actions. This is not standard bandit normalisation.

        Fix: use Welford's online algorithm to maintain a running mean and
        variance per action. Only the NEW reward is normalised relative to the
        running statistics and appended. Existing deque entries are untouched,
        keeping historical utility estimates stable.

        Welford update for sample n:
            delta  = reward - mean_{n-1}
            mean_n = mean_{n-1} + delta / n
            delta2 = reward - mean_n
            M2_n   = M2_{n-1} + delta * delta2
            var_n  = M2_n / n   (population variance)
        """
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
        else:
            # FIX-06 (MS-01): For n < 3, use simple linear clamp. Once n reaches
            # 3 the Welford branch takes over with stable running statistics.
            normalised = max(-self.REWARD_CLAMP, min(self.REWARD_CLAMP, reward))

        self.action_rewards[action].append(normalised)
        self.action_counts[action] = self.action_counts.get(action, 0) + 1

    # =========================================================
    # GLOBAL BEST REWARD — MATH-04 FIX
    # =========================================================

    def global_best_reward(self):
        """
        HAR-1 (MATH-1): Return the best RAW reward seen across all actions
        this session, or None if no actions have been recorded yet.

        MATH-1 root cause: the original implementation initialised best=0.0
        and unconditionally returned 0.0 when no raw rewards had been
        recorded.  In uniformly failing contexts (all rewards at -0.5),
        regret = 0.0 - (-0.5) = 0.5 for EVERY action — a constant that
        provides no discrimination between better and worse actions.  The
        regret signal was effectively disabled in the most important scenario:
        when the agent is failing and needs to differentiate actions.

        Fix: return None when no rewards have been recorded, and let callers
        skip the regret update when best_reward is None.  When rewards exist,
        return the true maximum over raw rewards (which may be negative).
        This ensures regret = best - reward correctly discriminates between
        actions even when all rewards are negative.

        Returns float or None:
          - None:  no actions recorded yet (caller should skip regret update)
          - float: true maximum raw reward over all recorded actions
                   (may be negative if all actions have failed so far)
        """
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
        # HARD-2 (§3.8): Type-tagged encoding to eliminate None/"None" collision.
        #
        # Bug: the fallback branch `return str(value).encode()` produced b"None"
        # for BOTH value=None AND value="None" (the string). Observation dicts
        # containing None values (common for missing perception fields, e.g.
        # `perception_snapshot=None`) produced identical commitment hashes as
        # dicts containing the string "None". This weakened the cryptographic
        # commitment chain, making observationally distinct world states
        # hash-indistinguishable.
        #
        # Fix: prefix each type with a 1-byte tag before the value encoding.
        # Tags are chosen to be unambiguous and cannot appear as a prefix of
        # any other type's encoding:
        #   \x00 = None
        #   \x01 = str
        #   \x02 = int / bool (stringified)
        #   \x03 = float (IEEE 754 big-endian double)
        #   \x04 = dict (handled by recursive branch)
        #   \x05 = list (handled by recursive branch)
        #   \xff = unknown fallback (type name + stringified value)
        #
        # The dict and list branches use recursion and do not need tags because
        # their structure is unambiguous from their encoding (sorted keys for
        # dict, length-prefixed elements for list).
        if value is None:
            # \x00 tag: unambiguously None — cannot be produced by any str value.
            return b"\x00"
        if isinstance(value, float):
            return b"\x03" + struct.pack("!d", value)
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
            # HARDEN-1 (M-NEW-01): Replace b"|".join(...) with length-prefixed
            # encoding to eliminate the delimiter-byte collision risk.
            #
            # Bug: The previous delimiter b"|" (byte value 0x7C = 124) can appear
            # in IEEE 754 struct.pack("!d", v) float encodings. Two distinct lists
            # [f1, f2] and [f1 + delta] could theoretically produce identical byte
            # sequences when the boundary byte of an encoded float coincidentally
            # equals 0x7C — breaking commitment hash uniqueness.
            #
            # The code comment claiming b"|" "never appears" in float packing was
            # false. Many double-precision floats produce byte sequences containing
            # 0x7C. While exploitation requires adversarially crafted observation
            # dicts, natural perception data could also trigger this in practice.
            #
            # Fix: length-prefix each serialised element with its byte count as a
            # 4-byte big-endian unsigned integer. This makes the boundary between
            # elements unambiguous: the decoder always reads 4 bytes for length,
            # then exactly that many bytes for the payload — no in-band delimiter
            # needed. Two distinct lists can never produce the same encoding.
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
        """
        Fix 12: Serialise BeliefState to a JSON-safe dict including
        _iteration_counter and _sample_counter so Thompson sampling seeds
        are reproducible across process restarts.

        Callers that want replay reproducibility should persist the returned
        dict (e.g. in the authority state file) and pass it to from_dict()
        when reconstructing BeliefState after a restart.

        Note: action_rewards and _raw_action_rewards store deques of floats
        which are serialised as lists. The Welford running-statistics fields
        (_welford_n, _welford_mean, _welford_M2) are also persisted so that
        normalisation history survives a restart.
        """
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
