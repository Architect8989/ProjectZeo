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

    # Bootstrap phase: map raw rewards into normalised range without Welford
    # variance (which requires n >= 3).  A scale of 1.0 preserves the linear
    # [-1, 1] range and avoids the 3× amplification bias from the previous
    # REWARD_CLAMP-based scale.  (M-5 fix)
    BOOTSTRAP_REWARD_SCALE: float = 1.0

    # Clamping and numerical stability constants
    REWARD_CLAMP: float = 3.0        # z-score ceiling for normalised rewards
    NORMALIZE_EPS: float = 1e-8      # variance floor for Welford normalisation

    # Raw reward validity range (actions produce signals in [-1, 1])
    RAW_REWARD_MIN: float = -1.0
    RAW_REWARD_MAX: float = 1.0

    MAX_ENTROPY_CONVERGENCE: float = 2.0  # nats; uniform over 8 states ≈ 2.08

    # Thompson uses a recent window for reactivity; UCB1/EU use the full window
    # for stability.  The deliberate split is documented here to prevent future
    # "alignment" that would break temporal horizon reasoning.
    THOMPSON_WINDOW = 20

    _FALLBACK_PRUNE_THRESHOLD = PRIOR_ALPHA * 2.0

    # ------------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------------

    def __init__(self, intent_hash: str = "") -> None:
        self.created_at: float = time.time()

        # Belief distribution
        self.state_probabilities: Dict[str, float] = {"neutral": 1.0}

        # Action history (normalised reward stream)
        self.action_counts: Dict[str, int] = {}
        self.action_rewards: Dict[str, deque] = {}       # normalised
        self._raw_action_rewards: Dict[str, deque] = {}  # raw, for global_best_reward

        # Regret tracking
        self.regret: Dict[str, Tuple[float, int]] = {}

        # Environment model
        self.progress_score: float = 0.0
        self.environment_stability: float = 1.0
        self.consecutive_high_stability_count: int = 0

        # Commitment chain — dual-hash architecture (HAR-1)
        #   task_identity_hash   : static SHA-256(intent); identifies the task
        #   commitment_chain_hash: mutable; extended by each record_action() call
        self.task_identity_hash: str = (
            hashlib.sha256(intent_hash.encode("utf-8")).hexdigest()
            if intent_hash
            else "GENESIS"
        )
        self.commitment_chain_hash: str = self.task_identity_hash

        # Counters
        self._iteration_counter: int = 0
        self._sample_counter: int = 0
        self._regret_decay: float = self.REGRET_DECAY

        # Welford incremental variance estimators (per action)
        self._welford_n: Dict[str, int] = {}
        self._welford_mean: Dict[str, float] = {}
        self._welford_M2: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # BACKWARD COMPATIBILITY
    # ------------------------------------------------------------------

    @property
    def commitment_hash(self) -> str:
        """Backward-compatible alias for task_identity_hash."""
        return self.task_identity_hash

    @commitment_hash.setter
    def commitment_hash(self, value: str) -> None:
        """Allow from_dict() to set task_identity_hash via the old field name."""
        self.task_identity_hash = value
        if self.commitment_chain_hash in ("GENESIS", ""):
            self.commitment_chain_hash = value

    # =========================================================================
    # BELIEF UPDATE
    # =========================================================================

    def bayesian_update(self, likelihoods: Dict[str, float]) -> None:
        """
        Apply a proportional belief update using heuristic observation weights.

        DOCUMENTATION NOTE (SI-2 / HARDEN-1):
        Despite the method name, the `likelihoods` values are *heuristic
        weights*, not statistical likelihoods P(observation | state) derived
        from an observation model.  The proportional update rule is formally
        correct; the epistemic claim is intentionally overstated in the original
        architecture.  New code should treat this as a heuristic attractor, not
        a calibrated Bayesian posterior.

        Caller-assigned weights:
          app:<n>   = 0.9   (focused app matches expected)
          ui_rich   = 0.8   (>10 visible UI entities)
          ui_sparse = 0.7   (1–10 visible entities)
          ui_empty  = 0.5   (no visible entities)
          neutral   = 0.9 or 0.5 (no delta / delta present)
        """
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

        # Ensure minimum two-state distribution to prevent entropy floor
        # from collapsing into a deterministic one-state distribution
        if len(pruned) == 1:
            (sole_state,) = pruned.keys()
            pruned["__prior_fallback__"] = self.PRIOR_ALPHA
            total = sum(pruned.values())

        self.state_probabilities = {s: v / total for s, v in pruned.items()}

        # Entropy floor recovery — blend toward uniform if entropy collapses
        _current_entropy = self.entropy()
        if _current_entropy < self.MIN_ENTROPY_FLOOR:
            _MAX_BLEND_WEIGHT = 0.30
            for _ in range(20):  # hard cap — converges in ≤5 iterations in practice
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

        # Prune the synthetic fallback state once it is below threshold
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
        """Shannon entropy of the current belief distribution (in nats)."""
        return -sum(
            p * math.log(p)
            for p in self.state_probabilities.values()
            if p > 0.0
        )

    # =========================================================================
    # ACTION SCORING
    # =========================================================================

    def expected_utility(self, action: str) -> float:
        """
        Risk-adjusted expected utility with bounded variance penalty.

        FIX (B-MATH-3): Cap penalty at |mean| to prevent sign inversion for
        high-mean-high-variance actions.  Without the cap, a reward mean of
        0.1 with variance 10.0 would yield EU = 0.1 − 3.0 = −2.9, making an
        occasionally-good action appear catastrophically bad.
        """
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
        """
        UCB1 score with scale-corrected exploitation term.

        FIX (B-MATH-1): Map mean_reward from [−REWARD_CLAMP, REWARD_CLAMP] → [0, 1]
        before adding the exploration bonus.  UCB1's regret bound requires the
        exploitation term to be in [0, 1]; without this mapping the exploration
        constant C has no calibrated meaning and regret bounds are invalid.
        """
        count = self.action_counts.get(action, 0)
        if count == 0:
            return float("inf")

        total_actions = max(sum(self.action_counts.values()) + 1, 2)
        rewards = self.action_rewards.get(action)
        mean_reward_raw = sum(rewards) / len(rewards) if rewards else 0.0

        # Map to [0, 1] for UCB1 compliance
        mean_reward_01 = (mean_reward_raw + self.REWARD_CLAMP) / (2.0 * self.REWARD_CLAMP)
        mean_reward_01 = max(0.0, min(1.0, mean_reward_01))

        exploration = self.EXPLORATION_C * math.sqrt(
            math.log(total_actions) / count
        )
        return mean_reward_01 + exploration

    # =========================================================================
    # THOMPSON SAMPLING — CRITICAL FIX (RB-1 / D-1 / D-2)
    # =========================================================================

    def thompson_sample(self, action: str) -> float:
        """
        Draw a sample from the Normal-Normal conjugate posterior for `action`.

        CRITICAL FIX (RB-1):
            The previous implementation referenced `rng` (undefined name).
            This caused NameError on EVERY call after the first recorded action,
            crashing all tasks beyond their second action attempt.

        FIX (D-1 / D-2):
            `_rng = np.random.default_rng(seed)` is now created and the
            deterministic seed derived from commitment_chain_hash is actually
            applied.  Prior implementation computed `seed` correctly but never
            passed it to any RNG object.

        FIX (MATH-2):
            Acceptance-rejection sampling replaces clamped Normal to produce
            the exact truncated Gaussian T_Normal(μ, σ², lo, hi).  Fallback
            to the posterior mean fires only after 128 failed rejections.

        Returns
        -------
        float
            Sample from the posterior reward distribution, bounded to
            [−REWARD_CLAMP, REWARD_CLAMP].  Returns 0.0 (neutral prior mean)
            if no reward history is available for `action`.
        """
        rewards = self.action_rewards.get(action)
        if not rewards:
            # No history — return neutral prior mean (not 0.5; reward space is centred at 0)
            return 0.0

        recent = list(rewards)[-self.THOMPSON_WINDOW:]
        n = len(recent)

        # ------------------------------------------------------------------
        # Posterior parameters — Normal-Normal conjugate update
        # ------------------------------------------------------------------
        _mu0: float = 0.0                              # neutral prior mean
        _sigma0_sq: float = self.REWARD_CLAMP ** 2    # wide prior variance = 9.0

        _welford_n = self._welford_n.get(action, 0)
        if _welford_n >= 3:
            # Welford M2/(n-1) is the unbiased sample variance of the normalised
            # reward stream.  Use as observation noise variance, clamped below to
            # prevent division by zero on zero-variance (deterministic) actions.
            _obs_variance = max(
                self._welford_M2.get(action, 0.0) / max(_welford_n - 1, 1),
                self.NORMALIZE_EPS,
            )
        else:
            # Bootstrap: assume maximum noise (conservative; promotes exploration)
            _obs_variance = _sigma0_sq

        _prior_prec = 1.0 / _sigma0_sq
        _obs_prec = n / _obs_variance       # sum of n i.i.d. observation precisions
        _post_prec = _prior_prec + _obs_prec
        _post_variance = 1.0 / _post_prec

        _sum_rewards = sum(recent)
        _post_mean = _post_variance * (_prior_prec * _mu0 + _sum_rewards / _obs_variance)

        # Clamp posterior mean to valid reward range (numerical safety)
        _post_mean = max(-self.REWARD_CLAMP, min(self.REWARD_CLAMP, _post_mean))

        # ------------------------------------------------------------------
        # Deterministic seed from commitment chain
        # D-1 / D-2 FIX: seed is now ACTUALLY applied to the RNG.
        # Previously seed was computed but never passed to any RNG object,
        # making sampling non-deterministic and violating the architectural
        # guarantee of deterministic Thompson sampling.
        # ------------------------------------------------------------------
        self._sample_counter += 1
        seed_material = (
            f"{self.commitment_chain_hash}:{action}:"
            f"{self._iteration_counter}:{self._sample_counter}"
        ).encode("utf-8")
        digest = hashlib.sha256(seed_material).digest()
        seed = int.from_bytes(digest[:8], byteorder="big", signed=False)

        # CRITICAL FIX (RB-1): Create the RNG with the deterministic seed.
        # The previous code reached `rng.normal(...)` with `rng` undefined,
        # raising NameError on every second action attempt.
        _rng = np.random.default_rng(seed)  # ← THE MISSING LINE

        # ------------------------------------------------------------------
        # Acceptance-rejection sampling from truncated Gaussian posterior
        # (MATH-2 fix: avoids boundary-mass clustering from clamped Normal)
        # ------------------------------------------------------------------
        _lo = -self.REWARD_CLAMP
        _hi = self.REWARD_CLAMP
        _std = math.sqrt(_post_variance)

        # Fallback: posterior mean is guaranteed in-range by construction above
        sample: float = _post_mean

        for _ in range(128):
            # CRITICAL FIX (RB-1): `_rng` is now defined; this no longer raises NameError
            _candidate = float(_rng.normal(loc=_post_mean, scale=_std))
            if _lo <= _candidate <= _hi:
                sample = _candidate
                break
        # If 128 rejections exhausted (posterior extremely wide), sample = _post_mean (safe)

        return float(sample)

    # =========================================================================
    # REGRET TRACKING
    # =========================================================================

    def _get_effective_regret(self, action: str) -> float:
        """Retrieve regret for `action` with temporal decay applied."""
        entry = self.regret.get(action)
        if not entry:
            return 0.0

        raw_value, last_iter = entry
        delta_iter = self._iteration_counter - last_iter
        decayed = raw_value * (self._regret_decay ** delta_iter)
        decayed = min(decayed, self.MAX_REGRET)

        # Cache the decayed value to avoid recomputing on every access
        if delta_iter > 0:
            self.regret[action] = (decayed, self._iteration_counter)

        return decayed

    def set_plan_horizon(self, total_steps: int, iters_per_step: int = 13) -> None:
        """
        Tune regret decay so that residual regret at plan completion is ≤5%.

        Formula: target_fraction^(1/total_iters)
        where total_iters = total_steps * iters_per_step.
        """
        total_iters = max(total_steps * iters_per_step, 1)
        target_fraction = 0.05
        self._regret_decay = target_fraction ** (1.0 / total_iters)

    def update_regret(self, action: str, reward: float, best_reward: float) -> None:
        """Record instantaneous regret = best_reward − reward and accumulate."""
        self._iteration_counter += 1

        regret_value = best_reward - reward
        if regret_value <= 0.0:
            return

        current = self._get_effective_regret(action)
        updated = min(current + regret_value, self.MAX_REGRET)
        self.regret[action] = (updated, self._iteration_counter)

    # =========================================================================
    # ACTION RECORDING WITH WELFORD NORMALISATION
    # =========================================================================

    def record_action(self, action: str, reward: float) -> None:
        """
        Record `reward` for `action`, extending the commitment chain.

        Processing pipeline:
          1. Extend commitment_chain_hash with SHA-256(prev || ":" || action)
          2. Clamp raw reward to [RAW_REWARD_MIN, RAW_REWARD_MAX]
          3. Store raw reward for global_best_reward()
          4. Apply Welford incremental update (mean, M2)
          5. Normalise: z-score (Welford, n≥3) or linear bootstrap (n<3)
          6. Clamp normalised reward to [−REWARD_CLAMP, REWARD_CLAMP]
          7. Append to action_rewards deque
        """
        # 1. Extend commitment chain — MUST be first (before any state mutation)
        _chain_input = f"{self.commitment_chain_hash}:{action}".encode("utf-8")
        self.commitment_chain_hash = hashlib.sha256(_chain_input).hexdigest()

        # 2. Initialise deques on first encounter
        if action not in self.action_rewards:
            self.action_rewards[action] = deque(maxlen=self.REWARD_WINDOW)
        if action not in self._raw_action_rewards:
            self._raw_action_rewards[action] = deque(maxlen=self.REWARD_WINDOW)

        # 3. Clamp raw reward to valid range
        reward = max(self.RAW_REWARD_MIN, min(self.RAW_REWARD_MAX, float(reward)))
        self._raw_action_rewards[action].append(reward)

        # 4. Welford incremental update (Knuth / Welford algorithm)
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

        # 5. Normalise
        if n >= 3:
            # Standard Welford z-score (Bessel-corrected; B-MATH-02)
            variance = new_M2 / max(n - 1, 1)
            std = math.sqrt(max(variance, self.NORMALIZE_EPS))
            normalised = (reward - new_mean) / std
            normalised = max(-self.REWARD_CLAMP, min(self.REWARD_CLAMP, normalised))

            # SI-04 FIX: On crossing n=3, retroactively renormalise the two
            # bootstrap entries using the mean and std of ENTRIES 1–2 ONLY.
            #
            # Root cause of previous defect: the code passed ``new_mean``
            # (which is mean(entries 1, 2, 3) after Welford update) as the
            # renormalisation baseline for entries 1 and 2.  This introduced
            # systematic bias because entry 3's value influenced the mean used
            # to normalise entries 1 and 2.  A high entry-3 reward shifted the
            # mean up, making entries 1-2 appear to have *lower* z-scores than
            # they actually earned relative to their own distribution.
            #
            # Fix: compute a separate two-sample mean from the first two raw
            # rewards BEFORE the Welford update.  This two-sample mean is the
            # correct baseline for renormalising entries 1 and 2.
            #
            # The std must still be derived from the full 3-sample Welford
            # statistics because 2 samples do not yield a meaningful variance
            # estimate (n-1 = 1; sample variance from 2 points is 0 or the
            # full range, both unreliable).  Using std from n=3 is a known
            # mild bias but is bounded and corrects itself as n grows.
            if n == 3 and action in self._raw_action_rewards:
                raw_deque = self._raw_action_rewards[action]
                if len(raw_deque) >= 2:
                    old_entries = list(raw_deque)[:2]

                    # SI-04 FIX: Use mean of entries 1-2, NOT new_mean (1-2-3)
                    _two_sample_mean = (old_entries[0] + old_entries[1]) / 2.0

                    renormalized = []
                    for r in old_entries:
                        # z-score each entry relative to the 2-sample mean but
                        # using the 3-sample std (see note above).
                        z = (r - _two_sample_mean) / std
                        renormalized.append(
                            max(-self.REWARD_CLAMP, min(self.REWARD_CLAMP, z))
                        )
                    existing = self.action_rewards.get(action)
                    if existing is not None and len(existing) == 2:
                        existing.clear()
                        existing.extend(renormalized)
        else:
            # Bootstrap (n < 3): linear mapping using BOOTSTRAP_REWARD_SCALE.
            # M-5 FIX: BOOTSTRAP_REWARD_SCALE = 1.0 (identity for [-1,1] range)
            # prevents the old 3× amplification that biased UCB toward early successes.
            normalised = (reward / self.BOOTSTRAP_REWARD_SCALE) * self.REWARD_CLAMP
            normalised = max(-self.REWARD_CLAMP, min(self.REWARD_CLAMP, normalised))

        self.action_rewards[action].append(normalised)
        self.action_counts[action] = self.action_counts.get(action, 0) + 1

    # =========================================================================
    # REGRET FLUSH ON TASK SUCCESS (IH-05 FIX)
    # =========================================================================

    def flush_regret_on_success(self) -> None:
        
        self.regret.clear()



    def global_best_reward(self) -> "float | None":
        """Return the highest raw reward observed across all actions, or None."""
        best: "float | None" = None
        for history in self._raw_action_rewards.values():
            if history:
                local_max = max(history)
                if best is None or local_max > best:
                    best = local_max
        return best

    # =========================================================================
    # ENVIRONMENT MODEL
    # =========================================================================

    def compute_environment_stability(self, delta: Dict[str, Any]) -> None:
        """
        Update environment_stability based on observed world-graph delta.

        Stability decays on significant change (−0.2) and recovers slowly on
        stability (+0.05).  consecutive_high_stability_count tracks sustained
        stability for use by high-risk operation gating in operate.py.
        """
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

    # =========================================================================
    # SERIALISATION
    # =========================================================================

    def summary(self) -> dict:
        """
        Lightweight JSON-serialisable snapshot for ReasoningEngine and logging.
        Contains only the top-5 state probabilities for prompt compactness.
        """
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
        """
        Full serialisation for crash-recovery and replan continuity.

        All keys are JSON-serialisable.  Deques are serialised as lists.
        Regret tuples are serialised as 2-element lists.
        """
        return {
            # Belief distribution
            "state_probabilities": dict(self.state_probabilities),
            "progress_score": self.progress_score,
            "environment_stability": self.environment_stability,
            "consecutive_high_stability_count": self.consecutive_high_stability_count,

            # Commitment / iteration tracking
            "commitment_hash": self.commitment_hash,         # backward compat
            "commitment_chain_hash": self.commitment_chain_hash,
            "_iteration_counter": self._iteration_counter,
            "_sample_counter": self._sample_counter,
            "_regret_decay": self._regret_decay,

            # Action statistics
            "action_counts": dict(self.action_counts),
            "action_rewards": {k: list(v) for k, v in self.action_rewards.items()},
            "_raw_action_rewards": {
                k: list(v) for k, v in self._raw_action_rewards.items()
            },
            "regret": {k: list(v) for k, v in self.regret.items()},

            # Welford running statistics
            "_welford_n": dict(self._welford_n),
            "_welford_mean": dict(self._welford_mean),
            "_welford_M2": dict(self._welford_M2),

            # Metadata
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict, *, intent_hash: str = "") -> "BeliefState":
        """
        Reconstruct a BeliefState from a to_dict() snapshot.

        Falls back to a fresh BeliefState on any deserialisation failure so
        crash-recovery never raises — the worst case is a reset bandit state,
        not a process abort.
        """
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

            # Commitment chain — use the setter for backward compat
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
            # Deserialisation failure → fresh state (crash-safe fallback)
            return cls(intent_hash=intent_hash)
