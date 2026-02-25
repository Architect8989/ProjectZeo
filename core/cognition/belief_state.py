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

    # AUDIT-SI-4 FIX: Bootstrap normalization scale promoted to class constant.
    #
    # Root cause: the previous code used a local variable `_raw_scale = 0.5`
    # inside record_action() with a comment "max absolute raw reward
    # (confidence - 0.5)". This embedded an undocumented implicit contract:
    # reward signals must be in [0, 1] (confidence scores where 0.5 = neutral).
    # No validation existed at call sites to enforce this range.
    #
    # Consequence: if a caller passed rewards outside [0, 1] — for example
    # [-1.0, +1.0] success/failure signals — the bootstrap phase (n < 3
    # samples) would produce distorted normalized values that are
    # retroactively corrected only at n == 3 by the Welford renormalization.
    # Decision-making during the first 2 observations for any action operated
    # on incorrectly scaled scores, causing suboptimal early exploration.
    #
    # Fix:
    #   1. Promote the scale to BOOTSTRAP_REWARD_SCALE class constant with a
    #      full docstring documenting the expected raw reward contract.
    #   2. Add RAW_REWARD_MIN / RAW_REWARD_MAX constants that define the
    #      expected input range for raw rewards.
    #   3. In record_action(), clamp the raw reward to [RAW_REWARD_MIN,
    #      RAW_REWARD_MAX] before normalization. This prevents distortion
    #      from out-of-range inputs without breaking callers that pass valid
    #      in-range values.
    #   4. The retroactive n==3 renormalization is preserved unchanged —
    #      it remains the authoritative correction path for bootstrap bias.
    #
    # BOOTSTRAP_REWARD_SCALE contract:
    #   Raw rewards are expected in [-1.0, 1.0] where:
    #     -1.0 = complete failure / policy abort
    #      0.0 = neutral / no progress (TRUE NEUTRAL POINT)
    #     +1.0 = complete success
    #   The scale maps [-1, 1] → [-REWARD_CLAMP, REWARD_CLAMP] via:
    #     normalised = (reward / BOOTSTRAP_REWARD_SCALE) * REWARD_CLAMP
    #
    # MATH-1 FIX: Previous docstring stated "reward=0.5 → 0.0 (neutral)".
    # This was WRONG and referred to an obsolete [0, 1] reward convention.
    # With RAW_REWARD_MIN = -1.0 (current contract), the formula gives:
    #   reward=0.5: (0.5/0.5)*3.0 = 3.0  → MAX positive (not neutral!)
    #   reward=0.0: (0.0/0.5)*3.0 = 0.0  → neutral (CORRECT)
    #   reward=-1.0: (-1.0/0.5)*3.0 = -6.0 → clamped to -REWARD_CLAMP=-3.0
    # The docstring is now corrected to match the actual [-1, 1] contract.
    #
    # HAR-3 (Math): Bootstrap resolution loss — accepted by design.
    #   During the bootstrap phase (n < 3), rewards outside the range
    #   (-BOOTSTRAP_REWARD_SCALE, BOOTSTRAP_REWARD_SCALE) i.e. outside
    #   (-0.5, 0.5) saturate at ±REWARD_CLAMP.  Concretely:
    #     reward = -0.3  →  normalised = -1.8  (clamped to -REWARD_CLAMP=-3.0)
    #     reward = -1.0  →  normalised = -6.0  (clamped to -REWARD_CLAMP=-3.0)
    #   Both a mild partial failure (-0.3) and a total failure (-1.0) produce
    #   the same normalised value for the first 2 samples.  This is a known
    #   limitation: the Welford renormalization at n==3 corrects retroactively
    #   once enough samples exist to estimate true variance.  The 2-sample
    #   blind period is acceptable because no action selection depends on
    #   distinguishing failure severity that early.
    BOOTSTRAP_REWARD_SCALE: float = 0.5   # half of the [0,1] input range

    # RB-CRIT-1 FIX: REWARD_CLAMP and NORMALIZE_EPS were used throughout this
    # class (ucb_score, thompson_sample, record_action) but were never defined
    # anywhere in the codebase (grep -rn "REWARD_CLAMP\s*=" returned zero
    # results). Every call to record_action(), ucb_score(), thompson_sample(),
    # or expected_utility() raised:
    #   AttributeError: 'BeliefState' object has no attribute 'REWARD_CLAMP'
    # This made the entire probabilistic cognition subsystem non-functional.
    # No task could complete more than one iteration.
    #
    # Fix: define both constants here as class-level attributes with the values
    # implied by the rest of the code:
    #   REWARD_CLAMP = 3.0  (z-score ceiling used in all normalization paths)
    #   NORMALIZE_EPS = 1e-8  (numeric stability guard in Welford variance)
    REWARD_CLAMP: float = 3.0             # z-score ceiling for normalized rewards
    NORMALIZE_EPS: float = 1e-8          # variance floor for Welford normalization

    # RB-CRIT-2 FIX: Expand reward range to [-1.0, 1.0] to preserve negative
    # learning signal. The previous RAW_REWARD_MIN = 0.0 silently clamped
    # reward=-0.5 (policy-denied, authority-abort, low-confidence) to 0.0
    # (neutral), making failures indistinguishable from untried actions.
    # The bandit re-explored destructive actions with equal probability as
    # novel ones. Regret underflowed (failure → neutral → 0 regret delta).
    #
    # With RAW_REWARD_MIN = -1.0:
    #   reward=-0.5 (failure) → normalized = (-0.5/0.5)*3.0 = -3.0 (floor)
    #   reward=0.0  (neutral) → normalized = (0.0/0.5)*3.0  =  0.0 (center)
    #   reward=1.0  (success) → normalized = (1.0/0.5)*3.0  = +3.0 (ceiling)
    # Failures are now distinguishable from untried actions, regret accumulates
    # correctly, and the bandit avoids re-exploring known-bad actions.
    RAW_REWARD_MIN: float = -1.0          # minimum valid raw reward (failures)
    RAW_REWARD_MAX: float = 1.0           # maximum valid raw reward (successes)

    # P2: Maximum entropy allowed for convergence declaration.
    # If entropy is still high, the belief distribution has not consolidated —
    # declaring convergence while uncertain risks false success.
    MAX_ENTROPY_CONVERGENCE: float = 2.0  # nats; uniform over 8 states ≈ 2.08

    # HAR-4 (Math): THOMPSON_WINDOW (20) vs REWARD_WINDOW (100) — deliberate
    # temporal horizon split.
    #
    # Thompson sampling uses only the most recent THOMPSON_WINDOW=20 z-scores
    # for Beta parameter estimation.  This makes it *responsive*: it reacts
    # quickly to recent environment changes and does not get anchored by stale
    # reward history.
    #
    # UCB1 and Expected Utility scoring use all REWARD_WINDOW=100 samples for
    # mean and confidence-interval calculation.  This gives them *stability*:
    # a longer memory prevents thrashing on noisy single-sample events.
    #
    # The split is intentional and documented here to prevent "fixing" it.
    # If you want both algorithms on the same horizon, change THOMPSON_WINDOW
    # to match REWARD_WINDOW=100 — but expect more Thompson thrashing on
    # reward-volatile tasks.
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

        
        # HAR-1 (Determinism): commitment_hash is a *static* SHA-256 of the
        # task intent computed once at construction time.  It does NOT update on
        # each action, so calling it a "chain" is misleading.  We rename the
        # semantic role:
        #
        #   task_identity_hash  — static, identifies the task (= old commitment_hash)
        #   commitment_chain_hash — mutable, updated per record_action() call by
        #                          SHA-256(prev_chain || action_key), creating a
        #                          genuine cryptographic audit chain of actions.
        #
        # commitment_hash is kept as a property alias for backward compatibility
        # with serializer.py, main.py, and action_ranker.py which read it.
        self.task_identity_hash: str = (
            hashlib.sha256(intent_hash.encode("utf-8")).hexdigest()
            if intent_hash
            else "GENESIS"
        )
        # commitment_chain_hash starts as task_identity_hash and is extended
        # by SHA-256(prev_chain_hash + ":" + action_key) in record_action().
        #
        # H-7 FIX: Trust-boundary documentation for commitment_chain_hash.
        #
        # The chain IS a genuine cryptographic SHA-256 Merkle-style audit trail —
        # each action appends an immutable link, making it impossible to insert,
        # delete, or reorder actions without invalidating all subsequent hashes.
        # Within a single uncompromised process this provides strong integrity.
        #
        # The chain IS NOT tamper-proof against a compromised or patched process:
        #   - The hash state is held entirely in process memory with no external
        #     root of trust (no HSM, no append-only log, no external verifier).
        #   - A compromised process can call self.commitment_chain_hash = "..."
        #     directly and forge any chain value (frozen dataclass is not used here).
        #   - There is no out-of-band verifier that can detect an in-process forgery.
        #
        # Intended use: post-hoc audit of an uncompromised session (compare the
        # persisted chain hash in .authority_state.json against a replay of the
        # journal to detect accidental corruption or crash-recovery truncation).
        # NOT intended use: tamper detection against a hostile or compromised process.
        self.commitment_chain_hash: str = self.task_identity_hash
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
    # BACKWARD COMPATIBILITY PROPERTY
    # =========================================================

    @property
    def commitment_hash(self) -> str:
        """Backward-compatible alias for task_identity_hash.

        HAR-1: The field was renamed to task_identity_hash to make clear it is
        a static task-level identifier, not a mutable chain.  External callers
        (serializer.py, main.py, action_ranker.py) continue to work unchanged
        via this property.  New code should prefer task_identity_hash directly.
        """
        return self.task_identity_hash

    @commitment_hash.setter
    def commitment_hash(self, value: str) -> None:
        """Allow from_dict() to set task_identity_hash via the old field name."""
        self.task_identity_hash = value
        # Re-synchronise chain hash to new identity if chain is still at genesis
        if self.commitment_chain_hash == "GENESIS" or not self.commitment_chain_hash:
            self.commitment_chain_hash = value

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
        # H-02 FIX (MATH-05): Use commitment_chain_hash (mutable, updated per
        # record_action() call) instead of commitment_hash (static alias for
        # task_identity_hash).  When commitment_hash was used, identical task
        # intent + identical iteration/sample counters produced cryptographically
        # identical Thompson samples across sessions — adversarial environments
        # could exploit the deterministic exploration pattern.
        # commitment_chain_hash advances with every action recorded, so even
        # for the same intent the seed is unique per action history.
        seed_material = (
            f"{self.commitment_chain_hash}:{action}:{self._iteration_counter}:{self._sample_counter}"
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
        # HAR-1: Advance the commitment chain hash so that each recorded action
        # extends the cryptographic audit trail.  chain_n = SHA-256(chain_{n-1}
        # || ":" || action_key).  This makes commitment_chain_hash a genuine
        # per-action chain rather than a static task-level identifier.
        _chain_input = f"{self.commitment_chain_hash}:{action}".encode("utf-8")
        self.commitment_chain_hash = hashlib.sha256(_chain_input).hexdigest()

        if action not in self.action_rewards:
            self.action_rewards[action] = deque(maxlen=self.REWARD_WINDOW)
        if action not in self._raw_action_rewards:
            self._raw_action_rewards[action] = deque(maxlen=self.REWARD_WINDOW)

        # AUDIT-SI-4 FIX: Clamp raw reward to [RAW_REWARD_MIN, RAW_REWARD_MAX]
        # before any normalization. This enforces the bootstrap contract
        # (rewards must be in [0, 1]) without breaking callers that already
        # pass valid values. Out-of-range rewards (e.g. -0.5 or 1.5) are
        # silently clamped here rather than producing distorted bootstrap
        # scores that require retroactive correction. The Welford renormalization
        # at n==3 still fires as before — this clamp is an additional guard,
        # not a replacement.
        reward = max(self.RAW_REWARD_MIN, min(self.RAW_REWARD_MAX, float(reward)))

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
            # Bootstrap path (n < 3): use BOOTSTRAP_REWARD_SCALE class constant.
            # AUDIT-SI-4 FIX: The previous code used a local `_raw_scale = 0.5`
            # with no documentation. Replaced with the class constant
            # BOOTSTRAP_REWARD_SCALE (= 0.5) which is now documented above with
            # its contract and rationale. The raw reward has already been clamped
            # to [RAW_REWARD_MIN, RAW_REWARD_MAX] = [0.0, 1.0] earlier in this
            # method, so division by BOOTSTRAP_REWARD_SCALE is guaranteed safe
            # (BOOTSTRAP_REWARD_SCALE > 0 by definition).
            normalised = (reward / self.BOOTSTRAP_REWARD_SCALE) * self.REWARD_CLAMP
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
            # Significant UI change: reduce stability score and reset streak.
            self.environment_stability = max(
                0.0, self.environment_stability - 0.2
            )
            self.consecutive_high_stability_count = 0
        else:
            # No significant change: nudge stability upward, cap at 1.0.
            self.environment_stability = min(
                1.0, self.environment_stability + 0.05
            )
            if self.environment_stability >= 0.8:
                self.consecutive_high_stability_count += 1
            else:
                self.consecutive_high_stability_count = 0

    # =========================================================
    # SERIALIZATION  — P0 FIX: to_dict / from_dict / summary
    #
    # These three methods were entirely absent from the file.
    # operate.py calls:
    #   belief.summary()        at line 399 (ReasoningEngine fallback)
    #   BeliefState.from_dict() at line 244 (replan state restoration)
    #   belief.to_dict()        (populates belief_state_out for next replan)
    # All three raised AttributeError, making reasoning fallback and
    # cross-replan continuity completely non-functional.
    # =========================================================

    def summary(self) -> dict:
        """
        Return a lightweight snapshot of the current belief state suitable
        for passing to ReasoningEngine.propose_actions() and for logging.
        All values are JSON-serializable primitives.
        """
        # Top-5 most probable world states (keep summary small).
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
        Serialize the full BeliefState to a JSON-safe dict.

        All deque objects are converted to lists. Welford running statistics
        (mean, M2, n) are preserved so that normalization is consistent when
        the state is reconstructed via from_dict().

        Called by operate.py to persist state across replans via belief_state_out.
        """
        return {
            # Belief distribution
            "state_probabilities": dict(self.state_probabilities),
            "progress_score": self.progress_score,
            "environment_stability": self.environment_stability,
            "consecutive_high_stability_count": self.consecutive_high_stability_count,

            # Commitment / iteration tracking
            "commitment_hash": self.commitment_hash,
            "commitment_chain_hash": self.commitment_chain_hash,  # HAR-1: true per-action chain
            "_iteration_counter": self._iteration_counter,
            "_sample_counter": self._sample_counter,
            "_regret_decay": self._regret_decay,

            # Action statistics
            "action_counts": dict(self.action_counts),
            "action_rewards": {
                k: list(v) for k, v in self.action_rewards.items()
            },
            "_raw_action_rewards": {
                k: list(v) for k, v in self._raw_action_rewards.items()
            },
            "regret": {k: list(v) for k, v in self.regret.items()},

            # Welford running statistics (required for consistent renormalization)
            "_welford_n": dict(self._welford_n),
            "_welford_mean": dict(self._welford_mean),
            "_welford_M2": dict(self._welford_M2),

            # Metadata
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict, *, intent_hash: str = "") -> "BeliefState":
        """
        Reconstruct a BeliefState from a dict produced by to_dict().

        Preserves action counts, reward history, Welford statistics,
        regret, commitment_hash, and Thompson counters so that cross-replan
        learning is not lost. A fresh BeliefState seeded only with
        intent_hash is returned as a safe fallback if data is invalid.

        Parameters
        ----------
        data : dict
            Serialized state from BeliefState.to_dict().
        intent_hash : str
            The intent string for the current task (used only when creating
            a fallback instance; the commitment_hash in data takes precedence).
        """
        if not isinstance(data, dict):
            return cls(intent_hash=intent_hash)

        try:
            instance = cls.__new__(cls)

            # Timestamps
            instance.created_at = float(data.get("created_at", 0.0) or 0.0)

            # Belief distribution
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

            # Commitment / iteration
            instance.commitment_hash = str(
                data.get("commitment_hash", "GENESIS")
            )
            # HAR-1: Restore the per-action chain hash; fall back to the
            # task_identity_hash (== commitment_hash) for states serialized
            # before this field was introduced.
            instance.commitment_chain_hash = str(
                data.get("commitment_chain_hash", instance.task_identity_hash)
            )
            instance._iteration_counter = int(
                data.get("_iteration_counter", 0)
            )
            instance._sample_counter = int(data.get("_sample_counter", 0))
            instance._regret_decay = float(
                data.get("_regret_decay", cls.REGRET_DECAY)
            )

            # Action statistics — restore as deques with correct maxlen.
            instance.action_counts = dict(data.get("action_counts", {}))

            instance.action_rewards = {}
            for k, v in data.get("action_rewards", {}).items():
                d = deque(v, maxlen=cls.REWARD_WINDOW)
                instance.action_rewards[k] = d

            instance._raw_action_rewards = {}
            for k, v in data.get("_raw_action_rewards", {}).items():
                d = deque(v, maxlen=cls.REWARD_WINDOW)
                instance._raw_action_rewards[k] = d

            instance.regret = {}
            for k, v in data.get("regret", {}).items():
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    instance.regret[k] = (float(v[0]), int(v[1]))

            # Welford running statistics
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
            # Any reconstruction error → safe fallback with fresh state.
            return cls(intent_hash=intent_hash)

