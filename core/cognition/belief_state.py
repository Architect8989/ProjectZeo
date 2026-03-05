from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
from collections import deque
import time
import math
import hashlib
import numpy as np


class BeliefState:
    

    # ------------------------------------------------------------------
    # Class-level constants
    # ------------------------------------------------------------------

    EXPLORATION_C: float = 1.4
    RISK_LAMBDA: float = 0.3
    REWARD_WINDOW: int = 100
    PRIOR_ALPHA: float = 0.01
    
    LIKELIHOOD_FLOOR: float = 1e-10
    REGRET_DECAY: float = 0.995
    MAX_STATES: int = 64
    MAX_REGRET: float = 100.0

    
    MIN_ENTROPY_FLOOR: float = 0.3

    
    BOOTSTRAP_REWARD_SCALE: float = 1.0

    REWARD_CLAMP: float = 3.0
    NORMALIZE_EPS: float = 1e-8

    RAW_REWARD_MIN: float = -1.0
    RAW_REWARD_MAX: float = 1.0

    MAX_ENTROPY_CONVERGENCE: float = 2.0

    THOMPSON_WINDOW: int = 20

    _FALLBACK_PRUNE_THRESHOLD: float = PRIOR_ALPHA * 2.0

    
    REGRET_SINGLE_STEP_CAP: float = 2.0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Commitment chain property
    # ------------------------------------------------------------------

    @property
    def commitment_hash(self) -> str:
        """Stable task-identity hash (SHA-256 of the original intent)."""
        return self.task_identity_hash

    @commitment_hash.setter
    def commitment_hash(self, value: str) -> None:
        
        self.task_identity_hash = value
        if self.commitment_chain_hash in ("GENESIS", ""):
            self.commitment_chain_hash = value

    # ------------------------------------------------------------------
    # Bayesian belief update
    # ------------------------------------------------------------------

    def bayesian_update(self, likelihoods: Dict[str, float]) -> None:
        
        if not likelihoods:
            return

        all_states = set(self.state_probabilities) | set(likelihoods)
        new_belief: Dict[str, float] = {}
        total = 0.0

        for state in all_states:
            prior = self.state_probabilities.get(state, self.PRIOR_ALPHA)
            # IH-01 FIX: Use LIKELIHOOD_FLOOR (1e-10) as the likelihood minimum,
            # NOT PRIOR_ALPHA (0.01).  This allows contradicting evidence to drive
            # a state's posterior effectively to zero, enabling proper Bayesian
            # convergence.  PRIOR_ALPHA is only used as the prior for unseen states.
            likelihood = max(likelihoods.get(state, self.PRIOR_ALPHA), self.LIKELIHOOD_FLOOR)
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
            
            _MAX_BLEND_WEIGHT = 0.50
            _MAX_ITERATIONS = 10  # analytic w* means far fewer iterations needed
            for _ in range(_MAX_ITERATIONS):
                if _current_entropy >= self.MIN_ENTROPY_FLOOR:
                    break
                _n = len(self.state_probabilities)
                if _n == 0:
                    break
                _max_entropy = math.log(_n) if _n > 1 else 0.0
                if _max_entropy <= _current_entropy:
                    break
                # Analytic weight: fraction of the gap to uniform needed to close deficit
                _w_analytic = (self.MIN_ENTROPY_FLOOR - _current_entropy) / (
                    _max_entropy - _current_entropy
                )
                _w = min(_w_analytic, _MAX_BLEND_WEIGHT)
                # Ensure a minimum blend step to prevent stalling at tiny deficits
                _w = max(_w, 0.01)
                blended = {
                    k: (1.0 - _w) * v + _w / _n
                    for k, v in self.state_probabilities.items()
                }
                _total = sum(blended.values())
                if _total > 0.0:
                    self.state_probabilities = {
                        k: v / _total for k, v in blended.items()
                    }
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

    # ------------------------------------------------------------------
    # Entropy
    # ------------------------------------------------------------------

    def entropy(self) -> float:
        """Shannon entropy of the current belief distribution (nats)."""
        return -sum(
            p * math.log(p)
            for p in self.state_probabilities.values()
            if p > 0.0
        )

    # ------------------------------------------------------------------
    # Action scoring
    # ------------------------------------------------------------------

    def expected_utility(self, action: str) -> float:
        """
        Risk-adjusted expected utility for *action*.

        EU = mean_reward − λ · min(variance, |mean_reward|)

        The variance penalty is bounded to ``|mean_reward|`` so a
        high-variance, high-reward action is never penalised below 0.
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
        Upper Confidence Bound score for *action*.

        Returns ``+inf`` for unvisited actions (UCB initialisation
        convention; handled via dedicated tiebreak in ActionRanker).
        """
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

    
    MIN_OBS_VARIANCE: float = 0.01      # minimum observation variance floor
    ZERO_VARIANCE_WARN_THRESHOLD: int = 5  # consecutive collapses before WARNING

    def thompson_sample(self, action: str) -> float:
        
        rewards = self.action_rewards.get(action)
        if not rewards:
            return 0.0

        recent = list(rewards)[-self.THOMPSON_WINDOW :]
        n = len(recent)

        _mu0: float = 0.0
        _sigma0_sq: float = self.REWARD_CLAMP ** 2

        _welford_n = self._welford_n.get(action, 0)
        _variance_floored = False  # H5: track whether the floor was applied

        if _welford_n >= 3:
            _raw_obs_variance = (
                self._welford_M2.get(action, 0.0) / max(_welford_n - 1, 1)
            )
            # MS-T FIX: Apply minimum variance floor. Without this, constant-
            # reward actions get _raw_obs_variance=0, which collapses Thompson
            # sampling to a point estimate (std=0 -> all samples equal mean).
            if _raw_obs_variance <= self.NORMALIZE_EPS * 10:
                _obs_variance = self.MIN_OBS_VARIANCE
                _variance_floored = True
            else:
                _obs_variance = max(_raw_obs_variance, self.NORMALIZE_EPS)
        else:
            _obs_variance = _sigma0_sq

        # H5 HARDENING: Track consecutive zero-variance collapse events per action.
        if not hasattr(self, "_zero_variance_count"):
            self._zero_variance_count: dict = {}

        if _variance_floored:
            self._zero_variance_count[action] = (
                self._zero_variance_count.get(action, 0) + 1
            )
            count = self._zero_variance_count[action]
            if count == self.ZERO_VARIANCE_WARN_THRESHOLD:
                import sys as _sys
                print(
                    f"[BeliefState] WARNING H5: Thompson sampling for action "
                    f"{action!r} has hit zero-variance floor {count} consecutive "
                    f"times. Reward history is constant (Welford M2~=0). "
                    f"MIN_OBS_VARIANCE={self.MIN_OBS_VARIANCE} floor applied to "
                    f"preserve exploration diversity. This is expected for reliable "
                    f"UI actions; suppress with PROJECTZEO_DISABLE_THOMPSON_WARN=1 "
                    f"if intended.",
                    file=_sys.stderr,
                )
        else:
            # Reset counter when variance is non-zero (action became variable)
            if action in self._zero_variance_count:
                self._zero_variance_count[action] = 0

        _prior_prec = 1.0 / _sigma0_sq
        _obs_prec = n / _obs_variance
        _post_prec = _prior_prec + _obs_prec
        _post_variance = 1.0 / _post_prec

        _sum_rewards = sum(recent)
        _post_mean = _post_variance * (
            _prior_prec * _mu0 + _sum_rewards / _obs_variance
        )

        _post_mean = max(-self.REWARD_CLAMP, min(self.REWARD_CLAMP, _post_mean))

        self._sample_counter += 1
        
        _action_key_hash = hashlib.sha256(action.encode("utf-8")).hexdigest()[:8]
        seed_material = (
            f"{self.commitment_chain_hash}:{_action_key_hash}:"
            f"{self._iteration_counter}:{self._sample_counter}"
        ).encode("utf-8")
        digest = hashlib.sha256(seed_material).digest()
        seed = int.from_bytes(digest[:8], byteorder="big", signed=False)

        _rng = np.random.default_rng(seed)

        _lo = -self.REWARD_CLAMP
        _hi = self.REWARD_CLAMP
        _std = math.sqrt(_post_variance)

        sample: float = _post_mean  # fallback if rejection loop exhausts

        for _ in range(128):
            _candidate = float(_rng.normal(loc=_post_mean, scale=_std))
            if _lo <= _candidate <= _hi:
                sample = _candidate
                break

        return float(sample)

    # ------------------------------------------------------------------
    # Regret
    # ------------------------------------------------------------------

    def _get_effective_regret(self, action: str) -> float:
        """
        Return the effective (decay-adjusted) regret for *action*.

        Applies lazy exponential decay: the stored raw value is decayed
        by ``REGRET_DECAY^Δiter`` where Δiter is the number of iterations
        since the last update.  The decayed value is written back to
        avoid repeated decay of the same delta.
        """
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


    # ==========================================================
    # SEMANTIC LOOP DETECTION (NEW)
    # ==========================================================

    _SEMANTIC_LOOP_WINDOW: int = 6      # look back N actions
    _SEMANTIC_LOOP_MIN_CYCLE: int = 2   # minimum cycle length to detect

    def detect_semantic_loop(self) -> bool:
        """
        NEW FIX: Detect action-level loops where each step succeeds individually
        but no net progress is made (A→B→A→B pattern).

        Stagnation counters only catch cases where no action succeeds. This
        catches the more subtle case where actions alternate without advancing
        the task — e.g. clicking to open a menu, then clicking outside to close
        it, then clicking to open it again.

        Returns True if a repeating action cycle is detected in the recent
        action history, False otherwise.
        """
        history = getattr(self, "_recent_action_key_history", [])
        if len(history) < self._SEMANTIC_LOOP_WINDOW:
            return False

        window = history[-self._SEMANTIC_LOOP_WINDOW:]

        # Detect repeating prefix: [A, B, A, B] or [A, B, C, A, B, C]
        for cycle_len in range(
            self._SEMANTIC_LOOP_MIN_CYCLE,
            len(window) // 2 + 1,
        ):
            pattern = window[-cycle_len:]
            preceding = window[-cycle_len * 2:-cycle_len]
            if len(preceding) == cycle_len and preceding == pattern:
                return True

        return False

    def record_action_key_for_loop_detection(self, action_key: str) -> None:
        """Record an action key for semantic loop detection."""
        if not hasattr(self, "_recent_action_key_history"):
            self._recent_action_key_history = []
        self._recent_action_key_history.append(action_key)
        # Bound to 2× window size
        max_size = self._SEMANTIC_LOOP_WINDOW * 2
        if len(self._recent_action_key_history) > max_size:
            self._recent_action_key_history = self._recent_action_key_history[-max_size:]

    def set_plan_horizon(self, total_steps: int, iters_per_step: int = 13) -> None:
        """
        Tune regret decay so that 5 % residual remains at plan end.

        Calibration: given ``T = total_steps × iters_per_step`` total
        iterations and a target residual fraction ``f = 0.05``,
        ``decay = f^(1/T)``.

        Parameters
        ----------
        total_steps:
            Number of non-DONE steps in the plan.
        iters_per_step:
            Expected iterations per step (default 13 — empirically tuned).
        """
        total_iters = max(total_steps * iters_per_step, 1)
        target_fraction = 0.05
        self._regret_decay = target_fraction ** (1.0 / total_iters)

    def update_regret(
        self, action: str, reward: float, best_reward: float
    ) -> None:
        
        
        best_reward = min(best_reward, 0.9)

        regret_value = best_reward - reward
        if regret_value <= 0.0:
            return
        regret_value = min(regret_value, self.REGRET_SINGLE_STEP_CAP)

        current = self._get_effective_regret(action)
        updated = min(current + regret_value, self.MAX_REGRET)
        self.regret[action] = (updated, self._iteration_counter)

    # ------------------------------------------------------------------
    # Record action
    # ------------------------------------------------------------------

    def record_action(self, action: str, reward: float) -> None:
        
        _chain_input = f"{self.commitment_chain_hash}:{action}".encode("utf-8")
        self.commitment_chain_hash = hashlib.sha256(_chain_input).hexdigest()

        
        self._iteration_counter += 1

        if action not in self.action_rewards:
            self.action_rewards[action] = deque(maxlen=self.REWARD_WINDOW)
        if action not in self._raw_action_rewards:
            self._raw_action_rewards[action] = deque(maxlen=self.REWARD_WINDOW)

        reward = max(self.RAW_REWARD_MIN, min(self.RAW_REWARD_MAX, float(reward)))
        self._raw_action_rewards[action].append(reward)

        n = self._welford_n.get(action, 0) + 1
        self._welford_n[action] = n

        
        if n > self.REWARD_WINDOW:
            _window = list(self._raw_action_rewards[action])
            if len(_window) > 0:
                _w_n = len(_window)
                _w_mean = sum(_window) / _w_n
                _w_M2 = sum((r - _w_mean) ** 2 for r in _window)
                self._welford_n[action] = _w_n
                self._welford_mean[action] = _w_mean
                self._welford_M2[action] = _w_M2
                n = _w_n
                new_mean = _w_mean
                new_M2 = _w_M2
            else:
                new_mean = self._welford_mean.get(action, 0.0)
                new_M2 = self._welford_M2.get(action, 0.0)
        else:
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
                # MF-4 FIX: Retroactive re-normalisation at the n=3 boundary.
                # All three entries must be z-scored around the *same*
                # 3-sample mean (new_mean) to avoid a distribution
                # discontinuity between entries 0–1 (previously normalised
                # around the 2-sample mean) and entry 2.
                raw_deque = self._raw_action_rewards[action]
                if len(raw_deque) >= 2:
                    old_entries = list(raw_deque)[:2]
                    renormalized = []
                    for r in old_entries:
                        z = (r - new_mean) / std
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

    # ------------------------------------------------------------------
    # Convenience mutators
    # ------------------------------------------------------------------

    def flush_regret_on_success(self) -> None:
        """Clear all accumulated regret on task success."""
        self.regret.clear()

    

    def reset_for_new_task(self, intent_hash: str = "") -> None:
        """
        Reset transient fields for a new task while preserving the
        class-level constants.

        Does **not** reset action history (counts, rewards, Welford
        stats) — those survive task boundaries to accelerate warm-start.
        """
        self._iteration_counter = 0
        self._sample_counter = 0
        self._regret_decay = self.REGRET_DECAY
        self.regret.clear()
        self.progress_score = 0.0

        
        if hasattr(self, "_zero_variance_count"):
            self._zero_variance_count.clear()

        new_identity = (
            hashlib.sha256(intent_hash.encode("utf-8")).hexdigest()
            if intent_hash
            else "GENESIS"
        )
        self.task_identity_hash = new_identity
        self.commitment_chain_hash = new_identity

    # ------------------------------------------------------------------
    # Global best reward
    # ------------------------------------------------------------------

    def global_best_reward(self) -> "Optional[float]":
        """
        Return the highest raw reward ever observed across all actions,
        or ``None`` if no actions have been recorded yet.

        .. warning::
           The DONE sentinel returns reward = 1.0.  After any DONE
           sub-goal this method permanently returns 1.0.  **Callers must
           cap the result at 0.9** before using it as the reference
           ``best_reward`` in ``update_regret()`` to prevent DONE from
           inflating regret for all subsequent actions (RT-05 / SI-04).
           See ``operate.py`` for the canonical cap.
        """
        best: "Optional[float]" = None
        for history in self._raw_action_rewards.values():
            if history:
                local_max = max(history)
                if best is None or local_max > best:
                    best = local_max
        return best

    # ------------------------------------------------------------------
    # Environment stability
    # ------------------------------------------------------------------

    def compute_environment_stability(self, delta: Dict[str, Any]) -> None:
        """
        Update ``environment_stability`` based on a world-graph delta.

        A ``significant_change`` flag in *delta* drops stability by 0.2
        (floor 0.0) and resets the consecutive-high-stability counter.
        Absence of significant change raises stability by 0.05 (cap 1.0)
        and increments the counter when stability ≥ 0.8.
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

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """
        Return a compact summary dict suitable for LLM context injection.
        """
        top_states = dict(
            sorted(
                self.state_probabilities.items(), key=lambda x: x[1], reverse=True
            )[:5]
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

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise the entire BeliefState to a JSON-safe dictionary.

        Every field written here must have a corresponding read in
        ``from_dict()`` to guarantee round-trip fidelity.
        """
        return {
            "state_probabilities": dict(self.state_probabilities),
            "progress_score": self.progress_score,
            "environment_stability": self.environment_stability,
            "consecutive_high_stability_count": self.consecutive_high_stability_count,
            # Canonical name for the task-identity hash; read back via
            # the raw attribute path in from_dict() (not the property setter).
            "commitment_hash": self.task_identity_hash,
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
            # IH-03 FIX: _zero_variance_count must survive crash-recovery.
            # Without this field, after a restart the counter resets to 0 and
            # the ZERO_VARIANCE_WARN_THRESHOLD can never fire even if the action
            # had already collapsed variance before the restart.
            "_zero_variance_count": dict(
                getattr(self, "_zero_variance_count", {})
            ),
        }

    @classmethod
    def from_dict(cls, data: dict, *, intent_hash: str = "") -> "BeliefState":
        """
        Reconstruct a ``BeliefState`` from a previously serialised dict.

        RT-01 FIX — AttributeError in property setter
        -----------------------------------------------
        The prior code called::

            instance.commitment_hash = str(data.get("commitment_hash", "GENESIS"))

        This invoked the ``commitment_hash`` property setter which reads
        ``self.commitment_chain_hash`` (to check for the GENESIS sentinel)
        *before* that attribute has been assigned on the freshly
        ``__new__``-ed instance → ``AttributeError``.  The outer
        ``except Exception`` silently swallowed the error and returned a
        fresh (GENESIS-state) instance, discarding every persisted field.

        Fix: assign ``task_identity_hash`` and ``commitment_chain_hash``
        **directly** as plain instance attributes — bypassing the property
        setter entirely — immediately after the instance is created.  The
        property setter is never called during deserialisation.

        Parameters
        ----------
        data:
            Dictionary produced by ``to_dict()``.
        intent_hash:
            Original intent string used to seed the task-identity hash.
            Provided as a consistency cross-check; the persisted
            ``commitment_hash`` field takes precedence.

        Returns
        -------
        BeliefState
            A fully restored instance, or a fresh ``BeliefState(intent_hash)``
            if *data* is malformed or an unexpected error occurs.
        """
        if not isinstance(data, dict):
            return cls(intent_hash=intent_hash)

        try:
            instance: "BeliefState" = cls.__new__(cls)

            

            _task_identity: str = str(data.get("commitment_hash", "GENESIS"))
            _chain_hash: str = str(
                data.get("commitment_chain_hash", _task_identity)
            )

            # Direct attribute writes — no property setter involved.
            instance.task_identity_hash = _task_identity
            instance.commitment_chain_hash = _chain_hash

            # ----------------------------------------------------------
            # Remaining fields (order does not matter now that the
            # commitment identity fields are already set)
            # ----------------------------------------------------------

            instance.created_at = float(data.get("created_at", 0.0) or 0.0)

            raw_probs = data.get("state_probabilities", {"neutral": 1.0})
            instance.state_probabilities = (
                dict(raw_probs)
                if isinstance(raw_probs, dict)
                else {"neutral": 1.0}
            )
            instance.progress_score = float(data.get("progress_score", 0.0))
            instance.environment_stability = float(
                data.get("environment_stability", 1.0)
            )
            instance.consecutive_high_stability_count = int(
                data.get("consecutive_high_stability_count", 0)
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

            # IH-03 FIX: Restore _zero_variance_count so ZERO_VARIANCE_WARN_THRESHOLD
            # fires correctly after crash-recovery, not only on fresh runs.
            instance._zero_variance_count = {
                k: int(v)
                for k, v in data.get("_zero_variance_count", {}).items()
            }

            return instance

        except Exception as _exc:
            # Structured fallback: log the root cause so that silent
            # state loss (as in the original RT-01 bug) is surfaced.
            import sys as _sys
            print(
                f"[BeliefState.from_dict] WARNING: deserialization failed "
                f"({type(_exc).__name__}: {_exc}). "
                "Returning fresh BeliefState. "
                "This indicates data corruption or a schema mismatch.",
                file=_sys.stderr,
            )
            return cls(intent_hash=intent_hash)
