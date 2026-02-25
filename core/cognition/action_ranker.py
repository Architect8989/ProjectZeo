# core/cognition/action_ranker.py

from typing import List, Dict, Any
import math
import hashlib
import json


class ActionRanker:


    MIN_TAU = 0.15
    MAX_TAU = 1.5
    MAX_EXPLORATION_BONUS = 1.0  # prevent UCB domination
    MIN_EXPLORATION_BONUS = 0.0  # FIX MATH-05: prevent exploitation suppression

    # MATH-NEW-02 FIX: Default saturation; tuned by set_plan_horizon().
    # See set_plan_horizon() docstring for rationale.
    _DEFAULT_EXPLOIT_SATURATION_N = 50

    def __init__(self) -> None:
        # MATH-NEW-02 FIX: _exploit_saturation_n is an instance attribute so
        # set_plan_horizon() can tune it per-task without affecting other
        # ActionRanker instances (e.g. concurrent replan scenarios).
        self._exploit_saturation_n: int = self._DEFAULT_EXPLOIT_SATURATION_N

    def set_plan_horizon(self, total_steps: int) -> None:
        
        self._exploit_saturation_n = max(10, total_steps * 2)

    # ==================================================
    # ACTION SELECTION (DETERMINISTIC except for tie-breaking)
    # ==================================================

    def select(
        self,
        actions: List[Dict[str, Any]],
        belief_state,
    ) -> Dict[str, Any]:

        if not actions:
            raise RuntimeError("No candidate actions")

        action_ids = [self.action_key(a) for a in actions]
        scores = []

        for action, key in zip(actions, action_ids):

            eu = belief_state.expected_utility(key)

            ucb_full = belief_state.ucb_score(key)

            rewards = belief_state.action_rewards.get(key)
            mean_reward = (
                sum(rewards) / len(rewards)
                if rewards else 0.0
            )

            # FIX MATH-05: clamp to [MIN_EXPLORATION_BONUS, MAX_EXPLORATION_BONUS].
            # When ucb_full < mean_reward (fully-exploited action), the raw
            # bonus is negative, incorrectly penalising actions that already
            # have a good track record. Clamping at zero means "no extra
            # exploration incentive" — not "this action is worse than we know."
            exploration_bonus = ucb_full - mean_reward
            exploration_bonus = max(
                self.MIN_EXPLORATION_BONUS,
                min(exploration_bonus, self.MAX_EXPLORATION_BONUS),
            )

            # Delegated Thompson (single coherent implementation)
            thompson = belief_state.thompson_sample(key)

            
            n = belief_state.action_counts.get(key, 0)
            _t = min(1.0, n / self._exploit_saturation_n)  # 0.0 (new) → 1.0 (mature)

            # Thompson shifts from 0.25 → 0.05 as action matures
            w_thompson = 0.25 - 0.20 * _t
            # Explore stays roughly constant (slight decay)
            w_explore = 0.25 - 0.05 * _t
            # EU absorbs the rest — grows from 0.50 → 0.90
            w_eu = 1.0 - w_thompson - w_explore

            combined = (
                w_eu * eu
                + w_explore * exploration_bonus
                + w_thompson * thompson
            )

            scores.append(combined)

        # --------------------------------------------------
        # Entropy-adaptive temperature
        # --------------------------------------------------

        entropy = belief_state.entropy()
        tau = self._entropy_temperature(entropy)

        # --------------------------------------------------
        # Deterministic softmax (argmax of probabilities)
        # --------------------------------------------------

        max_score = max(scores)
        shifted = [(s - max_score) / tau for s in scores]

        # Deterministic softmax — shift ensures max shifted value is 0.0,
        # so exp(s) ≤ 1 for all s. The min(50.0, s) clamp that was here is
        # dead code (MATH-05) and has been removed to avoid confusion.
        exp_scores = [math.exp(s) for s in shifted]
        total = sum(exp_scores)

        if total <= 0.0:
            return actions[scores.index(max_score)]

        probabilities = [s / total for s in exp_scores]

        
        max_prob = max(probabilities)
        candidates = [
            i for i, p in enumerate(probabilities)
            if abs(p - max_prob) < 1e-9
        ]
        if len(candidates) == 1:
            selected_index = candidates[0]
        else:
            
            try:
                _chain = getattr(belief_state, "commitment_chain_hash", None) \
                         or getattr(belief_state, "commitment_hash", "")
                _hash_bytes = bytes.fromhex(_chain[:16])
                _hash_int = int.from_bytes(_hash_bytes, "big")
                selected_index = candidates[_hash_int % len(candidates)]
            except Exception:
                selected_index = candidates[0]

        return actions[selected_index]

    # ==================================================
    # ENTROPY TEMPERATURE
    # ==================================================

    def _entropy_temperature(self, entropy: float) -> float:
        tau = math.tanh(entropy) * self.MAX_TAU
        return max(self.MIN_TAU, min(self.MAX_TAU, tau))

    # ==================================================
    # STABLE ACTION KEY
    # ==================================================

    def action_key(self, action: Dict[str, Any]) -> str:
        """
        Stable identity based only on logical operation
        and canonical essential fields.

        Prevents history invalidation if extra metadata
        fields are added to the action dict.
        """

        operation = str(action.get("operation", "")).strip()

        # canonical minimal identity subset
        canonical_subset = {
            "operation": operation,
            "target": action.get("target"),
            "text": action.get("text"),
            "keys": action.get("keys"),
        }

        canonical = json.dumps(
            canonical_subset,
            sort_keys=True,
            default=str,
        )

        return hashlib.sha256(
            canonical.encode()
        ).hexdigest()[:16]
