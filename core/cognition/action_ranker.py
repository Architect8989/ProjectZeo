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

    
    EXPLORATION_FLOOR_COUNT: int = 3
    EXPLORATION_FLOOR_BONUS: float = 0.1

    # MATH-NEW-02 FIX: Default saturation; tuned by set_plan_horizon().
    _DEFAULT_EXPLOIT_SATURATION_N = 50

    def __init__(self) -> None:
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

            # SI-05 FIX: Apply exploration floor for underexplored actions.
            # When count < EXPLORATION_FLOOR_COUNT, guarantee a minimum bonus of
            # EXPLORATION_FLOOR_BONUS so that underexplored actions always receive
            # meaningful exploration credit regardless of mean_reward magnitude.
            # FIX MATH-05: clamp to [MIN_EXPLORATION_BONUS, MAX_EXPLORATION_BONUS].
            exploration_bonus = ucb_full - mean_reward
            if n < self.EXPLORATION_FLOOR_COUNT:
                exploration_bonus = max(exploration_bonus, self.EXPLORATION_FLOOR_BONUS)
            exploration_bonus = max(
                self.MIN_EXPLORATION_BONUS,
                min(exploration_bonus, self.MAX_EXPLORATION_BONUS),
            )

            thompson = belief_state.thompson_sample(key)

            n = belief_state.action_counts.get(key, 0)
            _t = min(1.0, n / self._exploit_saturation_n)

            w_thompson = 0.25 - 0.20 * _t
            w_explore = 0.25 - 0.05 * _t
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
            
            selected_index = self._deterministic_tiebreak(
                candidates, belief_state
            )

        return actions[selected_index]

    def _deterministic_tiebreak(
        self,
        candidates: List[int],
        belief_state,
    ) -> int:
        """
        RT-08 FIX: Return a deterministic index into candidates using the
        commitment_chain_hash without risking ValueError on non-hex sentinels.

        Precedence:
          1. "GENESIS" sentinel → index 0 (no actions recorded yet; stable)
          2. Valid hex string (≥16 chars) → bytes.fromhex first 8 bytes → int
          3. Non-hex / short string → SHA-256 of the raw string → int

        The modulo ensures the result is always a valid index into candidates.
        """
        if len(candidates) == 1:
            return candidates[0]

        _chain = (
            getattr(belief_state, "commitment_chain_hash", None)
            or getattr(belief_state, "commitment_hash", "")
        )

        if not _chain or _chain == "GENESIS":
            # No actions recorded yet: deterministic first-element selection.
            return candidates[0]

        try:
            # Fast path: valid 64-char hex SHA-256 digest (normal operating state).
            # Use first 8 bytes (16 hex chars) for the integer.
            if len(_chain) >= 16 and all(c in "0123456789abcdefABCDEF" for c in _chain[:16]):
                _hash_int = int.from_bytes(bytes.fromhex(_chain[:16]), "big")
            else:
                # Fallback path: hash the chain string itself to get a stable int.
                # Covers partial-hex strings, short sentinels, or any future
                # non-hex chain formats.
                _hash_int = int.from_bytes(
                    hashlib.sha256(_chain.encode("utf-8")).digest()[:8], "big"
                )
            return candidates[_hash_int % len(candidates)]
        except Exception:
            # Last-resort: first candidate — never raises, always deterministic.
            return candidates[0]

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
