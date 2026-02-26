from typing import List, Dict, Any
import math
import hashlib
import json


class ActionRanker:
    MIN_TAU: float = 0.15
    MAX_TAU: float = 1.5

    MAX_EXPLORATION_BONUS: float = 1.0
    MIN_EXPLORATION_BONUS: float = 0.0

    EXPLORATION_FLOOR_COUNT: int = 3
    EXPLORATION_FLOOR_BONUS: float = 0.1

    _DEFAULT_EXPLOIT_SATURATION_N: int = 50
    _UNVISITED_SCORE_SENTINEL: float = 1e9

    def __init__(self) -> None:
        self._exploit_saturation_n: int = self._DEFAULT_EXPLOIT_SATURATION_N

    def set_plan_horizon(self, total_steps: int) -> None:
        self._exploit_saturation_n = max(10, total_steps * 2)

    def select(
        self,
        actions: List[Dict[str, Any]],
        belief_state,
    ) -> Dict[str, Any]:

        if not actions:
            raise RuntimeError(
                "ActionRanker.select(): received empty action list — "
                "caller must provide at least one candidate action."
            )

        action_ids = [self.action_key(a) for a in actions]
        scores: List[float] = []

        _reward_clamp = float(getattr(belief_state, "REWARD_CLAMP", 3.0))
        if _reward_clamp <= 0.0:
            _reward_clamp = 3.0

        for action, key in zip(actions, action_ids):

            n: int = belief_state.action_counts.get(key, 0)

            eu: float = belief_state.expected_utility(key)
            ucb_full: float = belief_state.ucb_score(key)

            rewards = belief_state.action_rewards.get(key)
            mean_reward_raw: float = (
                sum(rewards) / len(rewards) if rewards else 0.0
            )

            mean_reward_01: float = (
                (mean_reward_raw + _reward_clamp) / (2.0 * _reward_clamp)
            )
            mean_reward_01 = max(0.0, min(1.0, mean_reward_01))

            exploration_bonus: float = ucb_full - mean_reward_01

            if n < self.EXPLORATION_FLOOR_COUNT:
                exploration_bonus = max(exploration_bonus, self.EXPLORATION_FLOOR_BONUS)

            if not math.isinf(exploration_bonus):
                exploration_bonus = max(
                    self.MIN_EXPLORATION_BONUS,
                    min(exploration_bonus, self.MAX_EXPLORATION_BONUS),
                )

            thompson: float = belief_state.thompson_sample(key)

            _t: float = min(1.0, n / self._exploit_saturation_n)
            w_thompson: float = 0.25 - 0.20 * _t
            w_explore: float = 0.25 - 0.05 * _t
            w_eu: float = 1.0 - w_thompson - w_explore

            combined: float = (
                w_eu * eu
                + w_explore * exploration_bonus
                + w_thompson * thompson
            )

            scores.append(combined)

        inf_indices: List[int] = [
            i for i, s in enumerate(scores)
            if math.isinf(s) and s > 0.0
        ]

        if inf_indices:
            selected_index = self._deterministic_tiebreak(inf_indices, belief_state)
            return actions[selected_index]

        min_finite: float = min(
            (s for s in scores if not math.isnan(s)),
            default=0.0,
        )
        scores = [s if not math.isnan(s) else min_finite for s in scores]

        entropy: float = belief_state.entropy()
        tau: float = self._entropy_temperature(entropy)

        max_score: float = max(scores)

        if all(abs(s - max_score) < 1e-12 for s in scores):
            all_indices = list(range(len(actions)))
            return actions[self._deterministic_tiebreak(all_indices, belief_state)]

        shifted: List[float] = [(s - max_score) / tau for s in scores]
        exp_scores: List[float] = [math.exp(max(-500.0, s)) for s in shifted]
        total: float = sum(exp_scores)

        if total <= 0.0 or math.isnan(total):
            return actions[scores.index(max_score)]

        probabilities: List[float] = [s / total for s in exp_scores]

        max_prob: float = max(probabilities)
        candidates: List[int] = [
            i for i, p in enumerate(probabilities)
            if abs(p - max_prob) < 1e-9
        ]

        if not candidates:
            return actions[scores.index(max_score)]

        if len(candidates) == 1:
            selected_index = candidates[0]
        else:
            selected_index = self._deterministic_tiebreak(candidates, belief_state)

        return actions[selected_index]

    def _deterministic_tiebreak(
        self,
        candidates: List[int],
        belief_state,
    ) -> int:

        if not candidates:
            return 0

        if len(candidates) == 1:
            return candidates[0]

        _chain: str = (
            getattr(belief_state, "commitment_chain_hash", None)
            or getattr(belief_state, "commitment_hash", "")
            or ""
        )

        if not _chain or _chain == "GENESIS":
            return candidates[0]

        try:
            if (
                len(_chain) >= 16
                and all(c in "0123456789abcdefABCDEF" for c in _chain[:16])
            ):
                _hash_int = int.from_bytes(bytes.fromhex(_chain[:16]), "big")
            else:
                _hash_int = int.from_bytes(
                    hashlib.sha256(_chain.encode("utf-8")).digest()[:8],
                    "big",
                )
            return candidates[_hash_int % len(candidates)]
        except Exception:
            return candidates[0]

    def _entropy_temperature(self, entropy: float) -> float:
        entropy = max(0.0, float(entropy))
        tau = math.tanh(entropy) * self.MAX_TAU
        return max(self.MIN_TAU, min(self.MAX_TAU, tau))

    def action_key(self, action: Dict[str, Any]) -> str:
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

        return hashlib.sha256(canonical.encode()).hexdigest()[:16]
