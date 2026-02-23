# core/cognition/action_ranker.py

from typing import List, Dict, Any
import math
import hashlib
import json


class ActionRanker:
    """
    Deterministic action selector.

    Guarantees:
    - No randomness: tie-breaking is derived from belief_state.commitment_hash
      (a SHA-256 chain seeded per intent), not from Python's process-level RNG.
    - Stable action identity
    - Single Thompson source (delegated to BeliefState)
    - Bounded exploration bonus (both above AND below zero)

    FIX HARD-1 (SI-4 / §3.2): The previous implementation used random.choice()
    for tie-breaking, which was non-deterministic across runs. The docstring
    claimed "No randomness" while the code comments admitted "Uniform random
    tie-breaking" — a direct contradiction. Tie-breaking is now derived from
    commitment_hash bytes, making action selection fully reproducible for
    identical inputs.

    FIX MATH-05: exploration_bonus was previously only clamped at the top
    (min(bonus, MAX_EXPLORATION_BONUS)). When ucb_full < mean_reward (which
    happens for well-explored actions whose UCB exploration term has shrunk
    below the mean), exploration_bonus became negative and unclamped, applying
    a significant penalty to well-known high-performing actions. This
    suppressed exploitation of the best-known action in favour of less-explored
    alternatives — the opposite of correct UCB behaviour.

    Fix: clamp exploration_bonus to [0.0, MAX_EXPLORATION_BONUS]. A UCB
    score below the mean reward simply means "no additional exploration
    incentive" — it should never penalise the action. Zero is the correct
    floor.
    """

    MIN_TAU = 0.15
    MAX_TAU = 1.5
    MAX_EXPLORATION_BONUS = 1.0  # prevent UCB domination
    MIN_EXPLORATION_BONUS = 0.0  # FIX MATH-05: prevent exploitation suppression

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

            combined = (
                0.6 * eu
                + 0.2 * exploration_bonus
                + 0.2 * thompson
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

        # HARD-1 (SI-4 / §3.2): Deterministic tie-breaking among equally-scoring
        # actions, derived from belief_state.commitment_hash.
        #
        # Bug: random.choice(candidates) uses Python's process-level random module,
        # which is NOT seeded deterministically. The class docstring claims "No
        # randomness" and prior code comments stated "Uniform random tie-breaking"
        # — a direct contradiction. Across runs with identical belief state and
        # action set, different action selections were produced whenever multiple
        # candidates tied for maximum softmax probability (common at task start
        # when all actions are unseen). This made the commitment chain diverge
        # from run to run, breaking post-hoc replay.
        #
        # Fix: derive the tie-breaking index from commitment_hash, which is itself
        # a deterministic SHA-256 chain seeded per intent. For identical inputs
        # (same intent, same action sequence, same world state), commitment_hash
        # is identical, and therefore the same candidate is selected. The selection
        # rotates predictably as the chain advances — no process-level RNG needed.
        #
        # Implementation: take the first 8 bytes of commitment_hash as a big-endian
        # uint64, then index into candidates via modulo. This preserves uniform
        # coverage over candidates (no modulo bias for candidate lists ≤ 2^32).
        max_prob = max(probabilities)
        candidates = [
            i for i, p in enumerate(probabilities)
            if abs(p - max_prob) < 1e-9
        ]
        if len(candidates) == 1:
            selected_index = candidates[0]
        else:
            # Deterministic index derived from commitment_hash.
            # Falls back to index 0 if commitment_hash is unavailable or malformed.
            try:
                _hash_bytes = bytes.fromhex(belief_state.commitment_hash[:16])
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
