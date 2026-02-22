# core/cognition/action_ranker.py

from typing import List, Dict, Any
import math
import random
import hashlib
import json


class ActionRanker:
    """
    Deterministic action selector.

    Guarantees:
    - No randomness
    - Stable action identity
    - Single Thompson source (delegated to BeliefState)
    - Bounded exploration bonus (both above AND below zero)

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

        # MR-03 FIX: Uniform random tie-breaking among equally-scoring actions.
        # ─────────────────────────────────────────────────────────────────
        # Bug: max(range(len(actions)), key=lambda i: probabilities[i]) uses
        # Python's max() which returns the LAST maximum when multiple actions
        # share the highest probability. In the common case where all actions
        # are unseen (probabilities are equal), this always picks the last
        # action in the list — biasing exploration toward the end of whatever
        # ordering the planner happened to produce.
        #
        # Fix: collect all maximally-scoring actions and choose uniformly at
        # random. This gives true uniform exploration when no history exists
        # and preserves correct exploitation when one action is clearly better.
        # ─────────────────────────────────────────────────────────────────
        max_prob = max(probabilities)
        candidates = [
            i for i, p in enumerate(probabilities)
            if abs(p - max_prob) < 1e-9
        ]
        selected_index = random.choice(candidates)

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
