# core/cognition/action_ranker.py

from typing import List, Dict, Any


class ActionRanker:
    """
    Scores candidate actions based on belief state.
    """

    def rank(
        self,
        actions: List[Dict[str, Any]],
        belief_summary: Dict[str, Any],
    ) -> List[Dict[str, Any]]:

        scored = []

        for action in actions:
            score = 1.0

            # penalize repeated failed patterns
            for failure in belief_summary.get("recent_failures", []):
                if failure["action"] == action:
                    score *= 0.5

            scored.append((score, action))

        scored.sort(reverse=True, key=lambda x: x[0])

        return [a for _, a in scored]
