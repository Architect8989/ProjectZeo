# core/cognition/reasoning_engine.py

from typing import Dict, Any, List


class ReasoningEngine:
    """
    Multi-hypothesis LLM reasoning.
    """

    def __init__(self, llm_callable):
        self._llm = llm_callable

    def propose_actions(
        self,
        *,
        objective: str,
        belief_summary: Dict[str, Any],
        perception: Dict[str, Any],
        k: int = 3,
    ) -> List[Dict[str, Any]]:

        prompt = {
            "role": "user",
            "content": f"""
Objective: {objective}

Belief summary:
{belief_summary}

Perception:
{perception}

Propose {k} possible next actions.
Return JSON list of actions.
"""
        }

        result = self._llm(
            messages=[prompt],
            objective=objective,
            session_id="cognition",
        )

        if isinstance(result, list):
            return result

        if isinstance(result, dict) and "actions" in result:
            return result["actions"]

        return []
