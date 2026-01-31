"""
High-Level Task Decomposer

Purpose:
Transforms a raw human objective into a structured,
ordered execution plan suitable for KernelController.

This module:
- Does NOT execute
- Does NOT touch OS
- Does NOT call screen
- Does NOT assume UI

It produces abstract action skeletons that the
existing LLM executor later grounds into UI actions.
"""

from typing import List, Dict
import json

from core.telemetry.logger import log_info, log_warn


class DecompositionError(RuntimeError):
    pass


class TaskDecomposer:
    """
    Stateless planner.
    Deterministic interface.
    """

    MAX_STEPS = 50

    SYSTEM_PROMPT = """
You are a software planning engine.

You do NOT control a computer.
You do NOT describe UI clicks.
You ONLY decompose objectives into high-level steps.

Return STRICT JSON.

Schema:

{
  "steps": [
     {"id": 1, "goal": "short description"},
     {"id": 2, "goal": "short description"}
  ]
}

Rules:
- Steps must be ordered
- Each step must be independently verifiable
- No UI references
- No mouse, keyboard, click, type, open
- No commentary
- No markdown
"""

    def __init__(self, llm_call):
        """
        llm_call(prompt:str)->str
        """
        self.llm_call = llm_call

    # -----------------------------------------------------

    def decompose(self, objective: str) -> List[Dict]:
        """
        Returns list of {id, goal}
        """

        prompt = (
            self.SYSTEM_PROMPT.strip()
            + "\n\nOBJECTIVE:\n"
            + objective.strip()
        )

        raw = self.llm_call(prompt)

        try:
            parsed = json.loads(raw)
            steps = parsed["steps"]
        except Exception:
            raise DecompositionError(
                "Planner did not return valid JSON"
            )

        if not isinstance(steps, list):
            raise DecompositionError("steps must be list")

        if len(steps) == 0:
            raise DecompositionError("no steps produced")

        if len(steps) > self.MAX_STEPS:
            log_warn(
                f"[PLANNER] Truncating steps {len(steps)} -> {self.MAX_STEPS}"
            )
            steps = steps[: self.MAX_STEPS]

        for s in steps:
            if "id" not in s or "goal" not in s:
                raise DecompositionError("Invalid step schema")

            if not isinstance(s["goal"], str):
                raise DecompositionError("goal must be string")

        log_info(f"[PLANNER] Generated {len(steps)} steps")

        return steps
