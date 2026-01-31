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
import re

from core.telemetry.logger import log_info, log_warn


class DecompositionError(RuntimeError):
    pass


class TaskDecomposer:
    """
    Stateless planner.
    Deterministic interface.
    """

    MAX_STEPS = 50
    MAX_RAW_CHARS = 20_000
    MAX_RETRIES = 2

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
        if not isinstance(objective, str) or not objective.strip():
            raise DecompositionError("Objective must be non-empty string")

        prompt = (
            self.SYSTEM_PROMPT.strip()
            + "\n\nOBJECTIVE:\n"
            + objective.strip()
        )

        last_error = None

        for attempt in range(self.MAX_RETRIES + 1):

            raw = self.llm_call(prompt)

            if not isinstance(raw, str):
                last_error = "Planner returned non-string"
                continue

            raw = self._sanitize(raw)

            if len(raw) > self.MAX_RAW_CHARS:
                last_error = "Planner output too large"
                continue

            try:
                parsed = self._safe_json_extract(raw)
                steps = parsed["steps"]
            except Exception as e:
                last_error = str(e)
                continue

            self._validate_steps(steps)

            if len(steps) > self.MAX_STEPS:
                log_warn(
                    f"[PLANNER] Truncating steps {len(steps)} -> {self.MAX_STEPS}"
                )
                steps = steps[: self.MAX_STEPS]

            log_info(f"[PLANNER] Generated {len(steps)} steps")
            return steps

        raise DecompositionError(
            f"Planner failed after retries: {last_error}"
        )

    # -----------------------------------------------------
    # Internal Guards
    # -----------------------------------------------------

    def _sanitize(self, text: str) -> str:
        # remove nulls and control chars
        return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text).strip()

    def _safe_json_extract(self, text: str) -> Dict:
        """
        Extract first JSON object if model wrapped output.
        """
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise DecompositionError("No JSON object found")

        return json.loads(match.group(0))

    def _validate_steps(self, steps: List[Dict]) -> None:
        if not isinstance(steps, list):
            raise DecompositionError("steps must be list")

        if len(steps) == 0:
            raise DecompositionError("no steps produced")

        prev_id = 0

        for s in steps:
            if not isinstance(s, dict):
                raise DecompositionError("step must be object")

            if "id" not in s or "goal" not in s:
                raise DecompositionError("Invalid step schema")

            if not isinstance(s["goal"], str):
                raise DecompositionError("goal must be string")

            if not isinstance(s["id"], int):
                raise DecompositionError("id must be integer")

            if s["id"] <= prev_id:
                raise DecompositionError("steps not ordered")

            prev_id = s["id"]
