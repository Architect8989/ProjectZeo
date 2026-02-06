"""
High-Level Task Decomposer

Purpose:
Transforms a raw human objective into structured,
ordered high-level goals suitable for ExecutionPlanner.

This module:
- DOES NOT execute
- DOES NOT touch OS
- DOES NOT emit UI actions
- DOES NOT construct ExecutionPlan
- DOES NOT invent step types

It produces ONLY abstract goals.
"""

from typing import List, Dict
import json
import re

from core.telemetry.logger import log_info, log_warn


class DecompositionError(RuntimeError):
    pass


class TaskDecomposer:
    """
    Stateless, deterministic decomposer.
    """

    MAX_STEPS = 50
    MAX_RAW_CHARS = 20_000
    MAX_RETRIES = 2

    SYSTEM_PROMPT = """
You are a task decomposition engine.

You do NOT execute tasks.
You do NOT describe UI actions.
You do NOT mention mouse, keyboard, click, type, open.
You ONLY break an objective into ordered, high-level goals.

Return STRICT JSON ONLY.

Schema:
{
  "steps": [
     {"id": 1, "goal": "short, concrete, verifiable goal"},
     {"id": 2, "goal": "short, concrete, verifiable goal"}
  ]
}

Rules:
- Steps must be ordered
- Each step must be independently verifiable
- No UI references
- No commentary
- No markdown
"""

    def __init__(self, llm_call):
        self.llm_call = llm_call

    # -------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------

    def decompose(self, objective: str) -> List[Dict[str, str]]:
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
                last_error = "LLM returned non-string"
                continue

            raw = self._sanitize(raw)

            if len(raw) > self.MAX_RAW_CHARS:
                last_error = "LLM output too large"
                continue

            try:
                parsed = self._safe_json_extract(raw)
                steps = parsed.get("steps")
                self._validate_steps(steps)
            except Exception as e:
                last_error = str(e)
                continue

            if len(steps) > self.MAX_STEPS:
                log_warn(
                    f"[DECOMPOSER] Truncating steps {len(steps)} → {self.MAX_STEPS}"
                )
                steps = steps[: self.MAX_STEPS]

            log_info(f"[DECOMPOSER] Produced {len(steps)} goals")
            return steps

        raise DecompositionError(
            f"Task decomposition failed after retries: {last_error}"
        )

    # -------------------------------------------------
    # Internal guards
    # -------------------------------------------------

    def _sanitize(self, text: str) -> str:
        return re.sub(
            r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text
        ).strip()

    def _safe_json_extract(self, text: str) -> Dict:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise DecompositionError("No JSON object found")
        return json.loads(match.group(0))

    def _validate_steps(self, steps: List[Dict]) -> None:
        if not isinstance(steps, list) or not steps:
            raise DecompositionError("steps must be non-empty list")

        prev_id = 0

        for s in steps:
            if not isinstance(s, dict):
                raise DecompositionError("step must be object")

            if "id" not in s or "goal" not in s:
                raise DecompositionError("step missing id or goal")

            if not isinstance(s["id"], int):
                raise DecompositionError("id must be integer")

            if not isinstance(s["goal"], str) or not s["goal"].strip():
                raise DecompositionError("goal must be non-empty string")

            if s["id"] <= prev_id:
                raise DecompositionError("steps not strictly ordered")

            prev_id = s["id"]
