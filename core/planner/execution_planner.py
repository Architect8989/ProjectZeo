"""
Execution Planner

Purpose:
Authoritatively converts intent into a validated ExecutionPlan.

This module:
- DOES planning only
- DOES NOT execute
- DOES NOT touch OS / UI / observer
- DOES NOT invent tools beyond LLM output
- DOES emit strictly schema-valid execution graphs
"""

from typing import List, Dict, Any, Optional
import json
import time

from core.schemas.execution_plan import (
    ExecutionPlan,
    ExecutionStep,
    StepType,
)


class PlanningError(RuntimeError):
    """Authoritative planning failure."""


class ExecutionPlanner:
    """
    Deterministic execution planner.

    HARD CONTRACT:
    - Planner is the ONLY place intelligence exists
    - LLM output must be structured and validated
    - No side effects, no execution, no guessing
    """

    def __init__(
        self,
        *,
        llm_call,
        environment_fingerprint: Optional[Dict[str, Any]] = None,
    ):
        if not callable(llm_call):
            raise PlanningError("llm_call must be callable")

        self._llm_call = llm_call
        self._environment = environment_fingerprint or {}

    # ==================================================
    # PUBLIC API
    # ==================================================

    def create_plan(
        self,
        *,
        objective: str,
        requirements: Dict[str, Any],
        high_level_steps: List[Dict[str, Any]],
    ) -> ExecutionPlan:
        if not isinstance(objective, str) or not objective.strip():
            raise PlanningError("Objective must be non-empty string")

        if not isinstance(high_level_steps, list) or not high_level_steps:
            raise PlanningError("high_level_steps must be non-empty list")

        execution_steps: List[ExecutionStep] = []
        step_id = 1
        last_step_id: Optional[int] = None

        for hl in high_level_steps:
            goal = hl.get("goal")
            if not isinstance(goal, str) or not goal.strip():
                raise PlanningError("Invalid high-level goal entry")

            expanded = self._expand_goal(goal.strip())
            if not expanded:
                raise PlanningError(f"Goal produced no steps: {goal}")

            for spec in expanded:
                deps = [last_step_id] if last_step_id else []

                step = ExecutionStep(
                    id=step_id,
                    type=spec["type"],
                    description=spec["description"],
                    action=spec.get("action", {}),
                    verification=spec.get("verification", {}),
                    dependencies=deps,
                    estimated_duration=spec.get("estimated_duration", 0.0),
                    retryable=spec.get("retryable", True),
                )

                execution_steps.append(step)
                last_step_id = step_id
                step_id += 1

        if not execution_steps:
            raise PlanningError("No executable steps generated")

        # ---- mandatory DONE step ----
        execution_steps.append(
            ExecutionStep(
                id=step_id,
                type=StepType.DONE,
                description="Objective complete",
                action={
                    "operation": "done",
                    "summary": objective.strip(),
                },
                verification={},
                dependencies=[last_step_id] if last_step_id else [],
                estimated_duration=0.0,
                retryable=False,
            )
        )

        plan = ExecutionPlan(
            objective=objective.strip(),
            steps=execution_steps,
            required_tools=self._extract_required_tools(requirements),
            created_at=time.time(),
        )

        if not plan.validate():
            raise PlanningError("ExecutionPlan validation failed")

        return plan

    # ==================================================
    # INTERNAL HELPERS
    # ==================================================

    def _extract_required_tools(
        self, requirements: Dict[str, Any]
    ) -> List[str]:
        tools = requirements.get("tools", [])
        if not isinstance(tools, list):
            return []
        return [t for t in tools if isinstance(t, str)]

    # ==================================================
    # LLM-POWERED GOAL EXPANSION
    # ==================================================

    def _expand_goal(self, goal: str) -> List[Dict[str, Any]]:
        """
        Uses LLM to expand goal into executable, schema-valid steps.

        HARD RULES:
        - LLM MUST return valid JSON
        - Steps MUST conform to ExecutionStep schema
        - Planner validates everything
        """

        prompt = f"""
You are the planning brain of a self-operating computer.

Environment fingerprint:
{json.dumps(self._environment, indent=2)}

Task:
"{goal}"

Return ONLY valid JSON.

Schema:
[
  {{
    "type": "ui_interaction" | "command_execution" | "file_creation" | "tool_installation" | "verification",
    "description": "...",
    "action": {{ }},
    "verification": {{ }},
    "estimated_duration": number,
    "retryable": boolean
  }}
]

Rules:
- No hallucinated tools
- If a tool is required, emit tool_installation step
- Every step must be verifiable
- Be conservative and explicit
"""

        raw = self._llm_call(prompt)

        try:
            data = json.loads(raw)
        except Exception as e:
            raise PlanningError(f"LLM returned invalid JSON: {e}")

        if not isinstance(data, list) or not data:
            raise PlanningError("LLM produced empty or invalid step list")

        validated: List[Dict[str, Any]] = []

        for idx, step in enumerate(data):
            if not isinstance(step, dict):
                raise PlanningError(f"Invalid step at index {idx}")

            step_type = step.get("type")
            if step_type not in StepType.__members__.values() and step_type not in [
                t.value for t in StepType
            ]:
                raise PlanningError(f"Invalid step type: {step_type}")

            validated.append(
                {
                    "type": StepType(step_type),
                    "description": step.get("description", "").strip(),
                    "action": step.get("action", {}),
                    "verification": step.get("verification", {}),
                    "estimated_duration": float(
                        step.get("estimated_duration", 0.0)
                    ),
                    "retryable": bool(step.get("retryable", True)),
                }
            )

        return validated
