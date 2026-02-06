"""
Execution Planner

Purpose:
Converts decomposed planning output into a concrete, auditable ExecutionPlan.

This module:
- DOES planning
- DOES NOT execute
- DOES NOT touch OS
- DOES NOT read screen
- DOES NOT perform UI actions

It is a pure transformer from intent structure → execution structure.
"""

from typing import List, Dict, Any, Optional
import time

from core.schemas.execution_plan import (
    ExecutionPlan,
    ExecutionStep,
    StepType,
)


class PlanningError(RuntimeError):
    pass


class ExecutionPlanner:
    """
    Deterministic execution planner.

    HARD CONTRACT:
    - Inputs are already analyzed and decomposed
    - Output is a fully ordered, validated ExecutionPlan
    - No guessing, no execution logic
    """

    def __init__(
        self,
        *,
        llm_call,
        environment_fingerprint: Optional[Dict[str, Any]] = None,
    ):
        self.llm_call = llm_call
        self.environment = environment_fingerprint or {}

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

        if not high_level_steps:
            raise PlanningError("No high-level steps provided")

        execution_steps: List[ExecutionStep] = []
        step_id = 1
        last_step_id: Optional[int] = None

        # --------------------------------------------------
        # Expand high-level goals
        # --------------------------------------------------

        for hl in high_level_steps:
            goal = hl.get("goal")
            if not isinstance(goal, str) or not goal.strip():
                raise PlanningError("Invalid high-level step goal")

            expanded = self._expand_goal(goal.strip())

            for spec in expanded:
                depends_on: List[int] = []
                if last_step_id is not None:
                    depends_on.append(last_step_id)

                step = ExecutionStep(
                    id=step_id,
                    type=spec["type"],
                    description=spec["description"],
                    action=spec.get("action", {}),
                    verification=spec.get("verification", {}),
                    depends_on=depends_on,
                    estimated_duration=spec.get("estimated_duration", 0.0),
                    retryable=True,
                )

                execution_steps.append(step)
                last_step_id = step_id
                step_id += 1

        if not execution_steps:
            raise PlanningError("Execution plan has no executable steps")

        # --------------------------------------------------
        # Append FINAL DONE step (mandatory)
        # --------------------------------------------------

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
                depends_on=[last_step_id] if last_step_id else [],
                estimated_duration=0.0,
                retryable=False,
            )
        )

        plan = ExecutionPlan(
            objective=objective.strip(),
            steps=execution_steps,
            required_tools=[],  # tool resolution handled elsewhere
            created_at=time.time(),
        )

        # HARD VALIDATION (raises on failure)
        plan.validate()

        return plan

    # ==================================================
    # INTERNAL EXPANSION LOGIC
    # ==================================================

    def _expand_goal(self, goal: str) -> List[Dict[str, Any]]:
        """
        Expands a single high-level goal into conservative execution steps.

        Rules:
        - Deterministic
        - No UI emission
        - No tool invention
        - Always verifiable
        """

        normalized = goal.lower()
        steps: List[Dict[str, Any]] = []

        # ---- file system intent ----
        if any(k in normalized for k in ("create", "setup", "initialize", "scaffold")):
            steps.append(
                {
                    "type": StepType.FILE_WRITE,
                    "description": goal,
                    "action": {
                        "path": "./",
                        "content": "",
                    },
                    "verification": {
                        "exists": True,
                    },
                    "estimated_duration": 10.0,
                }
            )

            steps.append(
                {
                    "type": StepType.VERIFICATION,
                    "description": f"Verify filesystem effect: {goal}",
                    "action": {},
                    "verification": {},
                    "estimated_duration": 5.0,
                }
            )
            return steps

        # ---- command intent ----
        if any(k in normalized for k in ("run", "build", "generate", "start")):
            steps.append(
                {
                    "type": StepType.COMMAND,
                    "description": goal,
                    "action": {
                        "command": "",
                        "sudo": False,
                    },
                    "verification": {},
                    "estimated_duration": 20.0,
                }
            )

            steps.append(
                {
                    "type": StepType.VERIFICATION,
                    "description": f"Verify command effect: {goal}",
                    "action": {},
                    "verification": {},
                    "estimated_duration": 5.0,
                }
            )
            return steps

        # ---- fallback: verification-only ----
        steps.append(
            {
                "type": StepType.VERIFICATION,
                "description": goal,
                "action": {},
                "verification": {},
                "estimated_duration": 5.0,
            }
        )

        return steps
