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
        """
        Build an ExecutionPlan from decomposed steps.

        Raises:
            PlanningError on any structural violation
        """

        if not high_level_steps:
            raise PlanningError("No high-level steps provided")

        execution_steps: List[ExecutionStep] = []
        step_id = 1
        last_step_id: Optional[int] = None

        for hl in high_level_steps:
            goal = hl.get("goal")
            if not isinstance(goal, str):
                raise PlanningError("Invalid high-level step goal")

            expanded_steps = self._expand_goal(goal)

            for spec in expanded_steps:
                deps = []
                if last_step_id is not None:
                    deps.append(last_step_id)

                step = ExecutionStep(
                    id=step_id,
                    type=spec["type"],
                    description=spec["description"],
                    action=spec.get("action", {}),
                    verification=spec.get("verification", {}),
                    dependencies=deps,
                    estimated_duration=spec.get("estimated_duration", 30),
                )

                execution_steps.append(step)
                last_step_id = step_id
                step_id += 1

        if not execution_steps:
            raise PlanningError("Execution plan is empty")

        plan = ExecutionPlan(
            objective=objective,
            steps=execution_steps,
            required_tools=[],  # tool manager added later
            estimated_total_duration=sum(
                s.estimated_duration for s in execution_steps
            ),
            created_at=time.time(),
        )

        if not plan.validate():
            raise PlanningError("ExecutionPlan validation failed")

        return plan

    # ==================================================
    # INTERNAL EXPANSION LOGIC
    # ==================================================

    def _expand_goal(self, goal: str) -> List[Dict[str, Any]]:
        """
        Expands a single high-level goal into conservative execution steps.

        Rules:
        - Always produce verifiable structure
        - Never invent tools
        - Never emit UI ops here
        """

        normalized = goal.lower()
        steps: List[Dict[str, Any]] = []

        # ---- file system intent ----
        if any(k in normalized for k in ("create", "setup", "initialize", "scaffold")):
            steps.append(
                {
                    "type": StepType.FILE_CREATION,
                    "description": goal,
                    "action": {
                        "path": "./",
                        "content": "",
                    },
                    "verification": {
                        "exists": True,
                    },
                    "estimated_duration": 10,
                }
            )

            steps.append(
                {
                    "type": StepType.VERIFICATION,
                    "description": f"Verify filesystem effect: {goal}",
                    "action": {},
                    "verification": {},
                    "estimated_duration": 5,
                }
            )
            return steps

        # ---- command intent ----
        if any(k in normalized for k in ("run", "build", "generate", "start")):
            steps.append(
                {
                    "type": StepType.COMMAND_EXECUTION,
                    "description": goal,
                    "action": {
                        "command": "",
                        "sudo": False,
                    },
                    "verification": {},
                    "estimated_duration": 20,
                }
            )

            steps.append(
                {
                    "type": StepType.VERIFICATION,
                    "description": f"Verify command effect: {goal}",
                    "action": {},
                    "verification": {},
                    "estimated_duration": 5,
                }
            )
            return steps

        # ---- fallback: explicit verification-only step ----
        steps.append(
            {
                "type": StepType.VERIFICATION,
                "description": goal,
                "action": {},
                "verification": {},
                "estimated_duration": 5,
            }
        )
        return steps
