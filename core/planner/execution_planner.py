"""
Execution Planner

Purpose:
Bridges high-level task decomposition into a concrete ExecutionPlan.

This module:
- DOES planning
- DOES NOT execute
- DOES NOT touch OS
- DOES NOT read screen
- DOES NOT perform UI actions

It converts TaskDecomposer output into ExecutionPlan objects
consumable by operate.py.
"""

from typing import List, Dict, Callable
import time

from core.planner.task_decomposer import TaskDecomposer, DecompositionError
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
    - Input: raw human objective
    - Output: validated ExecutionPlan
    - No side effects
    """

    def __init__(self, llm_call: Callable[[str], str]):
        self._decomposer = TaskDecomposer(llm_call=llm_call)

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def create_plan(self, objective: str) -> ExecutionPlan:
        """
        Create a full ExecutionPlan from a human objective.
        """
        try:
            high_level_steps = self._decomposer.decompose(objective)
        except DecompositionError as e:
            raise PlanningError(f"Decomposition failed: {e}")

        execution_steps: List[ExecutionStep] = []
        step_id = 1
        previous_step_id = None

        for hl in high_level_steps:
            expanded = self._expand_goal(hl["goal"])

            for step in expanded:
                deps = []
                if previous_step_id is not None:
                    deps.append(previous_step_id)

                step = ExecutionStep(
                    id=step_id,
                    type=step["type"],
                    description=step["description"],
                    action=step["action"],
                    verification=step.get("verification", {}),
                    dependencies=deps,
                    estimated_duration=step.get("estimated_duration", 30),
                )

                execution_steps.append(step)
                previous_step_id = step_id
                step_id += 1

        plan = ExecutionPlan(
            objective=objective,
            steps=execution_steps,
            required_tools=[],  # tool discovery is future phase
            estimated_total_duration=sum(
                s.estimated_duration for s in execution_steps
            ),
            created_at=time.time(),
        )

        if not plan.validate():
            raise PlanningError("ExecutionPlan validation failed")

        return plan

    # -------------------------------------------------
    # Internal expansion logic
    # -------------------------------------------------

    def _expand_goal(self, goal: str) -> List[Dict]:
        """
        Expands a single high-level goal into concrete execution steps.

        CURRENT SCOPE (INTENTIONAL):
        - FILE_CREATION
        - COMMAND_EXECUTION
        - VERIFICATION

        This is conservative and deterministic.
        """
        steps: List[Dict] = []

        normalized = goal.lower()

        # ---- file-oriented goals ----
        if any(k in normalized for k in ("create", "setup", "initialize", "scaffold")):
            steps.append(
                {
                    "type": StepType.FILE_CREATION,
                    "description": goal,
                    "action": {
                        "path": "./",
                        "content": "",
                    },
                    "verification": {},
                    "estimated_duration": 10,
                }
            )

            steps.append(
                {
                    "type": StepType.VERIFICATION,
                    "description": f"Verify: {goal}",
                    "action": {},
                    "verification": {},
                    "estimated_duration": 5,
                }
            )
            return steps

        # ---- command-oriented goals ----
        if any(k in normalized for k in ("run", "install", "build", "generate", "start")):
            steps.append(
                {
                    "type": StepType.COMMAND_EXECUTION,
                    "description": goal,
                    "action": {
                        "command": "echo 'placeholder'",
                        "sudo": False,
                    },
                    "verification": {},
                    "estimated_duration": 20,
                }
            )

            steps.append(
                {
                    "type": StepType.VERIFICATION,
                    "description": f"Verify: {goal}",
                    "action": {},
                    "verification": {},
                    "estimated_duration": 5,
                }
            )
            return steps

        # ---- fallback (safe no-op verification) ----
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
