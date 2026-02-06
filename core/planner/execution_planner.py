"""
Execution Planner

Purpose:
Deterministically converts decomposed intent into a validated ExecutionPlan.

This module:
- DOES planning only
- DOES NOT execute
- DOES NOT touch OS / UI / observer
- DOES NOT invent tools
- DOES emit strictly schema-valid execution graphs
"""

from typing import List, Dict, Any, Optional
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
    - Inputs are already semantically decomposed
    - Output is a fully ordered, validated ExecutionPlan
    - No guessing, no side effects, no execution logic
    """

    def __init__(
        self,
        *,
        llm_call,
        environment_fingerprint: Optional[Dict[str, Any]] = None,
    ):
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
        """
        Build a fully validated ExecutionPlan.

        Raises:
            PlanningError on any structural violation.
        """

        if not isinstance(objective, str) or not objective.strip():
            raise PlanningError("Objective must be non-empty string")

        if not isinstance(high_level_steps, list) or not high_level_steps:
            raise PlanningError("high_level_steps must be non-empty list")

        execution_steps: List[ExecutionStep] = []
        step_id = 1
        last_step_id: Optional[int] = None

        # --------------------------------------------------
        # Expand high-level goals deterministically
        # --------------------------------------------------

        for hl in high_level_steps:
            goal = hl.get("goal")

            if not isinstance(goal, str) or not goal.strip():
                raise PlanningError("Invalid high-level goal entry")

            expanded_specs = self._expand_goal(goal.strip())

            if not expanded_specs:
                raise PlanningError(
                    f"Goal produced no executable steps: {goal}"
                )

            for spec in expanded_specs:
                dependencies: List[int] = []
                if last_step_id is not None:
                    dependencies.append(last_step_id)

                step = ExecutionStep(
                    id=step_id,
                    type=spec["type"],
                    description=spec["description"],
                    action=spec.get("action", {}),
                    verification=spec.get("verification", {}),
                    dependencies=dependencies,
                    estimated_duration=spec.get(
                        "estimated_duration", 0.0
                    ),
                    retryable=spec.get("retryable", True),
                )

                execution_steps.append(step)
                last_step_id = step_id
                step_id += 1

        if not execution_steps:
            raise PlanningError("No executable steps generated")

        # --------------------------------------------------
        # Mandatory terminal DONE step
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

        # --------------------------------------------------
        # HARD VALIDATION (authoritative)
        # --------------------------------------------------

        if not plan.validate():
            raise PlanningError("ExecutionPlan validation failed")

        return plan

    # ==================================================
    # INTERNAL HELPERS
    # ==================================================

    def _extract_required_tools(
        self, requirements: Dict[str, Any]
    ) -> List[str]:
        """
        Extracts required tools conservatively.
        Planning does not install or verify tools.
        """
        tools = requirements.get("tools", [])
        if not isinstance(tools, list):
            return []
        return [t for t in tools if isinstance(t, str)]

    # ==================================================
    # GOAL EXPANSION LOGIC
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

        # ---- filesystem intent ----
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
                    "retryable": False,
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
                    "retryable": False,
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
                "retryable": False,
            }
        )

        return steps
