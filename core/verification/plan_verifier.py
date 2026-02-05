"""
Plan Verifier

Purpose:
Authoritative pre-execution verification of an ExecutionPlan.

This module:
- DOES NOT execute
- DOES NOT touch OS
- DOES NOT read screen
- DOES NOT perform UI actions
- DOES NOT modify plans

It is the FINAL gate before execution.
If this verifier fails, execution MUST NOT start.
"""

from typing import Set

from core.schemas.execution_plan import ExecutionPlan, ExecutionStep, StepType


class PlanVerificationError(RuntimeError):
    pass


class PlanVerifier:
    """
    Deterministic execution-plan verifier.

    HARD CONTRACT:
    - Input: ExecutionPlan
    - Output: None (raises on failure)
    - No side effects
    """

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def verify(self, plan: ExecutionPlan) -> None:
        if not isinstance(plan, ExecutionPlan):
            raise PlanVerificationError("Invalid plan object")

        if not plan.steps:
            raise PlanVerificationError("ExecutionPlan contains no steps")

        self._verify_step_ids(plan)
        self._verify_dependencies(plan)
        self._verify_step_types(plan)
        self._verify_required_tools(plan)

    # -------------------------------------------------
    # Internal checks
    # -------------------------------------------------

    def _verify_step_ids(self, plan: ExecutionPlan) -> None:
        seen: Set[int] = set()

        for step in plan.steps:
            if not isinstance(step, ExecutionStep):
                raise PlanVerificationError("Invalid step object in plan")

            if not isinstance(step.id, int):
                raise PlanVerificationError("Step id must be integer")

            if step.id in seen:
                raise PlanVerificationError(f"Duplicate step id: {step.id}")

            seen.add(step.id)

    def _verify_dependencies(self, plan: ExecutionPlan) -> None:
        step_ids = {step.id for step in plan.steps}

        for step in plan.steps:
            if not isinstance(step.dependencies, list):
                raise PlanVerificationError(
                    f"Step {step.id} dependencies must be list"
                )

            for dep in step.dependencies:
                if dep not in step_ids:
                    raise PlanVerificationError(
                        f"Step {step.id} depends on missing step {dep}"
                    )

                if dep >= step.id:
                    raise PlanVerificationError(
                        f"Step {step.id} has invalid forward/self dependency {dep}"
                    )

    def _verify_step_types(self, plan: ExecutionPlan) -> None:
        for step in plan.steps:
            if not isinstance(step.type, StepType):
                raise PlanVerificationError(
                    f"Step {step.id} has invalid type {step.type}"
                )

            # Explicitly forbid TOOL_INSTALLATION until installer exists
            if step.type == StepType.TOOL_INSTALLATION:
                raise PlanVerificationError(
                    "TOOL_INSTALLATION steps are not yet supported"
                )

    def _verify_required_tools(self, plan: ExecutionPlan) -> None:
        # Planning phase must be explicit, even if empty
        if plan.required_tools is None:
            raise PlanVerificationError(
                "ExecutionPlan.required_tools must be explicitly set"
            )

        if not isinstance(plan.required_tools, list):
            raise PlanVerificationError(
                "ExecutionPlan.required_tools must be a list"
            )

        for tool in plan.required_tools:
            if not isinstance(tool, str) or not tool.strip():
                raise PlanVerificationError(
                    f"Invalid tool declaration: {tool}"
      )
