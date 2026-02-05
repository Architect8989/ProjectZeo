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

from typing import Set, Dict, List

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
            raise PlanVerificationError("Invalid ExecutionPlan object")

        if not isinstance(plan.steps, list) or not plan.steps:
            raise PlanVerificationError("ExecutionPlan contains no steps")

        self._verify_step_objects(plan)
        self._verify_step_ids(plan)
        self._verify_dependencies(plan)
        self._verify_step_types(plan)
        self._verify_step_actions(plan)
        self._verify_required_tools(plan)

    # -------------------------------------------------
    # Internal checks
    # -------------------------------------------------

    def _verify_step_objects(self, plan: ExecutionPlan) -> None:
        for step in plan.steps:
            if not isinstance(step, ExecutionStep):
                raise PlanVerificationError(
                    "ExecutionPlan contains non-ExecutionStep entry"
                )

    def _verify_step_ids(self, plan: ExecutionPlan) -> None:
        seen: Set[int] = set()

        for step in plan.steps:
            if not isinstance(step.id, int) or step.id <= 0:
                raise PlanVerificationError(
                    f"Invalid step id: {step.id}"
                )

            if step.id in seen:
                raise PlanVerificationError(
                    f"Duplicate step id detected: {step.id}"
                )

            seen.add(step.id)

    def _verify_dependencies(self, plan: ExecutionPlan) -> None:
        step_ids = {step.id for step in plan.steps}

        for step in plan.steps:
            deps = step.dependencies

            if not isinstance(deps, list):
                raise PlanVerificationError(
                    f"Step {step.id} dependencies must be a list"
                )

            for dep in deps:
                if not isinstance(dep, int):
                    raise PlanVerificationError(
                        f"Step {step.id} has non-integer dependency {dep}"
                    )

                if dep not in step_ids:
                    raise PlanVerificationError(
                        f"Step {step.id} depends on missing step {dep}"
                    )

                if dep >= step.id:
                    raise PlanVerificationError(
                        f"Step {step.id} has forward/self dependency {dep}"
                    )

    def _verify_step_types(self, plan: ExecutionPlan) -> None:
        for step in plan.steps:
            if not isinstance(step.type, StepType):
                raise PlanVerificationError(
                    f"Step {step.id} has invalid StepType {step.type}"
                )

            # Explicit guardrail: installer not integrated yet
            if step.type == StepType.TOOL_INSTALLATION:
                raise PlanVerificationError(
                    "TOOL_INSTALLATION steps are forbidden until installer exists"
                )

    def _verify_step_actions(self, plan: ExecutionPlan) -> None:
        for step in plan.steps:
            if not isinstance(step.action, dict):
                raise PlanVerificationError(
                    f"Step {step.id} action must be dict"
                )

            if step.type == StepType.COMMAND_EXECUTION:
                if "command" not in step.action:
                    raise PlanVerificationError(
                        f"Step {step.id} missing command action"
                    )

            if step.type == StepType.FILE_CREATION:
                if "path" not in step.action:
                    raise PlanVerificationError(
                        f"Step {step.id} missing file path"
                    )

            if step.type == StepType.UI_INTERACTION:
                if "op" not in step.action:
                    raise PlanVerificationError(
                        f"Step {step.id} missing UI operation"
                    )

    def _verify_required_tools(self, plan: ExecutionPlan) -> None:
        if plan.required_tools is None:
            raise PlanVerificationError(
                "ExecutionPlan.required_tools must be explicitly defined"
            )

        if not isinstance(plan.required_tools, list):
            raise PlanVerificationError(
                "ExecutionPlan.required_tools must be a list"
            )

        for tool in plan.required_tools:
            if not isinstance(tool, str) or not tool.strip():
                raise PlanVerificationError(
                    f"Invalid required tool declaration: {tool}"
                            )
