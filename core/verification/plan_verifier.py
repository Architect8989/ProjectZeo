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

PATCH (audit Bug #2):
  The original code unconditionally raised PlanVerificationError on any
  TOOL_INSTALLATION step ("forbidden until installer exists").  However:
    • ExecutionPlan.schema explicitly includes StepType.TOOL_INSTALLATION.
    • operate.py dispatches TOOL_INSTALLATION steps to
      AutonomousInstaller.install_tool().
    • main.py wires the installer into operate_main().
  The installer is therefore present and integrated.  Keeping the hard
  rejection caused every plan that contained tool-installation steps to
  fail the verification gate before execution could begin — a latent bug.

  Fix: TOOL_INSTALLATION steps are now accepted.  A TOOL_INSTALLATION
  step still must have a valid 'tool' dict with an 'official_url' that
  starts with 'https://' (enforcement delegated to the planner, verified
  here at the action-shape level).
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

            # PATCH (audit Bug #2): TOOL_INSTALLATION is a valid, integrated
            # step type handled by AutonomousInstaller in operate.py.
            # The original hard-rejection has been removed.
            # Action-shape validation for TOOL_INSTALLATION steps is
            # performed in _verify_step_actions() below.

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

            # PATCH (audit Bug #2): validate TOOL_INSTALLATION action shape.
            # Require a 'tool' dict with an 'official_url' starting with
            # 'https://' so that the installer can open the official download
            # page safely (mirrors the planner-level enforcement).
            if step.type == StepType.TOOL_INSTALLATION:
                tool = step.action.get("tool")
                if not isinstance(tool, dict):
                    raise PlanVerificationError(
                        f"Step {step.id} TOOL_INSTALLATION action must contain "
                        f"a 'tool' dict"
                    )
                official_url = tool.get("official_url", "")
                if not isinstance(official_url, str) or not official_url.startswith(
                    "https://"
                ):
                    raise PlanVerificationError(
                        f"Step {step.id} TOOL_INSTALLATION 'official_url' must "
                        f"start with 'https://'"
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
