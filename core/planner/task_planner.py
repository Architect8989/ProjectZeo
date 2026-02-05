"""
Authoritative Task Planner

Purpose:
Owns the PLANNING phase.

This is the ONLY module allowed to:
- Call external LLM for reasoning
- Analyze intent into requirements
- Coordinate decomposition + execution planning
- Decide whether execution is permitted

Hard rules:
- No OS calls
- No UI assumptions
- No execution
- No partial plans
"""

from typing import Dict, Any, Optional

from core.planner.task_decomposer import TaskDecomposer, DecompositionError
from core.planner.execution_planner import ExecutionPlanner
from core.schemas.execution_plan import ExecutionPlan
from core.telemetry.logger import log_info, log_warn, log_error


class PlanningError(RuntimeError):
    pass


class TaskPlanner:
    """
    Planning authority.

    Flow:
    intent
      -> requirement analysis (LLM)
      -> task decomposition
      -> execution plan synthesis
      -> plan validation
    """

    MAX_INTENT_CHARS = 10_000

    def __init__(
        self,
        *,
        llm_call,
        environment_fingerprint: Optional[Dict[str, Any]] = None,
    ):
        """
        llm_call(prompt: str) -> str
        environment_fingerprint is informational only
        """
        if llm_call is None:
            raise ValueError("llm_call is required")

        self.llm_call = llm_call
        self.environment = environment_fingerprint or {}

        self.decomposer = TaskDecomposer(llm_call=llm_call)
        self.execution_planner = ExecutionPlanner(
            llm_call=llm_call,
            environment_fingerprint=self.environment,
        )

    # ==================================================
    # PUBLIC API
    # ==================================================

    def analyze_and_plan(self, intent: str) -> ExecutionPlan:
        """
        Main entrypoint for PLANNING mode.

        Returns:
            ExecutionPlan (fully validated)

        Raises:
            PlanningError if plan cannot be produced safely
        """

        if not isinstance(intent, str) or not intent.strip():
            raise PlanningError("Intent must be non-empty string")

        intent = intent.strip()

        if len(intent) > self.MAX_INTENT_CHARS:
            raise PlanningError("Intent too large")

        log_info("[PLANNER] Starting planning phase")

        # ---- Phase 1: requirement analysis ----
        requirements = self._analyze_requirements(intent)

        # ---- Phase 2: high-level decomposition ----
        try:
            high_level_steps = self.decomposer.decompose(intent)
        except DecompositionError as e:
            raise PlanningError(f"Decomposition failed: {e}")

        if not high_level_steps:
            raise PlanningError("No steps produced by decomposer")

        # ---- Phase 3: execution plan synthesis ----
        execution_plan = self.execution_planner.create_plan(
            objective=intent,
            requirements=requirements,
            high_level_steps=high_level_steps,
        )

        # ---- Phase 4: hard validation ----
        if not isinstance(execution_plan, ExecutionPlan):
            raise PlanningError("Planner did not return ExecutionPlan")

        if not execution_plan.validate():
            raise PlanningError("ExecutionPlan validation failed")

        log_info(
            f"[PLANNER] Plan ready: {len(execution_plan.steps)} steps"
        )

        return execution_plan

    # ==================================================
    # INTERNAL ANALYSIS
    # ==================================================

    def _analyze_requirements(self, intent: str) -> Dict[str, Any]:
        """
        Uses LLM to extract explicit requirements and constraints.

        Output is informational but mandatory.
        """

        prompt = f"""
You are a requirement analysis engine.

Task:
Analyze the following objective and extract REQUIREMENTS ONLY.

Rules:
- No execution steps
- No UI actions
- No speculation
- Be explicit
- Return STRICT JSON

Schema:
{{
  "scope": "short description",
  "deliverables": [ "item", "item" ],
  "constraints": [ "constraint", "constraint" ],
  "assumptions": [ "assumption", "assumption" ],
  "risks": [ "risk", "risk" ]
}}

OBJECTIVE:
{intent}
""".strip()

        raw = self.llm_call(prompt)

        if not isinstance(raw, str):
            raise PlanningError("Requirement analysis returned non-string")

        try:
            import json
            import re

            match = re.search(r"\{.*\}", raw, re.S)
            if not match:
                raise ValueError("No JSON found")

            requirements = json.loads(match.group(0))
        except Exception as e:
            raise PlanningError(f"Requirement parsing failed: {e}")

        if not isinstance(requirements, dict):
            raise PlanningError("Invalid requirements structure")

        # minimal sanity checks
        for key in ("scope", "deliverables", "constraints"):
            if key not in requirements:
                raise PlanningError(f"Missing requirement field: {key}")

        return requirements
