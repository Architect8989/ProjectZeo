import json
import re
from typing import Dict, Any, Optional, List

from core.planner.task_decomposer import TaskDecomposer, DecompositionError
from core.planner.execution_planner import ExecutionPlanner
from core.schemas.execution_plan import ExecutionPlan
from core.telemetry.logger import log_info, log_warn, log_error


class PlanningError(RuntimeError):
    pass


class TaskPlanner:
    """
    Planning authority.

    HAR-2 WARNING: This class is currently DEAD CODE — it is not imported or
    instantiated anywhere in the production execution path.  It is preserved
    for potential future use.  The LLM interface bug in _analyze_requirements()
    has been fixed (see module docstring), but the class has not been
    integration-tested against the live execution path.

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
        llm_call(messages: List[dict], objective: str, session_id: str) -> str
            Must accept the standard three-argument interface used throughout
            the production execution path.  A plain string callable
            (prompt: str) -> str will raise TypeError at runtime.

        environment_fingerprint is informational only.
        """
        if llm_call is None:
            raise ValueError("llm_call is required")

        if not callable(llm_call):
            raise ValueError("llm_call must be callable")

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

        HAR-2 FIX: Updated to use the correct llm_call interface:
            llm_call(messages: List[dict], objective: str, session_id: str)

        Root cause of original bug: this method called self.llm_call(prompt)
        with a plain string.  The system's llm_call interface expects a list of
        message dicts as the first argument (the standard chat interface used
        throughout operate.py, ExecutionPlanner, and ReasoningEngine).  Passing
        a string triggered TypeError: argument after * must be an iterable
        (or similar depending on the adapter implementation) on the first call.
        The class was dead code so this was never caught.

        Fix: wrap prompt in the standard messages format and call with
        (messages, objective, session_id) so any conforming adapter works.
        """

        system_prompt = (
            "You are a requirement analysis engine. "
            "Analyze the objective and extract REQUIREMENTS ONLY. "
            "Rules: no execution steps, no UI actions, no speculation, be explicit. "
            "Return STRICT JSON matching the schema provided."
        )

        user_prompt = f"""Analyze the following objective and extract REQUIREMENTS ONLY.

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

Return ONLY valid JSON. No preamble, no explanation."""

        # HAR-2 FIX: Use the correct three-argument llm_call interface.
        # The original code passed a plain string; this raised TypeError in every
        # adapter because the first positional argument must be List[dict].
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

        try:
            raw = self.llm_call(messages, intent, "requirement_analysis")
        except Exception as e:
            raise PlanningError(f"LLM call for requirement analysis failed: {e}")

        # Adapters may return a list of action dicts (vision path) or a string
        # (text path).  Accept both.
        if isinstance(raw, list):
            # Vision adapter returned UI ops — cannot use for requirement analysis.
            # Construct a minimal requirements dict so planning can continue.
            log_warn(
                "[PLANNER] _analyze_requirements: LLM returned action list instead "
                "of text — using minimal fallback requirements."
            )
            return {
                "scope": intent[:200],
                "deliverables": [],
                "constraints": [],
                "assumptions": [],
                "risks": [],
            }

        if not isinstance(raw, str):
            raise PlanningError(
                f"Requirement analysis returned unexpected type: {type(raw)}"
            )

        try:
            match = re.search(r"\{.*\}", raw, re.S)
            if not match:
                raise ValueError("No JSON object found in LLM response")

            requirements = json.loads(match.group(0))
        except Exception as e:
            raise PlanningError(f"Requirement parsing failed: {e}")

        if not isinstance(requirements, dict):
            raise PlanningError("Invalid requirements structure (not a dict)")

        for key in ("scope", "deliverables", "constraints"):
            if key not in requirements:
                raise PlanningError(f"Missing requirement field: {key!r}")

        return requirements
