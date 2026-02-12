from typing import List, Dict, Any, Optional
import json
import time
import concurrent.futures

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
    - No side effects
    - Planner operates on FROZEN snapshot only
    """

    LLM_TIMEOUT_SECONDS = 30.0
    MAX_SCREEN_CHARS = 500
    MAX_ESTIMATED_DURATION = 600.0  # 10 minutes hard cap

    def __init__(
        self,
        *,
        llm_call,
        environment_fingerprint: Optional[Dict[str, Any]] = None,
        world_graph=None,
    ):
        if not callable(llm_call):
            raise PlanningError("llm_call must be callable")

        self._llm_call = llm_call
        self._environment = environment_fingerprint or {}
        self._world_snapshot: Optional[Dict[str, Any]] = None

        if world_graph is not None:
            try:
                snap = world_graph.snapshot()
                self._world_snapshot = json.loads(json.dumps(snap))
            except Exception:
                raise PlanningError("Failed to snapshot world_graph")

        if self._world_snapshot is None:
            self._world_snapshot = {
                "entities": [],
                "focused_app": None,
                "entity_count": 0,
                "timestamp": None,
            }

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
                description = spec["description"].strip()
                if not description:
                    raise PlanningError("LLM produced step without description")

                deps = [last_step_id] if last_step_id else []

                step = ExecutionStep(
                    id=step_id,
                    type=spec["type"],
                    description=description,
                    action=spec["action"],
                    verification=spec["verification"],
                    dependencies=deps,
                    estimated_duration=spec["estimated_duration"],
                    retryable=spec["retryable"],
                )

                execution_steps.append(step)
                last_step_id = step_id
                step_id += 1

        if not execution_steps:
            raise PlanningError("No executable steps generated")

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
    # NEW: SNAPSHOT UPDATE
    # ==================================================

    def update_world_snapshot(self, new_snapshot: Dict[str, Any]) -> None:
        """
        Replace frozen snapshot with new world snapshot.
        Deep-copied to preserve isolation.
        """

        if not isinstance(new_snapshot, dict):
            raise PlanningError("Invalid world snapshot")

        try:
            self._world_snapshot = json.loads(json.dumps(new_snapshot))
        except Exception:
            raise PlanningError("Failed to freeze new snapshot")

    # ==================================================
    # NEW: REPLAN DECISION ENGINE
    # ==================================================

    def should_replan(
        self,
        *,
        current_step_id: int,
        execution_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Decide whether re-planning is required.

        Returns structured decision.
        """

        if not isinstance(current_step_id, int):
            raise PlanningError("Invalid step id")

        if not isinstance(execution_history, list):
            execution_history = []

        if not self._world_snapshot:
            return {
                "replan_required": False,
                "confidence": 0.0,
                "reason": "No world snapshot available",
            }

        prompt = f"""
You are validating whether a deterministic execution plan must be revised.

Original snapshot:
{json.dumps(self._world_snapshot, indent=2)[:1000]}

Execution state:
- current_step_id: {current_step_id}
- recent_history:
{json.dumps(execution_history[-5:], indent=2)}

Return JSON only:
{{
  "replan_required": boolean,
  "confidence": 0.0-1.0,
  "reason": "short explanation"
}}

Rules:
- Replan only if snapshot divergence makes continuation unsafe.
- Be conservative.
"""

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._llm_call, prompt)
            try:
                raw = future.result(timeout=self.LLM_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                return {
                    "replan_required": False,
                    "confidence": 0.0,
                    "reason": "LLM timeout",
                }

        try:
            decision = json.loads(raw.strip())
        except Exception:
            return {
                "replan_required": False,
                "confidence": 0.0,
                "reason": "Invalid LLM response",
            }

        return {
            "replan_required": bool(decision.get("replan_required", False)),
            "confidence": float(decision.get("confidence", 0.0)),
            "reason": decision.get("reason", "LLM decision"),
        }

    # ==================================================
    # INTERNAL HELPERS
    # ==================================================

    def _extract_required_tools(
        self, requirements: Dict[str, Any]
    ) -> List[str]:
        tools = requirements.get("tools", [])
        if not isinstance(tools, list):
            return []
        return [t for t in tools if isinstance(t, str) and t.strip()]

    def _read_screen_context(self) -> str:
        if not self._world_snapshot:
            return ""

        text_chunks: List[str] = []
        entities = self._world_snapshot.get("entities", [])

        if not isinstance(entities, list):
            return ""

        for ent in entities:
            if not isinstance(ent, dict):
                continue

            label = ent.get("text")
            etype = ent.get("type")

            if isinstance(label, str) and label.strip():
                text_chunks.append(f"{etype}: {label}")

        context = "\n".join(text_chunks)
        return context[: self.MAX_SCREEN_CHARS]

    # ==================================================
    # LLM EXPANSION
    # ==================================================

    def _expand_goal(self, goal: str) -> List[Dict[str, Any]]:
        screen_context = self._read_screen_context()

        prompt = f"""
You are the planning brain of a deterministic execution kernel.

Environment:
{json.dumps(self._environment, indent=2)}

Frozen screen:
{screen_context}

Goal:
"{goal}"

Return JSON list only.

Schema:
[
  {{
    "type": "ui_interaction" | "command_execution" | "file_creation" | "verification",
    "description": "...",
    "action": {{ }},
    "verification": {{ }},
    "estimated_duration": number,
    "retryable": boolean
  }}
]

Rules:
- Do NOT emit DONE
- Do NOT emit tool_installation
- Every step must include verification
- Conservative estimates
"""

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._llm_call, prompt)
            try:
                raw = future.result(timeout=self.LLM_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                raise PlanningError(
                    f"LLM call timed out after {self.LLM_TIMEOUT_SECONDS}s"
                )

        try:
            data = json.loads(raw)
        except Exception as e:
            raise PlanningError(f"LLM returned invalid JSON: {e}")

        if not isinstance(data, list) or not data:
            raise PlanningError("LLM produced empty or invalid step list")

        validated: List[Dict[str, Any]] = []
        allowed_types = {
            StepType.UI_INTERACTION.value,
            StepType.COMMAND_EXECUTION.value,
            StepType.FILE_CREATION.value,
            StepType.VERIFICATION.value,
        }

        for idx, step in enumerate(data):
            if not isinstance(step, dict):
                raise PlanningError(f"Invalid step at index {idx}")

            step_type = step.get("type")
            if step_type not in allowed_types:
                raise PlanningError(f"Invalid step type: {step_type}")

            description = step.get("description")
            if not isinstance(description, str) or not description.strip():
                raise PlanningError("Step description must be non-empty string")

            action = step.get("action")
            if not isinstance(action, dict):
                raise PlanningError("Step action must be object")

            verification = step.get("verification")
            if not isinstance(verification, dict):
                raise PlanningError("Step verification must be object")

            try:
                duration = float(step.get("estimated_duration", 0.0))
            except Exception:
                raise PlanningError("Invalid estimated_duration")

            if duration < 0 or duration > self.MAX_ESTIMATED_DURATION:
                raise PlanningError("estimated_duration out of bounds")

            retryable = bool(step.get("retryable", True))

            validated.append(
                {
                    "type": StepType(step_type),
                    "description": description.strip(),
                    "action": action,
                    "verification": verification,
                    "estimated_duration": duration,
                    "retryable": retryable,
                }
            )

        return validated
