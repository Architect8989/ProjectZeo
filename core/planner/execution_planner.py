from typing import List, Dict, Any, Optional
import json
import time
import concurrent.futures
import hashlib

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
    - Only intelligence boundary
    - No side effects
    - Strict validation
    - Snapshot always frozen
    - Replan bounded and storm-protected
    """

    LLM_TIMEOUT_SECONDS = 30.0
    MAX_SCREEN_CHARS = 500
    MAX_ESTIMATED_DURATION = 600.0
    MAX_STEPS_PER_GOAL = 25

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
        self._last_replan_snapshot_hash: Optional[str] = None

        if world_graph is not None:
            self.update_world_snapshot(world_graph.snapshot())
        else:
            self._world_snapshot = {
                "entities": [],
                "focused_app": None,
                "entity_count": 0,
                "timestamp": None,
            }

    # ==================================================
    # PLAN CREATION
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

            if len(expanded) > self.MAX_STEPS_PER_GOAL:
                raise PlanningError("LLM produced too many steps")

            for spec in expanded:
                deps = [last_step_id] if last_step_id else []

                step = ExecutionStep(
                    id=step_id,
                    type=spec["type"],
                    description=spec["description"],
                    action=spec["action"],
                    verification=spec["verification"],
                    dependencies=deps,
                    estimated_duration=spec["estimated_duration"],
                    retryable=spec["retryable"],
                )

                execution_steps.append(step)
                last_step_id = step_id
                step_id += 1

        execution_steps.append(
            ExecutionStep(
                id=step_id,
                type=StepType.DONE,
                description="Objective complete",
                action={"operation": "done", "summary": objective.strip()},
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
    # SNAPSHOT UPDATE
    # ==================================================

    def update_world_snapshot(self, new_snapshot: Dict[str, Any]) -> None:
        if not isinstance(new_snapshot, dict):
            raise PlanningError("Invalid world snapshot")

        try:
            frozen = json.loads(json.dumps(new_snapshot))
        except Exception:
            raise PlanningError("Failed to freeze new snapshot")

        self._world_snapshot = frozen

    # ==================================================
    # REPLAN DECISION (HARDENED)
    # ==================================================

    def should_replan(
        self,
        *,
        current_step_id: int,
        execution_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        if not isinstance(current_step_id, int):
            raise PlanningError("Invalid step id")

        if not isinstance(execution_history, list):
            execution_history = []

        if not self._world_snapshot:
            return self._no_replan("No snapshot available")

        snapshot_hash = self._hash_snapshot(self._world_snapshot)

        # Prevent replan storm for identical snapshot
        if snapshot_hash == self._last_replan_snapshot_hash:
            return self._no_replan("Snapshot unchanged")

        prompt = f"""
Determine if replanning is required.

World snapshot:
{json.dumps(self._world_snapshot)[:1000]}

Current step: {current_step_id}
Recent history:
{json.dumps(execution_history[-5:])}

Return JSON only:
{{
  "replan_required": boolean,
  "confidence": 0.0-1.0,
  "reason": "short explanation"
}}

Be conservative.
"""

        try:
            raw = self._call_llm_with_timeout(prompt)
            decision = json.loads(raw.strip())
        except Exception:
            return self._no_replan("LLM failure")

        if not isinstance(decision, dict):
            return self._no_replan("Invalid schema")

        replan_required = bool(decision.get("replan_required", False))

        try:
            confidence = float(decision.get("confidence", 0.0))
        except Exception:
            confidence = 0.0

        confidence = max(0.0, min(confidence, 1.0))

        reason = str(decision.get("reason", ""))[:300]

        if replan_required:
            self._last_replan_snapshot_hash = snapshot_hash

        return {
            "replan_required": replan_required,
            "confidence": confidence,
            "reason": reason,
        }

    # ==================================================
    # INTERNAL HELPERS
    # ==================================================

    def _no_replan(self, reason: str) -> Dict[str, Any]:
        return {
            "replan_required": False,
            "confidence": 0.0,
            "reason": reason[:300],
        }

    def _hash_snapshot(self, snapshot: Dict[str, Any]) -> str:
        try:
            raw = json.dumps(snapshot, sort_keys=True)
        except Exception:
            raw = str(snapshot)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _call_llm_with_timeout(self, prompt: str) -> str:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._llm_call, prompt)
            try:
                result = future.result(timeout=self.LLM_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                raise PlanningError("LLM call timeout")

        if not isinstance(result, str):
            raise PlanningError("LLM must return string")

        return result

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

        return "\n".join(text_chunks)[: self.MAX_SCREEN_CHARS]

    # ==================================================
    # GOAL EXPANSION
    # ==================================================

    def _expand_goal(self, goal: str) -> List[Dict[str, Any]]:
        screen_context = self._read_screen_context()

        prompt = f"""
You are the deterministic planning brain.

Environment:
{json.dumps(self._environment)}

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
- No DONE
- No tool_installation
- Must include verification
- Conservative
"""

        raw = self._call_llm_with_timeout(prompt)

        try:
            data = json.loads(raw)
        except Exception:
            raise PlanningError("Invalid JSON from LLM")

        if not isinstance(data, list) or not data:
            raise PlanningError("LLM produced invalid step list")

        allowed_types = {
            StepType.UI_INTERACTION.value,
            StepType.COMMAND_EXECUTION.value,
            StepType.FILE_CREATION.value,
            StepType.VERIFICATION.value,
        }

        validated: List[Dict[str, Any]] = []

        for step in data:
            if not isinstance(step, dict):
                raise PlanningError("Invalid step format")

            step_type = step.get("type")
            if step_type not in allowed_types:
                raise PlanningError("Invalid step type")

            description = step.get("description")
            action = step.get("action")
            verification = step.get("verification")

            if not isinstance(description, str) or not description.strip():
                raise PlanningError("Invalid description")

            if not isinstance(action, dict):
                raise PlanningError("Invalid action")

            if not isinstance(verification, dict):
                raise PlanningError("Invalid verification")

            try:
                duration = float(step.get("estimated_duration", 0.0))
            except Exception:
                raise PlanningError("Invalid duration")

            if duration < 0 or duration > self.MAX_ESTIMATED_DURATION:
                raise PlanningError("Duration out of bounds")

            validated.append(
                {
                    "type": StepType(step_type),
                    "description": description.strip(),
                    "action": action,
                    "verification": verification,
                    "estimated_duration": duration,
                    "retryable": bool(step.get("retryable", True)),
                }
            )

        return validated
