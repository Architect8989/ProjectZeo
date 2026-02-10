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
    - No side effects, no execution, no guessing
    """

    LLM_TIMEOUT_SECONDS = 30.0
    MAX_SCREEN_CHARS = 500  # bounded advisory context

    def __init__(
        self,
        *,
        llm_call,
        environment_fingerprint: Optional[Dict[str, Any]] = None,
        observer=None,
        world_graph=None,
    ):
        if not callable(llm_call):
            raise PlanningError("llm_call must be callable")

        self._llm_call = llm_call
        self._environment = environment_fingerprint or {}
        self._observer = observer
        self._world_graph = world_graph

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
                description = spec.get("description", "").strip()
                if not description:
                    raise PlanningError(
                        "LLM produced step without description"
                    )

                deps = [last_step_id] if last_step_id else []

                step = ExecutionStep(
                    id=step_id,
                    type=spec["type"],
                    description=description,
                    action=spec.get("action", {}),
                    verification=spec.get("verification", {}),
                    dependencies=deps,
                    estimated_duration=spec.get("estimated_duration", 0.0),
                    retryable=spec.get("retryable", True),
                )

                execution_steps.append(step)
                last_step_id = step_id
                step_id += 1

        if not execution_steps:
            raise PlanningError("No executable steps generated")

        # ---- mandatory DONE step ----
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
        """
        Read-only grounding context.

        Uses world graph snapshot if available.
        Advisory only — never authoritative.
        """
        if not self._world_graph:
            return ""

        try:
            snapshot = self._world_graph.snapshot()
        except Exception:
            return ""

        text_chunks: List[str] = []

        for ent in snapshot.get("entities", []):
            label = ent.get("text")
            etype = ent.get("type")
            if isinstance(label, str) and label.strip():
                text_chunks.append(f"{etype}: {label}")

        context = "\n".join(text_chunks)
        return context[: self.MAX_SCREEN_CHARS]

    # ==================================================
    # LLM-POWERED GOAL EXPANSION (TIME-BOUNDED)
    # ==================================================

    def _expand_goal(self, goal: str) -> List[Dict[str, Any]]:
        screen_context = self._read_screen_context()

        prompt = f"""
You are the planning brain of a self-operating computer.

Environment fingerprint:
{json.dumps(self._environment, indent=2)}

Current screen state (may be empty):
{screen_context}

Task:
"{goal}"

Return ONLY valid JSON.

Schema:
[
  {{
    "type": "ui_interaction" | "command_execution" | "file_creation" | "tool_installation" | "verification",
    "description": "...",
    "action": {{ }},
    "verification": {{ }},
    "estimated_duration": number,
    "retryable": boolean
  }}
]

Rules:
- No hallucinated tools
- If a tool is required, emit tool_installation step
- Every step must be verifiable
- Be conservative and explicit
"""

        # ---- HARD TIMEOUT ENFORCEMENT ----
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

        for idx, step in enumerate(data):
            if not isinstance(step, dict):
                raise PlanningError(f"Invalid step at index {idx}")

            step_type = step.get("type")
            if step_type not in {t.value for t in StepType}:
                raise PlanningError(f"Invalid step type: {step_type}")

            validated.append(
                {
                    "type": StepType(step_type),
                    "description": step.get("description", ""),
                    "action": step.get("action", {}),
                    "verification": step.get("verification", {}),
                    "estimated_duration": float(
                        step.get("estimated_duration", 0.0)
                    ),
                    "retryable": bool(step.get("retryable", True)),
                }
            )

        return validated
