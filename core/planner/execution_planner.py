from typing import List, Dict, Any, Optional
import json
import time
import concurrent.futures
import re

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
    Hardened against malformed LLM output and thread leakage.
    """

    LLM_TIMEOUT_SECONDS = 30.0
    MAX_SCREEN_CHARS = 500
    MAX_ESTIMATED_DURATION = 600.0
    MAX_STEPS_PER_GOAL = 25

    SAFE_ENV_FIELDS = {
        "os",
        "architecture",
        "display_available",
        "tools",
        "running_in_container",
        "running_in_wsl",
        "ci_environment",
    }

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
        self._world_snapshot: Dict[str, Any] = {}

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
            frozen = json.loads(json.dumps(new_snapshot, sort_keys=True))
        except Exception:
            raise PlanningError("Failed to freeze snapshot")

        self._world_snapshot = frozen

    # ==================================================
    # LLM CALL (SAFE + BOUNDED)
    # ==================================================

    def _call_llm_with_timeout(self, prompt: str) -> str:

        payload = [
            {"role": "system", "content": "You are a deterministic planner."},
            {"role": "user", "content": prompt},
        ]

        def _invoke():
            result = self._llm_call(payload, None, "planning")

            if isinstance(result, tuple) and len(result) == 2:
                ops, err = result
                if err:
                    raise PlanningError(f"LLM adapter error: {err}")
                return ops

            return result

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_invoke)

        try:
            result = future.result(timeout=self.LLM_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise PlanningError("LLM call timeout")
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        if isinstance(result, str):
            return result.strip()

        if isinstance(result, list):
            return json.dumps(result)

        if isinstance(result, dict):
            content = result.get("content")
            if isinstance(content, str):
                return content.strip()

        raise PlanningError("Unsupported LLM return type")

    # ==================================================
    # TOOL EXTRACTION
    # ==================================================

    def _extract_required_tools(self, requirements: Dict[str, Any]) -> List[str]:
        tools = requirements.get("tools", [])
        if not isinstance(tools, list):
            return []
        return [t.strip() for t in tools if isinstance(t, str) and t.strip()]

    # ==================================================
    # SCREEN CONTEXT
    # ==================================================

    def _read_screen_context(self) -> str:
        entities = self._world_snapshot.get("entities", [])
        if not isinstance(entities, list):
            return ""

        text_chunks: List[str] = []

        for ent in entities:
            if not isinstance(ent, dict):
                continue
            label = ent.get("text")
            etype = ent.get("type")
            if isinstance(label, str) and label.strip():
                text_chunks.append(f"{etype}: {label}")

        return "\n".join(text_chunks)[: self.MAX_SCREEN_CHARS]

    # ==================================================
    # ROBUST JSON EXTRACTION
    # ==================================================

    def _extract_json(self, raw: str) -> Any:
        raw = raw.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # non-greedy fallback
            match = re.search(r"(\{.*?\}|\[.*?\])", raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except Exception:
                    pass

        raise PlanningError("Invalid JSON from LLM")

    # ==================================================
    # GOAL EXPANSION
    # ==================================================

    def _expand_goal(self, goal: str) -> List[Dict[str, Any]]:

        screen_context = self._read_screen_context()

        safe_env = {
            k: v for k, v in self._environment.items()
            if k in self.SAFE_ENV_FIELDS
        }

        prompt = f"""
You are a deterministic execution planner.

Environment:
{json.dumps(safe_env)}

Frozen screen:
{screen_context}

Goal:
"{goal}"

Return STRICT JSON list of steps.

Each step MUST contain:
- type
- description
- action
- verification
- estimated_duration
- retryable

Return JSON only.
"""

        raw = self._call_llm_with_timeout(prompt)
        data = self._extract_json(raw)

        if not isinstance(data, list) or not data:
            raise PlanningError("LLM produced invalid step list")

        allowed_types = {
            StepType.UI_INTERACTION.value,
            StepType.COMMAND_EXECUTION.value,
            StepType.FILE_CREATION.value,
            StepType.VERIFICATION.value,
            StepType.TOOL_INSTALLATION.value,
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

            # Minimal semantic validation
            if step_type == StepType.COMMAND_EXECUTION.value:
                if "command" not in action:
                    raise PlanningError("COMMAND_EXECUTION missing 'command'")

            if step_type == StepType.UI_INTERACTION.value:
                if "operation" not in action:
                    raise PlanningError("UI_INTERACTION missing 'operation'")

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
