from typing import List, Dict, Any, Optional
import json
import time
import re
import asyncio

from config.timeouts import LLM_CALL_TIMEOUT_SECONDS

from core.schemas.execution_plan import (
    ExecutionPlan,
    ExecutionStep,
    StepType,
)


class PlanningError(RuntimeError):
    pass


class ExecutionPlanner:

    MAX_SCREEN_CHARS = 500
    MAX_ESTIMATED_DURATION = 600.0
    MAX_STEPS_PER_GOAL = 25
    MAX_COMMAND_LENGTH = 512

    SAFE_ENV_FIELDS = {
        "os",
        "architecture",
        "display_available",
        "tools",
        "running_in_container",
        "running_in_wsl",
        "ci_environment",
    }

    # Strict destructive patterns
    DANGEROUS_PATTERNS = [
        r"\brm\s+-rf\b",
        r"\bsudo\b",
        r"\bdd\b",
        r"\bmkfs\b",
        r"\bformat\b",
        r"\bchmod\s+777\b",
        r"\bpython\s*-c\b",
        r"\bpython3\s*-c\b",
        r"\bbash\s*-c\b",
        r"\bsh\s*-c\b",
        r"\beval\b",
        r"\bexec\b",
        r"\bnc\b",
        r"\bnetcat\b",
        r"\bcrontab\b",
        r"\bat\b",
        r"\bbase64\b.*-d",
        r"\$\(",
        r";",
        r"&&",
        r"\|\|",
        r"\|",
        r">",
        r"<",
    ]

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
    # SAFE ASYNC LLM CALL (NO asyncio.run INSIDE LOOP)
    # ==================================================

    async def _call_llm_async(self, prompt: str) -> str:

        payload = [
            {"role": "system", "content": "You are a deterministic planner."},
            {"role": "user", "content": prompt},
        ]

        loop = asyncio.get_running_loop()

        def _invoke():
            return self._llm_call(payload, None, "planning")

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, _invoke),
                timeout=LLM_CALL_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            raise PlanningError("LLM call timeout")

        if isinstance(result, str):
            return result.strip()

        if isinstance(result, list):
            return json.dumps(result)

        if isinstance(result, dict):
            content = result.get("content")
            if isinstance(content, str):
                return content.strip()

        raise PlanningError("Unsupported LLM return type")

    def _call_llm_sync(self, prompt: str) -> str:
        try:
            loop = asyncio.get_running_loop()
            # already inside loop → schedule properly
            future = asyncio.run_coroutine_threadsafe(
                self._call_llm_async(prompt),
                loop,
            )
            return future.result(timeout=LLM_CALL_TIMEOUT_SECONDS)
        except RuntimeError:
            # no running loop → safe to create one
            return asyncio.run(self._call_llm_async(prompt))

    # ==================================================

    def _extract_required_tools(self, requirements: Dict[str, Any]) -> List[str]:
        tools = requirements.get("tools", [])
        if not isinstance(tools, list):
            return []
        return [t.strip() for t in tools if isinstance(t, str) and t.strip()]

    # ==================================================

    def _read_screen_context(self) -> str:
        entities = self._world_snapshot.get("entities", [])
        if not isinstance(entities, list):
            return ""

        chunks: List[str] = []

        for ent in entities:
            if not isinstance(ent, dict):
                continue
            label = ent.get("text")
            etype = ent.get("type")
            if isinstance(label, str) and label.strip():
                chunks.append(f"{etype}: {label}")

        return "\n".join(chunks)[: self.MAX_SCREEN_CHARS]

    # ==================================================

    def _extract_json(self, raw: str) -> Any:
        raw = raw.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            raise PlanningError("Invalid JSON from LLM")

    # ==================================================

    def _validate_command(self, cmd: str) -> None:

        cmd = cmd.strip()

        if not cmd:
            raise PlanningError("Empty command")

        if len(cmd) > self.MAX_COMMAND_LENGTH:
            raise PlanningError("Command too long")

        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, cmd, re.IGNORECASE):
                raise PlanningError("Dangerous command detected")

    # ==================================================

    def _expand_goal(self, goal: str) -> List[Dict[str, Any]]:

        screen_context = self._read_screen_context()

        safe_env = {
            k: v for k, v in self._environment.items()
            if k in self.SAFE_ENV_FIELDS
        }

        prompt = f"""
Environment:
{json.dumps(safe_env)}

Screen:
{screen_context}

Goal:
"{goal}"

Return STRICT JSON list of steps.
"""

        raw = self._call_llm_sync(prompt)
        data = self._extract_json(raw)

        if not isinstance(data, list) or not data:
            raise PlanningError("Invalid step list")

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

            action = step.get("action")
            if not isinstance(action, dict):
                raise PlanningError("Invalid action")

            if step_type == StepType.COMMAND_EXECUTION.value:
                cmd = action.get("command")
                if not isinstance(cmd, str):
                    raise PlanningError("Missing command")
                self._validate_command(cmd)

            try:
                duration = float(step.get("estimated_duration", 0.0))
            except Exception:
                raise PlanningError("Invalid duration")

            if duration < 0 or duration > self.MAX_ESTIMATED_DURATION:
                raise PlanningError("Duration out of bounds")

            description = step.get("description", "")
            if not isinstance(description, str) or not description.strip():
                raise PlanningError("Invalid description")

            validated.append(
                {
                    "type": StepType(step_type),
                    "description": description.strip(),
                    "action": action,
                    "verification": step.get("verification", {}),
                    "estimated_duration": duration,
                    "retryable": bool(step.get("retryable", True)),
                }
            )

        return validated
