"""
core/planner/execution_planner.py
===================================
PATCHES APPLIED (Audit Fixes):

  ✅  §R3  (was §planner-6): TaskDecomposer is now wired in.
           For complex objectives (>100 chars or explicit multi-step tasks),
           decompose() is called first to break the intent into ordered
           sub-goals. _expand_goal() is then called per sub-goal.
           This prevents single-shot planning failures on 20+ step tasks.

  ✅  §Evo4 (was §planner-7): Removed '$(' from DANGEROUS_PATTERNS.
           Command substitution $(...) is standard in legitimate install
           scripts.  Replaced with targeted pattern 'eval.*$(' which is
           the genuinely dangerous form.  curl|bash is kept as it is an
           audit-logged risk, not a hard block.

  ✅  §1.9 (original): MAX_COMMAND_LENGTH raised from 512 to 2048.
  ✅  §1.9 (original): _STEP_SCHEMA_BLOCK with full schema injected.
  ✅  §1.9 (original): _call_llm_sync raises if inside event loop.
  ✅  §1.9 (original): _call_llm_async exposed as proper coroutine.

All existing correct behaviours preserved:
  - ThreadPoolExecutor(max_workers=1)
  - Step type validation against StepType enum
  - Duration bounds [0, 600]
  - MAX_STEPS_PER_GOAL=25 (per sub-goal)
  - SAFE_ENV_FIELDS filtering
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
import json
import time
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor

from config.timeouts import LLM_CALL_TIMEOUT_SECONDS

from core.schemas.execution_plan import (
    ExecutionPlan,
    ExecutionStep,
    StepType,
)


class PlanningError(RuntimeError):
    pass


# -----------------------------------------------------------------------
# COMPLETE STEP SCHEMA — injected into every LLM planning prompt.
# -----------------------------------------------------------------------
_STEP_SCHEMA_BLOCK = """\
STEP SCHEMA (return exactly this structure for every element):
{
  "type": <string, one of: "ui_interaction" | "command_execution" | "file_creation" | "verification" | "tool_installation">,
  "description": <string, plain English description of what this step does>,
  "estimated_duration": <float, seconds this step is expected to take, 0.0–600.0>,
  "retryable": <boolean, true if the step is safe to retry on failure>,
  "verification": {
    "expected_state": <string, what the screen/system should look like after success>,
    "version_command": <string, optional shell command to verify (e.g. "node --version")>
  },
  "action": <object, shape depends on "type" — see below>
}

ACTION SHAPES BY TYPE:
  "ui_interaction":
    { "operation": "click|type|hotkey|scroll", "text": "...", "keys": [...] }

  "command_execution":
    { "operation": "command", "command": "<shell command string>" }

  "file_creation":
    { "operation": "file_create", "path": "<absolute path>", "content": "<file content>" }

  "verification":
    { "operation": "verify", "method": "screenshot|command", "command": "<optional>" }

  "tool_installation":
    {
      "operation": "install",
      "tool": {
        "name": "<tool name>",
        "official_url": "https://<official download page>",
        "version_command": "<e.g. node --version>",
        "min_version": "<optional semver string>",
        "install_commands": ["<platform-specific shell command>"]
      }
    }

RULES:
  - Return ONLY a JSON array (no prose, no markdown fences).
  - Every element must match the schema above exactly.
  - "type" must be one of the 5 values listed — no other values permitted.
  - Do not include a "done" step — it is appended automatically.
  - Prefer "command_execution" for CLI-based installs (apt, brew, npm, pip).
  - For tool installation via browser UI use "tool_installation".
  - For "tool_installation", always include "install_commands" with the
    recommended CLI install command for the current OS if one exists.
    This enables terminal-first installation without browser UI.
"""


class ExecutionPlanner:

    MAX_SCREEN_CHARS = 2000
    MAX_ESTIMATED_DURATION = 600.0
    MAX_STEPS_PER_GOAL = 25

    # Raised from 512 to 2048 to accommodate real-world install commands
    # e.g. `curl -fsSL https://deb.nodesource.com/setup_20.x | bash`
    MAX_COMMAND_LENGTH = 2048

    # Threshold: objectives longer than this character count are treated as
    # complex and routed through TaskDecomposer before planning.
    DECOMPOSE_THRESHOLD_CHARS = 100

    SAFE_ENV_FIELDS = {
        "os",
        "architecture",
        "display_available",
        "tools",
        "running_in_container",
        "running_in_wsl",
        "ci_environment",
    }

    # -------------------------------------------------------
    # DANGEROUS_PATTERNS — PATCH §Evo4
    #
    # REMOVED: '$(' — blocks legitimate command substitution
    #          e.g. `nvm install $(cat .nvmrc)` is valid
    #
    # RETAINED: genuinely destructive / privilege-escalation patterns
    # -------------------------------------------------------
    DANGEROUS_PATTERNS = [
        r"\brm\s+-rf\b",
        r"\bdd\b",
        r"\bmkfs\b",
        r"\bformat\b",
        r"\bchmod\s+777\b",
        r"\bnc\b",
        r"\bnetcat\b",
        r"\bcrontab\b",
        r"^\s*at\s",
        r"\bbase64\b.*-d",
        r"\beval\b.*\$\(",    # PATCH §Evo4: only block eval-with-substitution, not bare $()
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
        self._executor = ThreadPoolExecutor(max_workers=1)

        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.DANGEROUS_PATTERNS
        ]

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

    def update_world_snapshot(self, snapshot: Dict[str, Any]):
        if isinstance(snapshot, dict):
            self._world_snapshot = snapshot

    def refresh_environment(self, new_fingerprint: Dict[str, Any]) -> None:
        """
        PATCH §R5: Refresh environment fingerprint after tool installs.
        Called from main.py after successful execution so replans see
        newly installed tools (e.g. node, python3) and do not redundantly
        reinstall them.
        """
        if isinstance(new_fingerprint, dict):
            self._environment = new_fingerprint

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

        # PATCH §R3: For complex single-goal tasks, decompose into sub-goals first.
        # This prevents single-shot planning failures on long multi-step tasks.
        if (
            len(high_level_steps) == 1
            and isinstance(high_level_steps[0].get("goal"), str)
            and len(high_level_steps[0]["goal"]) > self.DECOMPOSE_THRESHOLD_CHARS
        ):
            high_level_steps = self._decompose_if_complex(
                high_level_steps[0]["goal"]
            )

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
    # PATCH §R3: DECOMPOSITION BRIDGE
    # ==================================================

    def _decompose_if_complex(
        self, objective: str
    ) -> List[Dict[str, Any]]:
        """
        Invoke TaskDecomposer to break a complex objective into ordered
        sub-goals. Falls back to single-step pass-through on any error
        so planner remains functional even if decomposer fails.
        """
        try:
            from core.planner.task_decomposer import TaskDecomposer

            def _llm_text_call(prompt: str) -> str:
                """
                Adapter: TaskDecomposer passes a plain string prompt and
                expects a plain string response.  We wrap the llm_callable
                (which expects messages list) accordingly.
                """
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a task decomposition engine. "
                            "Return ONLY valid JSON. No prose. No markdown."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ]
                result = self._llm_call(messages, None, "decomposition")
                # Normalise result to string
                if isinstance(result, str):
                    return result
                if isinstance(result, list):
                    return json.dumps(result)
                if isinstance(result, dict):
                    c = result.get("content")
                    if isinstance(c, str):
                        return c
                return json.dumps(result)

            decomposer = TaskDecomposer(llm_call=_llm_text_call)
            sub_goals = decomposer.decompose(objective)
            return [{"goal": s["goal"]} for s in sub_goals]

        except Exception:
            # Decomposition failed — fall back to single-goal pass-through
            return [{"goal": objective}]

    # ==================================================
    # LOOP-SAFE LLM CALL
    # ==================================================

    async def _call_llm_async(self, prompt: str) -> str:

        payload = [
            {"role": "system", "content": "You are a deterministic planner. Return only valid JSON arrays — no prose, no markdown."},
            {"role": "user", "content": prompt},
        ]

        loop = asyncio.get_running_loop()

        def _invoke():
            return self._llm_call(payload, None, "planning")

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(self._executor, _invoke),
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
        """
        Raises clearly if called inside a running event loop.
        Callers in async context should use `await _call_llm_async()` directly.
        """
        try:
            asyncio.get_running_loop()
            raise PlanningError(
                "ExecutionPlanner._call_llm_sync() must not be called "
                "inside a running event loop. Use _call_llm_async() instead."
            )
        except RuntimeError:
            # No running loop — safe to call asyncio.run()
            return asyncio.run(self._call_llm_async(prompt))

    # ==================================================

    def _extract_required_tools(self, requirements: Dict[str, Any]) -> List[str]:
        tools = requirements.get("tools", [])
        if not isinstance(tools, list):
            return []
        return [t.strip() for t in tools if isinstance(t, str) and t.strip()]

    # ==================================================
    # Deterministic prioritised screen context
    # ==================================================

    def _read_screen_context(self) -> str:
        entities = self._world_snapshot.get("entities", [])
        if not isinstance(entities, list):
            return ""

        def score(ent):
            text = ent.get("text", "")
            etype = ent.get("type", "")
            return (
                1 if etype in ("button", "input", "link") else 0,
                len(text),
            )

        ordered = sorted(
            [e for e in entities if isinstance(e, dict)],
            key=score,
            reverse=True,
        )

        chunks: List[str] = []
        for ent in ordered:
            label = ent.get("text")
            etype = ent.get("type")
            if isinstance(label, str) and label.strip():
                chunks.append(f"{etype}: {label.strip()}")

        return "\n".join(chunks)[: self.MAX_SCREEN_CHARS]

    # ==================================================

    def _extract_json(self, raw: str) -> Any:
        raw = raw.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()

        if len(raw) > 50_000:
            raise PlanningError("LLM response too large")

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PlanningError(f"Invalid JSON from LLM: {exc}") from exc

    # ==================================================

    def _validate_command(self, cmd: str) -> None:
        cmd = cmd.strip()

        if not cmd:
            raise PlanningError("Empty command")

        if len(cmd) > self.MAX_COMMAND_LENGTH:
            raise PlanningError(
                f"Command too long ({len(cmd)} chars, max {self.MAX_COMMAND_LENGTH})"
            )

        for pattern in self._compiled_patterns:
            if pattern.search(cmd):
                raise PlanningError(f"Dangerous command detected: {cmd[:80]!r}")

    # ==================================================
    # GOAL EXPANSION
    # ==================================================

    def _expand_goal(self, goal: str) -> List[Dict[str, Any]]:
        screen_context = self._read_screen_context()

        safe_env = {
            k: v
            for k, v in self._environment.items()
            if k in self.SAFE_ENV_FIELDS
        }

        prompt = (
            f"Environment:\n{json.dumps(safe_env, indent=2)}\n\n"
            f"Screen (visible UI elements):\n{screen_context or '(none visible)'}\n\n"
            f"Goal:\n\"{goal}\"\n\n"
            f"{_STEP_SCHEMA_BLOCK}\n"
            f"Return ONLY the JSON array of steps for this goal. "
            f"No preamble, no trailing text."
        )

        raw = self._call_llm_sync(prompt)
        data = self._extract_json(raw)

        if not isinstance(data, list) or not data:
            raise PlanningError("LLM returned empty or non-list step plan")

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
                raise PlanningError(f"Step is not a dict: {step!r}")

            step_type = step.get("type")
            if step_type not in allowed_types:
                raise PlanningError(
                    f"Invalid step type '{step_type}'. "
                    f"Allowed: {sorted(allowed_types)}"
                )

            action = step.get("action")
            if not isinstance(action, dict):
                raise PlanningError(
                    f"Step 'action' must be a dict, got: {type(action)}"
                )

            if step_type == StepType.COMMAND_EXECUTION.value:
                cmd = action.get("command")
                if not isinstance(cmd, str):
                    raise PlanningError("command_execution step missing 'command' string")
                self._validate_command(cmd)

            if step_type == StepType.FILE_CREATION.value:
                path = action.get("path")
                if not isinstance(path, str) or not path.strip():
                    raise PlanningError("file_creation step missing 'path' string")
                content = action.get("content")
                if not isinstance(content, str):
                    raise PlanningError("file_creation step 'content' must be a string")

            if step_type == StepType.TOOL_INSTALLATION.value:
                tool = action.get("tool")
                if not isinstance(tool, dict):
                    raise PlanningError("tool_installation step missing 'tool' dict")
                if not isinstance(tool.get("name"), str):
                    raise PlanningError("tool_installation.tool.name must be a string")
                official_url = tool.get("official_url", "")
                if not isinstance(official_url, str) or not official_url.startswith("https://"):
                    raise PlanningError(
                        "tool_installation.tool.official_url must be an https:// URL. "
                        "Provide the tool's official download page."
                    )

            try:
                duration = float(step.get("estimated_duration", 0.0))
            except Exception:
                raise PlanningError("estimated_duration must be a float")

            if duration < 0 or duration > self.MAX_ESTIMATED_DURATION:
                raise PlanningError(
                    f"estimated_duration {duration} out of bounds [0, {self.MAX_ESTIMATED_DURATION}]"
                )

            description = step.get("description", "")
            if not isinstance(description, str) or not description.strip():
                raise PlanningError("Step 'description' must be a non-empty string")

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
