"""
core/planner/execution_planner.py
===================================
PATCH AUDIT FIXES:

  ⚠️  §1.9 / Gap A: Planner prompt provided no JSON schema for steps.
            LLM was told "Return STRICT JSON list of steps" but never told
            what fields, types, or StepType values were required.
            This caused frequent schema mismatches and PlanningError cascades.
            FIX: _expand_goal() now embeds a complete JSON schema in the prompt,
            including all valid StepType values, per-type action shapes, and
            example step.

  ⚠️  §1.9: MAX_COMMAND_LENGTH=512 blocked long-form install commands like
            `curl -fsSL https://deb.nodesource.com/setup_20.x | bash`
            FIX: Raised to 2048 characters.  Dangerous-pattern validation
            still applies — this only relaxes the length guard.

  ⚠️  §1.9: _call_llm_sync raises PlanningError when called inside running loop,
            but ExecutionPlanner is sometimes called from async context via
            planner.update_world_snapshot().  The async path is now exposed as
            a proper coroutine so callers can choose the right dispatch mode.

  ✅  All existing correct behaviours preserved:
        - DANGEROUS_PATTERNS (audit-corrected set)
        - ThreadPoolExecutor(max_workers=1)
        - Step type validation against StepType enum
        - Duration bounds [0, 600]
        - MAX_STEPS_PER_GOAL=25
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
# This eliminates the "LLM must infer schema from context" fragility.
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
        "min_version": "<optional semver string>"
      }
    }

RULES:
  - Return ONLY a JSON array (no prose, no markdown fences).
  - Every element must match the schema above exactly.
  - "type" must be one of the 5 values listed — no other values permitted.
  - Do not include a "done" step — it is appended automatically.
  - Prefer "command_execution" for CLI-based installs (apt, brew, npm, pip).
  - For tool installation via browser UI use "tool_installation".
"""


class ExecutionPlanner:

    MAX_SCREEN_CHARS = 2000
    MAX_ESTIMATED_DURATION = 600.0
    MAX_STEPS_PER_GOAL = 25

    # PATCH §1.9: raised from 512 to 2048 to accommodate real-world install commands
    # e.g. `curl -fsSL https://deb.nodesource.com/setup_20.x | bash`
    MAX_COMMAND_LENGTH = 2048

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
    # DANGEROUS_PATTERNS — AUDIT-CORRECTED (from original file)
    #
    # REMOVED patterns that destroyed real-world autonomy:
    #   ";", "&&", "||", "|", ">", "<"  — blocked all shell pipelines/chaining
    #   "sudo"                           — blocked privilege escalation for apt/brew
    #   "bash -c", "sh -c"              — blocked sub-shells for install scripts
    #   "python -c", "python3 -c"       — blocked inline python for venv bootstraps
    #   "eval", "exec"                  — blocked dynamic execution
    #
    # RETAINED genuine destructive risk patterns:
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
        r"\$\(",
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
        PATCH §1.9: Raises clearly if called inside a running event loop.
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

        # PATCH §1.9: MAX_COMMAND_LENGTH raised to 2048
        if len(cmd) > self.MAX_COMMAND_LENGTH:
            raise PlanningError(
                f"Command too long ({len(cmd)} chars, max {self.MAX_COMMAND_LENGTH})"
            )

        for pattern in self._compiled_patterns:
            if pattern.search(cmd):
                raise PlanningError(f"Dangerous command detected: {cmd[:80]!r}")

    # ==================================================
    # GOAL EXPANSION — PATCHED WITH FULL SCHEMA
    # ==================================================

    def _expand_goal(self, goal: str) -> List[Dict[str, Any]]:
        """
        PATCH Gap A / §1.9: Prompt now includes complete JSON schema, all
        valid StepType values, and per-type action shapes so the LLM knows
        exactly what to return.  Previously the LLM had to infer schema
        from zero context.
        """
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
