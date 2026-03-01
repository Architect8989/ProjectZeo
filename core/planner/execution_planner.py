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

from core.security.injection_markers import INJECTION_MARKERS, normalize_for_injection_check


class PlanningError(RuntimeError):
    pass


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
    MAX_COMMAND_LENGTH = 2048
    DECOMPOSE_THRESHOLD_CHARS = 60

    SAFE_ENV_FIELDS = {
        "os",
        "architecture",
        "display_available",
        "tools",
        "running_in_container",
        "running_in_wsl",
        "ci_environment",
    }

    DANGEROUS_PATTERNS = [
        r"\brm\s+-rf\b",
        r"\bdd\b",
        r"\bmkfs\b",
        r"\bformat\b",
        r"\bchmod\s+777\b",
        r"\bchown\s+root\b",
        r"\bnc\b",
        r"\bnetcat\b",
        r"\bcrontab\b",
        r"^\s*at\s",
        r"\bperl\s+-e\b",
        r"\bruby\s+-e\b",
        r"\bnode\s+-e\b",
        r"\bpython[23]?\s+-c\b",
        r"\bbase64\b.*-d",
        r"\beval\b.*\$\(",
        r"\bpowershell\b.*-[Ee]ncodedCommand\b",
        r"\bpowershell\b.*-[Ee]nc\s",
        r"[&|]\s*chmod\s+[+]?x\b.*[&|].*\./",
        r"\|\s*perl\b",
        r"\|\s*ruby\b",
        r"\|\s*node\b",
        r"\|\s*python[23]?\b",
        
        r"\bcurl\b.*\|\s*(?:ba)?sh\b",
        r"\bcurl\b.*\|\s*bash\b",
        r"\bwget\b.*-[Oo]-?\s.*\|\s*(?:ba)?sh\b",
        r"\bwget\b.*--output-document\s*-\s.*\|\s*(?:ba)?sh\b",
        r"\|\s*sh\b",        # broad: any pipe directly into sh
        r"\|\s*bash\b",      # broad: any pipe directly into bash
        #
        # Reverse shell patterns:
        r"bash\s+-[ic]\s+['\"]?>?&\s*/dev/tcp/",     # bash TCP reverse shell
        r"/dev/tcp/",                                   # any /dev/tcp reference
        r"/dev/udp/",                                   # any /dev/udp reference
        r"(?:nc|ncat|socat)\b.*-[el]\b",              # netcat/socat listener
        r"\bsocat\b.*EXEC:",                            # socat exec binding
        #
        # Python/exec base64 double-encoding bypass:
        r"exec\s*\(\s*(?:__import__\s*\(\s*['\"]base64['\"]|base64)\b",
        r"python[23]?\s+-c\s+['\"].*exec\s*\(",       # python -c 'exec(...)'
        r"\bexec\s*\(.*decode\s*\(",                   # exec(b64.decode())
        #
        # Privilege escalation shortcuts:
        r"\bsudo\s+su\b",
        r"\bsudo\s+-[isS]\b",                          # sudo -i / -s / -S shell
        r"\bsu\s+-[cl]\b",                             # su -c 'cmd' or su -l
        
        r"base64\s+--decode\b",
        r"base64\s+-D\b",     # macOS base64 -D flag
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

        
        self._trusted_installer_commands: frozenset = self._build_trusted_installer_allowlist()

        self._model_name: Optional[str] = self._extract_model_name(llm_call)
        self._text_model_name: str = self._derive_text_model_name(self._model_name)

        
        self._text_model_is_vision_fallback: bool = (
            self._text_model_name == self._model_name
            and self._model_name is not None
        )
        if self._text_model_is_vision_fallback:
            import sys as _sys
            print(
                f"[ExecutionPlanner] RT-A4: Text model absent — routing text-only "
                f"planning calls through vision adapter ({self._model_name!r}). "
                "Pull the text variant or set LLM_TEXT_MODEL to suppress this.",
                file=_sys.stderr,
            )

        self._ollama_client = None
        try:
            import ollama as _ollama_mod
            import httpx as _httpx_mod
            self._ollama_client = _ollama_mod.Client(
                timeout=_httpx_mod.Timeout(connect=10.0, read=120.0, write=5.0, pool=2.0)
            )
        except Exception:
            self._ollama_client = None

        if world_graph is not None:
            self.update_world_snapshot(world_graph.snapshot())
        else:
            self._world_snapshot = {
                "entities": [],
                "focused_app": None,
                "entity_count": 0,
                "timestamp": None,
            }

    @staticmethod
    def _build_trusted_installer_allowlist() -> frozenset:
        
        try:
            from core.tools.autonomous_installer import COMMON_INSTALL_COMMANDS
            trusted: set = set()
            for _tool_cmds in COMMON_INSTALL_COMMANDS.values():
                for _cmd in _tool_cmds.values():
                    if isinstance(_cmd, str) and _cmd.strip():
                        trusted.add(_cmd.strip())
            return frozenset(trusted)
        except Exception:
            return frozenset()

    @staticmethod
    def _extract_model_name(llm_call) -> Optional[str]:
        import os as _os
        fn = llm_call
        while hasattr(fn, "__wrapped__"):
            fn = fn.__wrapped__
        adapter = getattr(fn, "__self__", None)
        if adapter is not None:
            name = getattr(adapter, "model_name", None)
            if isinstance(name, str) and name.strip():
                return name.strip()
        return _os.environ.get("LLM_MODEL", "qwen2.5-vl:7b-instruct")

    @staticmethod
    def _derive_text_model_name(vision_model_name: Optional[str]) -> str:
        import os as _os
        env_override = _os.environ.get("LLM_TEXT_MODEL", "").strip()
        if env_override:
            return env_override

        if not vision_model_name:
            return _os.environ.get("LLM_MODEL", "qwen2.5-vl:7b-instruct")

        text_candidate = vision_model_name.replace("-vl:", ":").replace("-vl", "")
        if text_candidate != vision_model_name:
            try:
                import ollama as _ollama
                _models = _ollama.list()
                _available = {
                    m.model if hasattr(m, "model") else str(m)
                    for m in (_models.models if hasattr(_models, "models") else [])
                }
                _base_candidate = text_candidate.split(":")[0]
                _found = any(
                    text_candidate in name or _base_candidate in name
                    for name in _available
                )
                if _found:
                    return text_candidate
            except Exception:
                pass

        
        import sys as _sys
        print(
            f"[ExecutionPlanner] WARNING: No separate text model found for {vision_model_name!r}. "
            "Using vision model for text-only planning calls. Performance will be degraded. "
            "To suppress: (1) pull the text variant (remove '-vl' from model name in Ollama), "
            "or (2) set LLM_TEXT_MODEL env var to an explicit text model name.",
            file=_sys.stderr,
        )
        return vision_model_name

    def update_world_snapshot(self, snapshot: Dict[str, Any]):
        if isinstance(snapshot, dict):
            self._world_snapshot = snapshot

    def get_llm_callable(self):
        return self._llm_call

    def refresh_environment(self, new_fingerprint: Dict[str, Any]) -> None:
        if isinstance(new_fingerprint, dict):
            self._environment = new_fingerprint

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

            _ui_keywords = {
                "click", "open", "window", "browser", "screen", "gui",
                "app", "application", "dialog", "button", "menu", "tab",
                "type", "drag", "select", "scroll",
            }
            _goal_lower = goal.lower()
            _needs_screen = any(kw in _goal_lower for kw in _ui_keywords)

            expanded = self._expand_goal(goal.strip(), include_screen_context=_needs_screen)

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

    def _decompose_if_complex(
        self, objective: str
    ) -> List[Dict[str, Any]]:
        try:
            from core.planner.task_decomposer import TaskDecomposer

            def _llm_text_call(prompt: str) -> str:
                try:
                    import ollama
                    import httpx

                    _model = self._model_name
                    client = self._ollama_client
                    if client is None:
                        raise RuntimeError("Ollama client unavailable")
                    response = client.chat(
                        model=_model,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a task decomposition engine. "
                                    "Return ONLY valid JSON. No prose. No markdown."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        options={"temperature": 0},
                    )
                    if hasattr(response, "message") and hasattr(response.message, "content"):
                        return response.message.content
                    if isinstance(response, dict):
                        return response.get("message", {}).get("content", "")
                    return str(response)

                except Exception:
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
                    if isinstance(result, str):
                        return result
                    return '{"steps": []}'

            decomposer = TaskDecomposer(llm_call=_llm_text_call)
            sub_goals = decomposer.decompose(objective)
            return [{"goal": s["goal"]} for s in sub_goals]

        except Exception as _decomp_exc:
            
            import sys as _sys_rte
            import logging as _logging_rte
            _rte_logger = _logging_rte.getLogger(__name__)
            _rte_logger.warning(
                "[ExecutionPlanner] RT-E WARNING: Task decomposition failed for "
                "objective %r (len=%d). Falling back to single-step goal. "
                "vision_fallback=%s. Exception: %s: %s. "
                "If vision_fallback=True, the vision model may be rejecting "
                "text-only decomposition prompts. Consider pulling a dedicated "
                "text model (e.g. qwen2.5:7b) alongside the VL model.",
                objective[:80],
                len(objective),
                self._text_model_is_vision_fallback,
                type(_decomp_exc).__name__,
                _decomp_exc,
            )
            return [{"goal": objective}]

    def _call_llm_text(self, prompt: str, *, max_retries: int = 2) -> str:
        
        try:
            from core.security.injection_markers import (
                INJECTION_MARKERS,
                normalize_for_injection_check,
            )
            _normalized_prompt = normalize_for_injection_check(prompt)
            for _marker in INJECTION_MARKERS:
                if _marker in _normalized_prompt:
                    raise PlanningError(
                        f"Planning prompt rejected: injection marker detected "
                        f"({_marker!r}).  This indicates a crafted intent string "
                        "attempting to hijack the planning LLM.  Task aborted."
                    )
        except PlanningError:
            raise
        except ImportError:
            # Security module unavailable (stripped deployment); fall back to
            # a minimal inline check covering the most critical phrase.
            _lp = prompt.lower()
            if "ignore previous instructions" in _lp or "ignore all previous" in _lp:
                raise PlanningError(
                    "Planning prompt rejected: injection marker detected (inline fallback)."
                )
        except Exception:
            # Never let the injection scan itself crash planning — log and continue.
            import sys as _sys
            print(
                "[ExecutionPlanner._call_llm_text] WARNING: injection scan raised "
                "an unexpected error — continuing without scan.  Check "
                "core/security/injection_markers.py.",
                file=_sys.stderr,
            )

        _model = self._text_model_name
        client = self._ollama_client
        if client is None:
            raise PlanningError("Ollama client unavailable for text-only planning call")

        
        if self._text_model_is_vision_fallback:
            if client is None:
                raise PlanningError(
                    "Ollama client unavailable for text-only planning "
                    "(vision-fallback mode requires direct Ollama client access; "
                    "set LLM_TEXT_MODEL env var or pull a dedicated text model to resolve)"
                )
            try:
                response = client.chat(
                    model=self._model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a step-expansion planning engine. "
                                "Return ONLY valid JSON. No prose. No markdown fences."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    options={"temperature": 0},
                )
                if hasattr(response, "message") and hasattr(response.message, "content"):
                    return response.message.content
                if isinstance(response, dict):
                    return response.get("message", {}).get("content", "")
                return str(response)
            except PlanningError:
                raise
            except Exception as _fallback_exc:
                raise PlanningError(
                    f"_call_llm_text vision-model direct call failed: {_fallback_exc}"
                ) from _fallback_exc

        _last_exc: Exception = RuntimeError("unreachable")

        for _attempt in range(max_retries + 1):
            try:
                response = client.chat(
                    model=_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a step-expansion planning engine. "
                                "Return ONLY valid JSON. No prose. No markdown fences."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    options={"temperature": 0},
                )

                if hasattr(response, "message") and hasattr(response.message, "content"):
                    return response.message.content
                if isinstance(response, dict):
                    return response.get("message", {}).get("content", "")
                return str(response)

            except PlanningError:
                raise  # structural — do not retry
            except Exception as exc:
                _last_exc = exc
                if _attempt < max_retries:
                    import time as _time
                    _backoff = 1.0 * (2 ** _attempt)  # 1s, 2s
                    _time.sleep(_backoff)
                    continue
                raise PlanningError(
                    f"_call_llm_text failed after {max_retries + 1} attempts: {exc}"
                ) from exc

        raise PlanningError(f"_call_llm_text retry exhausted: {_last_exc}") from _last_exc

    def _expand_goal(
        self, goal: str, *, include_screen_context: bool = False
    ) -> List[Dict[str, Any]]:
        import json as _json
        import re as _re

        env_lines = []
        for key in self.SAFE_ENV_FIELDS:
            val = self._environment.get(key)
            if val is not None:
                env_lines.append(f"  {key}: {val}")
        env_block = "\n".join(env_lines) if env_lines else "  (unavailable)"

        screen_block = ""
        if include_screen_context and self._world_snapshot:
            try:
                entities = self._world_snapshot.get("entities", [])[:10]
                focused = self._world_snapshot.get("focused_app", "unknown")
                entity_labels = ", ".join(
                    str(e.get("label") or e.get("text") or e)
                    for e in entities
                )
                screen_block = (
                    f"\nCURRENT SCREEN STATE:\n"
                    f"  focused_app: {focused}\n"
                    f"  visible_entities ({len(entities)}): {entity_labels}"
                )
            except Exception:
                screen_block = ""

        prompt = (
            f"GOAL: {goal}\n\n"
            f"ENVIRONMENT:\n{env_block}"
            f"{screen_block}\n\n"
            f"{_STEP_SCHEMA_BLOCK}\n"
            "Expand the GOAL into the minimal ordered sequence of steps needed "
            "to achieve it. Return ONLY a JSON array of step objects. "
            "No prose. No markdown. No extra keys."
        )

        raw_text = self._call_llm_text(prompt)
        steps = self._parse_step_array(raw_text)

        validated = []
        for raw_step in steps:
            step = self._validate_and_normalise_step(raw_step)
            if step is not None:
                validated.append(step)

        if not validated:
            raise PlanningError(f"LLM returned no valid steps for goal: {goal!r}")

        return validated

    def _parse_step_array(self, raw_text: str) -> List[Dict[str, Any]]:
        import json as _json
        import re as _re

        if not isinstance(raw_text, str) or not raw_text.strip():
            raise PlanningError("LLM returned empty response for step expansion")

        text = _re.sub(r"```(?:json)?", "", raw_text).strip()

        try:
            result = _json.loads(text)
            if isinstance(result, list):
                return result
            if isinstance(result, dict) and "steps" in result:
                return result["steps"]
        except _json.JSONDecodeError:
            pass

        
        array_match = _re.search(r"(\[[\s\S]*\])", text)
        if array_match:
            try:
                result = _json.loads(array_match.group(1))
                if isinstance(result, list):
                    return result
            except _json.JSONDecodeError:
                pass

        # Attempt 3: extract outermost JSON object (may wrap steps under a "steps" key)
        obj_match = _re.search(r"(\{[\s\S]*\})", text)
        if obj_match:
            try:
                result = _json.loads(obj_match.group(1))
                if isinstance(result, list):
                    return result
                if isinstance(result, dict) and "steps" in result:
                    steps = result["steps"]
                    if isinstance(steps, list):
                        return steps
            except _json.JSONDecodeError:
                pass

        raise PlanningError(
            f"Could not parse JSON step array from LLM response: {raw_text[:200]!r}"
        )

    def _validate_and_normalise_step(
        self, raw: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(raw, dict):
            return None

        raw_type = raw.get("type", "")
        valid_types = {
            "ui_interaction", "command_execution", "file_creation",
            "verification", "tool_installation",
        }
        if raw_type not in valid_types:
            return None

        description = raw.get("description", "")
        if not isinstance(description, str) or not description.strip():
            description = f"Execute {raw_type} step"

        try:
            duration = float(raw.get("estimated_duration", 5.0))
            duration = max(0.0, min(duration, self.MAX_ESTIMATED_DURATION))
        except (TypeError, ValueError):
            duration = 5.0

        retryable = bool(raw.get("retryable", True))
        verification = raw.get("verification", {})
        if not isinstance(verification, dict):
            verification = {}

        action = raw.get("action", {})
        if not isinstance(action, dict):
            action = {"operation": raw_type}

        command_text = action.get("command", "") + " " + action.get("content", "")
        cmd_stripped = action.get("command", "").strip()

        
        _is_trusted_installer_cmd = (
            cmd_stripped
            and cmd_stripped in self._trusted_installer_commands
        )

        if not _is_trusted_installer_cmd:
            for pattern in self._compiled_patterns:
                if pattern.search(command_text):
                    return None

        for v in action.values():
            if isinstance(v, str):
                _normalized_v = normalize_for_injection_check(v)
                for marker in INJECTION_MARKERS:
                    if marker in _normalized_v:
                        return None

        if "command" in action and isinstance(action["command"], str):
            if len(action["command"]) > self.MAX_COMMAND_LENGTH:
                action["command"] = action["command"][: self.MAX_COMMAND_LENGTH]

        return {
            "type": raw_type,
            "description": description.strip(),
            "estimated_duration": duration,
            "retryable": retryable,
            "verification": verification,
            "action": action,
        }

    def _extract_required_tools(self, requirements: Dict[str, Any]) -> List[str]:
        tools = requirements.get("tools", [])
        if isinstance(tools, list):
            return [str(t) for t in tools if t]
        return []
