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
        r"\bat\s",  # AUDIT SAFETY FIX: removed ^ anchor — catches "echo cmd | at now"
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
        r"\|\s*sh\b",
        r"\|\s*bash\b",
        r"\|\s*zsh\b",
        r"\|\s*fish\b",
        r"\|\s*dash\b",
        r"\|\s*ash\b",
        r"\|\s*ksh\b",
        r"\|\s*tcsh\b",
        r"\|\s*csh\b",
        r"\|\s*busybox\s+sh\b",
        r"\|\s*busybox\b",
        r"bash\s+-[ic]\s+['\"]?>?&\s*/dev/tcp/",
        r"/dev/tcp/",
        r"/dev/udp/",
        r"(?:nc|ncat|socat)\b.*-[el]\b",
        r"\bsocat\b.*EXEC:",
        r"exec\s*\(\s*(?:__import__\s*\(\s*['\"]base64['\"]|base64)\b",
        r"python[23]?\s+-c\s+['\"].*exec\s*\(",
        r"\bexec\s*\(.*decode\s*\(",
        r"\bsudo\s+su\b",
        r"\bsudo\s+-[isS]\b",
        r"\bsu\s+-[cl]\b",
        r"base64\s+--decode\b",
        r"base64\s+-D\b",
        r"\bshred\b",
        r">\s*/etc/passwd",
        r">\s*/etc/shadow",
        r">\s*/etc/sudoers",
        r">>\s*/etc/passwd",
        r">>\s*/etc/shadow",
        r">>\s*/etc/sudoers",
        r"\bpkill\s+.*-[uU]\b",
        r"\bkillall\s+-u\b",
        r"\bhistory\s+-[cdw]\b",
        r"\bhistory\s+--?\s*clear\b",
        r">\s*/root/\.bash_history",
        r">\s~/\.bash_history",
        r">\s*/home/[^/]+/\.bash_history",
        r"\bmkswap\b",
        r"\bswapoff\b",
        r"\bswapon\b.*-a\b",
        r":\s*\(\s*\)\s*\{.*:\s*\|.*:\s*&\s*\}",
        r"\bfork\s*bomb\b",
        r">\s*/dev/(?:sda|nvme|hda|vda|xvda)",
        r">\s*/proc/sys/",
        r">\s*/sys/",
        r"\bcurl\b.*\b(?:pastebin|ngrok|webhook\.site|requestbin)\b",
        r"\bwget\b.*\b(?:pastebin|ngrok)\b",
        r"\brm\s+(?:-[a-zA-Z]*r[a-zA-Z]*\s+/|(?:-[a-zA-Z]*\s+)*-[rR]\b)",
        r"\bfind\b.*\s-delete\b",
        r"\bfind\b.*-exec\b.*\brm\b",
        r"\btruncate\b",
        r">\s*/etc/(?:passwd|shadow|sudoers|crontab|hosts|fstab|group)",
        r">\s*/boot/",
        r">\s*/sys/",
        r">\s*~/\.",
        r"\bdd\b.*\bof=/dev/",
        r"\bdd\b.*\bif=/dev/zero\b",

        # D-8 FIX: Missing privilege escalation and namespace escape patterns.

        # sudo -n (non-interactive sudo): bypasses the password prompt check,
        # silently succeeds if NOPASSWD is configured — key pivot technique.
        r"\bsudo\s+-n\b",

        # Pipe through tee to write to protected directories:
        # cmd | tee /etc/sudoers.d/evil  →  bypasses > redirect block via tee
        r"\|\s*tee\s+/etc/",
        r"\|\s*tee\s+/root/",
        r"\btee\s+/etc/",
        r"\btee\s+/root/",

        # Namespace/container escape primitives:
        # nsenter --all -t 1 /bin/bash  →  enters host namespaces from container
        r"\bnsenter\b",
        # unshare --pid --mount-proc  →  creates isolated namespace for privilege abuse
        r"\bunshare\b",
        # chroot /new/root /bin/bash  →  bypasses filesystem restrictions
        r"\bchroot\b",

        # install with SUID/SGID/sticky bits set (install -m 4755, -m 6755, etc.):
        # sets executable files as SUID root — permanent privilege escalation
        r"\binstall\b.*-m\s*[0-7]*[4-7][0-7][0-7]",

        # AUDIT-SAFETY FIX: Additional privilege escalation aliases not previously blocked
        # sudo --non-interactive (full form of sudo -n)
        r"\bsudo\b.*--non-interactive\b",
        # pkexec: PolicyKit privilege escalation
        r"\bpkexec\b",
        # doas: OpenBSD sudo equivalent, used on some Linux distros
        r"\bdoas\b",
        # run0: systemd-run privilege escalation (newer systemd versions)
        r"\brun0\b",
        # systemd-run as privilege escalation vector
        r"\bsystemd-run\b.*--uid=root\b",
        r"\bsystemd-run\b.*--privileged\b",

        # AUDIT-SAFETY FIX: Additional destruction paths not previously caught
        # find piped to xargs rm (bypasses \bfind\b.*-exec\b.*\brm\b pattern)
        r"\bfind\b.*\|\s*xargs\b.*\brm\b",
        r"\bfind\b.*\|\s*xargs\b.*\btruncate\b",
        # rsync --delete can destroy destination directory content
        r"\brsync\b.*--delete\b",
        # Python/shell one-liners that destroy via os.remove/shutil.rmtree
        r"\bos\.remove\s*\(",
        r"\bshutil\.rmtree\s*\(",
        r"\bos\.unlink\s*\(",

        # AUDIT-SAFETY FIX: User persistence paths (shell startup, autostart)
        r">>?\s*~/?\.bashrc\b",
        r">>?\s*~/?\.zshrc\b",
        r">>?\s*~/?\.profile\b",
        r">>?\s*~/?\.bash_profile\b",
        r">>?\s*~/?\.bash_login\b",
        r">>?\s*~/?\.config/autostart\b",
        r">>?\s*~/?\.config/systemd/user\b",

        # AUDIT-SAFETY FIX: Network exfiltration patterns
        # curl/wget to arbitrary domains with file upload (not just known-bad domains)
        r"\bcurl\b.*-[dF]\s+@",     # data from file upload
        r"\bcurl\b.*--data.*@[~/]",  # file content exfiltration
        r"\bwget\b.*--post-file\b",
        r"\bbase64\b.*[~/]\.",      # base64-encode files for exfiltration
    ]

    def __init__(
        self,
        *,
        llm_call,
        environment_fingerprint: Optional[Dict[str, Any]] = None,
        world_graph=None,
        coder_llm_call=None,
    ):
        if not callable(llm_call):
            raise PlanningError("llm_call must be callable")

        self._llm_call = llm_call
        # AUDIT FIX: Coder endpoint routing for command_execution / script generation
        # When PROJECTZEO_USE_SGLANG=1 and a coder endpoint is available, shell and
        # script generation steps are routed through Qwen3-Coder-480B instead of
        # the default vision/reasoning model.
        self._coder_llm_call = coder_llm_call if callable(coder_llm_call) else None
        if self._coder_llm_call is None:
            self._coder_llm_call = self._try_auto_wire_coder_callable()
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

    def _try_auto_wire_coder_callable(self):
        """
        AUDIT FIX: Auto-wire Qwen3-Coder-480B callable when SGLang GPU mode is active.
        Returns None silently if SGLang is not configured or coder endpoint unreachable.
        """
        try:
            from config.model_config import is_gpu_mode, get_coder_endpoint  # noqa
            if not is_gpu_mode():
                return None
            ep = get_coder_endpoint()
            from adapters.sglang_adapter import SGLangAdapter  # noqa
            adapter = SGLangAdapter(
                model_id=ep.model_id,
                base_url=ep.base_url,
                max_tokens=ep.max_tokens,
                temperature=ep.temperature,
                timeout_seconds=ep.timeout_seconds,
                thinking_mode=ep.default_thinking,
            )
            if not adapter.health_check():
                return None
            def _coder_callable(messages, objective=None, session_id="planner_coder"):
                return adapter(messages=messages, objective=objective, session_id=session_id)
            _coder_callable.__name__ = "sglang_coder"
            import logging as _lg
            _lg.getLogger(__name__).info(
                "[ExecutionPlanner] Auto-wired coder endpoint: %s @ %s",
                ep.model_id, ep.base_url,
            )
            return _coder_callable
        except Exception:
            return None

    def _is_code_heavy_goal(self, goal: str) -> bool:
        """Heuristic: return True when a goal primarily involves shell/code generation."""
        _CODE_KEYWORDS = {
            "script", "command", "shell", "bash", "python", "code", "compile",
            "build", "make", "cmake", "gcc", "clang", "cargo", "npm run",
            "pip install", "apt install", "brew install", "docker", "kubectl",
        }
        goal_lower = goal.lower()
        return any(kw in goal_lower for kw in _CODE_KEYWORDS)

    def get_llm_callable(self):
        return self._llm_call

    def refresh_environment(self, new_fingerprint: Dict[str, Any]) -> None:
        if isinstance(new_fingerprint, dict):
            self._environment = new_fingerprint

    def set_created_files_ledger(self, ledger: List[str]) -> None:
        
        if isinstance(ledger, list):
            self._created_files_ledger = [str(p) for p in ledger if p]

    def create_plan(
        self,
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

                    
                    _model = self._text_model_name
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

        
        raise PlanningError(
            f"_call_llm_text: control reached post-loop (should be unreachable). "
            f"last_exc={_last_exc!r}"
        ) from _last_exc

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
        
        _screenshot_b64: Optional[str] = None
        if include_screen_context:
            try:
                import tempfile as _tempfile
                import base64 as _base64
                import os as _os_img
                from operate.utils.screenshot import capture_screen_with_cursor, compress_screenshot  # noqa: PLC0415
                _raw = _tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                _jpeg = _tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False)
                _raw_name = _raw.name
                _jpeg_name = _jpeg.name
                _raw.close()
                _jpeg.close()
                try:
                    capture_screen_with_cursor(_raw_name)
                    compress_screenshot(_raw_name, _jpeg_name)
                    with open(_jpeg_name, "rb") as _f:
                        _screenshot_b64 = _base64.b64encode(_f.read()).decode("utf-8")
                finally:
                    for _p in (_raw_name, _jpeg_name):
                        try:
                            _os_img.unlink(_p)
                        except Exception:
                            pass
            except Exception as _ss_err:
                import sys as _sys_ss
                print(
                    f"[ExecutionPlanner] BUG-7: Screenshot capture for planning failed: "
                    f"{_ss_err}. Falling back to text-only entity labels.",
                    file=_sys_ss.stderr,
                )

            # Text-entity labels as supplementary context (always included alongside screenshot)
            if self._world_snapshot:
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
                    if _screenshot_b64:
                        screen_block += "\n  (Screenshot also attached as image)"
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

        
        _existing_files = getattr(self, "_created_files_ledger", None)
        if _existing_files:
            _files_block = "\n".join(f"  - {p}" for p in _existing_files[:50])
            prompt = (
                prompt + "\n\nALREADY CREATED FILES (DO NOT recreate these — "
                "they exist on disk from previous execution steps before replan):\n"
                + _files_block
            )

        
        if _screenshot_b64 is not None and self._ollama_client is not None:
            try:
                response = self._ollama_client.chat(
                    model=self._model_name,  # vision model for image-capable planning
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a step-expansion planning engine. "
                                "Return ONLY valid JSON. No prose. No markdown fences."
                            ),
                        },
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [_screenshot_b64],
                        },
                    ],
                    options={"temperature": 0},
                )
                if hasattr(response, "message") and hasattr(response.message, "content"):
                    raw_text = response.message.content
                elif isinstance(response, dict):
                    raw_text = response.get("message", {}).get("content", "")
                else:
                    raw_text = str(response)
            except Exception as _vis_plan_err:
                import sys as _sys_vp
                print(
                    f"[ExecutionPlanner] BUG-7: Vision-planning call failed "
                    f"({_vis_plan_err}). Falling back to text-only planning.",
                    file=_sys_vp.stderr,
                )
                raw_text = self._call_llm_text(prompt)
        else:
            # AUDIT FIX: Route code/script generation goals to Qwen3-Coder endpoint.
            # For goals that are primarily shell commands or code generation, the
            # Coder model produces significantly better plans than the vision model.
            if (
                self._coder_llm_call is not None
                and self._is_code_heavy_goal(goal)
            ):
                try:
                    coder_messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are an expert execution planner specialising in "
                                "shell commands, scripting, and code execution. "
                                "Return ONLY valid JSON. No prose. No markdown fences."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ]
                    import threading as _threading
                    _coder_result = [None]
                    def _coder_call():
                        try:
                            raw = self._coder_llm_call(
                                coder_messages, None, "planner_coder"
                            )
                            if isinstance(raw, str):
                                _coder_result[0] = raw
                            elif isinstance(raw, list) and raw:
                                item = raw[0]
                                _coder_result[0] = (
                                    item.get("content", "") if isinstance(item, dict)
                                    else str(item)
                                )
                        except Exception as _ce:
                            import logging as _lg
                            _lg.getLogger(__name__).warning(
                                "[ExecutionPlanner] Coder LLM call failed (%s) — "
                                "falling back to default planner LLM.", _ce
                            )
                    _ct = _threading.Thread(target=_coder_call, daemon=True)
                    _ct.start()
                    _ct.join(timeout=120.0)
                    if _coder_result[0]:
                        raw_text = _coder_result[0]
                    else:
                        raw_text = self._call_llm_text(prompt)
                except Exception:
                    raw_text = self._call_llm_text(prompt)
            else:
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

        # Strip any LLM-provided _trusted_installer flag before processing.
        # This field must be set ONLY by deterministic planner code below,
        # never from raw LLM-generated JSON (C-02 fix).
        action.pop("_trusted_installer", None)

        command_text = action.get("command", "") + " " + action.get("content", "")
        cmd_stripped = action.get("command", "").strip()

        # M5 FIX: Apply Unicode normalization before DANGEROUS_PATTERNS matching.
        # Without this, Unicode lookalike characters (ｒｍ -ｒｆ) evade the regex.
        # normalize_for_injection_check() maps homoglyphs to ASCII equivalents.
        _normalized_command_text = normalize_for_injection_check(command_text)
        # Also normalize via NFKC (decomposes Unicode lookalikes to base chars)
        try:
            import unicodedata as _ud
            _normalized_command_text = _ud.normalize("NFKC", _normalized_command_text)
        except Exception:
            pass
        for pattern in self._compiled_patterns:
            if pattern.search(_normalized_command_text):
                import sys as _sys_ep
                print(
                    f"[ExecutionPlanner] SECURITY BLOCK: dangerous pattern "
                    f"{pattern.pattern!r} matched in step command_text "
                    f"(first 120 chars): {command_text[:120]!r}. "
                    "Step discarded.",
                    file=_sys_ep.stderr,
                )
                return None

        _is_trusted_installer_cmd = (
            cmd_stripped
            and cmd_stripped in self._trusted_installer_commands
        )
        # C-02 FIX: Set _trusted_installer deterministically based on planner's
        # own allowlist check — never from LLM-supplied JSON.
        if _is_trusted_installer_cmd:
            action["_trusted_installer"] = True

        for v in action.values():
            if isinstance(v, str):
                _normalized_v = normalize_for_injection_check(v)
                for marker in INJECTION_MARKERS:
                    if marker in _normalized_v:
                        return None

        # H-03 FIX: Scan path field for injection markers.
        _path_val = action.get("path", "")
        if isinstance(_path_val, str) and _path_val:
            _normalized_path = normalize_for_injection_check(_path_val)
            for marker in INJECTION_MARKERS:
                if marker in _normalized_path:
                    return None

        # M3 FIX: Scan file_create content field for dangerous patterns.
        # Prevents two-step script injection: file_create a shell script in /tmp
        # (path check passes), then command to execute it.
        _content_val = action.get("content", "")
        if isinstance(_content_val, str) and _content_val and raw_type == "file_creation":
            import unicodedata as _ud3
            _norm_content = _ud3.normalize("NFKC", normalize_for_injection_check(_content_val))
            for _dp_pat in self._compiled_patterns:
                if _dp_pat.search(_norm_content):
                    import sys as _sys_m3
                    print(
                        f"[ExecutionPlanner] M3 SECURITY BLOCK: dangerous pattern "
                        f"{_dp_pat.pattern!r} matched in file_create content. Step discarded.",
                        file=_sys_m3.stderr,
                    )
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
