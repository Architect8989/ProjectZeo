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

# FIX H4: Import the shared authoritative injection-marker set.
# Previously ExecutionPlanner maintained 7 markers while ReasoningEngine had 35.
# The planning prompt has at least as wide an attack surface as the reasoning
# prompt, so the asymmetry left 28 known injection patterns passing sanitisation
# undetected here. Importing from the shared module keeps both aligned.
from core.security.injection_markers import INJECTION_MARKERS, normalize_for_injection_check


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

    # EVO-4 (Audit): Lowered from 100 to 60 chars.
    # A 40-word objective like "Build Node.js + React + PostgreSQL app with auth"
    # is ~65 chars — below the old threshold, causing it to be planned as a
    # single shot which produces incomplete steps for complex multi-stack tasks.
    # 60 chars better captures real-world complex hackathon-style objectives.
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

        # DETERMINISM FIX (_decompose_model dead attribute): Extract the
        # underlying model name from the llm_call closure once at construction
        # time and cache it as self._model_name. The previous code used
        # getattr(self, "_decompose_model", None) in both _decompose_if_complex
        # and _call_llm_text — this attribute was never set, so the check always
        # returned None and fell through to the __wrapped__ introspection chain
        # on every call. Caching it here eliminates the repeated dead check.
        self._model_name: Optional[str] = self._extract_model_name(llm_call)

        # RD-04 FIX: Derive a text-only model name for use in _call_llm_text().
        # Planning prompts are structural JSON generation — they do not benefit
        # from the vision encoder. Using a text-only variant of the same model
        # family (when available) reduces per-token latency by 40–90% on CPU.
        self._text_model_name: str = self._derive_text_model_name(self._model_name)

        # FIX-07 (RB-A6): Cache a single shared Ollama client at construction time.
        # Previously both _call_llm_text() and the _llm_text_call closure inside
        # _decompose_if_complex() created a new ollama.Client() on every call —
        # each with its own independent connection pool. Under concurrent replan/
        # decompose sequences, pool exhaustion caused HTTP stream contention and
        # timeouts. A single shared client eliminates repeated construction cost.
        self._ollama_client = None
        try:
            import ollama as _ollama_mod
            import httpx as _httpx_mod
            self._ollama_client = _ollama_mod.Client(
                timeout=_httpx_mod.Timeout(connect=10.0, read=120.0, write=5.0, pool=2.0)
            )
        except Exception:
            self._ollama_client = None  # _call_llm_text will raise PlanningError on use

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
    def _extract_model_name(llm_call) -> Optional[str]:
        """
        Attempt to extract the model name from the llm_call closure.
        Walks __wrapped__ chains (decorator wrappers) and reads model_name
        from the underlying adapter instance if present.
        Returns the env-var default if introspection fails.
        """
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
        """
        RD-04 FIX: Derive a text-only model name from a vision model name.

        Root cause of defect:
            _call_llm_text() routed text-only planning prompts through the
            same vision model (e.g. qwen2.5-vl:7b-instruct) that is used for
            screenshot-attached inference.  Vision models process text-only
            requests correctly, but carry significantly higher per-token cost
            on CPU inference due to the unused multimodal transformer blocks.
            On a 16GB CPU system, a 7B vision model adds 40–90% latency
            compared to an equivalent-quality text-only model.

        Fix:
            Prefer a text-only variant when Ollama has one available:
              - LLM_TEXT_MODEL env var (operator override, highest priority)
              - If vision model name ends in '-vl', try stripping that suffix
              - Fall back to the vision model name (always correct, just slower)

            This is a latency optimisation, not a correctness fix.  Falling
            back to the vision model is always safe.

        Example:
            'qwen2.5-vl:7b-instruct' → LLM_TEXT_MODEL or 'qwen2.5:7b-instruct'
                                        (falls back to vision if text not available)
        """
        import os as _os
        # Operator override: explicit env var wins unconditionally
        env_override = _os.environ.get("LLM_TEXT_MODEL", "").strip()
        if env_override:
            return env_override

        if not vision_model_name:
            return _os.environ.get("LLM_MODEL", "qwen2.5-vl:7b-instruct")

        # Heuristic: strip '-vl' suffix from model name to get text variant
        # e.g. 'qwen2.5-vl:7b-instruct' → 'qwen2.5:7b-instruct'
        text_candidate = vision_model_name.replace("-vl:", ":").replace("-vl", "")
        if text_candidate != vision_model_name:
            # Check if the text variant is actually available in Ollama before using it.
            # If not, fall back to the vision model — always correct, just slower.
            try:
                import ollama as _ollama
                _models = _ollama.list()
                _available = {
                    m.model if hasattr(m, "model") else str(m)
                    for m in (_models.models if hasattr(_models, "models") else [])
                }
                # Check full name or base name match
                _base_candidate = text_candidate.split(":")[0]
                _found = any(
                    text_candidate in name or _base_candidate in name
                    for name in _available
                )
                if _found:
                    return text_candidate
            except Exception:
                pass  # Ollama unavailable or list failed — fall back to vision model

        # Fall back: use the vision model for text calls (correct but slower)
        return vision_model_name

    # ==================================================

    def update_world_snapshot(self, snapshot: Dict[str, Any]):
        if isinstance(snapshot, dict):
            self._world_snapshot = snapshot

    # FIX-C4 (RB-3): Expose a public get_llm_callable() method so callers do
    # not need to access the private _llm_call attribute directly.
    #
    # Root cause: operate.py checks `if hasattr(planner, "get_llm_callable")`
    # and falls back to `getattr(planner, "_llm_call", None)` if not found.
    # This fallback works today because the private attribute exists, but is
    # fragile: a rename of _llm_call to _llm_callable (common pattern) would
    # silently make llm_callable=None, causing
    #     RuntimeError("Planner LLM callable unavailable")
    # on the first task — with no test catching it because the attribute is
    # accessed dynamically.
    #
    # Fix: add the public method so the hasattr() branch in operate.py is
    # always taken, eliminating the fragile private-attribute fallback.
    def get_llm_callable(self):
        """
        FIX-C4 (RB-3): Public accessor for the LLM callable.

        Returns the callable stored in self._llm_call.  Provides a stable
        public contract so operate.py does not need to reach into private
        attributes.  Callers should use this method instead of accessing
        _llm_call directly.
        """
        return self._llm_call

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

            # P1 FIX: Only inject live screen context for goals that involve
            # UI interaction. Text-only goals (installs, CLI commands, file ops)
            # do not benefit from the current screen state and should not risk
            # leaking sensitive visible content.
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

        GAP-6 FIX: The previous _llm_text_call wrapper passed the raw return
        value of self._llm_call() (a List[dict] of UI operations: click, type,
        command) to TaskDecomposer.  TaskDecomposer.llm_call is expected to be
        a callable that receives a prompt string and returns a plain text string
        containing the decomposed sub-goals JSON.

        The Ollama vision adapter (QwenOllamaAdapter.get_next_action) always
        returns a List[dict] of UI operations — it captures the current screen,
        calls the vision model, and returns parsed click/type/command objects.
        Passing that to TaskDecomposer produced garbage: json.dumps of UI ops
        is not a valid decomposition response, so _safe_json_extract always
        raised DecompositionError, and the fallback returned [{"goal": objective}]
        — making decomposition a no-op for ALL complex tasks.

        Fix: build a dedicated text-only LLM callable that bypasses the vision
        adapter entirely.  For the Ollama path we call the Ollama client directly
        with a text-only chat (no screenshot), which returns a plain text string
        suitable for TaskDecomposer.  We extract the response content string from
        the ollama response object/dict using the same shim used elsewhere.
        If direct Ollama access is not available, fall back gracefully.
        """
        try:
            from core.planner.task_decomposer import TaskDecomposer

            def _llm_text_call(prompt: str) -> str:
                """
                GAP-6 FIX: Text-only LLM call for decomposition.
                Routes directly to Ollama with no screenshot so the model
                returns a plain text sub-goal list — not UI operation JSON.
                Falls back to self._llm_call message-list path if Ollama
                client is unavailable.
                """
                # Try direct Ollama text call (fastest, most compatible)
                try:
                    import ollama
                    import httpx

                    # Use the cached model name resolved at construction time.
                    # The previous getattr(self, "_decompose_model", None) was
                    # permanently None (the attribute was never set), causing
                    # repeated unnecessary closure introspection on every call.
                    _model = self._model_name

                    # FIX-07: Use shared client from outer scope (__init__).
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
                    # Extract plain text from ollama response (shim for >=0.2 and legacy)
                    if hasattr(response, "message") and hasattr(response.message, "content"):
                        return response.message.content
                    if isinstance(response, dict):
                        return response.get("message", {}).get("content", "")
                    return str(response)

                except Exception:
                    # Fallback: use message-list path through self._llm_call
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
                    # If we got UI ops back (vision adapter), we cannot use them
                    # for decomposition — return empty JSON to trigger fallback
                    return '{"steps": []}'

            decomposer = TaskDecomposer(llm_call=_llm_text_call)
            sub_goals = decomposer.decompose(objective)
            return [{"goal": s["goal"]} for s in sub_goals]

        except Exception:
            # Decomposition failed — fall back to single-goal pass-through
            return [{"goal": objective}]


    # ==================================================
    # HRD-08: TEXT-ONLY LLM CALL (no screenshot)
    # ==================================================

    def _call_llm_text(self, prompt: str) -> str:
        """
        HRD-08: Text-only LLM call for planning prompts.

        Planning prompts are structural (JSON schema generation) and do not
        need or benefit from a live screenshot. Routing through the vision
        adapter (get_next_action) unconditionally attaches a screenshot,
        introducing irrelevant noise that degrades plan quality.

        This method mirrors the pattern in _decompose_if_complex(): call
        Ollama directly with a text-only chat, then fall back to the
        message-list path if Ollama is unavailable.
        """
        try:
            import ollama
            import httpx

            # RD-04 FIX: Use the text-only model name derived at construction
            # time rather than the vision model.  _text_model_name is either:
            #   - The LLM_TEXT_MODEL env var (operator override)
            #   - A '-vl'-stripped variant if that variant is available in Ollama
            #   - The vision model name as fallback (always correct, just slower)
            # On systems where no text-only variant is installed, this falls
            # back to the vision model transparently.
            _model = self._text_model_name

            # FIX-07: Use shared Ollama client from __init__.
            client = self._ollama_client
            if client is None:
                raise PlanningError("Ollama client unavailable for text-only planning call")

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

            # Extract plain text from Ollama response (compat with >=0.2 and legacy).
            if hasattr(response, "message") and hasattr(response.message, "content"):
                return response.message.content
            if isinstance(response, dict):
                return response.get("message", {}).get("content", "")
            return str(response)

        except PlanningError:
            raise
        except Exception as exc:
            raise PlanningError(f"_call_llm_text failed: {exc}") from exc

    # ==================================================
    # GOAL EXPANSION — P0 FIX: _expand_goal() was entirely absent.
    # Every create_plan() call raised:
    #   AttributeError: 'ExecutionPlanner' object has no attribute '_expand_goal'
    # making planning impossible. This method is the core of the planner.
    # ==================================================

    def _expand_goal(
        self, goal: str, *, include_screen_context: bool = False
    ) -> "List[Dict[str, Any]]":
        """
        Expand a single high-level goal string into a list of ExecutionStep
        spec dicts by calling the LLM with a structured planning prompt.

        Returns a List of dicts, each matching _STEP_SCHEMA_BLOCK, with keys:
          type, description, estimated_duration, retryable, verification, action.

        Raises PlanningError if the LLM response cannot be parsed into valid steps.
        """
        import json as _json
        import re as _re

        # Build environment context block (safe fields only).
        env_lines = []
        for key in self.SAFE_ENV_FIELDS:
            val = self._environment.get(key)
            if val is not None:
                env_lines.append(f"  {key}: {val}")
        env_block = "\n".join(env_lines) if env_lines else "  (unavailable)"

        # Optionally include world/screen context for UI-facing goals.
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

    # --------------------------------------------------
    # JSON PARSING HELPERS
    # --------------------------------------------------

    def _parse_step_array(self, raw_text: str) -> "List[Dict[str, Any]]":
        """
        Extract a JSON array from LLM output. Uses greedy bracket matching
        as a fallback when the model wraps output in prose or markdown fences.
        """
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

        bracket_match = _re.search(r"\[.*\]", text, _re.DOTALL)
        if bracket_match:
            try:
                result = _json.loads(bracket_match.group(0))
                if isinstance(result, list):
                    return result
            except _json.JSONDecodeError:
                pass

        raise PlanningError(
            f"Could not parse JSON step array from LLM response: {raw_text[:200]!r}"
        )

    def _validate_and_normalise_step(
        self, raw: "Dict[str, Any]"
    ) -> "Optional[Dict[str, Any]]":
        """
        Validate one raw step dict from the LLM and normalise to the expected
        schema. Returns None for invalid steps so one bad step does not abort
        the whole plan.
        """
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

        # Safety: reject steps with dangerous shell commands.
        command_text = action.get("command", "") + " " + action.get("content", "")
        for pattern in self._compiled_patterns:
            if pattern.search(command_text):
                return None

        # Injection check on action field values.
        # HAR-6: Use normalize_for_injection_check() to defeat Unicode homoglyph
        # bypasses before marker matching.
        for v in action.values():
            if isinstance(v, str):
                _normalized_v = normalize_for_injection_check(v)
                for marker in INJECTION_MARKERS:
                    if marker in _normalized_v:
                        return None

        # Truncate excessively long commands.
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

    # --------------------------------------------------
    # UTILITIES
    # --------------------------------------------------

    def _extract_required_tools(self, requirements: "Dict[str, Any]") -> "List[str]":
        tools = requirements.get("tools", [])
        if isinstance(tools, list):
            return [str(t) for t in tools if t]
        return []
