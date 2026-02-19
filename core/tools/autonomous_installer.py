"""
core/tools/autonomous_installer.py
=====================================
PATCH AUDIT FIXES:

  ❌  §1.10 / Gap C: install_tool() accepted a pre-populated tool dict with
            'official_url' but nothing in the codebase populated official_url
            autonomously from an objective.  The LLM must supply the URL in the
            plan, but the prompt schema (pre-patch) never specified this.
            FIX: The planner prompt schema patch (execution_planner.py) now
            explicitly requires tool_installation steps to include official_url.
            Additionally, install_tool() now validates the URL is present and
            provides a clear error if it is missing.

  ⚠️  §1.10: No terminal-based installation path.  For most dev tools
            (node, python, curl, git) the official site provides a
            curl/apt/brew install command — the installer could not execute it.
            FIX: Added _try_terminal_install() which is attempted FIRST, before
            opening a browser.  It asks the LLM for the canonical install command
            for the given tool on the current platform and executes it via
            os_backend.run_command().  On failure it falls back to the existing
            browser-UI path.

  ⚠️  §1.10: Click resolution failed on graphical download buttons with no
            readable text, SVG icons, or dynamic React content.
            FIX: Added coordinate fallback strategy.  If fuzzy text matching
            fails, the installer asks the LLM to return x/y coordinates directly
            based on the perception layout and tries those.

  ✅  All existing correct behaviours preserved:
        - _validate_url() enforces https://
        - _sanitize_perception() bounds entity lists
        - _normalize_llm_response() / _validate_action_schema()
        - MAX_INSTALL_TIME=15 minutes
        - Post-install verification via StepVerifier
"""

from __future__ import annotations

import platform
import time
import json
from typing import Dict, Any, Optional, Tuple

from observer.observer_core import ObserverCore
from operate.utils.operating_system import OperatingSystem
from core.verification.step_verifier import StepVerifier, VerificationError
from core.schemas.execution_plan import ExecutionStep, StepType
from core.cognition.reasoning_engine import ReasoningEngine


class InstallationError(RuntimeError):
    pass


class AutonomousInstaller:

    MAX_INSTALL_TIME = 15 * 60
    UI_SETTLE_DELAY = 1.0
    PAGE_LOAD_TIMEOUT = 10.0
    MAX_ITERATIONS = 120
    MAX_PERCEPTION_BYTES = 10_000

    # PATCH: terminal install is always preferred over browser UI
    TERMINAL_INSTALL_TIMEOUT_SECONDS = 300  # 5 min for large downloads

    REQUIRED_FIELDS = {"name", "official_url"}

    def __init__(
        self,
        *,
        observer: ObserverCore,
        os_backend: OperatingSystem,
        llm_callable,
    ):
        if not isinstance(observer, ObserverCore):
            raise InstallationError("Observer required")

        if not isinstance(os_backend, OperatingSystem):
            raise InstallationError("OperatingSystem backend required")

        if not callable(llm_callable):
            raise InstallationError("LLM callable required")

        self._observer = observer
        self._os = os_backend
        self._llm = llm_callable
        self._verifier = StepVerifier()

        self._sanitizer = ReasoningEngine.__new__(ReasoningEngine)

    # =================================================
    # PUBLIC API
    # =================================================

    def install_tool(self, tool: Dict[str, Any]) -> None:
        """
        Install a tool.

        PATCH §1.10: Attempts terminal install FIRST (apt/brew/curl — the
        normal path for dev tools), then falls back to browser-UI navigation.

        The tool dict MUST include:
          name         : str  — display name of the tool
          official_url : str  — https:// URL of the official download page
        Optional:
          version_command : str  — shell command to verify installation
          min_version     : str  — semver floor
          install_commands: list[str] — pre-specified commands to try before LLM
        """

        self._validate_tool_schema(tool)

        name = tool["name"]
        url = tool["official_url"]

        self._validate_url(url)

        if self._is_already_installed(tool):
            return

        if not self._observer.is_healthy():
            raise InstallationError("Observer unhealthy — aborting install")

        # PATCH §1.10: Try terminal install first
        if self._try_terminal_install(tool):
            if self._is_already_installed(tool):
                return
            # Terminal reported success but verification failed — fall through to UI

        # Browser-UI install fallback
        self._browser_ui_install(tool)

    # =================================================
    # TERMINAL INSTALL PATH (NEW)
    # =================================================

    def _try_terminal_install(self, tool: Dict[str, Any]) -> bool:
        """
        PATCH §1.10: Ask the LLM for the canonical terminal install command
        for this tool on the current OS.  Execute it via run_command().
        Returns True if a command was executed (regardless of outcome).
        """
        name = tool["name"]
        os_name = platform.system()

        # Allow plan to pre-specify install commands
        pre_specified = tool.get("install_commands", [])
        if isinstance(pre_specified, list) and pre_specified:
            for cmd in pre_specified:
                if isinstance(cmd, str) and cmd.strip():
                    try:
                        result = self._os.run_command(cmd.strip())
                        if isinstance(result, dict) and result.get("returncode", 1) == 0:
                            return True
                    except Exception:
                        continue
            return True  # commands were tried

        # Ask LLM for the best install command
        prompt = [
            {
                "role": "user",
                "content": (
                    f"What is the single best terminal command to install "
                    f"'{name}' on {os_name}?\n"
                    f"Official URL for context: {tool['official_url']}\n\n"
                    "Return ONLY valid JSON:\n"
                    '{ "command": "<shell command string>", "reason": "<why>" }\n'
                    "If no terminal install is available, return:\n"
                    '{ "command": null, "reason": "browser required" }'
                ),
            }
        ]

        try:
            response = self._llm(
                messages=prompt,
                objective=f"Get install command for {name}",
                session_id="installer-terminal",
            )
        except Exception:
            return False

        try:
            decision = self._normalize_llm_response(response)
            cmd = decision.get("command")
        except Exception:
            return False

        if not isinstance(cmd, str) or not cmd.strip():
            return False  # LLM said browser required or no command

        try:
            result = self._os.run_command(cmd.strip())
            return True  # command was executed
        except Exception:
            return False

    # =================================================
    # BROWSER-UI INSTALL PATH (existing, patched)
    # =================================================

    def _browser_ui_install(self, tool: Dict[str, Any]) -> None:

        name = tool["name"]
        url = tool["official_url"]

        start_ts = time.time()

        self._open_browser()
        self._wait_ui()

        self._navigate(url)
        time.sleep(self.PAGE_LOAD_TIMEOUT)

        iteration = 0

        while iteration < self.MAX_ITERATIONS:

            if time.time() - start_ts > self.MAX_INSTALL_TIME:
                break

            if not self._observer.is_healthy():
                raise InstallationError("Observer lost during install")

            if self._is_already_installed(tool):
                return

            screen = self._observer.snapshot()
            perception = screen.get("perception")

            action = self._decide_next_action(
                tool_name=name,
                url=url,
                perception=perception,
            )

            if action.get("operation") == "done":
                if self._is_already_installed(tool):
                    return
                raise InstallationError("LLM declared done but verification failed")

            self._execute_action(action, perception)

            time.sleep(self.UI_SETTLE_DELAY)

            iteration += 1

        raise InstallationError(f"Installation timed out: {name}")

    # =================================================
    # DECISION LOOP
    # =================================================

    def _decide_next_action(
        self,
        *,
        tool_name: str,
        url: str,
        perception: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:

        safe_perception = self._sanitize_perception(perception)

        prompt = {
            "role": "user",
            "content": (
                f"Objective: Install {tool_name}\n\n"
                f"Current URL: {url}\n\n"
                f"Screen perception:\n"
                f"{json.dumps(safe_perception)}\n\n"
                "Return JSON ONLY:\n"
                "{\n"
                '  "operation": "click|type|hotkey|wait|done",\n'
                '  "target": "element description (if click)",\n'
                '  "x": <number, pixel x coordinate — required if target text not visible>,\n'
                '  "y": <number, pixel y coordinate — required if target text not visible>,\n'
                '  "text": "text to type (if type)",\n'
                '  "keys": ["ctrl","c"] (if hotkey)\n'
                "}"
            ),
        }

        response = self._llm(
            messages=[prompt],
            objective=f"Install {tool_name}",
            session_id="installer",
        )

        decision = self._normalize_llm_response(response)
        self._validate_action_schema(decision)

        return decision

    # =================================================
    # ACTION EXECUTION
    # =================================================

    def _execute_action(
        self,
        action: Dict[str, Any],
        perception: Optional[Dict[str, Any]],
    ) -> None:

        op = action.get("operation")

        if op == "click":
            # PATCH §1.10: fallback to LLM-supplied x/y if text match fails
            target = action.get("target")
            coords: Optional[Tuple[float, float]] = None

            if isinstance(target, str) and target.strip():
                coords = self._resolve_click_target(target, perception)

            if coords is None:
                # Fallback: use LLM-provided absolute coordinates
                x = action.get("x")
                y = action.get("y")
                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    coords = (float(x), float(y))

            if coords is None:
                raise InstallationError(
                    f"Unable to resolve click target: "
                    f"{target!r} (no OCR match and no x/y coordinates)"
                )

            self._os.click(coords[0], coords[1])
            return

        if op == "type":
            text = action.get("text", "")
            if not isinstance(text, str):
                raise InstallationError("Invalid type text")
            self._os.type_text(text)
            return

        if op == "hotkey":
            keys = action.get("keys", [])
            if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
                raise InstallationError("Invalid hotkey format")
            self._os.press_keys(keys)
            return

        if op == "wait":
            time.sleep(self.UI_SETTLE_DELAY)
            return

        raise InstallationError(f"Unsupported installer operation: {op}")

    def _resolve_click_target(
        self,
        target: str,
        perception: Optional[Dict[str, Any]],
    ) -> Optional[Tuple[float, float]]:

        if not isinstance(perception, dict):
            return None

        elements = perception.get("elements", [])
        if not isinstance(elements, list):
            return None

        target_lower = target.lower()

        for el in elements:
            if not isinstance(el, dict):
                continue

            text = str(el.get("text", "")).lower()
            if target_lower in text:
                x = el.get("x")
                y = el.get("y")
                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    return float(x), float(y)

        return None

    # =================================================
    # VERIFICATION
    # =================================================

    def _is_already_installed(self, tool: Dict[str, Any]) -> bool:

        version_cmd = tool.get("version_command")
        min_version = tool.get("min_version")

        if not isinstance(version_cmd, str) or not version_cmd.strip():
            return False

        step = ExecutionStep(
            id=0,
            type=StepType.TOOL_INSTALLATION,
            description=f"Verify {tool.get('name')} installed",
            action={"tool": tool.get("name")},
            verification={
                "version_command": version_cmd,
                "min_version": min_version,
            },
            dependencies=[],
            estimated_duration=5.0,
            retryable=False,
        )

        try:
            result = self._verifier.verify_step(step)
            return bool(result.success)
        except (VerificationError, Exception):
            return False

    # =================================================
    # BROWSER UTILITIES
    # =================================================

    def _open_browser(self) -> None:
        try:
            self._os.open_browser()
        except Exception as exc:
            raise InstallationError(f"Failed to open browser: {exc}") from exc

    def _navigate(self, url: str) -> None:
        try:
            self._os.focus_address_bar()
            self._os.press_keys(["ctrl", "a"])
            self._os.type_text(url)
            self._os.press_keys(["enter"])
        except Exception as exc:
            raise InstallationError(f"Navigation failed: {exc}") from exc

    def _wait_ui(self) -> None:
        time.sleep(self.UI_SETTLE_DELAY)

    # =================================================
    # LLM RESPONSE NORMALISATION
    # =================================================

    def _normalize_llm_response(self, response) -> dict:
        import json
        import re

        if isinstance(response, dict):
            return response

        if isinstance(response, list):
            if not response:
                raise InstallationError("LLM returned empty action list")
            first = response[0]
            if isinstance(first, dict):
                return first
            raise InstallationError(f"Unexpected list element type: {type(first)}")

        if isinstance(response, str):
            raw = response.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw).strip()
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise InstallationError(f"LLM response is not valid JSON: {exc}")
            if isinstance(parsed, list):
                if not parsed:
                    raise InstallationError("LLM returned empty JSON array")
                parsed = parsed[0]
            if isinstance(parsed, dict):
                return parsed
            raise InstallationError(f"LLM JSON parsed to unexpected type: {type(parsed)}")

        raise InstallationError(
            f"_normalize_llm_response: unhandled response type {type(response)}"
        )

    def _validate_action_schema(self, decision: dict) -> None:
        ALLOWED_OPS = {"click", "type", "hotkey", "wait", "done"}

        if not isinstance(decision, dict):
            raise InstallationError("Action must be a dict")

        op = decision.get("operation")
        if not isinstance(op, str) or op.lower() not in ALLOWED_OPS:
            raise InstallationError(
                f"Invalid operation '{op}'. Allowed: {ALLOWED_OPS}"
            )

        op = op.lower()

        if op == "click":
            target = decision.get("target")
            x = decision.get("x")
            y = decision.get("y")
            # PATCH §1.10: either target OR x/y coordinates are acceptable
            has_target = isinstance(target, str) and target.strip()
            has_coords = isinstance(x, (int, float)) and isinstance(y, (int, float))
            if not has_target and not has_coords:
                raise InstallationError(
                    "click action requires either 'target' string or 'x'+'y' coordinates"
                )

        if op == "type":
            text = decision.get("text")
            if not isinstance(text, str):
                raise InstallationError("type action requires 'text' string")

        if op == "hotkey":
            keys = decision.get("keys")
            if (
                not isinstance(keys, list)
                or not keys
                or not all(isinstance(k, str) for k in keys)
            ):
                raise InstallationError(
                    "hotkey action requires 'keys' as non-empty list of strings"
                )

    def _sanitize_perception(self, perception) -> dict:
        import json

        if not isinstance(perception, dict):
            return {"note": "perception unavailable"}

        MAX_ENTITIES = 15
        MAX_BYTES = self.MAX_PERCEPTION_BYTES

        safe = {}

        for key, value in perception.items():
            if key in ("elements", "entities"):
                if isinstance(value, list):
                    safe[key] = value[:MAX_ENTITIES]
                else:
                    safe[key] = []
            else:
                safe[key] = value

        try:
            serialised = json.dumps(safe)
        except (TypeError, ValueError):
            return {"note": "perception not serializable"}

        if len(serialised) > MAX_BYTES:
            entities = safe.get("elements") or safe.get("entities") or []
            while len(serialised) > MAX_BYTES and entities:
                entities = entities[:-1]
                safe["elements"] = entities
                safe["entities"] = entities
                try:
                    serialised = json.dumps(safe)
                except (TypeError, ValueError):
                    break

        return safe

    def _validate_tool_schema(self, tool: Dict[str, Any]) -> None:

        if not isinstance(tool, dict):
            raise InstallationError("Tool must be dictionary")

        missing = self.REQUIRED_FIELDS - set(tool.keys())
        if missing:
            raise InstallationError(
                f"Tool dict missing required fields: {missing}. "
                f"Ensure the planner includes 'official_url' in tool_installation steps."
            )

        if not isinstance(tool.get("name"), str) or not tool["name"].strip():
            raise InstallationError("Invalid tool name")

        if not isinstance(tool.get("official_url"), str) or not tool["official_url"].strip():
            raise InstallationError("Invalid official_url")

    def _validate_url(self, url: str) -> None:
        if not url.startswith("https://"):
            raise InstallationError("official_url must use https:// scheme")
