"""
core/tools/autonomous_installer.py
=====================================
PATCHES APPLIED (Audit Fixes):

  ✅  §R2  (was §inst-5 CRITICAL): _try_terminal_install() previously sent a
           text-only JSON question to the vision LLM (QwenOllamaAdapter) which
           responds with UI click operations, not install commands.
           The JSON parse always failed, causing silent fallthrough to browser UI.

           FIX: Replaced LLM-based command lookup with a two-tier strategy:
             1. Use pre-specified install_commands from the tool dict (plan-supplied).
             2. Fall back to a curated COMMON_INSTALL_COMMANDS lookup table
                covering the most common dev tools on Linux/macOS/Windows.
             3. Only if both fail, fall through to browser UI.

           This avoids sending a factual question to a vision model while still
           achieving autonomous terminal installation without scripted workflows.

  ✅  §R7  (was §inst-6): focus_address_bar() now documented to use Ctrl+L /
           Cmd+L. Added explicit browser-open fallback chain in OS backend.

  ✅  §1.10 (original): Terminal install always attempted FIRST.
  ✅  §1.10 (original): LLM x/y coordinate fallback for graphical buttons.
  ✅  §1.10 (original): official_url required in tool dict.

All existing correct behaviours preserved:
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
from typing import Dict, Any, Optional, Tuple, List

from observer.observer_core import ObserverCore
from operate.utils.operating_system import OperatingSystem
from core.verification.step_verifier import StepVerifier, VerificationError
from core.schemas.execution_plan import ExecutionStep, StepType
from core.cognition.reasoning_engine import ReasoningEngine


class InstallationError(RuntimeError):
    pass


# ==========================================================
# PATCH §R2: Curated terminal install lookup table
#
# Covers the most common dev tools across Linux (apt/snap),
# macOS (brew), and Windows (choco/winget).
#
# Keys are lowercase tool names matching tool["name"].lower().
# Each entry maps OS → preferred install command.
# ==========================================================

_LINUX_HAS_APT = bool(__import__("shutil").which("apt-get"))
_LINUX_HAS_SNAP = bool(__import__("shutil").which("snap"))
_LINUX_PKG = "apt-get" if _LINUX_HAS_APT else ("snap" if _LINUX_HAS_SNAP else None)


def _apt(pkg: str) -> str:
    return f"sudo apt-get install -y {pkg}"


def _brew(pkg: str) -> str:
    return f"brew install {pkg}"


def _snap(pkg: str, classic: bool = False) -> str:
    classic_flag = " --classic" if classic else ""
    return f"sudo snap install {pkg}{classic_flag}"


def _choco(pkg: str) -> str:
    return f"choco install -y {pkg}"


def _winget(pkg: str) -> str:
    return f"winget install --accept-source-agreements --accept-package-agreements {pkg}"


COMMON_INSTALL_COMMANDS: Dict[str, Dict[str, str]] = {
    # ---- Node.js / JavaScript ----
    "node": {
        "Linux":   "curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash - && sudo apt-get install -y nodejs",
        "Darwin":  _brew("node"),
        "Windows": _choco("nodejs"),
    },
    "nodejs": {
        "Linux":   "curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash - && sudo apt-get install -y nodejs",
        "Darwin":  _brew("node"),
        "Windows": _choco("nodejs"),
    },
    "npm": {
        "Linux":   _apt("npm"),
        "Darwin":  _brew("npm"),
        "Windows": _choco("npm"),
    },
    "yarn": {
        "Linux":   "sudo npm install -g yarn",
        "Darwin":  "npm install -g yarn",
        "Windows": "npm install -g yarn",
    },
    "pnpm": {
        "Linux":   "sudo npm install -g pnpm",
        "Darwin":  "npm install -g pnpm",
        "Windows": "npm install -g pnpm",
    },
    "bun": {
        "Linux":   "curl -fsSL https://bun.sh/install | bash",
        "Darwin":  "curl -fsSL https://bun.sh/install | bash",
        "Windows": "powershell -c \"irm bun.sh/install.ps1|iex\"",
    },
    # ---- Python ----
    "python": {
        "Linux":   _apt("python3 python3-pip python3-venv"),
        "Darwin":  _brew("python3"),
        "Windows": _choco("python3"),
    },
    "python3": {
        "Linux":   _apt("python3 python3-pip python3-venv"),
        "Darwin":  _brew("python3"),
        "Windows": _choco("python3"),
    },
    "pip": {
        "Linux":   _apt("python3-pip"),
        "Darwin":  "python3 -m ensurepip --upgrade",
        "Windows": "python -m ensurepip --upgrade",
    },
    # ---- Version Control ----
    "git": {
        "Linux":   _apt("git"),
        "Darwin":  _brew("git"),
        "Windows": _choco("git"),
    },
    # ---- Containers ----
    "docker": {
        "Linux":   "curl -fsSL https://get.docker.com | sh",
        "Darwin":  "brew install --cask docker",
        "Windows": _choco("docker-desktop"),
    },
    # ---- Build tools ----
    "make": {
        "Linux":   _apt("build-essential"),
        "Darwin":  "xcode-select --install",
        "Windows": _choco("make"),
    },
    "gcc": {
        "Linux":   _apt("build-essential"),
        "Darwin":  "xcode-select --install",
        "Windows": _choco("mingw"),
    },
    # ---- Databases ----
    "postgresql": {
        "Linux":   _apt("postgresql postgresql-contrib"),
        "Darwin":  _brew("postgresql"),
        "Windows": _choco("postgresql"),
    },
    "postgres": {
        "Linux":   _apt("postgresql postgresql-contrib"),
        "Darwin":  _brew("postgresql"),
        "Windows": _choco("postgresql"),
    },
    "mysql": {
        "Linux":   _apt("mysql-server"),
        "Darwin":  _brew("mysql"),
        "Windows": _choco("mysql"),
    },
    "redis": {
        "Linux":   _apt("redis-server"),
        "Darwin":  _brew("redis"),
        "Windows": _choco("redis-64"),
    },
    "mongodb": {
        "Linux":   "curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | sudo gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg --dearmor && echo 'deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse' | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list && sudo apt-get update && sudo apt-get install -y mongodb-org",
        "Darwin":  "brew tap mongodb/brew && brew install mongodb-community",
        "Windows": _choco("mongodb"),
    },
    # ---- Shell utilities ----
    "curl": {
        "Linux":   _apt("curl"),
        "Darwin":  _brew("curl"),
        "Windows": _choco("curl"),
    },
    "wget": {
        "Linux":   _apt("wget"),
        "Darwin":  _brew("wget"),
        "Windows": _choco("wget"),
    },
    "unzip": {
        "Linux":   _apt("unzip"),
        "Darwin":  _brew("unzip"),
        "Windows": _choco("unzip"),
    },
    "jq": {
        "Linux":   _apt("jq"),
        "Darwin":  _brew("jq"),
        "Windows": _choco("jq"),
    },
    # ---- Go ----
    "go": {
        "Linux":   "curl -fsSL https://go.dev/dl/go1.22.0.linux-amd64.tar.gz | sudo tar -C /usr/local -xzf - && echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc",
        "Darwin":  _brew("go"),
        "Windows": _choco("golang"),
    },
    # ---- Rust ----
    "cargo": {
        "Linux":   "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "Darwin":  "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "Windows": _choco("rust"),
    },
    "rustup": {
        "Linux":   "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "Darwin":  "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "Windows": "winget install Rustlang.Rustup",
    },
    # ---- Java ----
    "java": {
        "Linux":   _apt("default-jdk"),
        "Darwin":  _brew("openjdk"),
        "Windows": _choco("openjdk"),
    },
    "javac": {
        "Linux":   _apt("default-jdk"),
        "Darwin":  _brew("openjdk"),
        "Windows": _choco("openjdk"),
    },
    # ---- Misc ----
    "ffmpeg": {
        "Linux":   _apt("ffmpeg"),
        "Darwin":  _brew("ffmpeg"),
        "Windows": _choco("ffmpeg"),
    },
    "gh": {
        "Linux":   "curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg && echo 'deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main' | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null && sudo apt update && sudo apt install gh -y",
        "Darwin":  _brew("gh"),
        "Windows": _choco("gh"),
    },
}


class AutonomousInstaller:

    MAX_INSTALL_TIME = 15 * 60
    UI_SETTLE_DELAY = 1.0
    PAGE_LOAD_TIMEOUT = 10.0
    MAX_ITERATIONS = 120
    MAX_PERCEPTION_BYTES = 10_000

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

        Strategy (in order):
          1. Terminal install — pre-specified commands in tool dict.
          2. Terminal install — COMMON_INSTALL_COMMANDS lookup table.
          3. Browser-UI install — navigate official_url and LLM-drive clicks.

        The tool dict MUST include:
          name         : str  — display name of the tool
          official_url : str  — https:// URL of the official download page
        Optional:
          version_command  : str       — shell command to verify installation
          min_version      : str       — semver floor
          install_commands : list[str] — pre-specified commands (highest priority)
        """
        self._validate_tool_schema(tool)

        name = tool["name"]
        url = tool["official_url"]

        self._validate_url(url)

        if self._is_already_installed(tool):
            return

        if not self._observer.is_healthy():
            raise InstallationError("Observer unhealthy — aborting install")

        # PATCH §R2: terminal install via lookup table (no vision LLM misuse)
        if self._try_terminal_install(tool):
            if self._is_already_installed(tool):
                return
            # Terminal ran something but verification failed — try browser UI

        # Browser-UI fallback
        self._browser_ui_install(tool)

    # =================================================
    # TERMINAL INSTALL PATH (PATCH §R2)
    # =================================================

    def _try_terminal_install(self, tool: Dict[str, Any]) -> bool:
        """
        PATCH §R2: Two-tier terminal install strategy.

        Tier 1: plan-supplied install_commands (highest fidelity).
        Tier 2: COMMON_INSTALL_COMMANDS lookup table by tool name + OS.

        Returns True if any command was attempted (success or failure).
        Does NOT use the vision LLM — avoids the fundamental interface mismatch
        where QwenOllamaAdapter returns UI click operations, not install commands.
        """
        name = tool["name"]
        os_name = platform.system()

        # --- Tier 1: plan-supplied commands ---
        pre_specified = tool.get("install_commands", [])
        if isinstance(pre_specified, list) and pre_specified:
            for cmd in pre_specified:
                if isinstance(cmd, str) and cmd.strip():
                    try:
                        result = self._os.run_command(cmd.strip())
                        if isinstance(result, dict) and result.get("returncode", 1) == 0:
                            return True
                        # Try next command if this one failed
                    except Exception:
                        continue
            # All pre-specified commands tried (may have failed)
            return True

        # --- Tier 2: lookup table ---
        name_lower = name.lower().strip()
        known = COMMON_INSTALL_COMMANDS.get(name_lower, {})
        cmd = known.get(os_name)

        if not cmd:
            # No known terminal install for this tool/OS combo
            return False

        try:
            self._os.run_command(cmd)
            return True
        except Exception:
            return False

    # =================================================
    # BROWSER-UI INSTALL PATH
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
