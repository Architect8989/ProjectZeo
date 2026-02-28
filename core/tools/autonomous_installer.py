from __future__ import annotations

import platform
import time
import json
import re
from typing import Dict, Any, Optional, Tuple, List

from observer.observer_core import ObserverCore
from operate.utils.operating_system import OperatingSystem
from core.verification.step_verifier import StepVerifier, VerificationError
from core.schemas.execution_plan import ExecutionStep, StepType
from core.cognition.reasoning_engine import ReasoningEngine


class InstallationError(RuntimeError):
    pass


import shutil


def _get_linux_pkg_manager() -> str:
    if shutil.which("apt-get"):
        return "apt-get"
    if shutil.which("snap"):
        return "snap"
    return ""


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
    "node": {
        # SI-3 FIX: original "curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -"
        # matched the r"\|\s*bash\b" dangerous pattern and was silently dropped.
        # Replaced with a direct apt-get install that does not pipe to a shell interpreter.
        "Linux":   "sudo apt-get update -qq && sudo apt-get install -y nodejs npm",
        "Darwin":  _brew("node"),
        "Windows": _choco("nodejs"),
    },
    "nodejs": {
        "Linux":   "sudo apt-get update -qq && sudo apt-get install -y nodejs npm",
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
        # SI-3 FIX: original "curl -fsSL https://bun.sh/install | bash" matched
        # the r"\|\s*bash\b" dangerous pattern. Replaced with npm global install
        # which does not pipe to a shell interpreter.
        "Linux":   "npm install -g bun",
        "Darwin":  "curl -fsSL https://bun.sh/install | bash",
        "Windows": "powershell -c \"irm bun.sh/install.ps1|iex\"",
    },
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
    "git": {
        "Linux":   _apt("git"),
        "Darwin":  _brew("git"),
        "Windows": _choco("git"),
    },
    "docker": {
        # SI-3 FIX: original "curl -fsSL https://get.docker.com | sh" matched
        # both r"\bcurl\b.*\|\s*(?:ba)?sh\b" and r"\|\s*sh\b" dangerous patterns.
        # Replaced with apt-get install docker.io which is available in Ubuntu
        # default repositories without piping to a shell interpreter.
        "Linux":   "sudo apt-get update -qq && sudo apt-get install -y docker.io",
        "Darwin":  "brew install --cask docker",
        "Windows": _choco("docker-desktop"),
    },
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
    "go": {
        "Linux":   "curl -fsSL https://go.dev/dl/go1.22.0.linux-amd64.tar.gz | sudo tar -C /usr/local -xzf - && echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc",
        "Darwin":  _brew("go"),
        "Windows": _choco("golang"),
    },
    "cargo": {
        # SI-3 FIX: original "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"
        # matched r"\bcurl\b.*\|\s*(?:ba)?sh\b" and r"\|\s*sh\b" dangerous patterns.
        # Replaced with apt-get install cargo which is available in Ubuntu repos.
        "Linux":   "sudo apt-get update -qq && sudo apt-get install -y cargo",
        "Darwin":  "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "Windows": _choco("rust"),
    },
    "rustup": {
        # SI-3 FIX: same as cargo — replaced curl|sh with apt-get rustup.
        "Linux":   "sudo apt-get update -qq && sudo apt-get install -y rustup",
        "Darwin":  "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "Windows": "winget install Rustlang.Rustup",
    },
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
    "ffmpeg": {
        "Linux":   _apt("ffmpeg"),
        "Darwin":  _brew("ffmpeg"),
        "Windows": _choco("ffmpeg"),
    },
    "gh": {
        # SI-3 FIX: original command used "sudo dd of=/usr/share/keyrings/..."
        # which matched the r"\bdd\b" dangerous pattern and was silently dropped.
        # Replaced with sudo snap install which is available on Ubuntu and does
        # not trigger any dangerous patterns.
        "Linux":   "sudo snap install gh --classic",
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
    TERMINAL_INSTALL_TIMEOUT_SECONDS = 300
    REQUIRED_FIELDS = {"name", "official_url"}

    _DANGEROUS_PATTERNS = [
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
    ]

    _COORD_OVERSHOOT_FRACTION: float = 0.10

    def __init__(
        self,
        *,
        observer: ObserverCore,
        os_backend: OperatingSystem,
        llm_callable,
        shared_ollama_client=None,
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
        self._shared_ollama_client = shared_ollama_client

        self._compiled_dangerous = [
            re.compile(p, re.IGNORECASE) for p in self._DANGEROUS_PATTERNS
        ]

    def _validate_llm_command(self, cmd: str) -> None:
        for pattern in self._compiled_dangerous:
            if pattern.search(cmd):
                raise InstallationError(
                    f"LLM-generated install command rejected — "
                    f"matches dangerous pattern {pattern.pattern!r}: {cmd!r}"
                )

    def install_tool(self, tool: Dict[str, Any]) -> None:
        self._validate_tool_schema(tool)

        name = tool["name"]
        url = tool["official_url"]

        self._validate_url(url)

        if self._is_already_installed(tool):
            return

        if not self._observer.is_healthy():
            raise InstallationError("Observer unhealthy — aborting install")

        if self._try_terminal_install(tool):
            if self._is_already_installed(tool):
                return

        self._browser_ui_install(tool)

    def _try_terminal_install(self, tool: Dict[str, Any]) -> bool:
        name = tool["name"]
        os_name = platform.system()

        pre_specified = tool.get("install_commands", [])
        if isinstance(pre_specified, list) and pre_specified:
            for cmd in pre_specified:
                if isinstance(cmd, str) and cmd.strip():
                    try:
                        result = self._os.run_command(cmd.strip())
                        if hasattr(result, "returncode") and result.returncode == 0:
                            return True
                    except Exception:
                        continue
            return True

        name_lower = name.lower().strip()
        known = COMMON_INSTALL_COMMANDS.get(name_lower, {})
        cmd = known.get(os_name)

        if cmd:
            try:
                self._os.run_command(cmd)
                return True
            except Exception:
                return False

        try:
            import os as _os_mod
            import ollama as _ollama
            import httpx as _httpx

            _model = _os_mod.environ.get("LLM_MODEL", "qwen2.5-vl:7b-instruct")

            pkg_mgr = _get_linux_pkg_manager() if os_name == "Linux" else ""
            pkg_mgr_hint = f" Package manager available: {pkg_mgr}." if pkg_mgr else ""

            if self._shared_ollama_client is not None:
                client = self._shared_ollama_client
            else:
                client = _ollama.Client(
                    timeout=_httpx.Timeout(connect=10.0, read=60.0, write=5.0, pool=2.0)
                )

            response = client.chat(
                model=_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a shell command generator. "
                            "Respond with ONLY a single shell command string — "
                            "no JSON, no explanation, no markdown. "
                            "The command must install the requested tool on the target OS."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"OS: {os_name}.{pkg_mgr_hint}\n"
                            f"Tool to install: {name}\n"
                            "Return one shell command to install it."
                        ),
                    },
                ],
                options={"temperature": 0},
            )

            llm_cmd: Optional[str] = None
            if hasattr(response, "message") and hasattr(response.message, "content"):
                llm_cmd = response.message.content
            elif isinstance(response, dict):
                llm_cmd = response.get("message", {}).get("content")

            if isinstance(llm_cmd, str):
                llm_cmd = llm_cmd.strip().strip("`\"'")

            if llm_cmd and len(llm_cmd) > 4 and "\n" not in llm_cmd:
                self._validate_llm_command(llm_cmd)
                self._os.run_command(llm_cmd)
                return True

        except Exception:
            pass

        return False

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
                x = action.get("x")
                y = action.get("y")
                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    coords = (float(x), float(y))

            if coords is None:
                raise InstallationError(
                    f"Unable to resolve click target: "
                    f"{target!r} (no OCR match and no x/y coordinates)"
                )

            cx, cy = coords

            if cx > 1.0 or cy > 1.0:
                try:
                    sw, sh = self._os.screen_size()
                    if sw <= 0 or sh <= 0:
                        raise InstallationError(
                            f"screen_size() returned invalid dimensions: "
                            f"({sw}, {sh}) — cannot validate click coordinates"
                        )
                except InstallationError:
                    raise
                except Exception as _se:
                    raise InstallationError(
                        f"Cannot determine screen size for coordinate validation: "
                        f"{_se}"
                    ) from _se

                _max_x = sw * (1.0 + self._COORD_OVERSHOOT_FRACTION)
                _max_y = sh * (1.0 + self._COORD_OVERSHOOT_FRACTION)

                if cx > _max_x or cy > _max_y:
                    raise InstallationError(
                        f"LLM-generated click coordinates ({cx:.0f}, {cy:.0f}) "
                        f"exceed screen bounds {sw}×{sh} by more than "
                        f"{self._COORD_OVERSHOOT_FRACTION * 100:.0f}%. "
                        "This indicates the model hallucinated coordinates for "
                        "a different screen resolution. Retrying action selection."
                    )

                cx = cx / sw
                cy = cy / sh

            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))

            self._os.click(cx, cy)
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
