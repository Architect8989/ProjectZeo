import time
import json
from typing import Dict, Any, Optional

from observer.observer_core import ObserverCore
from operate.utils.operating_system import OperatingSystem
from core.verification.step_verifier import StepVerifier, VerificationError
from core.schemas.execution_plan import ExecutionStep, StepType
from core.cognition.reasoning_engine import ReasoningEngine


class InstallationError(RuntimeError):
    pass


class AutonomousInstaller:
    """
    Pure LLM-driven installer.

    HARD CONTRACT:
    - Browser-driven only
    - No shell installs
    - LLM decides actions dynamically
    - Verification is authoritative
    - Idempotent
    - Fail-closed
    """

    MAX_INSTALL_TIME = 15 * 60
    UI_SETTLE_DELAY = 1.0
    PAGE_LOAD_TIMEOUT = 10.0
    MAX_ITERATIONS = 120
    MAX_PERCEPTION_BYTES = 10_000

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

        # Sanitizer without full engine construction
        self._sanitizer = ReasoningEngine.__new__(ReasoningEngine)

    # =================================================
    # PUBLIC API
    # =================================================

    def install_tool(self, tool: Dict[str, Any]) -> None:

        self._validate_tool_schema(tool)

        name = tool["name"]
        url = tool["official_url"]

        if self._is_already_installed(tool):
            return

        if not self._observer.is_healthy():
            raise InstallationError("Observer unhealthy — aborting install")

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

            self._execute_action(action)

            time.sleep(self.UI_SETTLE_DELAY)

            iteration += 1

        raise InstallationError(f"Installation timed out: {name}")

    # =================================================
    # LLM DECISION LOOP (SANITIZED + FAIL-CLOSED)
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
    # PERCEPTION SANITIZATION
    # =================================================

    def _sanitize_perception(
        self,
        perception: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:

        if not isinstance(perception, dict):
            return {}

        try:
            safe = self._sanitizer._sanitize_perception(perception)
        except Exception:
            return {}

        try:
            serialized = json.dumps(safe)
            if len(serialized) > self.MAX_PERCEPTION_BYTES:
                return {}
        except Exception:
            return {}

        return safe

    # =================================================
    # RESPONSE NORMALIZATION
    # =================================================

    def _normalize_llm_response(self, response: Any) -> Dict[str, Any]:

        if isinstance(response, list):
            if not response:
                raise InstallationError("LLM returned empty list")
            response = response[0]

        if not isinstance(response, dict):
            raise InstallationError("LLM returned invalid decision format")

        return response

    def _validate_action_schema(self, action: Dict[str, Any]) -> None:

        if "operation" not in action:
            raise InstallationError("Installer decision missing operation")

        allowed = {"click", "type", "hotkey", "wait", "done"}
        if action["operation"] not in allowed:
            raise InstallationError(
                f"Unsupported installer operation: {action['operation']}"
            )

    # =================================================
    # ACTION EXECUTION
    # =================================================

    def _execute_action(self, action: Dict[str, Any]) -> None:

        op = action.get("operation")

        if op == "click":
            target = action.get("target")
            if not isinstance(target, str) or not target.strip():
                raise InstallationError("Missing click target")
            self._os.click(target)
            return

        if op == "type":
            text = action.get("text", "")
            if not isinstance(text, str):
                raise InstallationError("Invalid type text")
            self._os.type_text(text)
            return

        if op == "hotkey":
            keys = action.get("keys", [])
            if not isinstance(keys, list):
                raise InstallationError("Invalid hotkey format")
            self._os.press_keys(keys)
            return

        if op == "wait":
            time.sleep(self.UI_SETTLE_DELAY)
            return

        raise InstallationError(f"Unsupported installer operation: {op}")

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
        except Exception as e:
            raise InstallationError(f"Failed to open browser: {e}") from e

    def _navigate(self, url: str) -> None:
        try:
            self._os.focus_address_bar()
            self._os.press_keys(["ctrl", "a"])
            self._os.type_text(url)
            self._os.press_keys(["enter"])
        except Exception as e:
            raise InstallationError(f"Navigation failed: {e}") from e

    def _wait_ui(self) -> None:
        time.sleep(self.UI_SETTLE_DELAY)

    # =================================================
    # VALIDATION
    # =================================================

    def _validate_tool_schema(self, tool: Dict[str, Any]) -> None:

        if not isinstance(tool, dict):
            raise InstallationError("Tool must be dictionary")

        missing = self.REQUIRED_FIELDS - set(tool.keys())
        if missing:
            raise InstallationError(f"Tool missing required fields: {missing}")

        if not isinstance(tool.get("name"), str) or not tool["name"].strip():
            raise InstallationError("Invalid tool name")

        if not isinstance(tool.get("official_url"), str) or not tool["official_url"].strip():
            raise InstallationError("Invalid official_url")
