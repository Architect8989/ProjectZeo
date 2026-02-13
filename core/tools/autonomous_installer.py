import time
from typing import Dict, Any, Optional

from observer.observer_core import ObserverCore
from operate.utils.operating_system import OperatingSystem
from core.verification.step_verifier import StepVerifier, VerificationError
from core.schemas.execution_plan import ExecutionStep, StepType


class InstallationError(RuntimeError):
    pass


class AutonomousInstaller:
    """
    Human-style installer.

    HARD CONTRACT:
    - Browser-driven only
    - No silent shell installs
    - Deterministic wait logic
    - Verification is authoritative
    - Idempotent
    - Fail-closed
    """

    MAX_INSTALL_TIME = 15 * 60
    UI_SETTLE_DELAY = 1.0
    PAGE_LOAD_TIMEOUT = 10.0
    VERIFICATION_POLL_INTERVAL = 5.0

    REQUIRED_FIELDS = {"name", "official_url"}

    def __init__(
        self,
        *,
        observer: ObserverCore,
        os_backend: OperatingSystem,
    ):
        if not isinstance(observer, ObserverCore):
            raise InstallationError("Observer required")

        if not isinstance(os_backend, OperatingSystem):
            raise InstallationError("OperatingSystem backend required")

        self._observer = observer
        self._os = os_backend
        self._verifier = StepVerifier()

    # =================================================
    # PUBLIC API
    # =================================================

    def install_tool(self, tool: Dict[str, Any]) -> None:
        """
        Install tool via deterministic browser workflow.
        """

        self._validate_tool_schema(tool)

        name = tool["name"]
        url = tool["official_url"]

        # --------------------------------------------------
        # IDENTITY CHECK (STRICT)
        # --------------------------------------------------

        if self._is_already_installed(tool):
            return

        start_ts = time.time()

        # --------------------------------------------------
        # ENVIRONMENT SAFETY CHECK
        # --------------------------------------------------

        if not self._observer.is_healthy():
            raise InstallationError("Observer unhealthy — aborting install")

        # --------------------------------------------------
        # BROWSER WORKFLOW
        # --------------------------------------------------

        self._open_browser()
        self._wait_ui()

        self._navigate(url)
        self._wait_for_page_change()

        # --------------------------------------------------
        # INSTALL WAIT LOOP
        # --------------------------------------------------

        while time.time() - start_ts < self.MAX_INSTALL_TIME:

            if not self._observer.is_healthy():
                raise InstallationError("Observer lost during install")

            if self._is_already_installed(tool):
                return  # VERIFIED SUCCESS

            time.sleep(self.VERIFICATION_POLL_INTERVAL)

        raise InstallationError(
            f"Installation timed out without verification: {name}"
        )

    # =================================================
    # INTERNAL ACTIONS
    # =================================================

    def _open_browser(self) -> None:
        try:
            self._os.open_browser()
        except Exception as e:
            raise InstallationError(
                f"Failed to open browser: {e}"
            ) from e

    def _navigate(self, url: str) -> None:
        try:
            self._os.focus_address_bar()
            self._os.press(["ctrl", "a"])
            self._os.write(url)
            self._os.press(["enter"])
        except Exception as e:
            raise InstallationError(
                f"Navigation failed: {e}"
            ) from e

    def _wait_for_page_change(self) -> None:
        time.sleep(self.PAGE_LOAD_TIMEOUT)

    # =================================================
    # VERIFICATION
    # =================================================

    def _is_already_installed(self, tool: Dict[str, Any]) -> bool:
        """
        Single authoritative existence check.
        """

        version_cmd = tool.get("version_command")
        min_version = tool.get("min_version")

        if not isinstance(version_cmd, str) or not version_cmd.strip():
            # If no deterministic verification command exists,
            # installation cannot be trusted.
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

    # =================================================
    # UTIL
    # =================================================

    def _wait_ui(self) -> None:
        time.sleep(self.UI_SETTLE_DELAY)
