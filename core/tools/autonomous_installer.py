import time
from typing import Dict, Any

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
    - Browser-driven
    - UI-observed when vision exists
    - Deterministic fallbacks only
    - No silent shell installs
    - Idempotent (never reinstall if already present)

    WITHOUT VISION:
    - Navigation + wait + verification polling only
    - Success is NEVER assumed
    """

    MAX_INSTALL_TIME = 15 * 60  # seconds
    UI_SETTLE_DELAY = 1.0
    PAGE_LOAD_TIMEOUT = 10.0
    VERIFICATION_POLL_INTERVAL = 5.0

    def __init__(
        self,
        *,
        observer: ObserverCore,
        os_backend: OperatingSystem,
    ):
        self._observer = observer
        self._os = os_backend
        self._verifier = StepVerifier()

    # =================================================
    # PUBLIC API
    # =================================================

    def install_tool(self, tool: Dict[str, Any]) -> None:
        """
        Install a tool via browser + installer wizard.

        NEVER reports success unless verification passes.
        """

        name = tool.get("name")
        url = tool.get("official_url")

        if not isinstance(name, str) or not isinstance(url, str):
            raise InstallationError("Tool definition incomplete")

        # --------------------------------------------------
        # IDEMPOTENCE CHECK (AUTHORITATIVE)
        # --------------------------------------------------
        if self._is_already_installed(tool):
            return

        start_ts = time.time()

        # --------------------------------------------------
        # BROWSER-DRIVEN NAVIGATION
        # --------------------------------------------------
        self._open_browser()
        self._wait_ui()

        self._navigate(url)
        self._wait_for_page_change()

        # --------------------------------------------------
        # INSTALLATION WAIT LOOP (FAIL-CLOSED)
        # --------------------------------------------------
        while time.time() - start_ts < self.MAX_INSTALL_TIME:
            if self._is_already_installed(tool):
                return  # VERIFIED SUCCESS

            time.sleep(self.VERIFICATION_POLL_INTERVAL)

        # --------------------------------------------------
        # TIMEOUT → AUTHORITATIVE FAILURE
        # --------------------------------------------------
        raise InstallationError(
            f"Tool installation timed out without verification: {name}"
        )

    # =================================================
    # INTERNAL UI ACTIONS
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
                f"Browser navigation failed: {e}"
            ) from e

    def _wait_for_page_change(self) -> None:
        """
        Deterministic delay until vision-based detection exists.
        """
        time.sleep(self.PAGE_LOAD_TIMEOUT)

    # =================================================
    # VERIFICATION (AUTHORITATIVE)
    # =================================================

    def _is_already_installed(self, tool: Dict[str, Any]) -> bool:
        """
        Single source of truth for tool existence.
        """

        step = ExecutionStep(
            id=0,
            type=StepType.TOOL_INSTALLATION,
            description=f"Verify {tool.get('name')} installed",
            action={"tool": tool.get("name")},
            verification={
                "version_command": tool.get("version_command"),
                "min_version": tool.get("min_version"),
            },
        )

        try:
            result = self._verifier.verify_step(step)
            return bool(result.success)
        except (VerificationError, Exception):
            return False

    # =================================================
    # UTIL
    # =================================================

    def _wait_ui(self) -> None:
        time.sleep(self.UI_SETTLE_DELAY)
