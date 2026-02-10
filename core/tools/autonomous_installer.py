"""
Autonomous Installer

Purpose:
Installs missing tools using human-like interaction, not scripted installs.
"""

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
    - Browser-driven
    - UI-observed
    - Deterministic fallbacks only
    - No silent shell installs
    - Idempotent (never reinstall if already present)

    NOTE:
    Screen vision is currently stubbed.
    This class will not function fully until vision integration is complete.
    """

    MAX_INSTALL_TIME = 15 * 60  # seconds
    UI_SETTLE_DELAY = 1.0
    PAGE_LOAD_TIMEOUT = 10.0

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

        self._open_browser()
        self._wait_ui()

        self._navigate(url)
        self._wait_for_page_change()

        # From here onward, UI automation is unsafe without vision
        raise InstallationError(
            f"Autonomous installation disabled (vision unavailable) for {name}"
        )

    # =================================================
    # INTERNAL UI ACTIONS (PARTIAL)
    # =================================================

    def _open_browser(self) -> None:
        self._os.open_browser()

    def _navigate(self, url: str) -> None:
        self._os.focus_address_bar()
        self._os.press(["ctrl", "a"])
        self._os.write(url)
        self._os.press(["enter"])

    def _wait_for_page_change(self) -> None:
        """
        Stubbed.

        Previously used screen hash comparison.
        Will be replaced by vision-based state change detection.
        """
        time.sleep(self.PAGE_LOAD_TIMEOUT)

    # =================================================
    # VERIFICATION (UNCHANGED, SAFE)
    # =================================================

    def _is_already_installed(self, tool: Dict[str, Any]) -> bool:
        """
        Authoritative tool existence check.
        """

        step = ExecutionStep(
            id=0,
            type=StepType.TOOL_INSTALLATION,
            description=f"Verify {tool.get('name')} installed",
            action={
                "tool": tool.get("name"),
            },
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
