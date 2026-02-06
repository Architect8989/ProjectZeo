"""
Autonomous Installer

Purpose:
Installs missing tools using human-like interaction, not scripted installs.
"""

import time
from typing import Dict, Any, Optional

from observer.observer_core import ObserverCore
from observer.screenpipe_adapter import ScreenpipeAdapter, ScreenpipeBlindnessError
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
    """

    MAX_INSTALL_TIME = 15 * 60  # seconds
    UI_SETTLE_DELAY = 1.0
    PAGE_LOAD_TIMEOUT = 10.0

    def __init__(
        self,
        *,
        observer: ObserverCore,
        screenpipe: ScreenpipeAdapter,
        os_backend: OperatingSystem,
    ):
        self._observer = observer
        self._screenpipe = screenpipe
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

        if not self._click_download_button():
            raise InstallationError(
                f"Download button not found for {name}"
            )

        installer_path = self._wait_for_download(start_ts)
        if not installer_path:
            raise InstallationError(
                f"Installer download not detected for {name}"
            )

        self._os.exec(installer_path)
        self._wait_ui()

        self._navigate_installer_wizard(start_ts)

        # --------------------------------------------------
        # POST-INSTALL VERIFICATION
        # --------------------------------------------------
        if not self._is_already_installed(tool):
            raise InstallationError(
                f"Post-install verification failed for {name}"
            )

    # =================================================
    # INTERNAL UI ACTIONS
    # =================================================

    def _open_browser(self) -> None:
        self._os.open_browser()

    def _navigate(self, url: str) -> None:
        self._os.focus_address_bar()
        self._os.press(["ctrl", "a"])
        self._os.write(url)
        self._os.press(["enter"])

    def _wait_for_page_change(self) -> None:
        initial = self._safe_read().get("screen_hash")
        start = time.time()

        while time.time() - start < self.PAGE_LOAD_TIMEOUT:
            state = self._safe_read()
            if state.get("screen_hash") and state.get("screen_hash") != initial:
                time.sleep(self.UI_SETTLE_DELAY)
                return
            time.sleep(0.5)

        raise InstallationError("Page load timeout")

    def _click_download_button(self) -> bool:
        state = self._safe_read()
        text = state.get("text", "").lower()

        if "download" not in text:
            return False

        target = self._observer.find_click_target(contains="download")
        if not target:
            return False

        self._os.mouse(target)
        return True

    def _wait_for_download(self, start_ts: float) -> Optional[str]:
        while time.time() - start_ts < self.MAX_INSTALL_TIME:
            path = self._os.get_latest_download()
            if path:
                return path
            time.sleep(1.0)
        return None

    def _navigate_installer_wizard(self, start_ts: float) -> None:
        while time.time() - start_ts < self.MAX_INSTALL_TIME:
            state = self._safe_read()
            text = state.get("text", "").lower()

            if any(k in text for k in ("completed", "finish", "done")):
                self._click_button(["finish", "done"])
                return

            if self._click_button(["next", "agree", "install", "continue"]):
                self._wait_ui()
                continue

            time.sleep(1.0)

        raise InstallationError("Installer wizard timed out")

    def _click_button(self, labels) -> bool:
        for label in labels:
            target = self._observer.find_click_target(contains=label)
            if target:
                self._os.mouse(target)
                return True
        return False

    # =================================================
    # VERIFICATION (FIXED)
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

    def _safe_read(self) -> Dict[str, Any]:
        try:
            return self._screenpipe.read() or {}
        except ScreenpipeBlindnessError:
            raise
        except Exception:
            return {}
