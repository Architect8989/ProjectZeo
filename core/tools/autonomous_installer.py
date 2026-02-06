"""
Autonomous Installer

Purpose:
Installs missing tools using human-like interaction, not scripted installs.
"""

import time
from typing import Dict, Any, Optional

from observer.observer_core import ObserverCore
from observer.screenpipe_adapter import ScreenpipeAdapter
from operate.utils.operating_system import OperatingSystem
from core.verification.step_verifier import StepVerifier


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
    """

    MAX_INSTALL_TIME = 15 * 60  # seconds
    UI_SETTLE_DELAY = 1.0

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
        self._verifier = StepVerifier(os_backend=os_backend)

    # -------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------

    def install_tool(self, tool: Dict[str, Any]) -> None:
        """
        Install a tool via browser + installer wizard.
        """

        name = tool.get("name")
        url = tool.get("official_url")

        if not isinstance(name, str) or not isinstance(url, str):
            raise InstallationError("Tool definition incomplete")

        start_ts = time.time()

        self._open_browser()
        self._wait_ui()

        self._navigate(url)
        self._wait_ui()

        if not self._click_download_button(name):
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

        verification = self._verify_installation(tool)
        if not verification.success:
            raise InstallationError(
                f"Post-install verification failed: {verification.reason}"
            )

    # -------------------------------------------------
    # INTERNAL UI ACTIONS
    # -------------------------------------------------

    def _open_browser(self) -> None:
        self._os.open_browser()

    def _navigate(self, url: str) -> None:
        self._os.focus_address_bar()
        self._os.write(url)
        self._os.press(["enter"])

    def _click_download_button(self, tool_name: str) -> bool:
        """
        Conservative: click only when high confidence.
        """
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
        """
        Deterministic wizard navigation.
        """
        while time.time() - start_ts < self.MAX_INSTALL_TIME:
            state = self._safe_read()
            text = state.get("text", "").lower()

            if any(k in text for k in ("completed", "finish", "done")):
                self._click_button(["finish", "done"])
                return

            if self._click_button(["next", "agree", "install"]):
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

    # -------------------------------------------------
    # VERIFICATION
    # -------------------------------------------------

    def _verify_installation(self, tool: Dict[str, Any]):
        """
        Delegates to StepVerifier (authoritative).
        """
        step_like = {
            "operation": "tool_check",
            "tool": tool.get("name"),
            "version_command": tool.get("version_command"),
            "min_version": tool.get("min_version"),
        }

        return self._verifier.verify_step_like(step_like)

    # -------------------------------------------------
    # UTIL
    # -------------------------------------------------

    def _wait_ui(self) -> None:
        time.sleep(self.UI_SETTLE_DELAY)

    def _safe_read(self) -> Dict[str, Any]:
        try:
            return self._screenpipe.read() or {}
        except Exception:
            return {}
