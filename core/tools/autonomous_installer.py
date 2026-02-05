"""
Autonomous Installer

Purpose:
Installs missing tools using **human-like interaction**, not scripted shell installs.

This module:
- DOES interact with UI via observer + screenpipe
- DOES use OS backend for clicks / typing
- DOES NOT execute arbitrary shell installers silently
- DOES verify installation post-completion
- DOES fail hard if installer flow is ambiguous

This is the ONLY allowed mechanism for TOOL_INSTALLATION steps.
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

    MAX_INSTALL_TIME = 15 * 60  # 15 minutes per tool
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
        self._verifier = StepVerifier()

    # -------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------

    def install_tool(self, tool: Dict[str, Any]) -> None:
        """
        Install a tool via browser + installer wizard.

        tool schema (minimal):
        {
            "name": "node",
            "official_url": "https://nodejs.org",
            "verification": {...}
        }
        """

        name = tool.get("name")
        url = tool.get("official_url")

        if not name or not url:
            raise InstallationError("Tool definition incomplete")

        start_ts = time.time()

        # ---- open browser ----
        self._open_browser()
        self._wait_ui()

        # ---- navigate to official site ----
        self._navigate(url)
        self._wait_ui()

        # ---- locate download ----
        if not self._click_download_button(name):
            raise InstallationError(
                f"Could not locate download button for {name}"
            )

        # ---- wait for download ----
        installer_path = self._wait_for_download(start_ts)
        if not installer_path:
            raise InstallationError(
                f"Installer download not detected for {name}"
            )

        # ---- run installer ----
        self._os.exec(installer_path)
        self._wait_ui()

        # ---- wizard navigation ----
        self._navigate_installer_wizard(start_ts)

        # ---- verification ----
        if not self._verify_installation(tool):
            raise InstallationError(
                f"Post-install verification failed for {name}"
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
        Uses screen text to infer download CTA.
        Conservative: only clicks when high confidence.
        """
        state = self._screenpipe.read()
        text = state.get("text", "").lower()

        keywords = ["download", tool_name.lower()]
        if not all(k in text for k in keywords):
            return False

        button = self._observer.find_click_target(
            contains="download"
        )
        if not button:
            return False

        self._os.mouse(button)
        return True

    def _wait_for_download(self, start_ts: float) -> Optional[str]:
        """
        Poll downloads directory heuristically.
        """
        while time.time() - start_ts < self.MAX_INSTALL_TIME:
            path = self._os.get_latest_download()
            if path:
                return path
            time.sleep(1.0)
        return None

    def _navigate_installer_wizard(self, start_ts: float) -> None:
        """
        Generic installer wizard handler:
        clicks Next / Agree / Install until completion.
        """
        while time.time() - start_ts < self.MAX_INSTALL_TIME:
            state = self._screenpipe.read()
            text = state.get("text", "").lower()

            if any(k in text for k in ("finish", "completed", "done")):
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

    def _verify_installation(self, tool: Dict[str, Any]) -> bool:
        """
        Delegates to StepVerifier using tool_check semantics.
        """
        action = {
            "operation": "tool_check",
            "tool": tool.get("name"),
            "version_command": tool.get("version_command"),
            "min_version": tool.get("min_version"),
        }
        return self._verifier.verify_step(
            action=action,
            execution_result=None,
        )

    # -------------------------------------------------
    # UTIL
    # -------------------------------------------------

    def _wait_ui(self) -> None:
        time.sleep(self.UI_SETTLE_DELAY)
