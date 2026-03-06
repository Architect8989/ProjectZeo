from __future__ import annotations

import logging
import os
import subprocess
import threading
from typing import Any, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

# Lazy import of open-interpreter computer module
_computer = None
_COMPUTER_API_AVAILABLE: Optional[bool] = None
_COMPUTER_INIT_LOCK = threading.Lock()


def _check_computer_api() -> bool:
    global _computer, _COMPUTER_API_AVAILABLE
    if _COMPUTER_API_AVAILABLE is not None:
        return _COMPUTER_API_AVAILABLE
    with _COMPUTER_INIT_LOCK:
        if _COMPUTER_API_AVAILABLE is not None:
            return _COMPUTER_API_AVAILABLE
        try:
            from interpreter import computer  # noqa: PLC0415
            _computer = computer
            _COMPUTER_API_AVAILABLE = True
            _logger.info("[ComputerAPIBackend] Open Interpreter computer API available.")
        except ImportError:
            _COMPUTER_API_AVAILABLE = False
            _logger.info(
                "[ComputerAPIBackend] open-interpreter not installed. "
                "Using OperatingSystem fallback. "
                "Install: pip install open-interpreter"
            )
    return _COMPUTER_API_AVAILABLE


def _use_computer_api() -> bool:
    """Return True when Computer API is enabled and available."""
    return (
        os.environ.get("PROJECTZEO_USE_COMPUTER_API", "0").strip() in ("1", "true", "yes")
        and _check_computer_api()
    )


# ---------------------------------------------------------------------------
# Simple result type (mirrors OperatingSystem.ExecResult)
# ---------------------------------------------------------------------------

class ExecResult:
    """Command execution result compatible with OperatingSystem.ExecResult."""

    __slots__ = ("stdout", "stderr", "returncode")

    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


# ---------------------------------------------------------------------------
# ComputerAPIBackend
# ---------------------------------------------------------------------------

class ComputerAPIBackend:
    

    def __init__(self) -> None:
        self._use_api = _use_computer_api()
        self._os_fallback = None
        self._fallback_lock = threading.Lock()

        if self._use_api:
            _logger.info("[ComputerAPIBackend] Active — using Open Interpreter computer API.")
        else:
            _logger.info("[ComputerAPIBackend] Inactive — delegating to OperatingSystem.")

    def _get_os_fallback(self):
        """Lazy-initialise OperatingSystem fallback."""
        if self._os_fallback is not None:
            return self._os_fallback
        with self._fallback_lock:
            if self._os_fallback is None:
                try:
                    from operate.utils.operating_system import OperatingSystem  # noqa: PLC0415
                    self._os_fallback = OperatingSystem()
                except Exception as exc:
                    _logger.warning("[ComputerAPIBackend] OperatingSystem init failed: %s", exc)
        return self._os_fallback

    # =========================================================================
    # Screenshot
    # =========================================================================

    def screenshot(self) -> Optional[Any]:
        """
        Capture a screenshot.

        Returns:
            PIL Image if computer API is active, else bytes.
        """
        if self._use_api:
            try:
                return _computer.display.view()
            except Exception as exc:
                _logger.debug("[ComputerAPIBackend] screenshot via API failed: %s", exc)

        # Fallback: mss
        try:
            import mss  # noqa: PLC0415
            from PIL import Image  # noqa: PLC0415
            import io
            with mss.mss() as sct:
                m = sct.monitors[0]
                raw = sct.grab(m)
            img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
            return img
        except Exception as exc:
            _logger.debug("[ComputerAPIBackend] mss screenshot failed: %s", exc)
            return None

    # =========================================================================
    # Mouse control
    # =========================================================================

    def click(self, x: float, y: float, button: str = "left") -> None:
        
        if self._use_api:
            try:
                _computer.mouse.click(x, y)
                return
            except Exception as exc:
                _logger.debug("[ComputerAPIBackend] click via API failed: %s", exc)

        # Fallback: delegate to OperatingSystem
        fb = self._get_os_fallback()
        if fb is not None:
            fb.click(x, y, button=button)

    def move(self, x: float, y: float) -> None:
        """Move cursor to normalized [0,1] coordinates."""
        if self._use_api:
            try:
                _computer.mouse.move(x, y)
                return
            except Exception as exc:
                _logger.debug("[ComputerAPIBackend] move via API failed: %s", exc)

        fb = self._get_os_fallback()
        if fb is not None:
            try:
                fb.move(x, y)
            except Exception:
                pass

    def scroll(self, x: float, y: float, direction: str = "down", amount: int = 3) -> None:
        """Scroll at normalized coordinates."""
        if self._use_api:
            try:
                clicks = -amount if direction == "up" else amount
                _computer.mouse.scroll(x, y, clicks)
                return
            except Exception as exc:
                _logger.debug("[ComputerAPIBackend] scroll via API failed: %s", exc)

        fb = self._get_os_fallback()
        if fb is not None:
            try:
                fb.scroll(x, y, direction=direction, amount=amount)
            except Exception:
                pass

    # =========================================================================
    # Keyboard control
    # =========================================================================

    def write(self, text: str) -> None:
        """Type text using keyboard."""
        if self._use_api:
            try:
                _computer.keyboard.type(text)
                return
            except Exception as exc:
                _logger.debug("[ComputerAPIBackend] write via API failed: %s", exc)

        fb = self._get_os_fallback()
        if fb is not None:
            fb.write(text)

    def press(self, keys: List[str]) -> None:
        """Press a key or key combination."""
        if self._use_api:
            try:
                _computer.keyboard.hotkey(*keys)
                return
            except Exception as exc:
                _logger.debug("[ComputerAPIBackend] press via API failed: %s", exc)

        fb = self._get_os_fallback()
        if fb is not None:
            fb.press(keys)

    def hotkey(self, *keys: str) -> None:
        """Press a hotkey combination."""
        self.press(list(keys))

    # =========================================================================
    # Shell execution
    # =========================================================================

    def exec(self, command: str, timeout: int = 30) -> ExecResult:
        """
        Execute a shell command.

        Computer API wraps subprocess with timeout; fallback uses OperatingSystem.exec().
        """
        if self._use_api:
            try:
                output = _computer.os.run(command)
                return ExecResult(stdout=str(output), returncode=0)
            except Exception as exc:
                _logger.debug("[ComputerAPIBackend] exec via API failed: %s", exc)

        # Fallback: direct subprocess
        fb = self._get_os_fallback()
        if fb is not None:
            try:
                return fb.exec(command, timeout=timeout)
            except Exception:
                pass

        # Last resort: plain subprocess
        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return ExecResult(
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                returncode=proc.returncode,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(stdout="", stderr="Command timed out", returncode=1)
        except Exception as exc:
            return ExecResult(stdout="", stderr=str(exc), returncode=1)

    # =========================================================================
    # Browser operations
    # =========================================================================

    def browser_search(self, query: str) -> Optional[str]:
        """
        Open the browser and perform a search.
        Returns the page content/URL if accessible.
        """
        if self._use_api:
            try:
                result = _computer.browser.search(query)
                return str(result) if result else None
            except Exception as exc:
                _logger.debug("[ComputerAPIBackend] browser_search via API failed: %s", exc)

        # Fallback: open URL via xdg-open
        try:
            import urllib.parse  # noqa: PLC0415
            url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
            subprocess.Popen(["xdg-open", url])
            return url
        except Exception:
            return None

    # =========================================================================
    # Window management — delegate to OperatingSystem
    # =========================================================================

    def get_cursor_position(self) -> Optional[Dict[str, float]]:
        fb = self._get_os_fallback()
        return fb.get_cursor_position() if fb else None

    def set_cursor_position(self, coords: Dict) -> None:
        fb = self._get_os_fallback()
        if fb:
            fb.set_cursor_position(coords)

    def get_focused_window(self) -> Optional[Dict]:
        fb = self._get_os_fallback()
        return fb.get_focused_window() if fb else None

    def get_active_application(self) -> Optional[Dict]:
        fb = self._get_os_fallback()
        return fb.get_active_application() if fb else None

    def activate_application(self, app: Dict) -> None:
        fb = self._get_os_fallback()
        if fb:
            fb.activate_application(app)

    def focus_window(self, window: Dict) -> None:
        fb = self._get_os_fallback()
        if fb:
            fb.focus_window(window)

    def stop_automated_input(self) -> None:
        fb = self._get_os_fallback()
        if fb:
            try:
                fb.stop_automated_input()
            except Exception:
                pass

    def force_release_all(self, reason: str = "") -> None:
        fb = self._get_os_fallback()
        if fb:
            try:
                fb.force_release_all(reason=reason)
            except Exception:
                pass

    def mark_automation_inactive(self) -> None:
        fb = self._get_os_fallback()
        if fb:
            try:
                fb.mark_automation_inactive()
            except Exception:
                pass

    def is_computer_api_active(self) -> bool:
        return self._use_api

    def get_stats(self) -> Dict[str, Any]:
        return {
            "computer_api_active": self._use_api,
            "computer_api_available": _COMPUTER_API_AVAILABLE,
            "fallback_initialised": self._os_fallback is not None,
        }


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_os_backend() -> ComputerAPIBackend:
    """
    Create the best available OS control backend.

    Returns ComputerAPIBackend which transparently uses the Computer API
    when available and falls back to OperatingSystem otherwise.
    """
    return ComputerAPIBackend()
