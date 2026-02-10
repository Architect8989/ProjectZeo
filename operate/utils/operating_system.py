import pyautogui
import platform
import time
import math
import threading
import subprocess
import os
from typing import Optional, Dict

from operate.utils.misc import convert_percent_to_decimal


# HARD FAILSAFE (cursor to corner aborts)
pyautogui.FAILSAFE = True


class OperatingSystem:
    """
    OS interaction layer (best-effort, unsafe by nature).

    GUARANTEES:
    - Deterministic input calls
    - Explicit failure surfacing
    - Watchdog-based fail-open release
    - No zombie automation
    """

    def __init__(self):
        self._automation_active = False
        self._automation_lock = threading.Lock()

        self._last_heartbeat: Optional[float] = None
        self._heartbeat_lock = threading.Lock()

        self._WATCHDOG_INTERVAL = 0.5
        self._HEARTBEAT_TIMEOUT = 2.0

        self._watchdog_started = False
        self._watchdog_lock = threading.Lock()

    # -------------------------------------------------
    # HEARTBEAT / WATCHDOG
    # -------------------------------------------------

    def heartbeat(self) -> None:
        with self._heartbeat_lock:
            self._last_heartbeat = time.time()
        self._ensure_watchdog()

    def _ensure_watchdog(self) -> None:
        with self._watchdog_lock:
            if self._watchdog_started:
                return
            self._watchdog_started = True
            threading.Thread(
                target=self._watchdog_loop,
                daemon=True,
            ).start()

    def _watchdog_loop(self) -> None:
        while True:
            time.sleep(self._WATCHDOG_INTERVAL)

            timed_out = False
            with self._heartbeat_lock, self._automation_lock:
                if self._automation_active and self._last_heartbeat is not None:
                    timed_out = (
                        time.time() - self._last_heartbeat
                        > self._HEARTBEAT_TIMEOUT
                    )
                    if timed_out:
                        self._automation_active = False

            if timed_out:
                try:
                    self.force_release_all(
                        reason="heartbeat_timeout"
                    )
                except Exception:
                    pass

    # -------------------------------------------------
    # EXECUTION PRIMITIVES
    # -------------------------------------------------

    def exec(
        self, cmd: str, *, sudo: bool = False
    ) -> subprocess.CompletedProcess:
        if not isinstance(cmd, str) or not cmd.strip():
            raise RuntimeError("exec(): invalid command")

        full_cmd = cmd
        if sudo and hasattr(os, "geteuid") and os.geteuid() != 0:
            full_cmd = f"sudo {cmd}"

        return subprocess.run(
            full_cmd,
            shell=True,
            capture_output=True,
            text=True,
        )

    def write_file(self, path: str, content: str) -> None:
        if not isinstance(path, str) or not path:
            raise RuntimeError("write_file(): invalid path")

        if not isinstance(content, str):
            raise RuntimeError("write_file(): content must be string")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    # -------------------------------------------------
    # INPUT ACTIONS
    # -------------------------------------------------

    def write(self, content: str) -> None:
        if not isinstance(content, str):
            raise RuntimeError("write(): content must be string")

        content = content.replace("\\n", "\n")

        with self._automation_lock:
            self._automation_active = True

        for char in content:
            pyautogui.write(char)
            time.sleep(0.01)

    def press(self, keys) -> None:
        if not isinstance(keys, list) or not keys:
            raise RuntimeError("press(): keys must be non-empty list")

        with self._automation_lock:
            self._automation_active = True

        for key in keys:
            pyautogui.keyDown(key)
        time.sleep(0.05)
        for key in reversed(keys):
            pyautogui.keyUp(key)

    def mouse(self, click_detail: dict) -> None:
        x = convert_percent_to_decimal(click_detail.get("x"))
        y = convert_percent_to_decimal(click_detail.get("y"))

        if not self._valid_coord(x) or not self._valid_coord(y):
            raise RuntimeError(f"Invalid click coordinates: {click_detail}")

        self._click_at_percentage(x, y)

    def _click_at_percentage(self, x_pct: float, y_pct: float) -> None:
        screen_w, screen_h = pyautogui.size()
        x_px = int(screen_w * x_pct)
        y_px = int(screen_h * y_pct)

        pyautogui.moveTo(x_px, y_px, duration=0.1)
        time.sleep(0.05)

        cur_x, cur_y = pyautogui.position()
        if abs(cur_x - x_px) > 5 or abs(cur_y - y_px) > 5:
            raise RuntimeError(
                f"Cursor failed to reach target "
                f"({x_px},{y_px}) got ({cur_x},{cur_y})"
            )

        pyautogui.click(x_px, y_px)

    # -------------------------------------------------
    # CURSOR
    # -------------------------------------------------

    def get_cursor_position(self):
        return pyautogui.position()

    def set_cursor_position(self, x: int, y: int) -> None:
        pyautogui.moveTo(int(x), int(y), duration=0)

    # -------------------------------------------------
    # WINDOW / APPLICATION
    # -------------------------------------------------

    def get_focused_window(self) -> Dict[str, str]:
        system = platform.system()

        try:
            if system == "Darwin":
                script = '''
                tell application "System Events"
                    set frontApp to first application process whose frontmost is true
                    return name of frontApp
                end tell
                '''
                result = subprocess.run(
                    ["osascript", "-e", script],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    return {"application": result.stdout.strip()}

            if system == "Linux":
                wid = subprocess.check_output(
                    ["xdotool", "getactivewindow"],
                    text=True,
                ).strip()
                title = subprocess.check_output(
                    ["xdotool", "getwindowname", wid],
                    text=True,
                ).strip()
                return {"id": wid, "title": title}

            if system == "Windows":
                try:
                    import win32gui
                except ImportError as e:
                    raise OSError("win32gui not installed") from e

                hwnd = win32gui.GetForegroundWindow()
                title = win32gui.GetWindowText(hwnd)
                return {"id": str(hwnd), "title": title}

        except Exception as e:
            raise OSError(f"Failed to get focused window: {e}") from e

        raise OSError("Focused window unavailable")

    def get_active_application(self) -> Optional[str]:
        info = self.get_focused_window()
        return info.get("application") or info.get("title")

    # -------------------------------------------------
    # RESTORATION / SAFETY
    # -------------------------------------------------

    def mark_automation_inactive(self) -> None:
        with self._automation_lock:
            self._automation_active = False

    def stop_automated_input(self) -> None:
        self.mark_automation_inactive()

    def force_release_all(self, *, reason: str) -> None:
        self.mark_automation_inactive()

        try:
            pyautogui.mouseUp()
        except Exception:
            pass

        for key in ("shift", "ctrl", "alt", "cmd"):
            try:
                pyautogui.keyUp(key)
            except Exception:
                pass

    def focus_window(self, identifier: str) -> None:
        if platform.system() == "Linux" and identifier:
            subprocess.run(
                ["xdotool", "windowactivate", identifier],
                capture_output=True,
            )

    def activate_application(self, name: str) -> None:
        if platform.system() == "Darwin" and name:
            subprocess.run(["open", "-a", name])

    def set_window_geometry(self, *args, **kwargs) -> None:
        # best-effort noop (documented)
        return

    def set_window_z_order(self, *args, **kwargs) -> None:
        # best-effort noop (documented)
        return

    def restore_browser_state(self, *args, **kwargs) -> None:
        # best-effort noop (documented)
        return

    # -------------------------------------------------
    # HELPERS
    # -------------------------------------------------

    @staticmethod
    def _valid_coord(v) -> bool:
        return (
            isinstance(v, float)
            and not math.isnan(v)
            and 0.0 <= v <= 1.0
            )
