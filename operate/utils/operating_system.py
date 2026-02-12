import pyautogui
import platform
import time
import math
import threading
import subprocess
import os
from typing import Optional, Dict

from operate.utils.misc import convert_percent_to_decimal

pyautogui.FAILSAFE = True


class OperatingSystem:
    """
    Deterministic OS boundary.

    CONTRACT:
    - Cursor schema: {"x": int, "y": int}
    - Window schema: {"title": str}
    - Explicit failures only
    - No silent success
    - Post-condition verification required
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

    # =================================================
    # HEARTBEAT
    # =================================================

    def heartbeat(self) -> None:
        with self._heartbeat_lock:
            self._last_heartbeat = time.monotonic()
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
                if self._automation_active and self._last_heartbeat:
                    if (
                        time.monotonic() - self._last_heartbeat
                        > self._HEARTBEAT_TIMEOUT
                    ):
                        timed_out = True
                        self._automation_active = False

            if timed_out:
                self.force_release_all(reason="heartbeat_timeout")

    # =================================================
    # COMMANDS / FILES
    # =================================================

    def exec(self, cmd: str, *, sudo: bool = False) -> subprocess.CompletedProcess:
        if not isinstance(cmd, str) or not cmd.strip():
            raise RuntimeError("exec(): invalid command")

        full_cmd = cmd
        if sudo and hasattr(os, "geteuid") and os.geteuid() != 0:
            full_cmd = f"sudo {cmd}"

        result = subprocess.run(
            full_cmd,
            shell=True,
            capture_output=True,
            text=True,
        )

        return result

    def write_file(self, path: str, content: str) -> None:
        if not isinstance(path, str) or not path:
            raise RuntimeError("write_file(): invalid path")
        if not isinstance(content, str):
            raise RuntimeError("write_file(): content must be string")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    # =================================================
    # INPUT ACTIONS
    # =================================================

    def write(self, content: str) -> None:
        if not isinstance(content, str):
            raise RuntimeError("write(): content must be string")

        content = content.replace("\\n", "\n")

        with self._automation_lock:
            self._automation_active = True

        pyautogui.write(content, interval=0.01)

    def press(self, keys) -> None:
        if not isinstance(keys, list) or not keys:
            raise RuntimeError("press(): keys must be non-empty list")

        with self._automation_lock:
            self._automation_active = True

        pyautogui.hotkey(*keys)

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

        with self._automation_lock:
            self._automation_active = True

        pyautogui.moveTo(x_px, y_px, duration=0.05)

        cur_x, cur_y = pyautogui.position()
        if abs(cur_x - x_px) > 3 or abs(cur_y - y_px) > 3:
            raise RuntimeError("Cursor failed to reach target")

        pyautogui.click()

    # =================================================
    # CURSOR STATE
    # =================================================

    def get_cursor_position(self) -> Dict[str, int]:
        x, y = pyautogui.position()
        return {"x": int(x), "y": int(y)}

    def set_cursor_position(self, position: Dict[str, int]) -> None:
        if not isinstance(position, dict):
            raise RuntimeError("set_cursor_position(): invalid position")

        x = position.get("x")
        y = position.get("y")

        if not isinstance(x, int) or not isinstance(y, int):
            raise RuntimeError("set_cursor_position(): invalid coordinates")

        pyautogui.moveTo(x, y, duration=0.05)

    # =================================================
    # WINDOW / APPLICATION
    # =================================================

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
                if result.returncode == 0 and result.stdout.strip():
                    return {"title": result.stdout.strip()}

            elif system == "Linux":
                try:
                    title = subprocess.check_output(
                        ["xdotool", "getactivewindow", "getwindowname"],
                        text=True,
                    ).strip()
                except FileNotFoundError:
                    raise OSError("xdotool not installed")
                return {"title": title}

            elif system == "Windows":
                import win32gui
                hwnd = win32gui.GetForegroundWindow()
                title = win32gui.GetWindowText(hwnd)
                return {"title": title}

        except Exception as e:
            raise OSError(f"Failed to get focused window: {e}") from e

        raise OSError("Focused window unavailable")

    def get_active_application(self) -> Dict[str, str]:
        return self.get_focused_window()

    # =================================================
    # CRITICAL: APPLICATION ACTIVATION
    # =================================================

    def activate_application(self, app_spec: Dict[str, str]) -> None:
        if not isinstance(app_spec, dict):
            raise RuntimeError("activate_application(): invalid app_spec")

        title = app_spec.get("title")
        if not isinstance(title, str) or not title.strip():
            raise RuntimeError("activate_application(): missing title")

        system = platform.system()

        try:
            if system == "Darwin":
                script = f'''
                tell application "{title}"
                    activate
                end tell
                '''
                result = subprocess.run(
                    ["osascript", "-e", script],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode != 0:
                    raise OSError(result.stderr.strip() or "activation failed")

            elif system == "Linux":
                try:
                    result = subprocess.run(
                        ["wmctrl", "-a", title],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                except FileNotFoundError:
                    raise OSError("wmctrl not installed")

                if result.returncode != 0:
                    raise OSError(result.stderr.strip() or "activation failed")

            elif system == "Windows":
                import win32gui
                import win32con

                target_hwnd = None

                def enum_handler(hwnd, _):
                    nonlocal target_hwnd
                    if win32gui.IsWindowVisible(hwnd):
                        if title.lower() in win32gui.GetWindowText(hwnd).lower():
                            target_hwnd = hwnd
                            return False
                    return True

                win32gui.EnumWindows(enum_handler, None)

                if target_hwnd is None:
                    raise OSError(f"Application '{title}' not found")

                if win32gui.IsIconic(target_hwnd):
                    win32gui.ShowWindow(target_hwnd, win32con.SW_RESTORE)

                win32gui.SetForegroundWindow(target_hwnd)
                win32gui.BringWindowToTop(target_hwnd)

            else:
                raise OSError(f"Unsupported platform: {system}")

        except subprocess.TimeoutExpired:
            raise OSError(f"activate_application(): timeout for '{title}'")

        time.sleep(0.15)

        focused = self.get_focused_window()
        if title.lower() not in focused.get("title", "").lower():
            raise OSError("activate_application(): verification failed")

    # =================================================
    # RESTORATION / SAFETY
    # =================================================

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

    # =================================================
    # HELPERS
    # =================================================

    @staticmethod
    def _valid_coord(v) -> bool:
        return (
            isinstance(v, float)
            and not math.isnan(v)
            and 0.0 <= v <= 1.0
        )
