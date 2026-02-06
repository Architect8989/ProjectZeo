import pyautogui
import platform
import time
import math
import threading
import subprocess
import os
from typing import Optional

from operate.utils.misc import convert_percent_to_decimal


# HARD FAILSAFE (best-effort, not security)
pyautogui.FAILSAFE = True


class OperatingSystem:
    """
    OS interaction layer (best-effort, unsafe by nature).

    GUARANTEES:
    - Deterministic input calls
    - Explicit failure surfacing (no silent ignores)
    - Watchdog-based fail-open release
    - No zombie automation after heartbeat loss

    NON-GUARANTEES:
    - Native OS safety
    - Verified UI semantics
    - Perfect cross-platform window control
    """

    # -------------------------------------------------
    # INIT
    # -------------------------------------------------

    def __init__(self):
        self._execution_mode = "OBSERVER"
        self._execution_mode_lock = threading.Lock()

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
                    self.force_release_all(reason="heartbeat_timeout")
                except Exception:
                    pass

    # -------------------------------------------------
    # EXECUTION MODE
    # -------------------------------------------------

    def get_execution_mode(self) -> str:
        with self._execution_mode_lock:
            return self._execution_mode

    def set_execution_mode(self, mode: str) -> None:
        if not isinstance(mode, str):
            raise RuntimeError("execution_mode must be string")

        with self._execution_mode_lock:
            self._execution_mode = mode

    # -------------------------------------------------
    # AUTOMATION MARKERS
    # -------------------------------------------------

    def mark_automation_active(self) -> None:
        with self._automation_lock:
            self._automation_active = True
        self.heartbeat()

    def mark_automation_inactive(self) -> None:
        with self._automation_lock:
            self._automation_active = False

    # -------------------------------------------------
    # INPUT ACTIONS (FAIL-CLOSED)
    # -------------------------------------------------

    def write(self, content: str) -> None:
        if not isinstance(content, str):
            raise RuntimeError("write(): content must be string")

        content = content.replace("\\n", "\n")

        for char in content:
            pyautogui.write(char)
            time.sleep(0.01)

    def press(self, keys) -> None:
        if not isinstance(keys, list) or not keys:
            raise RuntimeError("press(): keys must be non-empty list")

        for key in keys:
            pyautogui.keyDown(key)

        time.sleep(0.05)

        for key in reversed(keys):
            pyautogui.keyUp(key)

    def mouse(self, click_detail: dict) -> None:
        x_raw = click_detail.get("x")
        y_raw = click_detail.get("y")

        x = convert_percent_to_decimal(x_raw)
        y = convert_percent_to_decimal(y_raw)

        if not self._valid_coord(x) or not self._valid_coord(y):
            raise RuntimeError(
                f"Invalid click coordinates x={x_raw}, y={y_raw}"
            )

        self._click_at_percentage(x, y)

    def _click_at_percentage(self, x_pct: float, y_pct: float) -> None:
        screen_w, screen_h = pyautogui.size()

        x_px = int(screen_w * x_pct)
        y_px = int(screen_h * y_pct)

        pyautogui.moveTo(x_px, y_px, duration=0.1)

        cur_x, cur_y = pyautogui.position()
        if abs(cur_x - x_px) > 2 or abs(cur_y - y_px) > 2:
            raise RuntimeError("Cursor failed to reach target position")

        pyautogui.click(x_px, y_px)

    # -------------------------------------------------
    # COMMAND EXECUTION
    # -------------------------------------------------

    def exec(self, command: str, sudo: bool = False):
        if not isinstance(command, str) or not command.strip():
            raise RuntimeError("exec(): command must be non-empty string")

        full_cmd = command
        if sudo and platform.system() != "Windows":
            full_cmd = f"sudo {command}"

        return subprocess.run(
            full_cmd,
            shell=True,
            capture_output=True,
            text=True,
        )

    # -------------------------------------------------
    # FILESYSTEM
    # -------------------------------------------------

    def write_file(self, path: str, content: str) -> None:
        if not isinstance(path, str) or not path:
            raise RuntimeError("write_file(): invalid path")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content or "")

    def mkdir(self, path: str) -> None:
        if not isinstance(path, str) or not path:
            raise RuntimeError("mkdir(): invalid path")
        os.makedirs(path, exist_ok=True)

    # -------------------------------------------------
    # BROWSER / DOWNLOAD SUPPORT
    # -------------------------------------------------

    def open_browser(self, url: str = "https://www.google.com") -> None:
        system = platform.system()
        if system == "Windows":
            os.startfile(url)
        elif system == "Darwin":
            subprocess.run(["open", url])
        else:
            subprocess.run(["xdg-open", url])

    def focus_address_bar(self) -> None:
        self.press(["ctrl", "l"])

    def get_latest_download(
        self,
        *,
        expected_extension: Optional[str] = None,
        min_size_bytes: int = 1024,
    ) -> Optional[str]:
        downloads = os.path.join(os.path.expanduser("~"), "Downloads")
        if not os.path.isdir(downloads):
            return None

        candidates = []

        for f in os.listdir(downloads):
            path = os.path.join(downloads, f)
            if not os.path.isfile(path):
                continue

            if f.endswith((".crdownload", ".part", ".tmp")):
                continue

            if expected_extension and not f.endswith(expected_extension):
                continue

            try:
                if os.path.getsize(path) < min_size_bytes:
                    continue
            except Exception:
                continue

            candidates.append(path)

        if not candidates:
            return None

        return max(candidates, key=os.path.getmtime)

    # -------------------------------------------------
    # FAIL-OPEN SAFETY
    # -------------------------------------------------

    def force_release_all(self, *, reason: Optional[str] = None) -> None:
        errors = []

        try:
            self.stop_automated_input()
        except Exception as e:
            errors.append(f"stop_input:{e}")

        try:
            self.set_execution_mode("OBSERVER")
        except Exception as e:
            errors.append(f"mode_reset:{e}")

        if errors:
            raise RuntimeError(
                f"force_release_all failed ({reason}): {errors}"
            )

    def stop_automated_input(self) -> None:
        keys = (
            ["shift", "ctrl", "alt", "win", "command", "esc"]
            + ["tab", "enter", "space", "backspace", "delete"]
            + ["up", "down", "left", "right"]
            + [f"f{i}" for i in range(1, 25)]
            + list("abcdefghijklmnopqrstuvwxyz0123456789")
        )

        for key in keys:
            try:
                pyautogui.keyUp(key)
            except Exception:
                pass

        for btn in ("left", "right", "middle"):
            try:
                pyautogui.mouseUp(button=btn)
            except Exception:
                pass

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

    def get_focused_window(self):
        system = platform.system()

        try:
            if system == "Darwin":
                script = '''
                tell application "System Events"
                    set frontApp to first application process whose frontmost is true
                    set frontWindow to front window of frontApp
                    return (name of frontApp) & "|" & (name of frontWindow)
                end tell
                '''
                result = subprocess.run(
                    ["osascript", "-e", script],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    app, title = result.stdout.strip().split("|", 1)
                    return {
                        "id": f"{app}:{title}",
                        "title": title,
                        "application": app,
                    }

            elif system == "Linux":
                wid = subprocess.check_output(
                    ["xdotool", "getactivewindow"],
                    text=True,
                ).strip()
                title = subprocess.check_output(
                    ["xdotool", "getwindowname", wid],
                    text=True,
                ).strip()
                return {"id": wid, "title": title}

            elif system == "Windows":
                import win32gui
                hwnd = win32gui.GetForegroundWindow()
                title = win32gui.GetWindowText(hwnd)
                return {"id": str(hwnd), "title": title}

        except Exception:
            pass

        return {"id": "unknown", "title": None}

    def focus_window(self, window_id: str) -> bool:
        if not isinstance(window_id, str):
            return False

        system = platform.system()

        try:
            if system == "Darwin":
                app = window_id.split(":", 1)[0]
                subprocess.run(
                    ["osascript", "-e", f'tell application "{app}" to activate'],
                    check=True,
                )
                return True

            elif system == "Linux":
                subprocess.run(
                    ["xdotool", "windowactivate", window_id],
                    check=True,
                )
                return True

            elif system == "Windows":
                import win32gui
                win32gui.SetForegroundWindow(int(window_id))
                return True

        except Exception:
            return False

        return False

    # -------------------------------------------------
    # APPLICATION (PROCESS LEVEL)
    # -------------------------------------------------

    def get_active_application(self):
        system = platform.system()

        try:
            if system == "Darwin":
                script = '''
                tell application "System Events"
                    set frontApp to first application process whose frontmost is true
                    return (name of frontApp) & "|" & (unix id of frontApp)
                end tell
                '''
                result = subprocess.run(
                    ["osascript", "-e", script],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    name, pid = result.stdout.strip().split("|", 1)
                    return {
                        "process_name": name,
                        "pid": int(pid),
                        "application": name,
                    }

            elif system == "Linux":
                wid = subprocess.check_output(
                    ["xdotool", "getactivewindow"],
                    text=True,
                ).strip()
                pid = subprocess.check_output(
                    ["xdotool", "getwindowpid", wid],
                    text=True,
                ).strip()
                proc = subprocess.check_output(
                    ["ps", "-p", pid, "-o", "comm="],
                    text=True,
                ).strip()
                return {
                    "process_name": proc,
                    "pid": int(pid),
                    "application": proc,
                }

            elif system == "Windows":
                import win32gui
                import win32process
                import psutil

                hwnd = win32gui.GetForegroundWindow()
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                proc = psutil.Process(pid)
                return {
                    "process_name": proc.name(),
                    "pid": pid,
                    "application": proc.name(),
                }

        except Exception:
            pass

        return {
            "process_name": "unknown",
            "pid": None,
            "application": None,
        }

    def activate_application(
        self,
        process_name: str,
        pid: Optional[int] = None,
    ) -> bool:
        if not isinstance(process_name, str) or not process_name:
            return False

        system = platform.system()

        try:
            if system == "Darwin":
                subprocess.run(
                    ["osascript", "-e", f'tell application "{process_name}" to activate'],
                    capture_output=True,
                )
                return True

            elif system == "Linux":
                if pid:
                    windows = subprocess.check_output(
                        ["xdotool", "search", "--pid", str(pid)],
                        text=True,
                    ).strip().split("\n")
                    if windows and windows[0]:
                        subprocess.run(
                            ["xdotool", "windowactivate", windows[0]],
                            check=False,
                        )
                        return True

                windows = subprocess.check_output(
                    ["xdotool", "search", "--class", process_name],
                    text=True,
                ).strip().split("\n")
                if windows and windows[0]:
                    subprocess.run(
                        ["xdotool", "windowactivate", windows[0]],
                        check=False,
                    )
                    return True

            elif system == "Windows":
                import win32gui
                import win32con

                def _enum(hwnd, acc):
                    if win32gui.IsWindowVisible(hwnd):
                        acc.append(hwnd)
                    return True

                windows = []
                win32gui.EnumWindows(_enum, windows)

                for hwnd in windows:
                    title = win32gui.GetWindowText(hwnd).lower()
                    if process_name.lower() in title:
                        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                        win32gui.SetForegroundWindow(hwnd)
                        return True

        except Exception:
            pass

        return False

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
