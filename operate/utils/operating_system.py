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
    - Cross-platform window control
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

            with self._heartbeat_lock, self._automation_lock:
                if not self._automation_active or self._last_heartbeat is None:
                    continue

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

    def exec(self, command: str, sudo: bool = False) -> None:
        if not isinstance(command, str) or not command.strip():
            raise RuntimeError("exec(): command must be non-empty string")

        full_cmd = command
        if sudo and platform.system() != "Windows":
            full_cmd = f"sudo {command}"

        result = subprocess.run(
            full_cmd,
            shell=True,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Command failed: {full_cmd}\n{result.stderr}"
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
    # BROWSER / DOWNLOAD SUPPORT (INSTALLER)
    # -------------------------------------------------

    def open_browser(self) -> None:
        system = platform.system()
        if system == "Windows":
            os.startfile("https://www.google.com")
        elif system == "Darwin":
            subprocess.run(["open", "https://www.google.com"])
        else:
            subprocess.run(["xdg-open", "https://www.google.com"])

    def focus_address_bar(self) -> None:
        self.press(["ctrl", "l"])

    def get_latest_download(self) -> Optional[str]:
        downloads = os.path.join(os.path.expanduser("~"), "Downloads")
        if not os.path.isdir(downloads):
            return None

        files = [
            os.path.join(downloads, f)
            for f in os.listdir(downloads)
        ]
        files = [f for f in files if os.path.isfile(f)]
        if not files:
            return None

        return max(files, key=os.path.getmtime)

    # -------------------------------------------------
    # FAIL-OPEN SAFETY (VISIBLE)
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
    # WINDOW / APPLICATION (INTENTIONAL STUBS)
    # -------------------------------------------------

    def get_focused_window(self):
        return {"id": "unknown", "title": None}

    def focus_window(self, window_id: str) -> bool:
        return False

    def get_active_application(self):
        return {"process_name": platform.system(), "pid": None}

    def activate_application(self, process_name: str, pid=None) -> bool:
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
