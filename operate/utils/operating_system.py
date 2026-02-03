import pyautogui
import platform
import time
import math
import threading
import os

from operate.utils.misc import convert_percent_to_decimal


# HARD FAILSAFE
pyautogui.FAILSAFE = True


class OperatingSystem:
    """
    OS interaction layer.

    Enforces:
    - Fail-open human reclaim
    - Crash-safe input release
    - Heartbeat watchdog (non-terminating)

    NOTE:
    - NO global singleton
    - NO signal handlers
    - NO atexit hooks
    """

    # -------------------------------------------------
    # INTERNAL AUTHORITY STATE
    # -------------------------------------------------

    def __init__(self):
        self._execution_mode_lock = threading.Lock()
        self._execution_mode = "OBSERVER"

        self._automation_active = False
        self._automation_lock = threading.Lock()

        self._last_heartbeat = None
        self._heartbeat_lock = threading.Lock()

        self._WATCHDOG_INTERVAL = 0.5
        self._HEARTBEAT_TIMEOUT = 2.0

        self._watchdog_thread_started = False
        self._watchdog_lock = threading.Lock()

    # -------------------------------------------------
    # WRITE / PRESS / CLICK
    # -------------------------------------------------

    def write(self, content):
        if not isinstance(content, str):
            raise RuntimeError("write(): content must be string")

        try:
            content = content.replace("\\n", "\n")
            for char in content:
                pyautogui.write(char)
        except Exception as e:
            raise RuntimeError(f"[OperatingSystem][write] {e}")

    def press(self, keys):
        if not isinstance(keys, list):
            raise RuntimeError("press(): keys must be list")

        try:
            for key in keys:
                pyautogui.keyDown(key)
            time.sleep(0.05)
            for key in keys:
                pyautogui.keyUp(key)
        except Exception as e:
            raise RuntimeError(f"[OperatingSystem][press] {e}")

    def mouse(self, click_detail):
        try:
            x = convert_percent_to_decimal(click_detail.get("x"))
            y = convert_percent_to_decimal(click_detail.get("y"))

            if not isinstance(x, float) or not isinstance(y, float):
                raise RuntimeError("Invalid click coordinates")

            self.click_at_percentage(x, y)

        except Exception as e:
            raise RuntimeError(f"[OperatingSystem][mouse] {e}")

    def click_at_percentage(
        self,
        x_percentage,
        y_percentage,
        duration=0.2,
        circle_radius=30,
        circle_duration=0.4,
    ):
        try:
            screen_width, screen_height = pyautogui.size()
            x_pixel = int(screen_width * float(x_percentage))
            y_pixel = int(screen_height * float(y_percentage))

            pyautogui.moveTo(x_pixel, y_pixel, duration=duration)

            start = time.time()
            while time.time() - start < circle_duration:
                angle = ((time.time() - start) / circle_duration) * 2 * math.pi
                x = x_pixel + math.cos(angle) * circle_radius
                y = y_pixel + math.sin(angle) * circle_radius
                pyautogui.moveTo(x, y, duration=0.05)

            pyautogui.click(x_pixel, y_pixel)

        except Exception as e:
            raise RuntimeError(f"[OperatingSystem][click] {e}")

    # -------------------------------------------------
    # EXECUTION MODE
    # -------------------------------------------------

    def get_execution_mode(self) -> str:
        with self._execution_mode_lock:
            return self._execution_mode

    def set_execution_mode(self, mode: str) -> None:
        with self._execution_mode_lock:
            self._execution_mode = mode

    # -------------------------------------------------
    # AUTOMATION MARKERS
    # -------------------------------------------------

    def mark_automation_active(self):
        with self._automation_lock:
            self._automation_active = True
        self._touch_heartbeat()
        self._ensure_watchdog()

    def mark_automation_inactive(self):
        with self._automation_lock:
            self._automation_active = False

    def is_automation_active(self) -> bool:
        with self._automation_lock:
            return bool(self._automation_active)

    def _touch_heartbeat(self):
        with self._heartbeat_lock:
            self._last_heartbeat = time.time()

    def heartbeat(self):
        self._touch_heartbeat()

    # -------------------------------------------------
    # WATCHDOG THREAD
    # -------------------------------------------------

    def _ensure_watchdog(self):
        with self._watchdog_lock:
            if self._watchdog_thread_started:
                return
            self._watchdog_thread_started = True
            t = threading.Thread(target=self._watchdog_loop, daemon=True)
            t.start()

    def _watchdog_loop(self):
        while True:
            time.sleep(self._WATCHDOG_INTERVAL)

            with self._heartbeat_lock, self._automation_lock:
                if not self._automation_active or self._last_heartbeat is None:
                    continue

                if time.time() - self._last_heartbeat > self._HEARTBEAT_TIMEOUT:
                    self._automation_active = False
                    timed_out = True
                else:
                    timed_out = False

            if timed_out:
                self.force_release_all()

    # -------------------------------------------------
    # HARD FAIL-OPEN SAFETY
    # -------------------------------------------------

    def force_release_all(self):
        try:
            self.stop_automated_input()
            self.enable_user_input()
            self.set_execution_mode("OBSERVER")
        except Exception:
            pass

    def stop_automated_input(self) -> None:
        try:
            keys = (
                ["shift", "ctrl", "alt", "win", "command", "esc", "capslock"]
                + ["tab", "enter", "space", "backspace", "delete",
                   "up", "down", "left", "right",
                   "home", "end", "pageup", "pagedown", "insert"]
                + [f"f{i}" for i in range(1, 25)]
                + list("abcdefghijklmnopqrstuvwxyz0123456789")
            )

            for key in keys:
                try:
                    pyautogui.keyUp(key)
                except Exception:
                    pass

            for btn in ["left", "right", "middle"]:
                try:
                    pyautogui.mouseUp(button=btn)
                except Exception:
                    pass
        except Exception:
            pass

    def enable_user_input(self) -> None:
        self.stop_automated_input()

    # -------------------------------------------------
    # CURSOR
    # -------------------------------------------------

    def get_cursor_position(self):
        return pyautogui.position()

    def set_cursor_position(self, x: int, y: int) -> None:
        pyautogui.moveTo(int(x), int(y), duration=0)

    # -------------------------------------------------
    # WINDOW / APPLICATION (STUB-SAFE)
    # -------------------------------------------------

    def get_focused_window(self):
        return {"id": "unknown", "title": None}

    def get_focused_window_id(self):
        return self.get_focused_window().get("id")

    def focus_window(self, window_id: str) -> bool:
        return False

    def get_active_application(self):
        return {"process_name": platform.system(), "pid": None}

    def activate_application(self, process_name: str, pid=None) -> bool:
        return False
