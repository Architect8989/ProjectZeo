import pyautogui
import platform
import time
import math
import threading
import atexit
import os
import signal

from operate.utils.misc import convert_percent_to_decimal


# HARD FAILSAFE
pyautogui.FAILSAFE = True


class OperatingSystem:
    """
    OS interaction layer.

    Enforces:
    - Fail-open human reclaim
    - Crash-safe input release
    - Heartbeat watchdog
    """

    # -------------------------------------------------
    # INTERNAL AUTHORITY STATE
    # -------------------------------------------------

    _execution_mode_lock = threading.Lock()
    _execution_mode = "OBSERVER"  # default-safe

    # -------------------------------------------------
    # AUTOMATION STATE
    # -------------------------------------------------

    _automation_active = False
    _automation_lock = threading.Lock()

    _last_heartbeat = time.time()
    _heartbeat_lock = threading.Lock()

    _WATCHDOG_INTERVAL = 0.5
    _HEARTBEAT_TIMEOUT = 2.0

    _watchdog_thread_started = False

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

    def _touch_heartbeat(self):
        with self._heartbeat_lock:
            self._last_heartbeat = time.time()

    def heartbeat(self):
        self._touch_heartbeat()

    # -------------------------------------------------
    # WATCHDOG THREAD
    # -------------------------------------------------

    def _ensure_watchdog(self):
        if self._watchdog_thread_started:
            return

        self._watchdog_thread_started = True
        t = threading.Thread(target=self._watchdog_loop, daemon=True)
        t.start()

    def _watchdog_loop(self):
        while True:
            time.sleep(self._WATCHDOG_INTERVAL)

            with self._heartbeat_lock:
                elapsed = time.time() - self._last_heartbeat

            with self._automation_lock:
                active = self._automation_active

            if active and elapsed > self._HEARTBEAT_TIMEOUT:
                print("[OperatingSystem] Heartbeat lost — forcing release")
                self.force_release_all()
                return

    # -------------------------------------------------
    # HARD FAIL-OPEN SAFETY
    # -------------------------------------------------

    def force_release_all(self):
        """
        Absolute safety valve.
        Physically releases all keys and mouse buttons.
        Never raises.
        """
        try:
            self.stop_automated_input()
            self.enable_user_input()
            self.set_execution_mode("OBSERVER")
        except Exception:
            pass

    def stop_automated_input(self) -> None:
        """
        Physically release everything.
        """
        try:
            # Modifiers
            for key in ["shift", "ctrl", "alt", "win", "command", "esc"]:
                try:
                    pyautogui.keyUp(key)
                except Exception:
                    pass

            # Letters
            for c in "abcdefghijklmnopqrstuvwxyz":
                try:
                    pyautogui.keyUp(c)
                except Exception:
                    pass

            # Mouse
            for btn in ["left", "right", "middle"]:
                try:
                    pyautogui.mouseUp(button=btn)
                except Exception:
                    pass

        except Exception:
            pass

    def enable_user_input(self) -> None:
        try:
            self.stop_automated_input()
        except Exception:
            pass

    # -------------------------------------------------
    # CURSOR
    # -------------------------------------------------

    def get_cursor_position(self):
        try:
            return pyautogui.position()
        except Exception as e:
            raise RuntimeError(e)

    def set_cursor_position(self, x: int, y: int) -> None:
        try:
            pyautogui.moveTo(int(x), int(y), duration=0)
        except Exception as e:
            raise RuntimeError(e)

    # -------------------------------------------------
    # WINDOW / APPLICATION (STUBS)
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


# -------------------------------------------------
# PROCESS-LEVEL FAIL-OPEN
# -------------------------------------------------

_OS_SINGLETON = OperatingSystem()


def _emergency_exit_handler(*args):
    try:
        _OS_SINGLETON.force_release_all()
    finally:
        os._exit(1)


atexit.register(_OS_SINGLETON.force_release_all)

for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, _emergency_exit_handler)
