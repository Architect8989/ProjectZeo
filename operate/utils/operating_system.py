import pyautogui
import platform
import time
import math
import threading
import atexit
import os
import signal

from operate.utils.misc import convert_percent_to_decimal

# ---------------------------
# GLOBAL HARD SAFETY
# ---------------------------

pyautogui.FAILSAFE = True   # moving mouse to corner aborts


class OperatingSystem:
    """
    OS interaction layer.

    Existing SOC execution logic is preserved.
    New logic enforces:
    - Fail-open human reclaim
    - Crash-safe input release
    - Out-of-band kill semantics
    """

    # -------------------------------------------------
    # INTERNAL AUTHORITY STATE (EXISTING)
    # -------------------------------------------------

    _execution_mode_lock = threading.Lock()
    _execution_mode = "OBSERVER"  # default-safe

    # -------------------------------------------------
    # HARD SAFETY STATE
    # -------------------------------------------------

    _automation_active = False
    _automation_lock = threading.Lock()

    _last_heartbeat = time.time()
    _heartbeat_lock = threading.Lock()

    _WATCHDOG_INTERVAL = 0.5
    _HEARTBEAT_TIMEOUT = 2.0  # seconds

    _watchdog_thread_started = False

    # -------------------------------------------------
    # SOC ACTIONS
    # -------------------------------------------------

    def write(self, content):
        try:
            content = content.replace("\\n", "\n")
            for char in content:
                pyautogui.write(char)
        except Exception as e:
            print("[OperatingSystem][write] error:", e)
            raise

    def press(self, keys):
        try:
            for key in keys:
                pyautogui.keyDown(key)
            time.sleep(0.1)
            for key in keys:
                pyautogui.keyUp(key)
        except Exception as e:
            print("[OperatingSystem][press] error:", e)
            raise

    def mouse(self, click_detail):
        try:
            x = convert_percent_to_decimal(click_detail.get("x"))
            y = convert_percent_to_decimal(click_detail.get("y"))

            if click_detail and isinstance(x, float) and isinstance(y, float):
                self.click_at_percentage(x, y)
            else:
                raise ValueError("Invalid click coordinates")
        except Exception as e:
            print("[OperatingSystem][mouse] error:", e)
            raise

    def click_at_percentage(
        self,
        x_percentage,
        y_percentage,
        duration=0.2,
        circle_radius=50,
        circle_duration=0.5,
    ):
        try:
            screen_width, screen_height = pyautogui.size()
            x_pixel = int(screen_width * float(x_percentage))
            y_pixel = int(screen_height * float(y_percentage))

            pyautogui.moveTo(x_pixel, y_pixel, duration=duration)

            start_time = time.time()
            while time.time() - start_time < circle_duration:
                angle = ((time.time() - start_time) / circle_duration) * 2 * math.pi
                x = x_pixel + math.cos(angle) * circle_radius
                y = y_pixel + math.sin(angle) * circle_radius
                pyautogui.moveTo(x, y, duration=0.1)

            pyautogui.click(x_pixel, y_pixel)
        except Exception as e:
            print("[OperatingSystem][click_at_percentage] error:", e)
            raise

    # -------------------------------------------------
    # AUTHORITY SUPPORT
    # -------------------------------------------------

    def get_execution_mode(self) -> str:
        with self._execution_mode_lock:
            return self._execution_mode

    def set_execution_mode(self, mode: str) -> None:
        with self._execution_mode_lock:
            self._execution_mode = mode

    # -------------------------------------------------
    # AUTOMATION LIFECYCLE
    # -------------------------------------------------

    def mark_automation_active(self):
        with self._automation_lock:
            self._automation_active = True
        self._touch_heartbeat()
        self._ensure_watchdog()

    def mark_automation_inactive(self):
        with self._automation_lock:
            self._automation_active = False

    def is_automation_active(self):
        with self._automation_lock:
            return self._automation_active

    def _touch_heartbeat(self):
        with self._heartbeat_lock:
            self._last_heartbeat = time.time()

    def heartbeat(self):
        self._touch_heartbeat()

    # -------------------------------------------------
    # WATCHDOG
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
                print("[OperatingSystem][WATCHDOG] Heartbeat lost — forcing input release")
                self.force_release_all()
                return

    # -------------------------------------------------
    # FORCED HUMAN RECLAIM
    # -------------------------------------------------

    def force_release_all(self):
        try:
            self.stop_automated_input()
            self.enable_user_input()
            self.set_execution_mode("OBSERVER")
        except Exception as e:
            print("[OperatingSystem][force_release_all] error:", e)

    # -------------------------------------------------
    # REAL INPUT RELEASE (FIXED)
    # -------------------------------------------------

    def stop_automated_input(self) -> None:
        try:
            pyautogui.keyUp("shift")
            pyautogui.keyUp("ctrl")
            pyautogui.keyUp("alt")
            pyautogui.mouseUp()
        except Exception:
            pass

    def enable_user_input(self) -> None:
        self.stop_automated_input()

    # -------------------------------------------------
    # CURSOR STATE
    # -------------------------------------------------

    def get_cursor_position(self):
        try:
            return pyautogui.position()
        except Exception as e:
            raise RuntimeError(f"Unable to get cursor position: {e}")

    def set_cursor_position(self, x: int, y: int) -> None:
        try:
            pyautogui.moveTo(int(x), int(y), duration=0)
        except Exception as e:
            raise RuntimeError(f"Unable to set cursor position: {e}")

    # -------------------------------------------------
    # WINDOW / APPLICATION FOCUS (STUBS)
    # -------------------------------------------------

    def get_focused_window(self):
        return {
            "id": "unknown",
            "title": None,
        }

    def get_focused_window_id(self):
        info = self.get_focused_window()
        return info.get("id")

    def focus_window(self, window_id: str) -> bool:
        return False

    def get_active_application(self):
        return {
            "process_name": platform.system(),
            "pid": None,
        }

    def activate_application(self, process_name: str, pid=None) -> bool:
        return False


# -------------------------------------------------
# PROCESS-LEVEL FAIL-OPEN GUARANTEES
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
