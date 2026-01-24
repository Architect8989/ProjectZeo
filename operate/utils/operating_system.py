import pyautogui
import platform
import time
import math
import threading

from operate.utils.misc import convert_percent_to_decimal


class OperatingSystem:
    """
    OS interaction layer.

    Existing SOC execution logic is preserved.
    New methods are added to support authority & restoration.
    """

    # -------------------------------------------------
    # INTERNAL AUTHORITY STATE (NEW, SAFE)
    # -------------------------------------------------

    _execution_mode_lock = threading.Lock()
    _execution_mode = "OBSERVER"  # default-safe

    # -------------------------------------------------
    # EXISTING SOC METHODS (UNCHANGED)
    # -------------------------------------------------

    def write(self, content):
        try:
            content = content.replace("\\n", "\n")
            for char in content:
                pyautogui.write(char)
        except Exception as e:
            print("[OperatingSystem][write] error:", e)

    def press(self, keys):
        try:
            for key in keys:
                pyautogui.keyDown(key)
            time.sleep(0.1)
            for key in keys:
                pyautogui.keyUp(key)
        except Exception as e:
            print("[OperatingSystem][press] error:", e)

    def mouse(self, click_detail):
        try:
            x = convert_percent_to_decimal(click_detail.get("x"))
            y = convert_percent_to_decimal(click_detail.get("y"))

            if click_detail and isinstance(x, float) and isinstance(y, float):
                self.click_at_percentage(x, y)

        except Exception as e:
            print("[OperatingSystem][mouse] error:", e)

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

    # -------------------------------------------------
    # NEW: AUTHORITY & RESTORATION SUPPORT
    # -------------------------------------------------

    # ---- Execution mode (AUTHORITY TRUTH)

    def get_execution_mode(self) -> str:
        """
        Returns current execution mode.

        This is authoritative for restoration & verification.
        """
        with self._execution_mode_lock:
            return self._execution_mode

    def set_execution_mode(self, mode: str) -> None:
        """
        Sets execution mode.

        Allowed values are enforced upstream.
        """
        with self._execution_mode_lock:
            self._execution_mode = mode

    # ---- Input control (SAFE, NO-OP FRIENDLY)

    def stop_automated_input(self) -> None:
        """
        Ceases any automated input.

        pyautogui is synchronous; this is a safety hook.
        """
        # No persistent input threads to stop currently
        return

    def enable_user_input(self) -> None:
        """
        Ensures user input is not blocked.

        No-op for pyautogui-based control.
        """
        return

    # ---- Cursor state (AUTHORITATIVE)

    def get_cursor_position(self):
        """
        Returns (x, y) in screen pixels.
        """
        try:
            return pyautogui.position()
        except Exception as e:
            raise RuntimeError(f"Unable to get cursor position: {e}")

    def set_cursor_position(self, x: int, y: int) -> None:
        """
        Moves cursor to absolute pixel position.
        """
        try:
            pyautogui.moveTo(int(x), int(y), duration=0)
        except Exception as e:
            raise RuntimeError(f"Unable to set cursor position: {e}")

    # ---- Window / application focus (BEST-EFFORT, SAFE)

    def get_focused_window(self):
        """
        Returns minimal focused window info.

        NOTE:
        pyautogui does not provide native window IDs.
        This is a conservative placeholder that preserves contract shape.
        """
        return {
            "id": "unknown",
            "title": None,
        }

    def get_focused_window_id(self):
        """
        Returns focused window ID if available.
        """
        info = self.get_focused_window()
        return info.get("id")

    def focus_window(self, window_id: str) -> bool:
        """
        Attempts to focus a window by ID.

        pyautogui cannot guarantee this.
        Returns False to allow fallback activation.
        """
        return False

    def get_active_application(self):
        """
        Returns active application info.

        Best-effort, platform-dependent.
        """
        return {
            "process_name": platform.system(),
            "pid": None,
        }

    def activate_application(self, process_name: str, pid=None) -> bool:
        """
        Attempts to activate an application.

        Best-effort only.
        Returns False if not supported.
        """
        return False
