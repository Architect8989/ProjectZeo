try:
    import pyautogui as _pyautogui_mod
    _pyautogui_mod.FAILSAFE = True
    _PYAUTOGUI_AVAILABLE: bool = True
except Exception as _pyautogui_import_err:
    _pyautogui_mod = None  # type: ignore[assignment]
    _PYAUTOGUI_AVAILABLE: bool = False
    _PYAUTOGUI_IMPORT_ERROR: str = str(_pyautogui_import_err)
else:
    _PYAUTOGUI_IMPORT_ERROR: str = ""

import platform
import time
import math
import threading
import subprocess
import os
from typing import Optional, Dict

from operate.utils.misc import convert_percent_to_decimal

try:
    from config.timeouts import INSTALL_COMMAND_TIMEOUT_SECONDS as _INSTALL_TIMEOUT
except ImportError:
    _INSTALL_TIMEOUT = 300


# ─────────────────────────────────────────────────────────────────────────────
# BUG-2 FIX: Wayland detection helpers
#
# On Ubuntu 22.04+ and 24.04, the default session is Wayland. xdotool is
# X11-only and silently fails (exit code 1, empty output) on Wayland.
# This causes get_focused_window() to return an empty string, which the
# snapshot provider falls through to __bare_desktop__, making ALL snapshots
# useless for restoration. wmctrl -a also fails on pure Wayland without
# XWayland.
#
# Fix:
#   1. _is_wayland() detects the session type from XDG_SESSION_TYPE.
#   2. get_focused_window() on Linux: if Wayland, try AT-SPI2 / ydotool
#      as alternatives before falling back gracefully.
#   3. Operators are warned at first call so the issue is visible in logs.
# ─────────────────────────────────────────────────────────────────────────────

def _is_wayland() -> bool:
    """Return True if the current session is a Wayland session."""
    session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
    if session_type == "wayland":
        return True
    # Also check WAYLAND_DISPLAY — present on Wayland even if XDG_SESSION_TYPE is unset
    if os.environ.get("WAYLAND_DISPLAY"):
        return True
    # Pure X11 or undetermined
    return False


def _get_focused_window_wayland() -> Dict[str, str]:
    """
    Best-effort focused window title on Wayland.

    Attempts (in order):
      1. ydotool getactivewindow (Wayland-native, requires ydotool daemon)
      2. AT-SPI2 via pyatspi (accessibility bus — works on Wayland when enabled)
      3. wmctrl -l (works on Wayland + XWayland if XWayland is running)
      4. Falls back to __wayland_unknown__ sentinel

    Returns {"title": <str>} consistent with X11 path.
    """
    import shutil

    # Attempt 1: ydotool (Wayland-native tool, requires ydotoold daemon)
    if shutil.which("ydotool"):
        try:
            result = subprocess.run(
                ["ydotool", "getactivewindow"],
                capture_output=True, text=True, timeout=3,
            )
            if result.returncode == 0 and result.stdout.strip():
                return {"title": result.stdout.strip()}
        except Exception:
            pass

    # Attempt 2: AT-SPI2 accessibility bus
    try:
        import pyatspi  # noqa: PLC0415
        desktop = pyatspi.Registry.getDesktop(0)
        for app in desktop:
            if app and app.getState().contains(pyatspi.STATE_ACTIVE):
                return {"title": app.name or "__wayland_app__"}
    except Exception:
        pass

    # Attempt 3: wmctrl with XWayland
    if shutil.which("wmctrl"):
        try:
            result = subprocess.run(
                ["wmctrl", "-l"],
                capture_output=True, text=True, timeout=3,
            )
            if result.returncode == 0:
                # wmctrl -l output: <wid> <desktop> <hostname> <title>
                for line in result.stdout.splitlines():
                    parts = line.split(None, 3)
                    if len(parts) >= 4:
                        return {"title": parts[3].strip()}
        except Exception:
            pass

    # All Wayland methods failed — return sentinel so snapshot uses __bare_desktop__
    import sys as _sys
    print(
        "[OperatingSystem] BUG-2: Wayland session detected but no window-title "
        "backend succeeded (ydotool, AT-SPI2, wmctrl all failed or not installed). "
        "Returning __wayland_unknown__ sentinel. "
        "Install ydotool: sudo apt-get install ydotool && ydotoold & "
        "OR enable AT-SPI2: gsettings set org.gnome.desktop.interface toolkit-accessibility true",
        file=_sys.stderr,
    )
    return {"title": "__wayland_unknown__"}


class OperatingSystemUnavailableError(RuntimeError):
    """Raised when a pyautogui operation is requested but pyautogui is
    unavailable (not installed or no X11 display)."""


def _require_pyautogui() -> object:
    """
    P0-1 FIX: Fail fast with a clear error when pyautogui is unavailable.

    Call this at the top of every method that delegates to pyautogui.
    Returns the pyautogui module on success. Raises OperatingSystemUnavailableError
    on failure.
    """
    if not _PYAUTOGUI_AVAILABLE:
        raise OperatingSystemUnavailableError(
            "pyautogui is not available. "
            f"Import error: {_PYAUTOGUI_IMPORT_ERROR!r}. "
            "Install with: pip install pyautogui\n"
            "On headless systems also install: pip install Pillow\n"
            "On Linux ensure DISPLAY is set or use a virtual framebuffer: "
            "Xvfb :99 & DISPLAY=:99 python main.py"
        )
    return _pyautogui_mod


class OperatingSystem:
    

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
    # SCREEN SIZE
    # =================================================

    def screen_size(self) -> tuple:
        """
        Return (width, height) in pixels.

        operate.py:_execute_decision() calls this whenever an OCR-resolved
        click has absolute pixel coordinates (x > 1.0 or y > 1.0).
        """
        pya = _require_pyautogui()
        w, h = pya.size()
        return int(w), int(h)

    # =================================================
    # COMMANDS / FILES
    # =================================================

    def exec(self, cmd: str, *, sudo: bool = False, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
        if not isinstance(cmd, str) or not cmd.strip():
            raise RuntimeError("exec(): invalid command")

        full_cmd = cmd
        if sudo and hasattr(os, "geteuid") and os.geteuid() != 0:
            full_cmd = f"sudo {cmd}"

        try:
            result = subprocess.run(
                full_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"exec(): command timed out after {timeout}s: {cmd!r}"
            ) from exc

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

        pya = _require_pyautogui()
        content = content.replace("\\n", "\n")

        with self._automation_lock:
            self._automation_active = True

        try:
            pya.write(content, interval=0.01)
        finally:
            with self._automation_lock:
                self._automation_active = False

    def press(self, keys) -> None:
        if not isinstance(keys, list) or not keys:
            raise RuntimeError("press(): keys must be non-empty list")

        pya = _require_pyautogui()

        with self._automation_lock:
            self._automation_active = True

        try:
            pya.hotkey(*keys)
        finally:
            with self._automation_lock:
                self._automation_active = False

    def mouse(self, click_detail: dict) -> None:
        if not isinstance(click_detail, dict):
            raise RuntimeError("mouse(): invalid click_detail")

        if "x" not in click_detail or "y" not in click_detail:
            raise RuntimeError("mouse(): requires x and y")

        x = convert_percent_to_decimal(click_detail.get("x"))
        y = convert_percent_to_decimal(click_detail.get("y"))

        if not self._valid_coord(x) or not self._valid_coord(y):
            raise RuntimeError(f"Invalid click coordinates: {click_detail}")

        self._click_at_percentage(float(x), float(y))

    def _click_at_percentage(self, x_pct: float, y_pct: float) -> None:
        pya = _require_pyautogui()

        screen_w, screen_h = pya.size()

        x_px = int(screen_w * x_pct)
        y_px = int(screen_h * y_pct)

        with self._automation_lock:
            self._automation_active = True

        try:
            pya.moveTo(x_px, y_px, duration=0.05)

            cur_x, cur_y = pya.position()
            if abs(cur_x - x_px) > 3 or abs(cur_y - y_px) > 3:
                raise RuntimeError("Cursor failed to reach target")

            pya.click()
        finally:
            with self._automation_lock:
                self._automation_active = False

    # =================================================
    # CURSOR STATE
    # =================================================

    def get_cursor_position(self) -> Dict[str, int]:
        pya = _require_pyautogui()
        x, y = pya.position()
        return {"x": int(x), "y": int(y)}

    def set_cursor_position(self, position: Dict[str, int]) -> None:
        if not isinstance(position, dict):
            raise RuntimeError("set_cursor_position(): invalid position")

        x = position.get("x")
        y = position.get("y")

        if not isinstance(x, int) or not isinstance(y, int):
            raise RuntimeError("set_cursor_position(): invalid coordinates")

        pya = _require_pyautogui()
        pya.moveTo(x, y, duration=0.05)

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
                # BUG-2 FIX: Detect Wayland and route to Wayland-capable backend.
                # xdotool is X11-only; on Wayland it exits with code 1 and empty
                # output, causing every snapshot to record __bare_desktop__ and
                # making all restoration a no-op.
                if _is_wayland():
                    return _get_focused_window_wayland()

                # X11 path (original behaviour)
                try:
                    result = subprocess.run(
                        ["xdotool", "getactivewindow", "getwindowname"],
                        capture_output=True,
                        text=True,
                        timeout=3,
                    )
                except FileNotFoundError:
                    raise OSError("xdotool not installed")

                # BUG-2 FIX: xdotool on X11 may succeed (rc=0) but return empty
                # output if no window is focused (bare desktop, screen locked, etc.).
                # Previously the empty string propagated to snapshot_provider which
                # fell through to __bare_desktop__ silently. Now we raise OSError
                # with a clear reason so callers get a diagnostic.
                if result.returncode != 0:
                    stderr_msg = result.stderr.strip() or "no error output"
                    raise OSError(
                        f"xdotool getwindowname failed (rc={result.returncode}): "
                        f"{stderr_msg}. "
                        "If running on Wayland, set XDG_SESSION_TYPE=wayland or "
                        "install ydotool for Wayland window-title support."
                    )

                title = result.stdout.strip()
                if not title:
                    return {"title": "__bare_desktop__"}

                return {"title": title}

            elif system == "Windows":
                # BUG-12 FIX: win32gui is from pywin32 which was missing from
                # requirements.txt. Import inside try/except to give a clear error.
                try:
                    import win32gui  # noqa: PLC0415
                except ImportError:
                    raise OSError(
                        "win32gui not available. Install pywin32: pip install pywin32"
                    )
                hwnd = win32gui.GetForegroundWindow()
                title = win32gui.GetWindowText(hwnd)
                return {"title": title}

        except OSError:
            raise
        except Exception as e:
            raise OSError(f"Failed to get focused window: {e}") from e

        raise OSError("Focused window unavailable")

    def get_active_application(self) -> Dict[str, str]:
        """
        M3 FIX: Return the name of the active application PROCESS, not the
        window title.

        The previous implementation was:
            return self.get_focused_window()
        This caused get_active_application() and get_focused_window() to return
        identical dicts, making the "application check" in snapshot_provider a
        no-op (it was comparing the window title against itself).  RestoreVerifier
        could not distinguish "wrong app focused" from "correct app, different window
        title", so cross-app restoration failures went silently undetected.

        Platform implementations:
          Linux:   xdotool getactivewindow getwindowpid → /proc/<pid>/comm
                   Falls back to get_focused_window() if xdotool or /proc unavailable.
          macOS:   AppleScript → frontmost process name (already process-level).
          Windows: win32gui.GetForegroundWindow() → win32process → psutil.Process.name()
                   Falls back to get_focused_window() if pywin32/psutil unavailable.

        Returns {"title": <process_name>} on success.  The "title" key is kept
        for backward compatibility with all callers (snapshot_provider, restore_provider,
        restore_verifier) that expect a {"title": str} dict shape.
        """
        system = platform.system()

        try:
            if system == "Linux":
                # Step 1: get the active window ID
                try:
                    _wid_result = subprocess.run(
                        ["xdotool", "getactivewindow"],
                        capture_output=True,
                        text=True,
                        timeout=3,
                    )
                except FileNotFoundError:
                    # xdotool not installed — fall back
                    return self.get_focused_window()

                if _wid_result.returncode != 0 or not _wid_result.stdout.strip():
                    return self.get_focused_window()

                _wid = _wid_result.stdout.strip()

                # Step 2: get the PID of that window
                _pid_result = subprocess.run(
                    ["xdotool", "getwindowpid", _wid],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                if _pid_result.returncode != 0 or not _pid_result.stdout.strip():
                    return self.get_focused_window()

                _pid_str = _pid_result.stdout.strip()

                # Step 3: resolve process name from /proc/<pid>/comm (Linux-specific)
                # /proc/<pid>/comm contains only the executable basename (≤15 chars),
                # which is what we want — not the full argv[0] path.
                _comm_path = f"/proc/{_pid_str}/comm"
                if os.path.exists(_comm_path):
                    try:
                        with open(_comm_path, "r", encoding="utf-8") as _f:
                            _proc_name = _f.read().strip()
                        if _proc_name:
                            return {"title": _proc_name}
                    except OSError:
                        pass

                # Step 4: psutil fallback (cross-distro, handles kernel threads)
                try:
                    import psutil as _psutil  # noqa: PLC0415
                    _p = _psutil.Process(int(_pid_str))
                    _proc_name = _p.name()
                    if _proc_name:
                        return {"title": _proc_name}
                except Exception:
                    pass

                # Fall back to window title if process name unresolvable
                return self.get_focused_window()

            elif system == "Darwin":
                # AppleScript returns the process name directly (not window title)
                _script = (
                    'tell application "System Events"\n'
                    '    set frontApp to first application process whose frontmost is true\n'
                    '    return name of frontApp\n'
                    'end tell'
                )
                _result = subprocess.run(
                    ["osascript", "-e", _script],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if _result.returncode == 0 and _result.stdout.strip():
                    return {"title": _result.stdout.strip()}
                return self.get_focused_window()

            elif system == "Windows":
                try:
                    import win32gui    # noqa: PLC0415
                    import win32process  # noqa: PLC0415
                    _hwnd = win32gui.GetForegroundWindow()
                    _, _pid = win32process.GetWindowThreadProcessId(_hwnd)
                    try:
                        import psutil as _psutil  # noqa: PLC0415
                        _proc_name = _psutil.Process(_pid).name()
                        if _proc_name:
                            return {"title": _proc_name}
                    except Exception:
                        pass
                    # psutil unavailable — return window title as fallback
                    return self.get_focused_window()
                except ImportError:
                    return self.get_focused_window()

        except Exception:
            pass

        # Final fallback: return window title so callers never get an exception
        return self.get_focused_window()

    # =================================================
    # WINDOW GEOMETRY
    # =================================================

    def get_window_geometry(self, window_id: str) -> Dict[str, int]:
        """
        Return geometry dict for a window identified by title.

        RestoreVerifier._verify_window_geometry() calls this when
        snapshot.metadata["extended"]["window_geometry"] is present.

        Linux: queries xdotool. macOS/Windows: raises OSError (best-effort).
        Callers (RestoreVerifier) swallow OSError and continue.
        """
        if not isinstance(window_id, str) or not window_id.strip():
            raise OSError("get_window_geometry(): window_id must be a non-empty string")

        system = platform.system()

        if system == "Linux":
            try:
                search = subprocess.run(
                    ["xdotool", "search", "--name", window_id.strip()],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                if search.returncode != 0 or not search.stdout.strip():
                    raise OSError(f"Window not found: {window_id!r}")

                wid = search.stdout.strip().split()[0]

                geo = subprocess.run(
                    ["xdotool", "getwindowgeometry", "--shell", wid],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                if geo.returncode != 0:
                    raise OSError(f"Could not get geometry for wid {wid}")

                result: Dict[str, int] = {}
                for line in geo.stdout.splitlines():
                    if "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip().lower()
                    if key in ("x", "y", "width", "height"):
                        try:
                            result[key] = int(val.strip())
                        except ValueError:
                            pass

                if len(result) < 4:
                    raise OSError("Incomplete geometry data from xdotool")

                return result

            except subprocess.TimeoutExpired:
                raise OSError("get_window_geometry(): xdotool timed out")
            except OSError:
                raise
            except Exception as e:
                raise OSError(f"get_window_geometry() failed: {e}") from e

        raise OSError(
            f"get_window_geometry() not implemented on {system}. "
            "RestoreVerifier will treat this as best-effort (soft failure)."
        )

    # =================================================
    # APPLICATION ACTIVATION
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
                # BUG-2 FIX: On Wayland, wmctrl -a fails without XWayland.
                # Route to ydotool or skip wmctrl on pure Wayland sessions.
                if _is_wayland():
                    import shutil as _shutil
                    # ydotool: Wayland-native xdotool replacement
                    if _shutil.which("ydotool"):
                        try:
                            result = subprocess.run(
                                ["ydotool", "search", "--name", title],
                                capture_output=True, text=True, timeout=5,
                            )
                            if result.returncode == 0:
                                time.sleep(self.POST_ACTION_DELAY if hasattr(self, "POST_ACTION_DELAY") else 0.15)
                                return
                        except Exception:
                            pass
                    # On Wayland without ydotool, log warning but do not raise
                    import sys as _sys
                    print(
                        f"[OperatingSystem] BUG-2: Wayland session — wmctrl not "
                        f"supported. Cannot activate '{title}'. "
                        "Install ydotool for Wayland window activation support.",
                        file=_sys.stderr,
                    )
                    return  # Best-effort on Wayland

                # X11 path (original behaviour)
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
    # ALIASED / MISSING API SHIMS
    # =================================================

    def click(self, x: float, y: float) -> None:
        """Click at pixel-absolute coordinates."""
        pya = _require_pyautogui()
        screen_w, screen_h = pya.size()
        if screen_w <= 0 or screen_h <= 0:
            raise RuntimeError("click(): unable to determine screen size")

        x_pct = float(x) / screen_w
        y_pct = float(y) / screen_h

        if not (0.0 <= x_pct <= 1.0) or not (0.0 <= y_pct <= 1.0):
            raise RuntimeError(
                f"click(): coordinates ({x}, {y}) out of screen bounds "
                f"({screen_w}x{screen_h})"
            )

        self._click_at_percentage(x_pct, y_pct)

    def type_text(self, text: str) -> None:
        """Alias for write() — required by operate.py."""
        self.write(text)

    def press_keys(self, keys) -> None:
        """Alias for press() — required by operate.py and autonomous_installer.py."""
        self.press(keys)

    def run_command(self, command: str, *, timeout: Optional[int] = _INSTALL_TIMEOUT) -> subprocess.CompletedProcess:
        """
        Alias for exec() — required by operate.py and autonomous_installer.py.
        Returns CompletedProcess so callers can inspect stdout/stderr/returncode.
        """
        return self.exec(command, timeout=timeout)

    def open_browser(self) -> None:
        """Open the system default web browser."""
        system = platform.system()

        try:
            if system == "Darwin":
                result = subprocess.run(
                    ["open", "-a", "Safari"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode != 0:
                    subprocess.run(["open", "about:blank"], timeout=10)

            elif system == "Linux":
                browser = os.environ.get("BROWSER", "xdg-open")
                result = subprocess.run(
                    [browser, "about:blank"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode != 0:
                    raise OSError(result.stderr.strip() or "browser open failed")

            elif system == "Windows":
                subprocess.run(
                    ["cmd", "/c", "start", "", "about:blank"],
                    timeout=10,
                )
            else:
                raise OSError(f"Unsupported platform: {system}")

        except subprocess.TimeoutExpired:
            raise OSError("open_browser(): timeout waiting for browser")

        time.sleep(1.5)

    def focus_address_bar(self) -> None:
        """Focus the browser address bar (Ctrl+L / Cmd+L)."""
        system = platform.system()
        if system == "Darwin":
            self.press(["command", "l"])
        else:
            self.press(["ctrl", "l"])
        time.sleep(0.2)

    def focus_window(self, spec: dict) -> None:
        """
        Bring a window to the foreground by title substring.
        Required by restore_provider.py during the RESTORING phase.
        """
        if not isinstance(spec, dict):
            raise RuntimeError("focus_window(): spec must be a dict")

        title = spec.get("title")
        if not isinstance(title, str) or not title.strip():
            raise RuntimeError("focus_window(): spec must contain a non-empty 'title'")

        self.activate_application({"title": title.strip()})

    # =================================================
    # RESTORATION / SAFETY
    # =================================================

    def is_automation_active(self) -> bool:
        """
        P0-2 FIX: Expose automation state for RestoreVerifier._verify_input_released().

        Previously absent. RestoreVerifier checks hasattr(self._os, 'is_automation_active')
        and silently skips the check when the method is missing — making the
        'fail-closed verification' claim false. This method makes the check active.

        Thread-safe: reads under _automation_lock.
        """
        with self._automation_lock:
            return self._automation_active

    def mark_automation_inactive(self) -> None:
        with self._automation_lock:
            self._automation_active = False

    def stop_automated_input(self) -> None:
        self.mark_automation_inactive()

    def force_release_all(self, *, reason: str) -> None:
        self.mark_automation_inactive()

        if _PYAUTOGUI_AVAILABLE and _pyautogui_mod is not None:
            try:
                _pyautogui_mod.mouseUp()
            except Exception:
                pass

            for key in ("shift", "ctrl", "alt", "cmd"):
                try:
                    _pyautogui_mod.keyUp(key)
                except Exception:
                    pass

    

    def get_window_z_order(self, window_id: str) -> int:
        
        raise NotImplementedError(
            "get_window_z_order() is not yet implemented for this platform. "
            "RestoreVerifier will treat this as a soft-fail (verification skipped). "
            "Implement this method to activate Z-order restoration verification."
        )

    def get_browser_state(self) -> dict:
        """
        AUDIT-SI-3 STUB: Return current browser state as {"url": str, "title": str}.

        RestoreVerifier compares against the snapshot value. Raises
        NotImplementedError until a CDP/Marionette integration is provided
        (soft-fail in verifier).

        Platform notes:
          Chrome/Chromium: CDP via websocket on port 9222 (--remote-debugging-port).
          Firefox:         Marionette protocol.
        """
        raise NotImplementedError(
            "get_browser_state() is not yet implemented. "
            "A CDP integration (Chrome DevTools Protocol) is required. "
            "RestoreVerifier will treat this as a soft-fail (verification skipped)."
        )

    def get_media_playback_position(self) -> float:
        
        raise NotImplementedError(
            "get_media_playback_position() is not yet implemented. "
            "An MPRIS/osascript/COM integration is required. "
            "RestoreVerifier will treat this as a soft-fail (verification skipped)."
        )

    # =================================================
    # HELPERS
    # =================================================

    @staticmethod
    def _valid_coord(v) -> bool:
        if not isinstance(v, (int, float)):
            return False
        try:
            v = float(v)
        except Exception:
            return False
        return not math.isnan(v) and 0.0 <= v <= 1.0
