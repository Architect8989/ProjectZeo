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




def _ydotool_available() -> bool:
    
    import shutil as _shutil
    if not _shutil.which("ydotool"):
        return False  # binary not installed at all

    # Live daemon probe: zero-displacement relative move — no visible effect.
    try:
        result = subprocess.run(
            ["ydotool", "mousemove", "--relative", "-x", "0", "-y", "0"],
            capture_output=True,
            timeout=3,
        )
        return result.returncode == 0
    except Exception:
        return False


def _wayland_click(x_px: int, y_px: int) -> None:
    
    if _ydotool_available():
        result = subprocess.run(
            ["ydotool", "mousemove", "--absolute", "-x", str(x_px), "-y", str(y_px)],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ydotool mousemove failed (rc={result.returncode}): {result.stderr.strip()}"
            )
        result = subprocess.run(
            ["ydotool", "click", "0xC0"],  # 0xC0 = left button down+up
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ydotool click failed (rc={result.returncode}): {result.stderr.strip()}"
            )
        return

    # xdotool via XWayland fallback
    import shutil as _shutil
    if _shutil.which("xdotool"):
        result = subprocess.run(
            ["xdotool", "mousemove", "--", str(x_px), str(y_px)],
            capture_output=True, text=True, timeout=5,
        )
        result2 = subprocess.run(
            ["xdotool", "click", "1"],
            capture_output=True, text=True, timeout=5,
        )
        if result2.returncode == 0:
            return

    raise RuntimeError(
        "Wayland click failed: neither ydotool nor xdotool (XWayland) succeeded. "
        "Install ydotool: sudo apt-get install ydotool && sudo ydotoold &  "
        "OR launch the agent from a GNOME-on-Xorg session (gear at login → Ubuntu on Xorg)."
    )


def _wayland_type(text: str) -> None:
    
    if _ydotool_available():
        result = subprocess.run(
            ["ydotool", "type", "--", text],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return
        raise RuntimeError(
            f"ydotool type failed (rc={result.returncode}): {result.stderr.strip()}"
        )

    import shutil as _shutil
    if _shutil.which("xdotool"):
        result = subprocess.run(
            ["xdotool", "type", "--", text],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return

    raise RuntimeError(
        "Wayland type failed: neither ydotool nor xdotool (XWayland) succeeded."
    )


# Map pyautogui key names → ydotool key names (X11 keysym subset)
_YDOTOOL_KEY_MAP: Dict[str, str] = {
    "enter": "Return", "return": "Return",
    "tab": "Tab", "space": "space",
    "backspace": "BackSpace", "delete": "Delete",
    "escape": "Escape", "esc": "Escape",
    "up": "Up", "down": "Down", "left": "Left", "right": "Right",
    "home": "Home", "end": "End", "pageup": "Prior", "pagedown": "Next",
    "f1": "F1", "f2": "F2", "f3": "F3", "f4": "F4", "f5": "F5",
    "f6": "F6", "f7": "F7", "f8": "F8", "f9": "F9", "f10": "F10",
    "f11": "F11", "f12": "F12",
    "ctrl": "ctrl", "control": "ctrl",
    "alt": "alt", "shift": "shift",
    "cmd": "super", "command": "super", "win": "super",
    "l": "l", "c": "c", "v": "v", "x": "x", "z": "z", "a": "a",
    "t": "t", "w": "w", "r": "r", "n": "n", "s": "s", "q": "q",
}


def _wayland_hotkey(keys: list) -> None:
   
    if _ydotool_available():
        mapped = [_YDOTOOL_KEY_MAP.get(k.lower(), k) for k in keys]
        combo = "+".join(mapped)
        result = subprocess.run(
            ["ydotool", "key", "--", combo],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return
        raise RuntimeError(
            f"ydotool key failed (rc={result.returncode}): {result.stderr.strip()}"
        )

    import shutil as _shutil
    if _shutil.which("xdotool"):
        result = subprocess.run(
            ["xdotool", "key", "--", "+".join(keys)],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return

    raise RuntimeError(
        "Wayland hotkey failed: neither ydotool nor xdotool (XWayland) succeeded."
    )


def _get_focused_window_wayland() -> Dict[str, str]:
    
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

        full_cmd = cmd.strip()
        if sudo and hasattr(os, "geteuid") and os.geteuid() != 0:
            full_cmd = f"sudo {full_cmd}"

        # Re-validate against dangerous patterns at execution boundary.
        # This is a defence-in-depth check: planning-time validation should have
        # already blocked dangerous commands, but dynamic candidates and installer
        # code paths may bypass the planner.  Unicode normalization mirrors the
        # planner's pipeline so homoglyph-obfuscated commands are caught here too.
        try:
            from core.planner.execution_planner import ExecutionPlanner as _EP  # noqa: PLC0415
            from core.security.injection_markers import normalize_for_injection_check as _norm  # noqa: PLC0415
            _compiled = getattr(_EP, "_exec_compiled_patterns", None)
            if _compiled is None:
                import re as _re
                _compiled = [_re.compile(p, _re.IGNORECASE) for p in _EP.DANGEROUS_PATTERNS]
                _EP._exec_compiled_patterns = _compiled
            _normalized_cmd = _norm(full_cmd)
            for _pat in _compiled:
                if _pat.search(_normalized_cmd):
                    raise RuntimeError(
                        f"exec(): command blocked by dangerous-pattern filter "
                        f"(pattern={_pat.pattern!r}): {full_cmd[:120]!r}"
                    )
        except RuntimeError:
            raise
        except Exception:
            pass  # Validator unavailable — proceed; planning-time filter is primary

        # Build a restricted environment: inherit PATH and common tool vars but
        # exclude LD_PRELOAD, LD_LIBRARY_PATH, and PYTHONPATH which could be used
        # to hijack the executed subprocess.
        _safe_env = {
            k: v for k, v in os.environ.items()
            if k not in ("LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH",
                         "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH")
        }

        try:
            result = subprocess.run(
                full_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=_safe_env,
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

        target_dir = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(target_dir, exist_ok=True)

        # Write to a temp file in the same directory so os.replace() stays on
        # the same filesystem (required for POSIX rename(2) atomicity).
        import tempfile as _tempfile
        tmp_fd, tmp_path = _tempfile.mkstemp(dir=target_dir, prefix=".zeo_write_")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                fh.write(content)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, path)
        except Exception:
            # Best-effort cleanup of the temp file on any failure.
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # =================================================
    # INPUT ACTIONS
    # =================================================

    def write(self, content: str) -> None:
        if not isinstance(content, str):
            raise RuntimeError("write(): content must be string")

        content = content.replace("\\n", "\n")

        # MAJOR-1 FIX: Route through Wayland-native backend when applicable.
        if platform.system() == "Linux" and _is_wayland():
            with self._automation_lock:
                self._automation_active = True
            try:
                _wayland_type(content)
            finally:
                with self._automation_lock:
                    self._automation_active = False
            return

        pya = _require_pyautogui()

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

        # MAJOR-1 FIX: Route through Wayland-native backend when applicable.
        if platform.system() == "Linux" and _is_wayland():
            with self._automation_lock:
                self._automation_active = True
            try:
                _wayland_hotkey(keys)
            finally:
                with self._automation_lock:
                    self._automation_active = False
            return

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
        # MAJOR-1 FIX: Route through Wayland-native backend when applicable.
        # pyautogui uses X11 XTest — silently fails on pure Wayland.
        if platform.system() == "Linux" and _is_wayland():
            # Get screen dimensions from mss (works on both X11 and Wayland)
            try:
                import mss as _mss
                with _mss.mss() as _sct:
                    _mon = _sct.monitors[1] if len(_sct.monitors) > 1 else _sct.monitors[0]
                    _screen_w = _mon["width"]
                    _screen_h = _mon["height"]
            except Exception:
                # Last-resort fallback: try pyautogui (may work via XWayland)
                if _PYAUTOGUI_AVAILABLE and _pyautogui_mod is not None:
                    _screen_w, _screen_h = _pyautogui_mod.size()
                else:
                    raise RuntimeError(
                        "_click_at_percentage(): cannot determine screen size on Wayland "
                        "— mss not installed and pyautogui unavailable."
                    )

            x_px = int(_screen_w * x_pct)
            y_px = int(_screen_h * y_pct)

            with self._automation_lock:
                self._automation_active = True
            try:
                _wayland_click(x_px, y_px)
            finally:
                with self._automation_lock:
                    self._automation_active = False
            return

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
        
        if platform.system() == "Linux" and _is_wayland():
            import shutil as _shutil
            if _shutil.which("xdotool"):
                try:
                    result = subprocess.run(
                        ["xdotool", "getmouselocation", "--shell"],
                        capture_output=True,
                        text=True,
                        timeout=3,
                    )
                    if result.returncode == 0:
                        # Output is:
                        #   X=1234\nY=567\nSCREEN=0\nWINDOW=1234567\n
                        x_val: Optional[int] = None
                        y_val: Optional[int] = None
                        for line in result.stdout.splitlines():
                            if line.startswith("X="):
                                try:
                                    x_val = int(line[2:].strip())
                                except ValueError:
                                    pass
                            elif line.startswith("Y="):
                                try:
                                    y_val = int(line[2:].strip())
                                except ValueError:
                                    pass
                        if x_val is not None and y_val is not None:
                            return {"x": x_val, "y": y_val}
                except Exception:
                    pass

            
            import sys as _sys_cur
            _sys_cur.stderr.write(
                "[OperatingSystem] CRIT-2: Wayland cursor query via xdotool failed "
                "(xdotool not installed or XWayland unavailable). "
                "Cursor position will be (0,0). Install xdotool for correct "
                "cursor snapshot/restore on Wayland: sudo apt-get install xdotool\n"
            )

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
                if _is_wayland():
                    import shutil as _shutil
                    
                    if _shutil.which("wmctrl"):
                        try:
                            result = subprocess.run(
                                ["wmctrl", "-a", title],
                                capture_output=True,
                                text=True,
                                timeout=5,
                            )
                            if result.returncode == 0:
                                time.sleep(0.15)
                                return
                            # wmctrl returns non-zero if title substring not found;
                            # fall through to next method rather than raising.
                        except Exception:
                            pass  # daemon not running or compositor mismatch

                    # Attempt 2: AT-SPI2 accessibility bus (compositor-native)
                    try:
                        import pyatspi  # noqa: PLC0415
                        desktop = pyatspi.Registry.getDesktop(0)
                        for app in desktop:
                            if app and title.lower() in (app.name or "").lower():
                                # Raise a "activate" action on the first window
                                for win in app:
                                    if win is not None:
                                        try:
                                            win.queryAction().doAction(
                                                win.queryAction().getActionIndex("activate")
                                            )
                                            time.sleep(0.15)
                                            return
                                        except Exception:
                                            pass
                    except Exception:
                        pass

                    # Attempt 3: xdotool via XWayland (hybrid sessions)
                    if _shutil.which("xdotool"):
                        try:
                            result = subprocess.run(
                                ["xdotool", "search", "--name", title,
                                 "windowactivate", "--sync"],
                                capture_output=True,
                                text=True,
                                timeout=5,
                            )
                            if result.returncode == 0:
                                time.sleep(0.15)
                                return
                        except Exception:
                            pass

                    # All Wayland activation methods failed — log warning (best-effort)
                    import sys as _sys_mod
                    print(
                        f"[OperatingSystem] BLOCKER-4: Could not activate window "
                        f"'{title}' on Wayland — wmctrl, AT-SPI2, and xdotool all "
                        "failed or unavailable.  Install wmctrl for best results: "
                        "sudo apt-get install wmctrl",
                        file=_sys_mod.stderr,
                    )
                    return  # Best-effort: don't raise so restoration proceeds

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
