import os
import platform
import subprocess


try:
    import pyautogui as _pyautogui  # noqa: F401
    _PYAUTOGUI_AVAILABLE: bool = True
except ImportError:
    _pyautogui = None  # type: ignore[assignment]
    _PYAUTOGUI_AVAILABLE: bool = False

from PIL import Image, ImageDraw, ImageGrab


class VisionUnavailableError(RuntimeError):
    """Raised when no screenshot backend is available in the current environment."""
    pass


def capture_screen_with_cursor(file_path):
    
    user_platform = platform.system()

    if user_platform == "Windows":
        if not _PYAUTOGUI_AVAILABLE:
            raise VisionUnavailableError(
                "pyautogui is not installed. "
                "Install it with: pip install pyautogui"
            )
        screenshot = _pyautogui.screenshot()
        screenshot.save(file_path)

    elif user_platform == "Linux":
        display = os.environ.get("DISPLAY", "").strip()

        if display:
            # ── Normal X11 path ───────────────────────────────────────────
            # PATCH: deferred Xlib import — only loaded on Linux at call time,
            # avoiding ImportError on macOS / Windows.
            try:
                import Xlib.display  # noqa: PLC0415
                import Xlib.X        # noqa: PLC0415
            except ImportError:
                # python-xlib absent → fall through to pyautogui, then headless.
                pass
            else:
                # Xlib present — attempt X11 capture.
                try:
                    screen = Xlib.display.Display().screen()
                    size = screen.width_in_pixels, screen.height_in_pixels
                    screenshot = ImageGrab.grab(bbox=(0, 0, size[0], size[1]))
                    screenshot.save(file_path)
                    return
                except Exception:
                    # X11 reachable but capture failed (race on display close,
                    # compositor restart, etc.) — fall through to pyautogui.
                    pass

            # RB-4 / RB-MED-1 FIX: pyautogui fallback when Xlib unavailable OR
            # X11 capture raised an exception.  Now gated on _PYAUTOGUI_AVAILABLE
            # so the absence of pyautogui falls cleanly through to headless CLI
            # backends rather than raising AttributeError.
            if _PYAUTOGUI_AVAILABLE:
                try:
                    screenshot = _pyautogui.screenshot()
                    screenshot.save(file_path)
                    return
                except Exception:
                    # pyautogui failed — fall through to headless backends.
                    pass

            # Last resort: headless CLI backends (scrot, import, gnome-screenshot).
            # _headless_capture() raises VisionUnavailableError if none succeed.
            _headless_capture(file_path)

        else:
            # ── Headless fallback path ────────────────────────────────────
            # Try CLI screenshot backends that can operate without DISPLAY.
            # Each backend writes to stdout (binary PNG/image data).
            _headless_capture(file_path)

    elif user_platform == "Darwin":  # macOS
        # screencapture -C includes the cursor in the capture.
        subprocess.run(["screencapture", "-C", file_path], check=False)

    else:
        print(
            f"The platform you're using ({user_platform}) is not currently supported"
        )


def _headless_capture(file_path: str) -> None:
    
    # Backend 1: scrot (lightweight X11-free capable on some distros)
    try:
        result = subprocess.run(
            ["scrot", "-"],
            capture_output=True,
            timeout=1.5,  # AUDIT-RD-1: reduced from 5s → 1.5s
        )
        
        if result.returncode == 0 and len(result.stdout) > 0:
            with open(file_path, "wb") as f:
                f.write(result.stdout)
            return
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    
    _display = os.environ.get("DISPLAY", "").strip()
    _imagemagick_allowed = False
    if _display:
        try:
            _probe = subprocess.run(
                ["xdpyinfo", "-display", _display],
                capture_output=True,
                timeout=0.5,
            )
            _imagemagick_allowed = (_probe.returncode == 0)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            _imagemagick_allowed = False

    if _imagemagick_allowed:
        try:
            result = subprocess.run(
                ["import", "-window", "root", "png:-"],
                capture_output=True,
                timeout=1.5,  # AUDIT-RD-1: reduced from 5s → 1.5s
            )
            if result.returncode == 0 and result.stdout:
                with open(file_path, "wb") as f:
                    f.write(result.stdout)
                return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # Backend 3: gnome-screenshot (Wayland / GNOME environments)
    try:
        result = subprocess.run(
            ["gnome-screenshot", "-f", file_path],
            capture_output=True,
            timeout=1.5,  # AUDIT-RD-1: reduced from 5s → 1.5s
        )
        if result.returncode == 0 and os.path.exists(file_path):
            return
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    raise VisionUnavailableError(
        "No screenshot backend available in headless environment. "
        "Set DISPLAY to a reachable X server, or install one of: "
        "scrot, imagemagick (import), gnome-screenshot."
    )


def compress_screenshot(raw_screenshot_filename, screenshot_filename):
    """
    Convert a raw screenshot to JPEG, compositing transparency over white.

    FIX RTB-06: PIL palette-mode ("P") images with transparency are a
    single-band image — img.split() returns one band, so split()[3] raises
    IndexError. The transparency information on palette images is stored in
    img.info["transparency"], not as a fourth channel. Convert "P" images to
    "RGBA" first so the standard RGBA compositing path can apply uniformly.
    """
    with Image.open(raw_screenshot_filename) as img:
        # Step 1: Convert palette-mode (P/PA) to RGBA so transparency is
        # represented as a proper alpha channel that split()[3] can access.
        #
        # RB-NEW-03 FIX: Also handle "PA" (palette + alpha) mode.
        # PIL's "PA" mode is distinct from "P" (palette-only). A palette+alpha
        # PNG opened by PIL remains in "PA" mode; the previous check
        # `img.mode == "P"` did not match it. PA.split() returns 2 bands
        # (palette index + alpha), so bands[3] would raise IndexError.
        # Fix: check for both "P" and "PA" and convert both to "RGBA".
        if img.mode in ("P", "PA"):
            img = img.convert("RGBA")

        # Step 2: Composite any alpha channel over a white background.
        if img.mode in ("RGBA", "LA"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            # split()[3] is valid here: RGBA has 4 bands, LA has 2 bands with
            # alpha at index 1 — handle both.
            bands = img.split()
            alpha = bands[3] if img.mode == "RGBA" else bands[1]
            background.paste(img, mask=alpha)
            background.save(screenshot_filename, "JPEG", quality=85)
        else:
            img.convert("RGB").save(screenshot_filename, "JPEG", quality=85)
