import os
import platform
import subprocess
import pyautogui
from PIL import Image, ImageDraw, ImageGrab


class VisionUnavailableError(RuntimeError):
    """Raised when no screenshot backend is available in the current environment."""
    pass


def capture_screen_with_cursor(file_path):
    """
    Capture the current screen to file_path.

    RB-02 FIX: Headless Linux support.
    ─────────────────────────────────────────────────────────────────────────
    Original code called ImageGrab.grab() unconditionally on Linux, which
    requires an X11 DISPLAY environment variable. On headless SSH sessions
    and CI runners, DISPLAY is not set, causing an OSError at every capture.

    New flow (Linux):
      1. If DISPLAY is set → use existing Xlib / ImageGrab path (unchanged).
      2. If DISPLAY is absent → try CLI backends in order:
           a. scrot -  (writes PNG to stdout)
           b. import -window root png:- (ImageMagick)
           c. gnome-screenshot -f <path>
      3. If all CLI backends fail → raise VisionUnavailableError so the
         caller (observer) can surface a clean error instead of crashing.

    Windows: still uses pyautogui (unchanged — RB-03 is the Windows fix
    for the vision *adapter*, not the capture path).
    macOS: unchanged (screencapture -C).
    ─────────────────────────────────────────────────────────────────────────
    """
    user_platform = platform.system()

    if user_platform == "Windows":
        screenshot = pyautogui.screenshot()
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
                # Headless / no python-xlib: fall back to pyautogui
                screenshot = pyautogui.screenshot()
                screenshot.save(file_path)
                return

            screen = Xlib.display.Display().screen()
            size = screen.width_in_pixels, screen.height_in_pixels
            screenshot = ImageGrab.grab(bbox=(0, 0, size[0], size[1]))
            screenshot.save(file_path)

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
    """
    Attempt headless screenshot capture using available CLI tools.

    Tries in order: scrot, ImageMagick import, gnome-screenshot.
    Raises VisionUnavailableError if none succeed.
    """
    # Backend 1: scrot (lightweight X11-free capable on some distros)
    try:
        result = subprocess.run(
            ["scrot", "-"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout:
            with open(file_path, "wb") as f:
                f.write(result.stdout)
            return
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Backend 2: ImageMagick import (requires virtual framebuffer or Xvfb)
    try:
        result = subprocess.run(
            ["import", "-window", "root", "png:-"],
            capture_output=True,
            timeout=5,
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
            timeout=5,
        )
        if result.returncode == 0 and os.path.exists(file_path):
            return
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    raise VisionUnavailableError(
        "No screenshot backend available in headless environment. "
        "Set DISPLAY or install one of: scrot, imagemagick (import), gnome-screenshot."
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
        if img.mode == "P":
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
