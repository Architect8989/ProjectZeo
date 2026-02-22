import os
import platform
import subprocess
import pyautogui
from PIL import Image, ImageDraw, ImageGrab


def capture_screen_with_cursor(file_path):
    user_platform = platform.system()

    if user_platform == "Windows":
        screenshot = pyautogui.screenshot()
        screenshot.save(file_path)

    elif user_platform == "Linux":
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

    elif user_platform == "Darwin":  # macOS
        # screencapture -C includes the cursor in the capture.
        subprocess.run(["screencapture", "-C", file_path], check=False)

    else:
        print(
            f"The platform you're using ({user_platform}) is not currently supported"
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
