"""
screenshot.py — Cross-platform screen capture utility.

PATCH (audit Bug #1):
  Xlib imports moved from module-level into the Linux branch.
  Previously `import Xlib.display`, `Xlib.X`, `Xlib.Xutil` at the top
  of the file raised ImportError on macOS and Windows, making the entire
  module non-importable on those platforms.  The fix defers those imports
  to the Linux code path so Windows and macOS can import this module
  without python-xlib installed.
"""

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
    """
    with Image.open(raw_screenshot_filename) as img:
        # Handle alpha channel (RGBA/LA/P-with-transparency)
        if img.mode in ("RGBA", "LA") or (
            img.mode == "P" and "transparency" in img.info
        ):
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])  # 3 = alpha channel
            background.save(screenshot_filename, "JPEG", quality=85)
        else:
            img.convert("RGB").save(screenshot_filename, "JPEG", quality=85)
