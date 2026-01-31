import time
from typing import Dict, Any, Optional


def verify_step(
    action: Dict[str, Any],
    screenshot: Optional[Dict[str, Any]],
    previous_screenshot: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Evidence-aware step verifier.

    Contract:
    - If vision unavailable -> fail
    - For mutating ops, screen hash should change OR timestamp advance
    - 'done' always passes
    """

    if action is None:
        return False

    op = action.get("operation")

    if op == "done":
        return True

    if screenshot is None:
        return False

    if not screenshot.get("available"):
        return False

    frame_ts = screenshot.get("frame_ts")
    screen_hash = screenshot.get("screen_text_hash")

    if frame_ts is None or screen_hash is None:
        return False

    # If no previous reference, assume first step ok
    if previous_screenshot is None:
        return True

    prev_hash = previous_screenshot.get("screen_text_hash")
    prev_ts = previous_screenshot.get("frame_ts")

    # Screen must change OR advance in time
    if screen_hash != prev_hash:
        return True

    if prev_ts is not None and frame_ts > prev_ts:
        return True

    return False
