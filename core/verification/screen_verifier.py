from typing import List, Dict, Any, Optional

try:
    import easyocr
    _OCR_AVAILABLE = True
    _reader = easyocr.Reader(["en"], gpu=False)
except Exception:
    _OCR_AVAILABLE = False
    _reader = None


# -------------------------------------------------
# OCR (best-effort, secondary signal)
# -------------------------------------------------

def _extract_text(image) -> List[str]:
    if not _OCR_AVAILABLE or image is None:
        return []

    try:
        result = _reader.readtext(image)
        return [r[1].lower() for r in result if isinstance(r, list) and len(r) >= 2]
    except Exception:
        return []


# -------------------------------------------------
# Main Verifier
# -------------------------------------------------

def verify_execution(
    actions: List[Dict[str, Any]],
    screenshot: Optional[Dict[str, Any]],
    previous_screenshot: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Evidence-based execution verifier.

    Contract:
    - Vision must be available
    - Screen must not be blind
    - Screen hash or timestamp must advance
    - OCR text presence used only as weak fallback
    """

    if screenshot is None:
        return False

    # Screenpipe style contract
    if not screenshot.get("available"):
        return False

    if screenshot.get("blind"):
        return False

    frame_ts = screenshot.get("frame_ts")
    screen_hash = screenshot.get("screen_text_hash")

    if frame_ts is None or screen_hash is None:
        return False

    # If first verification, accept
    if previous_screenshot is None:
        return True

    prev_hash = previous_screenshot.get("screen_text_hash")
    prev_ts = previous_screenshot.get("frame_ts")

    # Strong signal: hash changed
    if screen_hash != prev_hash:
        return True

    # Medium signal: timestamp advanced
    if prev_ts is not None and frame_ts > prev_ts:
        return True

    # Weak fallback: OCR detects visible text
    texts = _extract_text(screenshot.get("image"))
    if len(texts) > 0:
        return True

    return False
