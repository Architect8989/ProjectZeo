from typing import List, Dict, Any, Optional

try:
    import easyocr
    _OCR_AVAILABLE = True
    _reader = easyocr.Reader(["en"], gpu=False)
except Exception:
    _OCR_AVAILABLE = False
    _reader = None


# -------------------------------------------------
# OCR (best-effort, weak signal only)
# -------------------------------------------------

def _extract_text(image) -> List[str]:
    if not _OCR_AVAILABLE or image is None:
        return []

    try:
        result = _reader.readtext(image)
        return [
            r[1].strip().lower()
            for r in result
            if isinstance(r, list) and len(r) >= 2 and isinstance(r[1], str)
        ]
    except Exception:
        return []


# -------------------------------------------------
# Main Verifier (HARDENED)
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
    - Hash change required for strong signal
    - OCR only used as secondary fallback
    - Timestamp advancement alone is NOT evidence
    """

    if not isinstance(screenshot, dict):
        return False

    if not screenshot.get("available"):
        return False

    if screenshot.get("blind"):
        return False

    screen_hash = screenshot.get("screen_text_hash")
    if not isinstance(screen_hash, str) or not screen_hash:
        return False

    # First observation — cannot prove change yet
    if previous_screenshot is None:
        return False

    prev_hash = previous_screenshot.get("screen_text_hash")

    # -----------------------------
    # STRONG SIGNAL: content changed
    # -----------------------------
    if isinstance(prev_hash, str) and screen_hash != prev_hash:
        return True

    # -----------------------------
    # WEAK SIGNAL: OCR delta
    # -----------------------------
    current_texts = set(_extract_text(screenshot.get("image")))
    prev_texts = set(_extract_text(previous_screenshot.get("image")))

    # Require real delta, not just presence
    if current_texts and current_texts != prev_texts:
        return True

    # No evidence of change
    return False
