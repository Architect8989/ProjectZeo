from typing import List, Dict, Any, Optional



_OCR_AVAILABLE: bool = False
_reader = None
_reader_init_attempted: bool = False


def _get_reader():
    """Lazily initialise and return the easyocr Reader. Returns None if unavailable."""
    global _OCR_AVAILABLE, _reader, _reader_init_attempted
    if _reader_init_attempted:
        return _reader  # either a Reader or None — already determined

    _reader_init_attempted = True
    try:
        import easyocr as _easyocr
        _reader = _easyocr.Reader(["en"], gpu=False)
        _OCR_AVAILABLE = True
    except Exception:
        _OCR_AVAILABLE = False
        _reader = None
    return _reader


# -------------------------------------------------
# OCR (best-effort, weak signal only)
# -------------------------------------------------

def _extract_text(image) -> List[str]:
    reader = _get_reader()
    if reader is None or image is None:
        return []

    try:
        result = reader.readtext(image)
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
