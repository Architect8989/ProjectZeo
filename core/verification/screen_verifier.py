# core/verification/screen_verifier.py

import easyocr

reader = easyocr.Reader(["en"], gpu=False)


def extract_text(image):
    try:
        result = reader.readtext(image)
        return [r[1].lower() for r in result]
    except Exception:
        return []


def verify_execution(actions, screenshot) -> bool:
    """
    Baseline invariant:
    Screen must be alive and readable.
    """
    texts = extract_text(screenshot)
    return len(texts) > 0
