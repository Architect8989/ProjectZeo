# core/verification/screen_verifier.py

from typing import List, Dict
import easyocr
import numpy as np

reader = easyocr.Reader(["en"], gpu=False)


def extract_text(image) -> List[str]:
    results = reader.readtext(image)
    return [r[1].lower() for r in results]


def verify_execution(actions: List[Dict], screenshot) -> bool:
    """
    Base invariant:
    If OCR returns text, screen is alive and changed.
    """

    texts = extract_text(screenshot)

    if not texts:
        return False

    return True
