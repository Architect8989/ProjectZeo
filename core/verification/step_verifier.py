# core/verification/step_verifier.py

def verify_step(action: dict, screenshot) -> bool:
    """
    Conservative default:
    If screenshot exists -> assume step executed.
    Advanced heuristics can be layered later.
    """

    if screenshot is None:
        return False

    op = action.get("operation")

    if op in ("click", "write", "press"):
        return True

    if op == "done":
        return True

    return False
