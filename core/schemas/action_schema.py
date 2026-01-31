from typing import Dict, Any, List

# -------------------------------------------------
# Allowed surface
# -------------------------------------------------

ALLOWED_OPERATIONS = {"click", "write", "press", "done"}

COMMON_FIELDS = {"operation", "thought"}

REQUIRED_FIELDS = {
    "click": {"operation"},
    "write": {"operation", "content"},
    "press": {"operation", "keys"},
    "done": {"operation", "summary"},
}

OPTIONAL_FIELDS = {
    "click": {"x", "y", "label", "text", "thought"},
    "write": {"content", "thought"},
    "press": {"keys", "thought"},
    "done": {"summary", "thought"},
}

# -------------------------------------------------

def _is_string(v):
    return isinstance(v, str) and len(v.strip()) > 0


def _is_list(v):
    return isinstance(v, list) and len(v) > 0


def _is_number_string(v):
    try:
        float(v)
        return True
    except Exception:
        return False


# -------------------------------------------------
# Validators
# -------------------------------------------------

def validate_click(action: Dict[str, Any]) -> bool:
    # Must specify exactly one targeting mode
    modes = 0

    if "x" in action and "y" in action:
        if not (_is_number_string(action["x"]) and _is_number_string(action["y"])):
            return False
        modes += 1

    if "label" in action:
        if not _is_string(action["label"]):
            return False
        modes += 1

    if "text" in action:
        if not _is_string(action["text"]):
            return False
        modes += 1

    return modes == 1


def validate_write(action: Dict[str, Any]) -> bool:
    return _is_string(action.get("content"))


def validate_press(action: Dict[str, Any]) -> bool:
    keys = action.get("keys")
    if not _is_list(keys):
        return False
    return all(_is_string(k) for k in keys)


def validate_done(action: Dict[str, Any]) -> bool:
    return _is_string(action.get("summary"))


# -------------------------------------------------
# Structural validation
# -------------------------------------------------

def validate_action(action: Dict[str, Any]) -> bool:
    if not isinstance(action, dict):
        return False

    op = action.get("operation")
    if op not in ALLOWED_OPERATIONS:
        return False

    # Required fields
    if not REQUIRED_FIELDS[op].issubset(action.keys()):
        return False

    # No unknown fields
    allowed_fields = COMMON_FIELDS | OPTIONAL_FIELDS[op]
    if not set(action.keys()).issubset(allowed_fields):
        return False

    # Operation-specific
    if op == "click":
        return validate_click(action)

    if op == "write":
        return validate_write(action)

    if op == "press":
        return validate_press(action)

    if op == "done":
        return validate_done(action)

    return False


def validate_actions(actions: Any) -> bool:
    if not isinstance(actions, list):
        return False

    if len(actions) == 0:
        return False

    for action in actions:
        if not validate_action(action):
            return False

    return True
