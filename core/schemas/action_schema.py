# core/schemas/action_schema.py

from typing import Dict, Any, List

ALLOWED_OPERATIONS = {"click", "write", "press", "done"}

REQUIRED_FIELDS = {
    "click": {"operation"},
    "write": {"operation", "content"},
    "press": {"operation", "keys"},
    "done": {"operation", "summary"},
}


def _is_string(v):
    return isinstance(v, str) and len(v) > 0


def _is_list(v):
    return isinstance(v, list) and len(v) > 0


def _is_number_string(v):
    try:
        float(v)
        return True
    except Exception:
        return False


def validate_click(action: Dict[str, Any]) -> bool:
    # support x/y OR label OR text
    if "x" in action and "y" in action:
        return _is_number_string(action["x"]) and _is_number_string(action["y"])
    if "label" in action:
        return _is_string(action["label"])
    if "text" in action:
        return _is_string(action["text"])
    return False


def validate_write(action: Dict[str, Any]) -> bool:
    return _is_string(action.get("content"))


def validate_press(action: Dict[str, Any]) -> bool:
    keys = action.get("keys")
    if not _is_list(keys):
        return False
    return all(_is_string(k) for k in keys)


def validate_done(action: Dict[str, Any]) -> bool:
    return _is_string(action.get("summary"))


def validate_action(action: Dict[str, Any]) -> bool:
    if not isinstance(action, dict):
        return False

    op = action.get("operation")
    if op not in ALLOWED_OPERATIONS:
        return False

    if not REQUIRED_FIELDS[op].issubset(action.keys()):
        return False

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

    for a in actions:
        if not validate_action(a):
            return False

    return True
