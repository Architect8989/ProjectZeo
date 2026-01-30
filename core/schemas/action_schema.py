# core/schemas/action_schema.py

from typing import Any, Dict, List


ALLOWED_OPERATIONS = {
    "click",
    "write",
    "press",
    "done"
}

# Required keys per operation
REQUIRED_FIELDS = {
    "click": {"operation"},
    "write": {"operation", "content"},
    "press": {"operation", "keys"},
    "done": {"operation", "summary"}
}


def _is_string(v):
    return isinstance(v, str)


def _is_list(v):
    return isinstance(v, list)


def validate_click(action: Dict[str, Any]) -> bool:
    # click may use x/y OR label OR text
    if "x" in action and "y" in action:
        return True
    if "label" in action:
        return True
    if "text" in action:
        return True
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

    if "operation" not in action:
        return False

    op = action["operation"]

    if op not in ALLOWED_OPERATIONS:
        return False

    # Required base fields
    for field in REQUIRED_FIELDS[op]:
        if field not in action:
            return False

    # Operation-specific validation
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
