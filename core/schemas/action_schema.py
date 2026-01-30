# core/schemas/action_schema.py

ALLOWED_ACTIONS = {
    "click",
    "type",
    "key_press",
    "open_app",
    "navigate",
    "wait",
    "scroll",
    "screenshot"
}

def validate_action(action: dict) -> bool:
    if "action" not in action:
        return False
    if action["action"] not in ALLOWED_ACTIONS:
        return False
    return True


def validate_plan(plan: dict) -> bool:
    if "steps" not in plan:
        return False
    if not isinstance(plan["steps"], list):
        return False

    for step in plan["steps"]:
        if not validate_action(step):
            return False

    return True
