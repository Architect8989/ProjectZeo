import time
import uuid
from typing import Dict, Any, List


INTERACTIVE_ROLES = {
    "push button",
    "button",
    "menu item",
    "link",
    "check box",
    "radio button",
    "text",
    "entry",
    "combo box",
    "list item",
    "tab",
}


def _is_interactive(node) -> bool:
    try:
        role = (node.getRoleName() or "").lower()
        state = node.getState()
        return (
            role in INTERACTIVE_ROLES
            and state.contains(1)  # STATE_VISIBLE
            and state.contains(7)  # STATE_SENSITIVE
        )
    except Exception:
        return False


def _allowed_actions(role: str) -> List[str]:
    role = role.lower()
    if "text" in role or "entry" in role:
        return ["type"]
    return ["click"]


def serialize(nodes: Dict[str, Any]) -> Dict[str, Any]:
    apps: Dict[str, Dict] = {}

    for node_id in sorted(nodes.keys()):
        node = nodes[node_id]
        try:
            if not _is_interactive(node):
                continue

            app_obj = node.getApplication()
            app_name = app_obj.name.lower() if app_obj else "unknown"

            role = node.getRoleName()
            name = node.name or ""

            if app_name not in apps:
                apps[app_name] = {
                    "app": app_name,
                    "controls": [],
                }

            apps[app_name]["controls"].append({
                "id": node_id,
                "role": role,
                "label": name,
                "actions": _allowed_actions(role),
            })

        except Exception:
            continue

    return {
        "version": "ESS-1.0",
        "snapshot_id": str(uuid.uuid4()),
        "timestamp": time.time(),
        "applications": list(apps.values()),
    }
