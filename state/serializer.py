import time
import uuid
import json
import os
import threading
from typing import Dict, Any, List, Optional

# ============================================================
# EXISTING LOGIC — UNCHANGED
# ============================================================

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


def serialize(
    nodes: Dict[str, Any],
    snapshot_id: str = None,
    timestamp: float = None,
) -> Dict[str, Any]:
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

            apps[app_name]["controls"].append(
                {
                    "id": node_id,
                    "role": role,
                    "label": name,
                    "actions": _allowed_actions(role),
                }
            )

        except Exception:
            continue

    return {
        "version": "ESS-1.0",
        "snapshot_id": snapshot_id or str(uuid.uuid4()),
        "timestamp": timestamp or time.time(),
        "applications": list(apps.values()),
    }


# ============================================================
# AUTHORITY STATE — HARDENED
# ============================================================

_AUTH_STATE_VERSION = "AUTH-STATE-1"

_DEFAULT_STATE = {
    "version": _AUTH_STATE_VERSION,
    "execution_mode": "OBSERVER",
    "automation_active": False,
    "restore_required": False,
    "last_snapshot_id": None,
    "dirty": False,
    "updated_at": None,
}


class AuthorityStateError(RuntimeError):
    pass


class AuthorityStateSerializer:
    """
    Crash-proof authority state persistence.

    GUARANTEES:
    - Atomic replace
    - No temp-file collision
    - No TOCTOU window
    - No cleanup race
    - Safe under concurrency
    """

    def __init__(self, state_path: str):
        self._state_path = state_path
        self._lock = threading.Lock()

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def load(self) -> Dict[str, Any]:
        """
        Load persisted authority state.
        Corruption or mismatch → safe defaults.
        """
        try:
            if not os.path.exists(self._state_path):
                return dict(_DEFAULT_STATE)

            with open(self._state_path, "r", encoding="utf-8") as f:
                state = json.load(f)

            if state.get("version") != _AUTH_STATE_VERSION:
                return dict(_DEFAULT_STATE)

            return state

        except Exception:
            return dict(_DEFAULT_STATE)

    def persist(
        self,
        *,
        execution_mode: str,
        automation_active: bool,
        restore_required: bool,
        last_snapshot_id: Optional[str],
        dirty: bool,
    ) -> None:
        """
        Persist authority state atomically.
        """
        state = {
            "version": _AUTH_STATE_VERSION,
            "execution_mode": execution_mode,
            "automation_active": bool(automation_active),
            "restore_required": bool(restore_required),
            "last_snapshot_id": last_snapshot_id,
            "dirty": bool(dirty),
            "updated_at": time.time(),
        }

        with self._lock:
            self._atomic_write(state)

    def force_safe_state(self) -> None:
        """
        Force pessimistic recovery state.
        """
        self.persist(
            execution_mode="OBSERVER",
            automation_active=False,
            restore_required=True,
            last_snapshot_id=None,
            dirty=True,
        )

    # --------------------------------------------------
    # Internal — TOCTOU SAFE
    # --------------------------------------------------

    def _atomic_write(self, state: Dict[str, Any]) -> None:
        """
        Atomic write using:
        - unique temp filename
        - fsync
        - os.replace
        """
        directory = os.path.dirname(self._state_path) or "."
        os.makedirs(directory, exist_ok=True)

        tmp_name = (
            f".auth_state_"
            f"{os.getpid()}_"
            f"{threading.get_ident()}_"
            f"{time.time_ns()}.tmp"
        )
        tmp_path = os.path.join(directory, tmp_name)

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        # Atomic on POSIX + Windows
        os.replace(tmp_path, self._state_path)
