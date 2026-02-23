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

_REQUIRED_KEYS = {
    "version",
    "execution_mode",
    "automation_active",
    "restore_required",
    "last_snapshot_id",
    "dirty",
    "updated_at",
}

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
    - fsync on file + directory (where supported)
    - No temp-file collision
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
        Any corruption or schema mismatch → safe defaults.
        """
        try:
            if not os.path.exists(self._state_path):
                return dict(_DEFAULT_STATE)

            with open(self._state_path, "r", encoding="utf-8") as f:
                state = json.load(f)

            if (
                not isinstance(state, dict)
                or state.get("version") != _AUTH_STATE_VERSION
                or not _REQUIRED_KEYS.issubset(state.keys())
            ):
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
        thompson_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Persist authority state atomically.

        FIX H-17: Optional thompson_state persists BeliefState counters so
        Thompson sampling is reproducible across process restarts.

        thompson_state dict schema:
          {
            "iteration_counter": int,   # belief._iteration_counter
            "sample_counter":    int,   # belief._sample_counter
            "commitment_hash":   str,   # belief.commitment_hash (hex)
          }

        Usage in main.py (before the operate_main call):
          auth_state.persist(
              ...,
              thompson_state={
                  "iteration_counter": belief._iteration_counter,
                  "sample_counter":    belief._sample_counter,
                  "commitment_hash":   belief.commitment_hash,
              },
          )

        On restore (after auth_state.load()):
          ts = persisted.get("thompson_state") or {}
          belief._iteration_counter = ts.get("iteration_counter", 0)
          belief._sample_counter    = ts.get("sample_counter", 0)
          if "commitment_hash" in ts:
              belief.commitment_hash = ts["commitment_hash"]
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
        if thompson_state is not None:
            state["thompson_state"] = {
                "iteration_counter": int(thompson_state.get("iteration_counter", 0)),
                "sample_counter":    int(thompson_state.get("sample_counter", 0)),
                "commitment_hash":   str(thompson_state.get("commitment_hash", "")),
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
    # Internal — Atomic & Durable
    # --------------------------------------------------

    def _atomic_write(self, state: Dict[str, Any]) -> None:
        directory = os.path.dirname(self._state_path) or "."
        os.makedirs(directory, exist_ok=True)

        tmp_name = (
            f".auth_state_"
            f"{os.getpid()}_"
            f"{threading.get_ident()}_"
            f"{time.time_ns()}.tmp"
        )
        tmp_path = os.path.join(directory, tmp_name)

        # --- write temp file ---
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        # --- atomic replace ---
        os.replace(tmp_path, self._state_path)

        # --- fsync directory (POSIX only, best-effort) ---
        try:
            dir_fd = os.open(directory, os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except Exception:
            # Windows / restricted FS — safe to ignore
            pass
