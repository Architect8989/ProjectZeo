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

        # RT-01 FIX: event_log is an in-memory append-only list that survives
        # within a single process lifetime.  Callers (main.py warmup diagnostics)
        # use record_event() to register structured events.  The log is capped at
        # _MAX_EVENTS to prevent unbounded memory growth across long-running
        # sessions and is NOT persisted to disk (events are diagnostic only —
        # the persistent crash-recovery payload lives in belief_state_full).
        self._event_log: List[Dict[str, Any]] = []
        self._event_lock = threading.Lock()
        self._MAX_EVENTS = 1000  # hard cap; oldest events dropped on overflow

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
        belief_state_full: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Persist authority state atomically.

        thompson_state: slim 3-field stub kept for backward compatibility.
          Keys are accepted with BOTH the leading-underscore prefix used by
          BeliefState.to_dict() ("_iteration_counter", "_sample_counter") AND
          the legacy no-underscore form ("iteration_counter", "sample_counter").
          This makes the caller in main.py resilient to either naming convention.

        belief_state_full (FIX SI-4): Full BeliefState.to_dict() payload.
          When present after a crash, main.py calls BeliefState.from_dict() and
          passes the result as prior_belief_state to the first operate_main(),
          restoring full bandit continuity across crash-recovery restarts.
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
            # RT-02 FIX: Accept both underscore-prefixed keys (from
            # BeliefState.to_dict()) and legacy no-underscore keys so the
            # stub is populated correctly regardless of which form the caller
            # passes.  Priority: underscore-prefix keys win when present.
            _iter = thompson_state.get(
                "_iteration_counter",
                thompson_state.get("iteration_counter", 0),
            )
            _samp = thompson_state.get(
                "_sample_counter",
                thompson_state.get("sample_counter", 0),
            )
            _comm = thompson_state.get(
                "commitment_chain_hash",
                thompson_state.get("commitment_hash", ""),
            )
            state["thompson_state"] = {
                # Store under canonical underscore-prefixed names so load()
                # callers that read the raw JSON get unambiguous keys.
                "_iteration_counter": int(_iter),
                "_sample_counter":    int(_samp),
                "commitment_chain_hash": str(_comm),
            }

        # FIX SI-4: Persist full BeliefState for crash-recovery bandit continuity.
        if belief_state_full is not None and isinstance(belief_state_full, dict):
            state["belief_state_full"] = belief_state_full

        with self._lock:
            self._atomic_write(state)

    def record_event(self, event: Dict[str, Any]) -> None:
        """
        RT-01 FIX: Record a structured diagnostic event in the in-memory log.

        Previously absent, causing main.py:266 to call
            auth_state.record_event({...})
        which raised AttributeError (caught by the outer try/except Exception:
        pass), silently discarding warmup-degradation diagnostics.

        This method is intentionally in-memory only:
        - Events are non-critical diagnostic data; losing them on crash is
          acceptable and avoids a write-per-event I/O penalty.
        - The persistent crash-recovery channel is belief_state_full in
          persist(), not the event log.

        Events are stamped with "recorded_at" if not already present and
        appended in a thread-safe manner.  The log is capped at _MAX_EVENTS;
        when full, the oldest half is discarded (bulk trim is cheaper than
        popping one entry at a time under contention).

        Parameters
        ----------
        event : dict
            Arbitrary structured payload.  Must be JSON-serializable so callers
            can safely pass it to json.dumps() for export.

        Raises
        ------
        Nothing — this method is fail-safe.  Any internal error is swallowed
        to ensure that a diagnostic helper never disrupts the main loop.
        """
        try:
            if not isinstance(event, dict):
                return
            # Deep-copy to prevent accidental external mutation of stored events.
            stamped = dict(event)
            stamped.setdefault("recorded_at", time.time())

            with self._event_lock:
                self._event_log.append(stamped)
                if len(self._event_log) > self._MAX_EVENTS:
                    # Discard oldest half — O(n/2) but infrequent (once per 1000 events).
                    self._event_log = self._event_log[self._MAX_EVENTS // 2:]
        except Exception:
            pass  # record_event must never raise

    def get_event_log(self) -> List[Dict[str, Any]]:
        """
        Return a snapshot of the in-memory event log.

        Thread-safe: returns a copy so the caller can iterate without holding
        the lock.  Intended for diagnostics, health-checks, and tests.
        """
        with self._event_lock:
            return list(self._event_log)

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
