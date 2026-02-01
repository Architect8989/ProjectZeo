import os
import time
import json
from typing import Optional

RESTART_DIR = "memory"
RESTART_FILE = os.path.join(RESTART_DIR, "restart_guard.json")

MAX_RESTARTS = 5
WINDOW_SECONDS = 300  # 5 minutes


# -------------------------------------------------
# INTERNAL
# -------------------------------------------------

def _ensure_dir():
    os.makedirs(RESTART_DIR, exist_ok=True)


def _now():
    return time.time()


# -------------------------------------------------
# PUBLIC API
# -------------------------------------------------

def record_restart():
    """
    Record a kernel restart attempt.
    Does NOT enforce policy.
    """

    _ensure_dir()

    now = _now()
    data = load_restart_state()

    if not data:
        data = {
            "first_ts": now,
            "count": 0,
        }

    # Reset window if expired
    if now - data["first_ts"] > WINDOW_SECONDS:
        data = {
            "first_ts": now,
            "count": 0,
        }

    data["count"] += 1

    _atomic_write(data)


def load_restart_state() -> Optional[dict]:
    if not os.path.exists(RESTART_FILE):
        return None

    try:
        with open(RESTART_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def clear_restart_state():
    """
    Explicit reset (used after successful stable run).
    """
    try:
        if os.path.exists(RESTART_FILE):
            os.remove(RESTART_FILE)
    except Exception:
        pass


def restart_allowed() -> bool:
    """
    Authoritative restart gate.

    Rules:
    - No state → allowed
    - Window expired → auto-clear → allowed
    - Count <= MAX → allowed
    - Count exceeded inside window → blocked
    """

    data = load_restart_state()
    if not data:
        return True

    now = _now()
    first_ts = data.get("first_ts")
    count = data.get("count", 0)

    if not first_ts:
        clear_restart_state()
        return True

    # Window expired → auto-recover
    if now - first_ts > WINDOW_SECONDS:
        clear_restart_state()
        return True

    return count <= MAX_RESTARTS


# -------------------------------------------------
# INTERNAL — SAFE WRITE
# -------------------------------------------------

def _atomic_write(data: dict):
    """
    Crash-safe atomic write.
    """
    _ensure_dir()

    tmp_path = f"{RESTART_FILE}.{os.getpid()}.{int(time.time_ns())}.tmp"

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, RESTART_FILE)
