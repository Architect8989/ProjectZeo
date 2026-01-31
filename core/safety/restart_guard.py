# core/safety/restart_guard.py

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
    Call at kernel boot.
    """

    _ensure_dir()

    data = load_restart_state() or {
        "first_ts": _now(),
        "count": 0,
    }

    now = _now()

    # Reset window
    if now - data["first_ts"] > WINDOW_SECONDS:
        data = {
            "first_ts": now,
            "count": 0,
        }

    data["count"] += 1

    with open(RESTART_FILE, "w") as f:
        json.dump(data, f)


def load_restart_state() -> Optional[dict]:
    if not os.path.exists(RESTART_FILE):
        return None

    try:
        with open(RESTART_FILE) as f:
            return json.load(f)
    except Exception:
        return None


def clear_restart_state():
    try:
        if os.path.exists(RESTART_FILE):
            os.remove(RESTART_FILE)
    except Exception:
        pass


def restart_allowed() -> bool:
    data = load_restart_state()
    if not data:
        return True

    return data.get("count", 0) <= MAX_RESTARTS
