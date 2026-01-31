# core/safety/checkpoint_store.py

import json
import os
import tempfile
import hashlib
from typing import Optional, Dict

CHECKPOINT_DIR = "memory"
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "kernel_checkpoint.json")
CHECKPOINT_TMP = os.path.join(CHECKPOINT_DIR, "kernel_checkpoint.tmp")


# -------------------------------------------------
# INTERNAL
# -------------------------------------------------

def _ensure_dir():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def _checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# -------------------------------------------------
# PUBLIC API
# -------------------------------------------------

def save_checkpoint(state: Dict):
    """
    Atomic, crash-safe checkpoint write.

    - Write temp
    - fsync
    - Rename over target
    """

    _ensure_dir()

    payload = {
        "checksum": None,
        "state": state,
    }

    raw = json.dumps(payload["state"]).encode("utf-8")
    payload["checksum"] = _checksum(raw)

    serialized = json.dumps(payload)

    with open(CHECKPOINT_TMP, "w") as f:
        f.write(serialized)
        f.flush()
        os.fsync(f.fileno())

    os.replace(CHECKPOINT_TMP, CHECKPOINT_FILE)


def load_checkpoint() -> Optional[Dict]:
    """
    Loads checkpoint only if checksum valid.
    """

    if not os.path.exists(CHECKPOINT_FILE):
        return None

    try:
        with open(CHECKPOINT_FILE) as f:
            payload = json.load(f)

        raw = json.dumps(payload["state"]).encode("utf-8")
        if payload.get("checksum") != _checksum(raw):
            raise RuntimeError("Checksum mismatch")

        return payload["state"]

    except Exception:
        return None


def clear_checkpoint():
    """
    Best-effort deletion.
    """
    try:
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
    except Exception:
        pass
