# core/safety/checkpoint_store.py

import json
import os
import tempfile
import hashlib
from typing import Optional, Dict, Any


import pathlib as _pathlib
CHECKPOINT_DIR = str(_pathlib.Path(__file__).resolve().parents[2] / "memory")
del _pathlib
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "kernel_checkpoint.json")


# -------------------------------------------------
# INTERNAL
# -------------------------------------------------

def _ensure_dir() -> None:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def _stable_json_bytes(obj: Any) -> bytes:
    """
    Deterministic JSON serialization.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fsync_dir(path: str) -> None:
    """
    Ensure directory metadata is flushed (POSIX durability).
    """
    try:
        dir_fd = os.open(path, os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        # Non-POSIX systems may not support O_DIRECTORY
        pass


# -------------------------------------------------
# PUBLIC API
# -------------------------------------------------

def save_checkpoint(state: Dict) -> None:
    
    if not isinstance(state, dict):
        raise TypeError("Checkpoint state must be dict")

    _ensure_dir()

    state_bytes = _stable_json_bytes(state)

    payload = {
        "checksum": _checksum(state_bytes),
        "state": state,
    }

    serialized_bytes = _stable_json_bytes(payload)

    # Create temp file in same directory (avoids cross-device rename)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=CHECKPOINT_DIR,
        delete=False,
    ) as tmp_file:

        tmp_file.write(serialized_bytes)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
        tmp_path = tmp_file.name

    # Atomic replace
    os.replace(tmp_path, CHECKPOINT_FILE)

    # Ensure directory entry durability
    _fsync_dir(CHECKPOINT_DIR)


def load_checkpoint() -> Optional[Dict]:
    

    if not os.path.exists(CHECKPOINT_FILE):
        return None

    try:
        with open(CHECKPOINT_FILE, "rb") as f:
            raw_bytes = f.read()

        payload = json.loads(raw_bytes.decode("utf-8"))

        if not isinstance(payload, dict):
            return None

        state = payload.get("state")
        checksum = payload.get("checksum")

        if not isinstance(state, dict) or not isinstance(checksum, str):
            return None

        recalculated = _checksum(_stable_json_bytes(state))

        if recalculated != checksum:
            return None

        return state

    except Exception:
        # Corrupt, partial, or invalid JSON
        return None


def clear_checkpoint() -> None:
    

    try:
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            _fsync_dir(CHECKPOINT_DIR)
    except Exception:
        pass


def checkpoint_exists() -> bool:
    
    return load_checkpoint() is not None


def get_checkpoint_step_index() -> Optional[int]:
    
    state = load_checkpoint()
    if state is None:
        return None
    idx = state.get("step_index")
    return int(idx) if isinstance(idx, (int, float)) else None


def get_checkpoint_execution_log() -> Optional[Dict[str, Any]]:
    
    state = load_checkpoint()
    if state is None:
        return None
    log = state.get("execution_log")
    return log if isinstance(log, dict) else None

