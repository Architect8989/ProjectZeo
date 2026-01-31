import os
import json
import hashlib
import tempfile
from typing import List, Dict, Optional

PLAYBOOK_DIR = "memory/playbooks"


# -------------------------------------------------

def _ensure_dir():
    os.makedirs(PLAYBOOK_DIR, exist_ok=True)


def _hash_intent(intent: str) -> str:
    return hashlib.sha256(intent.lower().encode("utf-8")).hexdigest()


def _compute_checksum(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# -------------------------------------------------
# Public API
# -------------------------------------------------

def save_playbook(intent: str, actions: List[Dict]) -> None:
    _ensure_dir()

    key = _hash_intent(intent)
    path = os.path.join(PLAYBOOK_DIR, f"{key}.json")
    tmp_fd, tmp_path = tempfile.mkstemp(dir=PLAYBOOK_DIR)

    try:
        payload = {
            "schema": 1,
            "intent": intent,
            "actions": actions,
        }

        raw = json.dumps(payload, indent=2)
        checksum = _compute_checksum(raw)

        wrapper = {
            "checksum": checksum,
            "payload": payload,
        }

        with os.fdopen(tmp_fd, "w") as f:
            json.dump(wrapper, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, path)

    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass


def load_playbook(intent: str) -> Optional[List[Dict]]:
    key = _hash_intent(intent)
    path = os.path.join(PLAYBOOK_DIR, f"{key}.json")

    if not os.path.exists(path):
        return None

    try:
        with open(path, "r") as f:
            wrapper = json.load(f)

        checksum = wrapper.get("checksum")
        payload = wrapper.get("payload")

        if not checksum or not payload:
            return None

        raw = json.dumps(payload, indent=2)
        if _compute_checksum(raw) != checksum:
            return None

        return payload.get("actions")

    except Exception:
        return None
