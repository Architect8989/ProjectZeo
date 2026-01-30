# core/memory/playbook_store.py

import os
import json
import hashlib
from typing import List, Dict

PLAYBOOK_DIR = "memory/playbooks"


def _ensure_dir():
    os.makedirs(PLAYBOOK_DIR, exist_ok=True)


def _hash_intent(intent: str) -> str:
    return hashlib.sha256(intent.lower().encode()).hexdigest()


def save_playbook(intent: str, actions: List[Dict]):
    _ensure_dir()
    key = _hash_intent(intent)
    path = os.path.join(PLAYBOOK_DIR, f"{key}.json")

    with open(path, "w") as f:
        json.dump(actions, f, indent=2)


def load_playbook(intent: str):
    key = _hash_intent(intent)
    path = os.path.join(PLAYBOOK_DIR, f"{key}.json")

    if not os.path.exists(path):
        return None

    with open(path) as f:
        return json.load(f)
