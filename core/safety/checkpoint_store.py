# core/safety/checkpoint_store.py

import json
import os

CHECKPOINT_FILE = "memory/kernel_checkpoint.json"


def save_checkpoint(state: dict):
    os.makedirs("memory", exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(state, f)


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return None

    with open(CHECKPOINT_FILE) as f:
        return json.load(f)


def clear_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
