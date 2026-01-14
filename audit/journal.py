import json
import time
import os
import sys

class ActionJournal:
    def __init__(self, path="action_audit.jsonl"):
        self.path = path
        self.index = 0
        self._initialize_log()

    def _initialize_log(self):
        try:
            with open(self.path, "a") as f:
                f.write(f"\n--- SESSION START: {time.ctime()} ---\n")
        except Exception as e:
            print(f"[CRITICAL AUDIT FAILURE] Journal initialization failed: {e}")
            sys.exit(1)

    def record(self, entry: dict):
        self.index += 1
        record_payload = {
            "index": self.index,
            "wall_time": time.time(),
            "monotonic_time": time.monotonic(),
            **entry
        }
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(record_payload) + "\n")
                f.flush()
                os.fsync(f.fileno()) # Guarantee physical persistence
        except Exception as e:
            print(f"[CRITICAL AUDIT FAILURE] Journal write failed: {e}")
            sys.exit(1)

    def seal(self, reason="NORMAL"):
        try:
            with open(self.path, "a") as f:
                f.write(f"--- SESSION SEALED ({reason}): {time.ctime()} ---\n")
        except:
            pass
