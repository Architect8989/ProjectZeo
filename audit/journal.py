import json
import time
import hashlib
import os

class ActionJournal:
    """
    CRYPTOGRAPHIC EXECUTION LEDGER.
    Non-repudiable evidence chain. Fail-closed on all integrity violations.
    """
    def __init__(self, path="action_audit.jsonl"):
        self.path = path
        self.last_hash = "0" * 64  # Genesis state
        self.last_intent_hash = None
        self._initialize_session()

    def _canonical_hash(self, payload: dict) -> str:
        """Computes SHA-256 over deterministic JSON representation."""
        try:
            # Enforce stable key order and separators for non-repudiation
            serialized = json.dumps(payload, sort_keys=True, separators=(',', ':'))
            return hashlib.sha256(serialized.encode()).hexdigest()
        except Exception as e:
            raise RuntimeError(f"AUDIT_INTEGRITY_FAILURE: Serialization error: {e}")

    def _persist(self, payload: dict):
        """Atomic write-and-sync. Failure here must terminate the process."""
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(payload, sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            # Crash the system. Unrecorded execution is a security breach.
            raise SystemExit(f"CRITICAL_AUDIT_FAILURE: Persistence failed: {e}")

    def _initialize_session(self):
        """Starts the hash chain. Callers do not provide chaining data."""
        entry = {
            "type": "SESSION_START",
            "timestamp": time.time()
        }
        self.record(entry)

    def record(self, entry: dict):
        """
        Appends evidence. 
        Enforces (1) Hash Chaining, (2) Intent-Effect Binding, (3) No dangling Intent.
        """
        phase = entry.get("phase")
        entry_type = entry.get("type")

        # 1. Invariant: No new INTENT or SEAL while an EFFECT is pending
        if (phase == "INTENT" or entry_type == "SESSION_SEAL") and self.last_intent_hash:
            raise RuntimeError("AUDIT_INTEGRITY_FAILURE: Unresolved INTENT without EFFECT.")

        # 2. Invariant: No EFFECT without a preceding INTENT
        if phase == "EFFECT":
            if self.last_intent_hash is None:
                raise RuntimeError("AUDIT_INTEGRITY_FAILURE: EFFECT recorded without active INTENT.")
            entry["intent_ref"] = self.last_intent_hash
            self.last_intent_hash = None # Resolve intent

        # 3. Ownership: Ledger alone controls the chain
        entry["prev_hash"] = self.last_hash
        
        # 4. Finalize cryptography
        current_hash = self._canonical_hash(entry)
        entry["hash"] = current_hash
        
        # 5. Internal state updates
        if phase == "INTENT":
            self.last_intent_hash = current_hash
            
        self.last_hash = current_hash
        
        # 6. Commit
        self._persist(entry)

    def seal(self, reason="NORMAL"):
        """Closes the ledger. Cannot succeed if an intent is unresolved."""
        entry = {
            "type": "SESSION_SEAL",
            "reason": reason,
            "timestamp": time.time()
        }
        self.record(entry)
