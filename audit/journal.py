import json
import time
import hashlib
import os


class ActionJournal:
    """
    CRYPTOGRAPHIC EXECUTION LEDGER.
    Non-repudiable evidence chain.
    Fail-closed on all integrity violations.
    """

    def __init__(self, path="action_audit.jsonl"):
        self.path = path
        self.last_hash = "0" * 64  # Genesis state
        self.last_intent_hash = None

        # Initialize session ONLY after object is fully constructed
        try:
            self._initialize_session()
        except Exception as e:
            raise RuntimeError(
                f"JOURNAL_INITIALIZATION_FAILURE: {e}"
            ) from e

    # -------------------------------------------------

    def _canonical_hash(self, payload: dict) -> str:
        """Computes SHA-256 over deterministic JSON representation."""
        try:
            serialized = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            )
            return hashlib.sha256(serialized.encode()).hexdigest()
        except Exception as e:
            raise RuntimeError(
                f"AUDIT_INTEGRITY_FAILURE: Serialization error: {e}"
            ) from e

    def _persist(self, payload: dict) -> None:
        """
        Atomic write-and-sync.
        Failure here MUST terminate the process.
        """
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(payload, sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            # Unrecorded execution is a security breach
            raise SystemExit(
                f"CRITICAL_AUDIT_FAILURE: Persistence failed: {e}"
            ) from e

    # -------------------------------------------------

    def _initialize_session(self) -> None:
        """
        Starts the hash chain.
        No chaining data accepted from callers.
        """
        entry = {
            "type": "SESSION_START",
            "timestamp": time.time(),
        }
        self.record(entry)

    # -------------------------------------------------

    def record(self, entry: dict) -> None:
        """
        Appends evidence.

        Enforces:
        1. Hash chaining
        2. Intent → Effect binding
        3. No dangling intent
        """

        phase = entry.get("phase")
        entry_type = entry.get("type")

        # 1. No new INTENT or SEAL while EFFECT pending
        if (
            (phase == "INTENT" or entry_type == "SESSION_SEAL")
            and self.last_intent_hash
        ):
            raise RuntimeError(
                "AUDIT_INTEGRITY_FAILURE: Unresolved INTENT without EFFECT."
            )

        # 2. EFFECT must reference an active INTENT
        if phase == "EFFECT":
            if self.last_intent_hash is None:
                raise RuntimeError(
                    "AUDIT_INTEGRITY_FAILURE: EFFECT without active INTENT."
                )
            entry["intent_ref"] = self.last_intent_hash
            self.last_intent_hash = None

        # 3. Ledger owns chaining
        entry["prev_hash"] = self.last_hash

        # 4. Cryptographic finalization
        current_hash = self._canonical_hash(entry)
        entry["hash"] = current_hash

        # 5. Internal state update
        if phase == "INTENT":
            self.last_intent_hash = current_hash

        self.last_hash = current_hash

        # 6. Commit (fsynced)
        self._persist(entry)

    # -------------------------------------------------

    def seal(self, reason="NORMAL") -> None:
        """
        Closes the ledger.
        Cannot succeed if an intent is unresolved.
        """
        entry = {
            "type": "SESSION_SEAL",
            "reason": reason,
            "timestamp": time.time(),
        }
        self.record(entry)
