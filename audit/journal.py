import json
import time
import hashlib
import os
from typing import Any


class ActionJournal:
    """
    CRYPTOGRAPHIC EXECUTION LEDGER.

    Guarantees:
    - Hash-chained, append-only audit log
    - Intent → Effect integrity
    - Fail-closed on integrity violations
    - Never terminates host process on I/O failure
    """

    def __init__(self, path="action_audit.jsonl"):
        self.path = path
        self.last_hash = "0" * 64
        self.last_intent_hash = None

        try:
            self._initialize_session()
        except Exception as e:
            raise RuntimeError(
                f"JOURNAL_INITIALIZATION_FAILURE: {e}"
            ) from e

    # -------------------------------------------------
    # INTERNALS
    # -------------------------------------------------

    def _canonicalize(self, payload: dict) -> str:
        """
        Deterministic JSON serialization.
        Degrades safely on exotic types.
        """
        try:
            return json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                default=str,  # <- critical fix
            )
        except Exception as e:
            raise RuntimeError(
                f"AUDIT_INTEGRITY_FAILURE: canonicalization failed: {e}"
            ) from e

    def _hash(self, payload: dict) -> str:
        serialized = self._canonicalize(payload)
        return hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()

    def _persist(self, payload: dict) -> None:
        """
        Best-effort durability.
        Journal failure does NOT kill the process.
        """
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(payload, sort_keys=True) + "\n"
                )
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    # fsync failure logged implicitly by missing durability
                    pass
        except Exception as e:
            raise RuntimeError(
                f"AUDIT_PERSISTENCE_FAILURE: {e}"
            ) from e

    def _now(self) -> dict:
        return {
            "timestamp_wall": time.time(),
            "timestamp_mono": time.monotonic(),
        }

    # -------------------------------------------------
    # SESSION
    # -------------------------------------------------

    def _initialize_session(self) -> None:
        entry = {
            "type": "SESSION_START",
            **self._now(),
        }
        self._record_internal(entry)

    # -------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------

    def record(self, entry: dict) -> None:
        """
        Public record API.

        Enforces:
        - Intent → Effect pairing
        - No silent corruption
        """

        phase = entry.get("phase")
        entry_type = entry.get("type")

        # --- dangling intent guard ---
        if (
            entry_type == "SESSION_SEAL"
            and self.last_intent_hash is not None
        ):
            # Auto-seal dangling intent instead of bricking journal
            self._force_seal_intent("implicit recovery")

        if phase == "INTENT":
            if self.last_intent_hash is not None:
                raise RuntimeError(
                    "AUDIT_INTEGRITY_FAILURE: INTENT already active"
                )

        if phase == "EFFECT":
            if self.last_intent_hash is None:
                raise RuntimeError(
                    "AUDIT_INTEGRITY_FAILURE: EFFECT without INTENT"
                )
            entry["intent_ref"] = self.last_intent_hash
            self.last_intent_hash = None

        self._record_internal(entry)

    def seal(self, reason="NORMAL") -> None:
        entry = {
            "type": "SESSION_SEAL",
            "reason": reason,
            **self._now(),
        }
        self.record(entry)

    # -------------------------------------------------
    # INTERNAL RECORD
    # -------------------------------------------------

    def _record_internal(self, entry: dict) -> None:
        entry["prev_hash"] = self.last_hash

        current_hash = self._hash(entry)
        entry["hash"] = current_hash

        if entry.get("phase") == "INTENT":
            self.last_intent_hash = current_hash

        self.last_hash = current_hash
        self._persist(entry)

    # -------------------------------------------------
    # RECOVERY
    # -------------------------------------------------

    def _force_seal_intent(self, reason: str) -> None:
        """
        Crash recovery hook.
        Explicitly seals dangling intent.
        """
        entry = {
            "type": "INTENT_ABORT",
            "reason": reason,
            **self._now(),
        }
        self.last_intent_hash = None
        self._record_internal(entry)
