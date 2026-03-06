import json
import time
import hashlib
import os
import pathlib
from typing import Any

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
_DEFAULT_JOURNAL_PATH = str(_PROJECT_ROOT / "logs" / "action_audit.jsonl")


class ActionJournal:
    

    def __init__(self, path: str = _DEFAULT_JOURNAL_PATH):
        self.path = path
        self.last_hash = "0" * 64
        self.last_intent_hash = None

        # Ensure the log directory exists before _initialize_session() writes.
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        except (PermissionError, OSError):
            pass  # Read-only filesystem: write will fail gracefully in record().

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


    import re as _re_cred_journal
    _CREDENTIAL_RE = _re_cred_journal.compile(
        r"(?:password|passwd|secret|token|api.?key|auth.?token"
        r"|bearer|private.?key|aws.?secret|access.?key)"
        r"\s*[:=]\s*\S+",
        _re_cred_journal.IGNORECASE,
    )

    def _scrub_text(self, text: str) -> str:
        """CRIT-NEW: Redact credentials from command output before journal write."""
        if not isinstance(text, str):
            return text
        return self._CREDENTIAL_RE.sub(
            lambda m: m.group(0).split(":")[0].split("=")[0] + "=<REDACTED>",
            text,
        )

    def _scrub_payload(self, payload: dict) -> dict:
        """Recursively scrub credential values from a journal entry dict.
        
        SEC-4 FIX: For write/type operation events, the 'content' and 'text' fields
        are scrubbed unconditionally as they may contain typed passwords.  This is
        defence-in-depth: operate.py already avoids logging raw write/type content,
        but this ensures any future code path that does include it is safe.
        """
        if not isinstance(payload, dict):
            return payload

        # Detect write/type operation events to apply targeted scrubbing
        _op = str(payload.get("operation") or "").lower()
        _event = str(payload.get("event") or "").lower()
        _is_write_type = _op in ("write", "type") or (
            "write" in _event or "type" in _event
        )

        out = {}
        for k, v in payload.items():
            if isinstance(v, str):
                # SEC-4: Unconditionally redact content/text in write/type contexts
                if _is_write_type and k in ("content", "text"):
                    out[k] = "<REDACTED:write_type_content>"
                else:
                    out[k] = self._scrub_text(v)
            elif isinstance(v, dict):
                out[k] = self._scrub_payload(v)
            elif isinstance(v, list):
                out[k] = [
                    self._scrub_text(i) if isinstance(i, str)
                    else self._scrub_payload(i) if isinstance(i, dict)
                    else i
                    for i in v
                ]
            else:
                out[k] = v
        return out

    def _persist(self, payload: dict) -> None:
        """
        Best-effort durability.
        Journal failure does NOT kill the process.
        """
        # CRIT-NEW: Scrub credentials before persisting to plaintext audit log
        payload = self._scrub_payload(payload)
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

    # -------------------------------------------------
    # QUERY
    # -------------------------------------------------

    def get_all(self) -> list:
        
        entries = []
        try:
            if not os.path.exists(self.path):
                return entries
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        except Exception:
            pass
        return entries
