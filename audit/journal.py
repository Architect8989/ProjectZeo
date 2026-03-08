"""
audit/journal.py — Tamper-evident action journal with credential scrubbing.

HIGH-2 FIX (March 2026):
  _scrub_payload() previously only scrubbed key=value patterns from strings.
  Shell CLI args like `curl -u admin:pass` or `--token SECRET` were stored
  plaintext in the journal.

  Fix 1: Extended _CREDENTIAL_RE to catch CLI flag patterns:
    --password <value>, -p <value>, -u user:pass, --token <value>,
    Authorization: Bearer <token>, https://user:pass@host URLs.

  Fix 2: Added _scrub_command_string() for shell command fields.
    Applied in _scrub_payload() for all keys in _COMMAND_FIELD_NAMES:
    'command', 'cmd', 'args', 'action_command', 'install_command', etc.

  Fix 3: List items in command-like fields are individually scrubbed
    (e.g. install_commands: ["pip install x --index-url https://user:pass@..."])
"""
import json
import time
import hashlib
import os
import re
import pathlib
from typing import Any

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
_DEFAULT_JOURNAL_PATH = str(_PROJECT_ROOT / "logs" / "action_audit.jsonl")

# ---------------------------------------------------------------------------
# Credential scrubbing regexes
# ---------------------------------------------------------------------------

# Pattern 1: key=value and key:value forms
_KV_CRED_RE = re.compile(
    r"(?:password|passwd|secret|token|api.?key|auth.?token"
    r"|bearer|private.?key|aws.?secret|access.?key"
    r"|database.?url|db.?password|connection.?string"
    r"|encryption.?key|signing.?key|client.?secret"
    r"|x.?api.?key|authorization|api_token"
    r")\s*[:=]\s*\S+",
    re.IGNORECASE,
)

# Pattern 2: CLI flag patterns (HIGH-2 FIX)
_CLI_CRED_RE = re.compile(
    r"(?:"
    r"--(?:password|passwd|secret|token|api-?key|auth-?token|bearer"
    r"|private-?key|aws-?secret|access-?key|client-?secret"
    r"|api-?token|auth|credential|credentials)\s+\S+"
    r"|(?<!\w)-(?:p|P|w|W|k)\s+\S+"
    r"|(?:https?://)[^@\s]+:[^@\s]+@\S+"
    r"|Authorization:\s*(?:Bearer|Basic|Token)\s+\S+"
    r"|(?<!\w)-u\s+[^:\s]+:[^\s]+"
    r")",
    re.IGNORECASE,
)

# Pattern 3: AWS access key IDs
_AWS_KEY_RE = re.compile(r"\b(AKIA[A-Z0-9]{16})\b")

# Fields that are shell command strings — apply CLI scrubber too
_COMMAND_FIELD_NAMES: frozenset = frozenset({
    "command", "cmd", "args", "action_command", "install_command",
    "shell_command", "exec_command", "run_command", "command_string",
    "full_cmd", "last_command",
})


class ActionJournal:
    """Tamper-evident, credential-scrubbed, append-only action journal."""

    def __init__(self, path: str = _DEFAULT_JOURNAL_PATH):
        self.path = path
        self.last_hash = "0" * 64
        self.last_intent_hash = None
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        except (PermissionError, OSError):
            pass
        try:
            self._initialize_session()
        except Exception as e:
            raise RuntimeError(f"JOURNAL_INITIALIZATION_FAILURE: {e}") from e

    # =========================================================================
    # Credential scrubbing
    # =========================================================================

    def _scrub_text(self, text: str) -> str:
        """Scrub key=value and key:value credential patterns."""
        if not isinstance(text, str):
            return text
        text = _KV_CRED_RE.sub(
            lambda m: m.group(0).split(":")[0].split("=")[0] + "=<REDACTED>",
            text,
        )
        text = _AWS_KEY_RE.sub("<REDACTED:AWS_KEY>", text)
        return text

    def _scrub_command_string(self, cmd: str) -> str:
        """HIGH-2 FIX: Scrub CLI credential patterns from shell command strings."""
        if not isinstance(cmd, str):
            return cmd
        cmd = self._scrub_text(cmd)
        cmd = _CLI_CRED_RE.sub("<REDACTED:CLI_CREDENTIAL>", cmd)
        return cmd

    def _scrub_payload(self, payload: dict) -> dict:
        """
        Recursively scrub credentials from a journal entry dict.

        HIGH-2 FIX: Command-string fields in _COMMAND_FIELD_NAMES now have
        the CLI credential scrubber applied. Previously only KV patterns
        were caught; CLI args like `curl -u user:pass` were stored plaintext.
        """
        if not isinstance(payload, dict):
            return payload

        _op = str(payload.get("operation") or "").lower()
        _event = str(payload.get("event") or "").lower()
        _is_write_type = _op in ("write", "type") or (
            "write" in _event or "type" in _event
        )

        out = {}
        for k, v in payload.items():
            if isinstance(v, str):
                if _is_write_type and k in ("content", "text"):
                    out[k] = "<REDACTED:write_type_content>"
                elif k in _COMMAND_FIELD_NAMES:
                    out[k] = self._scrub_command_string(v)
                else:
                    out[k] = self._scrub_text(v)
            elif isinstance(v, dict):
                out[k] = self._scrub_payload(v)
            elif isinstance(v, list):
                scrubbed = []
                for item in v:
                    if isinstance(item, str):
                        scrubbed.append(self._scrub_command_string(item))
                    elif isinstance(item, dict):
                        scrubbed.append(self._scrub_payload(item))
                    else:
                        scrubbed.append(item)
                out[k] = scrubbed
            else:
                out[k] = v
        return out

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _canonicalize(self, payload: dict) -> str:
        try:
            return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        except Exception as e:
            raise RuntimeError(f"AUDIT_INTEGRITY_FAILURE: canonicalization failed: {e}") from e

    def _hash(self, payload: dict) -> str:
        return hashlib.sha256(self._canonicalize(payload).encode("utf-8")).hexdigest()

    def _persist(self, payload: dict) -> None:
        payload = self._scrub_payload(payload)
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, sort_keys=True) + "\n")
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
        except Exception as e:
            raise RuntimeError(f"AUDIT_PERSISTENCE_FAILURE: {e}") from e

    def _now(self) -> dict:
        return {"timestamp_wall": time.time(), "timestamp_mono": time.monotonic()}

    # =========================================================================
    # Session management
    # =========================================================================

    def _initialize_session(self) -> None:
        self._record_internal({"type": "SESSION_START", **self._now()})

    # =========================================================================
    # Public API
    # =========================================================================

    def record(self, entry: dict) -> None:
        phase      = entry.get("phase")
        entry_type = entry.get("type")

        if entry_type == "SESSION_SEAL" and self.last_intent_hash is not None:
            self._force_seal_intent("implicit recovery")

        if phase == "INTENT":
            if self.last_intent_hash is not None:
                raise RuntimeError("AUDIT_INTEGRITY_FAILURE: INTENT already active")

        if phase == "EFFECT":
            if self.last_intent_hash is None:
                raise RuntimeError("AUDIT_INTEGRITY_FAILURE: EFFECT without INTENT")
            entry["intent_ref"]   = self.last_intent_hash
            self.last_intent_hash = None

        self._record_internal(entry)

    def seal(self, reason: str = "NORMAL") -> None:
        self.record({"type": "SESSION_SEAL", "reason": reason, **self._now()})

    # =========================================================================
    # Internal record + recovery
    # =========================================================================

    def _record_internal(self, entry: dict) -> None:
        entry["prev_hash"] = self.last_hash
        current_hash       = self._hash(entry)
        entry["hash"]      = current_hash
        if entry.get("phase") == "INTENT":
            self.last_intent_hash = current_hash
        self.last_hash = current_hash
        self._persist(entry)

    def _force_seal_intent(self, reason: str) -> None:
        self.last_intent_hash = None
        self._record_internal({"type": "INTENT_ABORT", "reason": reason, **self._now()})

    # =========================================================================
    # Query
    # =========================================================================

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
