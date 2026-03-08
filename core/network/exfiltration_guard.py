from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

_logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Decision constants
# ─────────────────────────────────────────────────────────────────────────────
ALLOW                      = "ALLOW"
DENY                       = "DENY"
REQUIRE_HUMAN_CONFIRMATION = "REQUIRE_HUMAN_CONFIRMATION"


# ─────────────────────────────────────────────────────────────────────────────
# Sensitive path patterns
# ─────────────────────────────────────────────────────────────────────────────

_SENSITIVE_PATHS: List[re.Pattern] = [
    re.compile(r"/etc/(?:passwd|shadow|sudoers|hosts|ssh)", re.IGNORECASE),
    re.compile(r"/root/", re.IGNORECASE),
    re.compile(r"~?/\.ssh/(?:id_|known_hosts|config)", re.IGNORECASE),
    re.compile(r"~?/\.(?:aws|gcp|azure|config)/", re.IGNORECASE),
    re.compile(r"~?/\.env(?:rc)?$|\\.env(?:rc)?\\b", re.IGNORECASE),
    re.compile(r"(?:api[_.]?key|secret|password|token|credential)", re.IGNORECASE),
    re.compile(r"/var/(?:log|db|lib)/", re.IGNORECASE),
    re.compile(r"~?/\.(?:gnupg|netrc|gitconfig|npmrc|pypirc)", re.IGNORECASE),
    re.compile(r"authorized_keys|id_rsa|id_ed25519", re.IGNORECASE),
    re.compile(r"(?:wallet|keystore|keychain|vault)", re.IGNORECASE),
]

# ─────────────────────────────────────────────────────────────────────────────
# Upload/POST patterns
# ─────────────────────────────────────────────────────────────────────────────

# curl / wget upload flags
_CURL_UPLOAD: re.Pattern = re.compile(
    r"\bcurl\b.*?(?:--data|-d|--upload-file|-T|--form|-F|--data-binary|--data-raw"
    r"|--data-urlencode|--json|--post-data)",
    re.IGNORECASE | re.DOTALL,
)
_WGET_UPLOAD: re.Pattern = re.compile(
    r"\bwget\b.*?(?:--post-data|--post-file|--method=POST)",
    re.IGNORECASE | re.DOTALL,
)

# Netcat piped output
_NC_PIPE: re.Pattern = re.compile(
    r"(?:cat\s+\S+|<\s*\S+)\s*\|\s*nc\b|\bnc\b.*?<\s*\S+",
    re.IGNORECASE,
)

# Python requests.post / http.post / urllib.request.urlopen with data
_PYTHON_POST: re.Pattern = re.compile(
    r"requests\.post|http\.post|urllib.*urlopen.*data=|httpx\.post",
    re.IGNORECASE,
)

# Node.js http POST with data
_NODE_POST: re.Pattern = re.compile(
    r"""(?:https?\.request|fetch|axios\.post|superagent\.post)""",
    re.IGNORECASE,
)

# SCP / rsync file send (source → remote)
_SCP_SEND: re.Pattern = re.compile(
    r"\bscp\b\s+\S+\s+\w+@",
    re.IGNORECASE,
)

# Base64-encoded data piped to network
_B64_PIPE: re.Pattern = re.compile(
    r"base64.*\|\s*(?:curl|nc|netcat|socat|wget)",
    re.IGNORECASE,
)

# DNS exfiltration via nslookup / dig with encoded data
_DNS_EXFIL: re.Pattern = re.compile(
    r"(?:nslookup|dig)\b.*(?:\$\(|`|base64)",
    re.IGNORECASE,
)

# git push to external (non-localhost) remote
_GIT_PUSH_EXTERNAL: re.Pattern = re.compile(
    r"\bgit\s+push\b(?!.*localhost)",
    re.IGNORECASE,
)

# All upload patterns to check (pattern, description, decision_on_match)
_UPLOAD_PATTERNS: List[Tuple[re.Pattern, str, str]] = [
    (_CURL_UPLOAD,        "curl data upload",       REQUIRE_HUMAN_CONFIRMATION),
    (_WGET_UPLOAD,        "wget data upload",        REQUIRE_HUMAN_CONFIRMATION),
    (_NC_PIPE,            "netcat pipe exfil",       DENY),
    (_PYTHON_POST,        "Python HTTP POST",        REQUIRE_HUMAN_CONFIRMATION),
    (_NODE_POST,          "Node.js HTTP POST",       REQUIRE_HUMAN_CONFIRMATION),
    (_SCP_SEND,           "SCP file send",           REQUIRE_HUMAN_CONFIRMATION),
    (_B64_PIPE,           "base64→network pipe",     DENY),
    (_DNS_EXFIL,          "DNS exfiltration",        DENY),
    (_GIT_PUSH_EXTERNAL,  "git push external",       REQUIRE_HUMAN_CONFIRMATION),
]


# ─────────────────────────────────────────────────────────────────────────────
# Guard
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExfiltrationResult:
    decision:    str                   # ALLOW | DENY | REQUIRE_HUMAN_CONFIRMATION
    reason:      str
    pattern:     str = ""
    sensitive:   bool = False


def check_command(command: str) -> ExfiltrationResult:
    """
    Inspect a shell command for data exfiltration patterns.

    Returns ExfiltrationResult with:
      decision = ALLOW | DENY | REQUIRE_HUMAN_CONFIRMATION
      reason   = human-readable explanation
      sensitive = True if sensitive paths were also detected
    """
    if not isinstance(command, str) or not command.strip():
        return ExfiltrationResult(ALLOW, "Empty command")

    cmd = command  # keep original case for path checks

    # 1. Check for upload patterns
    matched_pattern = ""
    matched_decision = ALLOW

    for regex, pattern_name, default_decision in _UPLOAD_PATTERNS:
        if regex.search(cmd):
            matched_pattern = pattern_name
            matched_decision = default_decision
            _logger.debug("[ExfilGuard] Upload pattern matched: %s", pattern_name)
            break

    if matched_decision == ALLOW:
        # No upload pattern — allow
        return ExfiltrationResult(ALLOW, "No upload pattern detected")

    # 2. Check if sensitive paths are involved
    is_sensitive = any(pat.search(cmd) for pat in _SENSITIVE_PATHS)

    if is_sensitive:
        # Sensitive path + upload → always DENY
        reason = (
            f"Data exfiltration blocked: {matched_pattern} "
            f"combined with sensitive path/data reference."
        )
        _logger.warning("[ExfilGuard] DENY: %s | cmd=%r", reason, cmd[:100])
        return ExfiltrationResult(DENY, reason, matched_pattern, sensitive=True)

    # 3. Non-sensitive upload → escalate to human
    if matched_decision == DENY:
        reason = (
            f"High-risk exfiltration pattern: {matched_pattern}. "
            f"Command blocked — likely data exfiltration."
        )
        _logger.warning("[ExfilGuard] DENY: %s | cmd=%r", reason, cmd[:100])
        return ExfiltrationResult(DENY, reason, matched_pattern, sensitive=False)

    reason = (
        f"Network upload detected ({matched_pattern}). "
        f"Requires human confirmation before sending data."
    )
    _logger.info("[ExfilGuard] REQUIRE_HUMAN_CONFIRMATION: %s | cmd=%r",
                 reason[:100], cmd[:80])
    return ExfiltrationResult(
        REQUIRE_HUMAN_CONFIRMATION, reason, matched_pattern, sensitive=False
    )


def check_action(action: dict) -> ExfiltrationResult:
    """
    Inspect a GII action dict for exfiltration.
    Checks command field and any content fields.
    """
    if not isinstance(action, dict):
        return ExfiltrationResult(ALLOW, "Not an action dict")

    op = str(action.get("operation", "")).lower()
    if op not in ("command", "file_create", "install"):
        return ExfiltrationResult(ALLOW, f"Operation {op!r} is not a network concern")

    command = str(action.get("command", ""))
    if command:
        return check_command(command)

    # For file_create, check if content looks like it's being staged for exfil
    content = str(action.get("content", ""))
    if content:
        is_sensitive = any(pat.search(content) for pat in _SENSITIVE_PATHS)
        if is_sensitive:
            _logger.warning(
                "[ExfilGuard] file_create with sensitive content — flagging."
            )
            return ExfiltrationResult(
                REQUIRE_HUMAN_CONFIRMATION,
                "file_create contains potentially sensitive content",
            )

    return ExfiltrationResult(ALLOW, "No exfiltration pattern found")
