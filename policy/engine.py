from __future__ import annotations

try:
    from core.network.policy_enforcer import NetworkPolicyEnforcer as _NetworkPolicyEnforcer
    _NETWORK_ENFORCER_AVAILABLE = True
except ImportError:
    _NetworkPolicyEnforcer = None  # type: ignore
    _NETWORK_ENFORCER_AVAILABLE = False

import os
import re
import threading
import logging
import secrets
from typing import FrozenSet, Optional, Set, Tuple, List

_logger = logging.getLogger(__name__)


class PolicyViolationError(RuntimeError):
    """Raised when a caller attempts an operation that violates policy."""


# ---------------------------------------------------------------------------
# Shell metacharacter detection (SEC-NEW)
# Checked AFTER trusted-installer prefix match to block suffix injection.
# ---------------------------------------------------------------------------
_SHELL_METACHAR_RE = re.compile(
    r"""
    (?:
      ;                        # command separator
    | \|\|                     # OR-chain
    | &&                       # AND-chain
    | \|(?!\|)                 # pipe (not ||)
    | `[^`]*`                  # backtick subshell
    | \$\(                     # $() subshell
    | >[>]?                    # output redirect
    | <\(                      # process substitution
    | \beval\b                 # eval keyword
    | \bexec\s                 # exec command
    )
    """,
    re.VERBOSE,
)

# Paths that are ALWAYS denied for file_create regardless of policy.yaml
_HARDCODED_DENIED_PATHS: FrozenSet[str] = frozenset({
    "/etc/cron.d", "/etc/cron.hourly", "/etc/cron.daily",
    "/etc/cron.weekly", "/etc/cron.monthly", "/etc/crontab",
    "/etc/passwd", "/etc/shadow", "/etc/sudoers", "/etc/sudoers.d",
    "/etc/hosts", "/etc/hostname", "/etc/fstab",
    "/etc/systemd", "/etc/init.d",
    "/etc/profile", "/etc/profile.d", "/etc/environment",
    "/etc/bash.bashrc", "/etc/ssh",
    "/root", "/boot", "/proc", "/sys", "/dev",
    "/bin", "/sbin", "/usr/bin", "/usr/sbin",
    "/usr/lib", "/lib", "/lib64",
})


class PolicyEngine:
    """
    Stateful policy gate with ALL policy sections enforced.

    Entrypoints:
      validate_action_dict()  — primary (dict-based, no AT-SPI)
      validate()              — secondary (AT-SPI node-based)

    Thread safety: all mutable state protected by _apps_lock.
    """

    ALLOW = "ALLOW"
    DENY = "DENY"
    REQUIRE_HUMAN_CONFIRMATION = "REQUIRE_HUMAN_CONFIRMATION"

    _OP_TO_SYNTHETIC_ROLE: dict = {
        "click":       "push button",
        "write":       "text",
        "type":        "text",
        "press":       "keyboard",
        "hotkey":      "keyboard",
        "key":         "keyboard",
        "command":     "terminal",
        "install":     "terminal",
        "file_create": "file",
        "scroll":      "scroll",
        "verify":      "verify",
        "done":        "done",
    }

    _HIGH_RISK_OPERATIONS: FrozenSet[str] = frozenset({
        "command", "install", "file_create",
    })

    _TRUSTED_INSTALLER_PREFIXES: FrozenSet[str] = frozenset({
        "apt-get install", "apt install", "apt-get update", "apt update",
        "dnf install", "yum install",
        "pacman -s", "pacman -sy",
        "brew install", "brew upgrade",
        "npm install", "npm i", "npx",
        "pip install", "pip3 install",
        "pip install --break-system-packages",
        "pip3 install --break-system-packages",
        "node --version", "node -v",
        "npm --version", "npm -v",
        "python --version", "python3 --version",
        "pip --version", "pip3 --version",
        "java -version", "java --version",
        "go version", "rustc --version", "cargo --version",
        "git --version", "docker --version", "docker-compose --version",
    })

    _DEFAULT_ALLOWED_APPS: FrozenSet[str] = frozenset({
        "__unknown_app__",
        "google-chrome", "firefox", "chromium", "chromium-browser",
        "brave-browser", "microsoft-edge",
        "libreoffice", "soffice", "libreoffice-writer",
        "libreoffice-calc", "libreoffice-impress",
        "gedit", "kate", "code", "code-oss", "sublime_text", "atom",
        "mousepad", "pluma",
        "gnome-terminal", "xterm", "konsole", "xfce4-terminal",
        "mate-terminal", "tilix", "alacritty", "terminal",
        "iterm", "iterm2", "hyper",
        "nautilus", "thunar", "nemo", "dolphin", "finder", "pcmanfm",
        "evince", "eog", "gpicview", "totem", "vlc",
        "blender", "gimp", "gnumeric",
        "bash", "sh", "zsh", "fish",
        "files",
    })

    _APPROVAL_SIGNAL_DIR: str = "/tmp"
    _APPROVAL_SIGNAL_PREFIX: str = "projectzeo_approve_"

    def __init__(
        self,
        allowed_apps: Optional[Set[str]] = None,
        *,
        denied_apps: Optional[Set[str]] = None,
        high_risk_apps: Optional[Set[str]] = None,
        allowed_write_paths: Optional[List[str]] = None,
        denied_write_paths: Optional[List[str]] = None,
    ) -> None:
        self._apps_lock = threading.RLock()

        self._allowed_apps: Set[str] = (
            {str(a).lower() for a in allowed_apps}
            if allowed_apps is not None
            else set(self._DEFAULT_ALLOWED_APPS)
        )

        # M2 FIX
        self._denied_apps: FrozenSet[str] = frozenset(
            str(a).lower() for a in (denied_apps or [])
        )
        self._high_risk_apps: FrozenSet[str] = frozenset(
            str(a).lower() for a in (high_risk_apps or [])
        )

        # M1 FIX
        self._allowed_write_paths: Optional[List[str]] = (
            [str(p).rstrip("/") for p in allowed_write_paths]
            if allowed_write_paths else None
        )
        _policy_denied = frozenset(
            str(p).rstrip("/") for p in (denied_write_paths or [])
        )
        self._denied_write_paths: FrozenSet[str] = (
            _HARDCODED_DENIED_PATHS | _policy_denied
        )

        self.denied_roles: Set[str] = {"password text", "alert"}
        self.high_risk_name_patterns = [
            re.compile(r"delete",  re.IGNORECASE),
            re.compile(r"remove",  re.IGNORECASE),
            re.compile(r"format",  re.IGNORECASE),
            re.compile(r"erase",   re.IGNORECASE),
        ]

        _logger.info(
            "PolicyEngine initialised. allowed=%d denied=%d high_risk=%d "
            "write_allowlist=%s denied_paths=%d",
            len(self._allowed_apps), len(self._denied_apps),
            len(self._high_risk_apps),
            self._allowed_write_paths is not None,
            len(self._denied_write_paths),
        )

        # AUDIT-CRIT-1 FIX: Network policy enforcer (set via from_policy_yaml)
        self._network_policy = None  # type: Optional[_NetworkPolicyEnforcer]

    # =========================================================================
    # CLASS METHOD: from policy.yaml (M1 + M2 wired in one call)
    # =========================================================================

    @classmethod
    def from_policy_yaml(cls, policy_cfg: dict) -> "PolicyEngine":
        """
        M1 + M2 FIX: Construct PolicyEngine from parsed policy.yaml dict,
        loading all policy sections that were previously ignored.
        """
        if not isinstance(policy_cfg, dict):
            _logger.warning("[PolicyEngine.from_policy_yaml] Not a dict — using defaults.")
            return cls()

        allowed_apps_raw = policy_cfg.get("allowed_apps")
        allowed_apps = set(allowed_apps_raw) if isinstance(allowed_apps_raw, list) else None

        # M2: denied_apps
        denied_raw = policy_cfg.get("denied_apps")
        denied_apps = (
            {str(a) for a in denied_raw if a} if isinstance(denied_raw, list) else None
        )
        if denied_apps:
            _logger.info("[PolicyEngine] M2: denied_apps=%s", sorted(denied_apps))

        # M2: high_risk_apps
        hr_raw = policy_cfg.get("high_risk_apps")
        high_risk_apps = (
            {str(a) for a in hr_raw if a} if isinstance(hr_raw, list) else None
        )
        if high_risk_apps:
            _logger.info("[PolicyEngine] M2: high_risk_apps=%s", sorted(high_risk_apps))

        # M1: filesystem policy
        fs_cfg = policy_cfg.get("filesystem", {}) or {}
        aw_raw = fs_cfg.get("allowed_write_paths")
        allowed_write_paths = (
            [str(p) for p in aw_raw if p] if isinstance(aw_raw, list) else None
        )
        dw_raw = fs_cfg.get("denied_write_paths")
        denied_write_paths = (
            [str(p) for p in dw_raw if p] if isinstance(dw_raw, list) else None
        )
        if allowed_write_paths:
            _logger.info("[PolicyEngine] M1: allowed_write_paths=%s", allowed_write_paths)
        if denied_write_paths:
            _logger.info("[PolicyEngine] M1: denied_write_paths=%s", denied_write_paths)

        instance = cls(
            allowed_apps=allowed_apps,
            denied_apps=denied_apps,
            high_risk_apps=high_risk_apps,
            allowed_write_paths=allowed_write_paths,
            denied_write_paths=denied_write_paths,
        )

        # AUDIT-CRIT-1 FIX: Wire network policy enforcement
        # Previously the `network` key was silently discarded — zero enforcement.
        # Now NetworkPolicyEnforcer is instantiated and stored so that
        # validate_action_dict() can check commands against it.
        network_cfg = policy_cfg.get("network")
        if isinstance(network_cfg, dict) and _NETWORK_ENFORCER_AVAILABLE:
            try:
                instance._network_policy = _NetworkPolicyEnforcer.from_network_cfg(network_cfg)
                _logger.info(
                    "[PolicyEngine] AUDIT-CRIT-1: NetworkPolicyEnforcer wired from policy.yaml. "
                    "network=%s", sorted(network_cfg.keys()),
                )
            except Exception as net_err:
                _logger.error(
                    "[PolicyEngine] AUDIT-CRIT-1: NetworkPolicyEnforcer init failed: %s "
                    "— network policy NOT enforced.", net_err,
                )
        elif network_cfg is not None and not _NETWORK_ENFORCER_AVAILABLE:
            _logger.warning(
                "[PolicyEngine] AUDIT-CRIT-1: network section present in policy.yaml "
                "but NetworkPolicyEnforcer module not available — network policy NOT enforced."
            )

        return instance

    # =========================================================================
    # M1 FIX — FILESYSTEM PATH ENFORCEMENT
    # =========================================================================

    def _validate_file_path(self, path: str) -> Tuple[str, Optional[str]]:
        """M1 FIX: Enforce allowed/denied write paths for file_create."""
        if not path:
            return self.DENY, "file_create: empty path"

        norm_path = os.path.normpath(os.path.expanduser(path))

        for denied in self._denied_write_paths:
            denied_norm = os.path.normpath(denied)
            if norm_path == denied_norm or norm_path.startswith(denied_norm + os.sep):
                reason = (
                    f"file_create DENIED: path {path!r} is in a protected "
                    f"directory ({denied!r}). System paths are always forbidden."
                )
                _logger.warning("[PolicyEngine] M1 DENY: %s", reason)
                return self.DENY, reason

        if self._allowed_write_paths is not None:
            allowed = any(
                norm_path == os.path.normpath(os.path.expanduser(p)) or
                norm_path.startswith(os.path.normpath(os.path.expanduser(p)) + os.sep)
                for p in self._allowed_write_paths
            )
            if not allowed:
                reason = (
                    f"file_create DENIED: path {path!r} not under any "
                    f"allowed_write_path. Permitted: {self._allowed_write_paths}."
                )
                _logger.warning("[PolicyEngine] M1 DENY: %s", reason)
                return self.DENY, reason

        return self.ALLOW, None

    def _validate_command_redirect(self, command: str) -> Tuple[str, Optional[str]]:
        """M1 FIX: Block command redirects writing to denied paths."""
        if not command:
            return self.ALLOW, None
        redirect_targets = re.findall(r">>?\s*([^\s;&|]+)", command)
        for target in redirect_targets:
            target = target.strip()
            if not target:
                continue
            verdict, reason = self._validate_file_path(target)
            if verdict == self.DENY:
                full_reason = (
                    f"command DENIED: shell redirect to protected path "
                    f"{target!r} in command: {command[:80]!r}."
                )
                _logger.warning("[PolicyEngine] M1 REDIRECT-DENY: %s", full_reason)
                return self.DENY, full_reason
        return self.ALLOW, None

    def _validate_file_content(self, content: str) -> Tuple[str, Optional[str]]:
        """M3 FIX: Scan file_create content for dangerous patterns."""
        if not content:
            return self.ALLOW, None
        for pat in self.high_risk_name_patterns:
            if pat.search(content):
                reason = (
                    f"file_create content BLOCKED: dangerous pattern "
                    f"{pat.pattern!r} detected. Prevents script-injection bypass."
                )
                _logger.warning("[PolicyEngine] M3 DENY: %s", reason)
                return self.REQUIRE_HUMAN_CONFIRMATION, reason
        # Executable script (shebang) requires confirmation
        if re.search(r"^#!\s*/", content, re.MULTILINE):
            reason = (
                "file_create content requires confirmation: "
                "file contains a shebang (executable script)."
            )
            _logger.warning("[PolicyEngine] M3 CONFIRM: %s", reason)
            return self.REQUIRE_HUMAN_CONFIRMATION, reason
        return self.ALLOW, None

    # =========================================================================
    # TRUSTED INSTALLER (GAP-3 + SEC-NEW metacharacter check)
    # =========================================================================

    def _is_trusted_installer_command(self, command: str) -> bool:
        if not command:
            return False
        cmd_lower = command.strip().lower()
        for prefix in self._TRUSTED_INSTALLER_PREFIXES:
            if cmd_lower.startswith(prefix):
                suffix = cmd_lower[len(prefix):]
                if _SHELL_METACHAR_RE.search(suffix):
                    _logger.warning(
                        "[PolicyEngine] TRUSTED_INSTALLER rejected — "
                        "shell metacharacters in suffix: %r", command[:120],
                    )
                    return False
                return True
        return False

    # =========================================================================
    # DYNAMIC ALLOWLIST
    # =========================================================================

    @property
    def allowed_apps(self) -> FrozenSet[str]:
        with self._apps_lock:
            return frozenset(self._allowed_apps)

    def allow_app(self, app_name: str) -> None:
        if not isinstance(app_name, str) or not app_name.strip():
            return
        normalised = app_name.strip().lower()
        # M2 FIX: denied_apps cannot be un-denied via allow_app()
        if normalised in self._denied_apps:
            _logger.warning(
                "[PolicyEngine] allow_app: BLOCKED — %r is in denied_apps.", normalised
            )
            return
        with self._apps_lock:
            already = normalised in self._allowed_apps
            self._allowed_apps.add(normalised)
        if not already:
            _logger.info("[PolicyEngine] allow_app: %r added (total=%d).",
                         normalised, len(self._allowed_apps))

    def allow_apps(self, app_names) -> None:
        for name in app_names:
            self.allow_app(name)

    def warn_if_unlisted(self, app_name: str) -> None:
        if not app_name:
            return
        normalised = app_name.lower()
        with self._apps_lock:
            listed = normalised in self._allowed_apps
        if not listed:
            _logger.warning(
                "[PolicyEngine] warn_if_unlisted: %r NOT in allowlist — "
                "all autonomous interactions will be DENIED.", app_name,
            )

    # =========================================================================
    # HUMAN APPROVAL SIGNAL-FILE SUPPORT
    # =========================================================================

    @staticmethod
    def generate_action_key() -> str:
        """
        M4 FIX: Cryptographically secure 16-byte hex action key.
        Replaces abs(hash(cmd)) % 999999 — eliminates collision TOCTOU race.
        """
        return secrets.token_hex(16)

    def approval_signal_path(self, action_key: str) -> str:
        return os.path.join(
            self._APPROVAL_SIGNAL_DIR,
            f"{self._APPROVAL_SIGNAL_PREFIX}{action_key}.signal",
        )

    def check_human_approval(self, action_key: str) -> bool:
        """
        C-05 FIX: FAIL-CLOSED on filesystem error.
        Returns True only when signal file is confirmed absent.
        Returns False on OSError (previously returned True — security hole).
        """
        path = self.approval_signal_path(action_key)
        try:
            approved = not os.path.exists(path)
        except OSError as exc:
            _logger.warning(
                "[PolicyEngine] check_human_approval: stat %r failed: %s "
                "— FAIL-CLOSED: NOT approved.", path, exc,
            )
            return False  # C-05 FIX: fail-closed
        if approved:
            _logger.info(
                "[PolicyEngine] HUMAN_APPROVAL_GRANTED: key=%r path=%r",
                action_key, path,
            )
        return approved

    # =========================================================================
    # PRIMARY ENTRY POINT — dict-based
    # =========================================================================

    def validate_action_dict(
        self,
        action: dict,
        *,
        focused_app: str = "__unknown_app__",
    ) -> Tuple[str, Optional[str]]:
        try:
            return self._validate_action_dict_inner(action, focused_app)
        except Exception as exc:
            _logger.error(
                "[PolicyEngine] validate_action_dict: unexpected error (fail-closed): %s", exc
            )
            return self.DENY, f"Policy validation error (fail-closed): {exc}"

    def _validate_action_dict_inner(
        self, action: dict, focused_app: str,
    ) -> Tuple[str, Optional[str]]:
        if not isinstance(action, dict):
            return self.DENY, "Action must be a dict"

        op = str(action.get("operation") or "").lower().strip()
        if not op:
            return self.DENY, "Action has no 'operation' field"

        if op == "done":
            return self.ALLOW, None

        # M2 FIX: denied_apps — checked BEFORE allowlist
        app = str(focused_app or "__unknown_app__").lower().strip()
        if app in self._denied_apps:
            reason = (
                f"Application {app!r} is in denied_apps — permanently forbidden. "
                "Remove from denied_apps in policy.yaml to permit."
            )
            _logger.warning("[PolicyEngine] M2 DENY (denied_apps): op=%r app=%r", op, app)
            return self.DENY, reason

        # Allowlist check
        with self._apps_lock:
            app_allowed = app in self._allowed_apps
        if not app_allowed:
            reason = (
                f"Unauthorized application: {app!r}. "
                "Add to allowed_apps or call allow_app() after installation."
            )
            _logger.warning("[PolicyEngine] DENY: op=%r app=%r — %s", op, app, reason)
            return self.DENY, reason

        # M2 FIX: high_risk_apps — require human confirmation
        if app in self._high_risk_apps:
            reason = (
                f"Application {app!r} is in high_risk_apps — "
                "all interactions require human confirmation."
            )
            _logger.warning(
                "[PolicyEngine] M2 REQUIRE_HUMAN_CONFIRMATION (high_risk_apps): "
                "op=%r app=%r", op, app,
            )
            return self.REQUIRE_HUMAN_CONFIRMATION, reason

        # Synthetic role check
        synthetic_role = self._OP_TO_SYNTHETIC_ROLE.get(op, op)
        if synthetic_role in self.denied_roles:
            reason = f"Forbidden operation role: {synthetic_role!r}"
            _logger.warning("[PolicyEngine] DENY: %s", reason)
            return self.DENY, reason

        # Semantic type/write into non-text target
        if op in ("write", "type") and action.get("target_role"):
            target_role = str(action["target_role"]).lower()
            if "text" not in target_role and "entry" not in target_role:
                reason = (
                    f"Semantic violation: type/write into role {target_role!r}. "
                    "Only 'text' and 'entry' roles are writable."
                )
                _logger.warning("[PolicyEngine] DENY: %s", reason)
                return self.DENY, reason

        # M1 FIX: Filesystem policy for file_create
        if op == "file_create":
            path = str(action.get("path") or "").strip()
            path_verdict, path_reason = self._validate_file_path(path)
            if path_verdict != self.ALLOW:
                return path_verdict, path_reason
            # M3 FIX: Scan content
            content = str(action.get("content") or "")
            content_verdict, content_reason = self._validate_file_content(content)
            if content_verdict != self.ALLOW:
                return content_verdict, content_reason

        # M1 FIX: Block command redirect to denied paths
        if op == "command":
            cmd_str = str(action.get("command") or "")
            redirect_verdict, redirect_reason = self._validate_command_redirect(cmd_str)
            if redirect_verdict != self.ALLOW:
                return redirect_verdict, redirect_reason

        # AUDIT-CRIT-1 FIX: Network policy enforcement for command/install ops
        # Previously the network section of policy.yaml was NEVER enforced.
        # Now every command is checked against denied_domains, SSH policy, HTTP policy.
        if op in ("command", "install") and self._network_policy is not None:
            _net_cmd = str(action.get("command") or action.get("tool", {}).get("name", "") or "")
            if _net_cmd:
                _net_decision = self._network_policy.validate_command(_net_cmd)
                if _net_decision.verdict != "ALLOW":
                    _net_reason = (
                        f"Network policy violation: {_net_decision.reason} "
                        f"(rule={_net_decision.matched_rule!r})"
                    )
                    _logger.warning(
                        "[PolicyEngine] AUDIT-CRIT-1 DENY: op=%r net_reason=%r cmd=%r",
                        op, _net_decision.reason, _net_cmd[:80],
                    )
                    return self.DENY, _net_reason

        # High-risk content check (GAP-3 trusted-installer bypass preserved)
        _trusted_flag: bool = bool(action.get("_trusted_installer", False))
        content_to_check = self._extract_risk_content(op, action)

        if content_to_check.strip():
            _bypass = (
                _trusted_flag
                and op in ("command", "install")
                and self._is_trusted_installer_command(content_to_check)
            )
            if _bypass:
                _logger.info(
                    "[PolicyEngine] GAP-3 TRUSTED_INSTALLER bypass: op=%r cmd=%r",
                    op, content_to_check[:80],
                )
            else:
                for pat in self.high_risk_name_patterns:
                    if pat.search(content_to_check):
                        reason = (
                            f"High-risk content (pattern={pat.pattern!r}): "
                            f"{content_to_check[:80]!r}"
                        )
                        _logger.warning(
                            "[PolicyEngine] REQUIRE_HUMAN_CONFIRMATION: "
                            "op=%r app=%r pattern=%r", op, app, pat.pattern,
                        )
                        return self.REQUIRE_HUMAN_CONFIRMATION, reason

        # Unknown operation — fail closed
        if op not in self._OP_TO_SYNTHETIC_ROLE:
            reason = f"Unknown operation: {op!r}"
            _logger.warning("[PolicyEngine] DENY: %s", reason)
            return self.DENY, reason

        return self.ALLOW, None

    def _extract_risk_content(self, op: str, action: dict) -> str:
        if op == "command":
            return str(action.get("command") or "")
        if op in ("write", "type"):
            return str(action.get("content") or action.get("text") or "")
        if op in ("press", "hotkey", "key"):
            keys = action.get("keys") or action.get("key") or []
            if isinstance(keys, str):
                keys = [keys]
            return " ".join(str(k) for k in keys)
        if op == "file_create":
            # M3 FIX: include content in risk scan
            path = str(action.get("path") or "")
            content = str(action.get("content") or "")
            return f"{path} {content}"
        return ""

    # =========================================================================
    # SECONDARY ENTRY POINT — AT-SPI node-based
    # =========================================================================

    def validate(self, node, action: str) -> Tuple[str, Optional[str]]:
        try:
            role = (node.getRoleName() or "unknown").lower()
        except Exception as exc:
            return self.DENY, f"Cannot read node role: {exc}"
        try:
            name = (node.name or "").lower()
        except Exception as exc:
            return self.DENY, f"Cannot read node name: {exc}"
        try:
            app_obj = node.getApplication()
            app = app_obj.name.lower() if app_obj and app_obj.name else "unknown"
        except Exception as exc:
            return self.DENY, f"Cannot read application identity: {exc}"

        if app == "unknown":
            return self.DENY, "Application identity unavailable"

        # M2 FIX in AT-SPI path
        if app in self._denied_apps:
            return self.DENY, f"Application {app!r} is in denied_apps."

        with self._apps_lock:
            app_allowed = app in self._allowed_apps
        if not app_allowed:
            return self.DENY, f"Unauthorized application: {app!r}."

        if app in self._high_risk_apps:
            return self.REQUIRE_HUMAN_CONFIRMATION, (
                f"Application {app!r} is in high_risk_apps — confirmation required."
            )

        if role in self.denied_roles:
            return self.DENY, f"Forbidden role: {role}"

        if action == "type" and "text" not in role and "entry" not in role:
            return self.DENY, f"Semantic violation: type into role '{role}'"

        for pat in self.high_risk_name_patterns:
            if pat.search(name):
                _logger.warning(
                    "[PolicyEngine] REQUIRE_HUMAN_CONFIRMATION (AT-SPI): "
                    "role=%r name=%r app=%r", role, name, app,
                )
                return self.REQUIRE_HUMAN_CONFIRMATION, (
                    f"High-risk label detected: {pat.pattern}"
                )

        return self.ALLOW, None
