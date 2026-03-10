from __future__ import annotations

try:
    from core.network.policy_enforcer import NetworkPolicyEnforcer as _NetworkPolicyEnforcer
    _NETWORK_ENFORCER_AVAILABLE = True
except ImportError:
    _NetworkPolicyEnforcer = None  # type: ignore
    _NETWORK_ENFORCER_AVAILABLE = False

import logging
import os
import re
import secrets
import tempfile
import threading
from typing import FrozenSet, List, Optional, Set, Tuple

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shell metacharacter detection
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

# Paths ALWAYS denied for file_create regardless of policy.yaml.
_HARDCODED_DENIED_PATHS: FrozenSet[str] = frozenset({
    # System paths
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
    # User persistence paths (prevent agent from establishing autostart)
    "~/.bashrc", "~/.zshrc", "~/.profile", "~/.bash_profile",
    "~/.bash_login", "~/.zprofile", "~/.zshenv",
    "~/.config/systemd/user",
    "~/.config/autostart",
    "~/.local/share/systemd/user",
})


# ---------------------------------------------------------------------------
# Per-session secure signal directory (prevents /tmp enumeration attacks)
# ---------------------------------------------------------------------------

def _init_signal_dir() -> str:
    _token = secrets.token_hex(16)
    _base  = tempfile.gettempdir()
    _path  = os.path.join(_base, f"projectzeo_{_token}")
    try:
        os.makedirs(_path, mode=0o700, exist_ok=True)
        os.chmod(_path, 0o700)
    except OSError:
        _path = _base
    return _path


_SESSION_SIGNAL_DIR: str = _init_signal_dir()

# ---------------------------------------------------------------------------
# Module-level ConsequenceReasoner singleton for Sin-3 CR-first routing
# ---------------------------------------------------------------------------
_policy_cr_instance = None
_policy_cr_lock = threading.Lock()


def _get_policy_consequence_reasoner():
    """Return a lazily-initialised ConsequenceReasoner for policy-level CR routing."""
    global _policy_cr_instance
    if _policy_cr_instance is not None:
        return _policy_cr_instance
    with _policy_cr_lock:
        if _policy_cr_instance is not None:
            return _policy_cr_instance
        try:
            from core.safety.consequence_reasoner import ConsequenceReasoner
            _policy_cr_instance = ConsequenceReasoner()
            _logger.info("[PolicyEngine] ConsequenceReasoner singleton initialised for CR-first routing.")
        except Exception as exc:
            _logger.warning(
                "[PolicyEngine] ConsequenceReasoner init failed — CR-first routing unavailable: %s", exc
            )
            _policy_cr_instance = None
    return _policy_cr_instance


class PolicyViolationError(RuntimeError):
    pass


class PolicyEngine:
    

    ALLOW                    = "ALLOW"
    DENY                     = "DENY"
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
        "wait":        "wait",
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

    # HIGH-4 FIX: Default allowed_write_paths uses current user's home,
    # not the entire /home/ tree.
    @staticmethod
    def _default_allowed_write_paths() -> List[str]:
        home = os.path.expanduser("~")
        return [
            os.path.join(home, "Desktop"),
            os.path.join(home, "Documents"),
            os.path.join(home, "Downloads"),
            os.path.join(home, "projects"),
            os.path.join(home, "workspace"),
            os.path.join(home, ".projectzeo"),
            "/tmp",
            "/var/tmp",
        ]

    _APPROVAL_SIGNAL_DIR:    str = _SESSION_SIGNAL_DIR
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

        self._denied_apps: FrozenSet[str] = frozenset(
            str(a).lower() for a in (denied_apps or [])
        )
        self._high_risk_apps: FrozenSet[str] = frozenset(
            str(a).lower() for a in (high_risk_apps or [])
        )

        # HIGH-4 FIX: default to current-user home subdirs only.
        _default_paths = self._default_allowed_write_paths()
        self._allowed_write_paths: Optional[List[str]] = (
            [str(p).rstrip("/") for p in allowed_write_paths]
            if allowed_write_paths is not None
            else _default_paths
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
            "write_allowlist=%s denied_paths=%d signal_dir=%r",
            len(self._allowed_apps), len(self._denied_apps),
            len(self._high_risk_apps),
            self._allowed_write_paths is not None,
            len(self._denied_write_paths),
            self._APPROVAL_SIGNAL_DIR,
        )

        self._network_policy = None  # type: Optional[object]

    # =========================================================================
    # CLASS METHOD: from policy.yaml
    # =========================================================================

    @classmethod
    def from_policy_yaml(cls, policy_cfg: dict) -> "PolicyEngine":
        if not isinstance(policy_cfg, dict):
            _logger.warning("[PolicyEngine.from_policy_yaml] Not a dict — using defaults.")
            return cls()

        allowed_apps_raw = policy_cfg.get("allowed_apps")
        allowed_apps = set(allowed_apps_raw) if isinstance(allowed_apps_raw, list) else None

        denied_raw = policy_cfg.get("denied_apps")
        denied_apps = (
            {str(a) for a in denied_raw if a} if isinstance(denied_raw, list) else None
        )

        hr_raw = policy_cfg.get("high_risk_apps")
        high_risk_apps = (
            {str(a) for a in hr_raw if a} if isinstance(hr_raw, list) else None
        )

        fs_cfg = policy_cfg.get("filesystem", {}) or {}
        aw_raw = fs_cfg.get("allowed_write_paths")
        # HIGH-4 FIX: expand ~ in paths from policy.yaml
        allowed_write_paths = (
            [os.path.expanduser(str(p)) for p in aw_raw if p]
            if isinstance(aw_raw, list) else None
        )
        dw_raw = fs_cfg.get("denied_write_paths")
        denied_write_paths = (
            [os.path.expanduser(str(p)) for p in dw_raw if p]
            if isinstance(dw_raw, list) else None
        )

        instance = cls(
            allowed_apps=allowed_apps,
            denied_apps=denied_apps,
            high_risk_apps=high_risk_apps,
            allowed_write_paths=allowed_write_paths,
            denied_write_paths=denied_write_paths,
        )

        network_cfg = policy_cfg.get("network")
        if isinstance(network_cfg, dict) and _NETWORK_ENFORCER_AVAILABLE:
            try:
                instance._network_policy = _NetworkPolicyEnforcer.from_network_cfg(network_cfg)
                _logger.info(
                    "[PolicyEngine] NetworkPolicyEnforcer wired. network=%s",
                    sorted(network_cfg.keys()),
                )
            except Exception as net_err:
                _logger.error(
                    "[PolicyEngine] NetworkPolicyEnforcer init failed: %s "
                    "— network policy NOT enforced.", net_err,
                )
        elif network_cfg is not None and not _NETWORK_ENFORCER_AVAILABLE:
            _logger.warning(
                "[PolicyEngine] network section present but NetworkPolicyEnforcer "
                "module not available — network policy NOT enforced."
            )

        return instance

    # =========================================================================
    # FILESYSTEM PATH ENFORCEMENT
    # =========================================================================

    def _validate_file_path(self, path: str) -> Tuple[str, Optional[str]]:
        if not path:
            return self.DENY, "file_create: empty path"

        # Resolve symlinks to prevent symlink-based path traversal.
        norm_path = os.path.realpath(os.path.normpath(os.path.expanduser(path)))

        for denied in self._denied_write_paths:
            denied_norm = os.path.realpath(os.path.normpath(os.path.expanduser(denied)))
            if norm_path == denied_norm or norm_path.startswith(denied_norm + os.sep):
                reason = (
                    f"file_create DENIED: path {path!r} is in a protected "
                    f"directory ({denied!r})."
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
        """Block command redirects writing to denied paths."""
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
        """Apply DANGEROUS_PATTERNS to file content at dispatch time."""
        if not content:
            return self.ALLOW, None

        try:
            from core.planner.execution_planner import ExecutionPlanner as _EP
            _dp_compiled = [re.compile(p, re.IGNORECASE) for p in _EP.DANGEROUS_PATTERNS]
            for _dp_pat in _dp_compiled:
                for line in content.splitlines():
                    if _dp_pat.search(line):
                        reason = (
                            f"file_create content BLOCKED: dangerous command pattern "
                            f"in file body: {_dp_pat.pattern[:80]!r}"
                        )
                        _logger.warning("[PolicyEngine] DISPATCH-CONTENT DENY: %s", reason)
                        return self.DENY, reason
        except Exception as _dp_err:
            _logger.debug("[PolicyEngine] DANGEROUS_PATTERNS file content check failed: %s", _dp_err)

        # High-risk content patterns that require human confirmation
        for pat in self.high_risk_name_patterns:
            if pat.search(content):
                reason = (
                    f"file_create content requires confirmation: dangerous pattern "
                    f"{pat.pattern!r} detected."
                )
                _logger.warning("[PolicyEngine] M3 CONFIRM: %s", reason)
                return self.REQUIRE_HUMAN_CONFIRMATION, reason

        if re.search(r"^#!\s*/", content, re.MULTILINE):
            reason = (
                "file_create content requires confirmation: "
                "file contains a shebang (executable script)."
            )
            _logger.warning("[PolicyEngine] M3 CONFIRM: %s", reason)
            return self.REQUIRE_HUMAN_CONFIRMATION, reason

        return self.ALLOW, None

    # =========================================================================
    # CONSEQUENCE-FIRST ROUTING (Blueprint §3 Sin-3 complete fix)
    # =========================================================================

    def _evaluate_unknown_app_via_consequence_reasoner(
        self,
        action: dict,
        app: str,
        op: str,
    ) -> str:
        """
        Route unknown apps through ConsequenceReasoner BEFORE human escalation.

        Returns one of: ALLOW, DENY, REQUIRE_HUMAN_CONFIRMATION

        Logic:
          - If CR returns SAFE + COHERENT → ALLOW (reversible low-risk action)
          - If CR returns HARMFUL or score < 0.15 → DENY
          - If CR returns UNCERTAIN or score in [0.15, 0.45) → REQUIRE_HUMAN_CONFIRMATION
          - If CR is unavailable → REQUIRE_HUMAN_CONFIRMATION (fail-safe)
          - High-risk operations (command, file_create, install) always → REQUIRE_HUMAN
            unless score >= 0.75

        This is the core GII architectural change: consequence reasoning as the
        primary gate, human approval as the fallback, allowlist as the shortcut.
        """
        # Fast-path: high-risk operations always require human unless CR is very confident
        _HIGH_RISK_OPS = frozenset({"command", "install", "file_create"})

        try:
            from core.safety.consequence_reasoner import (
                ConsequenceReasoner,
                SafetyDecision,
            )
        except ImportError:
            _logger.debug(
                "[PolicyEngine] ConsequenceReasoner not importable — "
                "falling back to REQUIRE_HUMAN_CONFIRMATION for unknown app %r.", app
            )
            return self.REQUIRE_HUMAN_CONFIRMATION

        # Lazy-init a module-level CR instance for policy use
        cr = _get_policy_consequence_reasoner()
        if cr is None:
            return self.REQUIRE_HUMAN_CONFIRMATION

        try:
            result = cr.evaluate(
                action=action,
                goal_description=f"Agent operating in unknown app: {app}",
                world_state={"focused_app": app, "operation": op},
            )

            score = getattr(result, "numeric_score", 0.5)
            decision = getattr(result, "decision", None)
            decision_str = decision.value if hasattr(decision, "value") else str(decision)

            _logger.debug(
                "[PolicyEngine] CR-FIRST result: app=%r op=%r score=%.3f decision=%s",
                app, op, score, decision_str,
            )

            if decision_str == SafetyDecision.DENY.value or score < 0.15:
                return self.DENY

            # High-risk operations require CR score >= 0.75 to auto-allow
            if op in _HIGH_RISK_OPS:
                if score >= 0.75 and decision_str == SafetyDecision.ALLOW.value:
                    return self.ALLOW
                return self.REQUIRE_HUMAN_CONFIRMATION

            if decision_str == SafetyDecision.ALLOW.value and score >= 0.55:
                return self.ALLOW

            if score >= 0.45:
                return self.REQUIRE_HUMAN_CONFIRMATION

            return self.DENY

        except Exception as exc:
            _logger.warning(
                "[PolicyEngine] CR-FIRST evaluation error for app=%r: %s — "
                "falling back to REQUIRE_HUMAN_CONFIRMATION.", app, exc
            )
            return self.REQUIRE_HUMAN_CONFIRMATION

    # =========================================================================
    # TRUSTED INSTALLER
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
        if normalised in self._denied_apps:
            _logger.warning(
                "[PolicyEngine] allow_app: BLOCKED — %r is in denied_apps.", normalised
            )
            return
        with self._apps_lock:
            already = normalised in self._allowed_apps
            self._allowed_apps.add(normalised)
        if not already:
            _logger.info(
                "[PolicyEngine] allow_app: %r added (total=%d).",
                normalised, len(self._allowed_apps),
            )

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
                "consequence evaluation will be required.", app_name,
            )

    # =========================================================================
    # HUMAN APPROVAL SIGNAL-FILE SUPPORT
    # =========================================================================

    @staticmethod
    def generate_action_key() -> str:
        return secrets.token_hex(16)

    def approval_signal_path(self, action_key: str) -> str:
        return os.path.join(
            self._APPROVAL_SIGNAL_DIR,
            f"{self._APPROVAL_SIGNAL_PREFIX}{action_key}.signal",
        )

    # =========================================================================
    # CANONICAL APPROVAL PATTERN (operate.py reference)
    # =========================================================================
    #
    # The ONLY correct approval check pattern in ProjectZeo:
    #
    #   signal_path  = policy_engine.approval_signal_path(action_key)
    #   approve_path = signal_path + ".APPROVE"
    #   # … write signal file, notify operator …
    #   approved = False
    #   while elapsed < timeout:
    #       time.sleep(0.5)
    #       if os.path.exists(approve_path):
    #           os.remove(approve_path)   # consume
    #           approved = True
    #           break
    #   # Finally: clean up signal_path unconditionally
    #
    #   Semantic: approved = .APPROVE file EXPLICITLY EXISTS
    #             denied   = .APPROVE absent after timeout
    #
    # CRITICAL-1 FIX: check_human_approval() and check_human_approval_legacy()
    # have been DELETED.  Both had inverted semantics (returned True when the
    # signal file was absent — approving every unconfirmed action).
    # Any remaining call site will raise AttributeError at import time,
    # which is the correct fail-loud behavior.
    # =========================================================================

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

        # Wait and done are always allowed (no-harm operations)
        if op in ("done", "wait"):
            return self.ALLOW, None

        app = str(focused_app or "__unknown_app__").lower().strip()

        # Step 1: hard-denied apps
        if app in self._denied_apps:
            reason = f"Application {app!r} is in denied_apps — permanently forbidden."
            _logger.warning("[PolicyEngine] M2 DENY (denied_apps): op=%r app=%r", op, app)
            return self.DENY, reason

        with self._apps_lock:
            app_allowed = app in self._allowed_apps

        # Step 2: unknown app → CONSEQUENCE-FIRST routing (Blueprint §3, Sin-3 complete fix)
        # GII design: ConsequenceReasoner is the PRIMARY gate for unknown apps.
        # Only escalate to human if CR deems action CRITICAL/UNCERTAIN/IRREVERSIBLE.
        # ALLOW if CR returns COHERENT + SAFE (low-risk reversible).
        # This replaces the old allowlist-as-primary-gate architecture.
        if not app_allowed:
            cr_decision = self._evaluate_unknown_app_via_consequence_reasoner(action, app, op)
            if cr_decision == self.ALLOW:
                _logger.info(
                    "[PolicyEngine] CR-FIRST: unknown app %r op=%r ALLOWED by ConsequenceReasoner.",
                    app, op,
                )
                return self.ALLOW, None
            elif cr_decision == self.DENY:
                reason = (
                    f"Unknown app {app!r} op={op!r}: ConsequenceReasoner assessed HIGH/CRITICAL risk. "
                    "Add to policy.yaml allowed_apps if this is a legitimate application."
                )
                _logger.warning("[PolicyEngine] CR-FIRST DENY: app=%r op=%r", app, op)
                return self.DENY, reason
            else:
                # REQUIRE_HUMAN_CONFIRMATION — CR said CRITICAL/IRREVERSIBLE or unavailable
                reason = (
                    f"Unknown app {app!r} op={op!r}: ConsequenceReasoner requires human review. "
                    "Add to policy.yaml allowed_apps to suppress this prompt for trusted apps."
                )
                _logger.warning("[PolicyEngine] CR-FIRST CONFIRM: app=%r op=%r", app, op)
                return self.REQUIRE_HUMAN_CONFIRMATION, reason

        # Step 3: high-risk apps → confirmation always
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

        # Step 4: denied roles
        synthetic_role = self._OP_TO_SYNTHETIC_ROLE.get(op, op)
        if synthetic_role in self.denied_roles:
            reason = f"Forbidden operation role: {synthetic_role!r}"
            _logger.warning("[PolicyEngine] DENY: %s", reason)
            return self.DENY, reason

        # Step 5: semantic role check for write/type
        if op in ("write", "type") and action.get("target_role"):
            target_role = str(action["target_role"]).lower()
            if "text" not in target_role and "entry" not in target_role:
                reason = (
                    f"Semantic violation: type/write into role {target_role!r}. "
                    "Only 'text' and 'entry' roles are writable."
                )
                _logger.warning("[PolicyEngine] DENY: %s", reason)
                return self.DENY, reason

        # Step 6: filesystem path checks
        if op == "file_create":
            path = str(action.get("path") or "").strip()
            path_verdict, path_reason = self._validate_file_path(path)
            if path_verdict != self.ALLOW:
                return path_verdict, path_reason
            content = str(action.get("content") or "")
            content_verdict, content_reason = self._validate_file_content(content)
            if content_verdict != self.ALLOW:
                return content_verdict, content_reason

        # Step 6b: redirect protection for commands
        if op == "command":
            cmd_str = str(action.get("command") or "")
            redirect_verdict, redirect_reason = self._validate_command_redirect(cmd_str)
            if redirect_verdict != self.ALLOW:
                return redirect_verdict, redirect_reason

        # Step 7: network policy
        if op in ("command", "install") and self._network_policy is not None:
            _net_cmd = str(
                action.get("command") or
                action.get("tool", {}).get("name", "") or ""
            )
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

        # Step 8: high-risk content / trusted installer bypass
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
                    "[PolicyEngine] TRUSTED_INSTALLER bypass: op=%r cmd=%r",
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
            path    = str(action.get("path") or "")
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

        if app in self._denied_apps:
            return self.DENY, f"Application {app!r} is in denied_apps."

        with self._apps_lock:
            app_allowed = app in self._allowed_apps
        if not app_allowed:
            return self.REQUIRE_HUMAN_CONFIRMATION, (
                f"Unknown application {app!r}. Consequence evaluation required. "
                "Approve or add to allowed_apps in policy.yaml."
            )

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

    # =========================================================================
    # GII-FIX: Policy hot-reload with file watcher (Blueprint §12.3)
    # =========================================================================

    def reload(self, policy_yaml_path: Optional[str] = None) -> bool:
        """
        Hot-reload policy configuration from policy.yaml without restarting.
        Thread-safe: acquires _apps_lock before swapping all mutable state.
        Returns True if reload succeeded.
        """
        try:
            import yaml as _yaml_mod
        except ImportError:
            _logger.error("[PolicyEngine.reload] PyYAML not installed — cannot reload.")
            return False

        search_paths = []
        if policy_yaml_path:
            search_paths.append(policy_yaml_path)
        search_paths += [
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "policy.yaml"),
            os.path.join(os.path.expanduser("~"), ".projectzeo", "policy.yaml"),
            "/etc/projectzeo/policy.yaml",
        ]

        loaded_path = None
        policy_cfg: dict = {}
        for path in search_paths:
            if path and os.path.isfile(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        policy_cfg = _yaml_mod.safe_load(f) or {}
                    loaded_path = path
                    break
                except Exception as exc:
                    _logger.warning("[PolicyEngine.reload] Failed to load %r: %s", path, exc)

        if not loaded_path or not isinstance(policy_cfg, dict):
            _logger.error("[PolicyEngine.reload] No valid policy.yaml found.")
            return False

        # Parse new values
        allowed_apps_raw = policy_cfg.get("allowed_apps")
        new_allowed = ({str(a).lower() for a in allowed_apps_raw}
                       if isinstance(allowed_apps_raw, list) else None)

        denied_raw = policy_cfg.get("denied_apps")
        new_denied = (frozenset(str(a).lower() for a in denied_raw if a)
                      if isinstance(denied_raw, list) else self._denied_apps)

        hr_raw = policy_cfg.get("high_risk_apps")
        new_high_risk = (frozenset(str(a).lower() for a in hr_raw if a)
                         if isinstance(hr_raw, list) else self._high_risk_apps)

        fs_cfg = policy_cfg.get("filesystem", {}) or {}
        aw_raw = fs_cfg.get("allowed_write_paths")
        new_allowed_write = ([os.path.expanduser(str(p)) for p in aw_raw if p]
                             if isinstance(aw_raw, list) else None)

        dw_raw = fs_cfg.get("denied_write_paths")
        extra_denied = (frozenset(os.path.expanduser(str(p)).rstrip("/") for p in dw_raw if p)
                        if isinstance(dw_raw, list) else frozenset())
        new_denied_write = _HARDCODED_DENIED_PATHS | extra_denied

        with self._apps_lock:
            if new_allowed is not None:
                self._allowed_apps = new_allowed
            self._denied_apps = new_denied
            self._high_risk_apps = new_high_risk
            if new_allowed_write is not None:
                self._allowed_write_paths = new_allowed_write
            self._denied_write_paths = new_denied_write

        # ── SICA: Load auto-applied SICA rules from policy.yaml append block ──
        # When SICA auto-applies a rule (confidence >= 0.8), it appends to
        # policy.yaml under sica_auto_rules: section. Load and store these.
        sica_rules_raw = policy_cfg.get("sica_auto_rules", [])
        if isinstance(sica_rules_raw, list) and sica_rules_raw:
            try:
                with self._apps_lock:
                    if not hasattr(self, "_sica_auto_rules"):
                        self._sica_auto_rules = []
                    self._sica_auto_rules = [
                        r for r in sica_rules_raw
                        if isinstance(r, dict) and r.get("operation") and r.get("rule")
                    ]
                _logger.info(
                    "[PolicyEngine.reload] Loaded %d SICA auto-rules.",
                    len(self._sica_auto_rules),
                )
            except Exception as _sica_load_exc:
                _logger.debug("[PolicyEngine.reload] SICA rule load error: %s", _sica_load_exc)

        _logger.info(
            "[PolicyEngine.reload] Reloaded from %r. allowed=%d denied=%d high_risk=%d",
            loaded_path, len(self._allowed_apps),
            len(self._denied_apps), len(self._high_risk_apps),
        )
        return True

    def start_file_watcher(
        self,
        policy_yaml_path: Optional[str] = None,
        poll_interval_seconds: float = 30.0,
    ) -> None:
        """
        Start a background daemon thread that polls policy.yaml for mtime
        changes and calls reload() automatically. No inotify dependency.
        """
        if not policy_yaml_path:
            for path in [
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "policy.yaml"),
                os.path.join(os.path.expanduser("~"), ".projectzeo", "policy.yaml"),
            ]:
                if os.path.isfile(path):
                    policy_yaml_path = path
                    break

        if not policy_yaml_path or not os.path.isfile(policy_yaml_path):
            _logger.warning("[PolicyEngine] start_file_watcher: policy.yaml not found.")
            return

        watch_path = policy_yaml_path
        _state: List[float] = [os.path.getmtime(watch_path)]

        def _watcher() -> None:
            while True:
                try:
                    import time as _t
                    _t.sleep(poll_interval_seconds)
                    mtime = os.path.getmtime(watch_path)
                    if mtime != _state[0]:
                        _logger.info(
                            "[PolicyEngine] policy.yaml changed — hot-reloading..."
                        )
                        if self.reload(watch_path):
                            _state[0] = mtime
                except Exception as exc:
                    _logger.warning("[PolicyEngine] File watcher error: %s", exc)

        t = threading.Thread(target=_watcher, daemon=True, name="policy-watcher")
        t.start()
        _logger.info(
            "[PolicyEngine] Policy watcher started: %r (every %.0fs).",
            watch_path, poll_interval_seconds,
        )
