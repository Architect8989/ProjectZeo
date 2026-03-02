from __future__ import annotations

import os
import re
import threading
import logging
from typing import FrozenSet, Optional, Set, Tuple

_logger = logging.getLogger(__name__)


class PolicyViolationError(RuntimeError):
    """Raised when a caller attempts an operation that violates policy
    in a context where exceptions are preferred over return codes."""


class PolicyEngine:
    """
    Stateless(ish) policy gate.

    validate_action_dict() is the primary entry point for the main execution
    loop.  validate() is the secondary entry point for AT-SPI node-based checks
    (used when accessibility metadata is available).

    Thread safety
    -------------
    allowed_apps is protected by _apps_lock so allow_app() / allow_apps()
    can be called safely from the AutonomousInstaller background thread.

    GAP #3 FIX — Autonomous package installation
    =============================================
    The original ``high_risk_name_patterns`` list contained a ``sudo`` pattern
    that triggered ``REQUIRE_HUMAN_CONFIRMATION`` for *every* package install.
    Combined with ``_CONFIRM_TIMEOUT_SECONDS = 60``, any unattended install
    timed out to DENY.  The operator had to manually delete a signal file for
    each installation step — completely defeating "autonomously install tools".

    Fix: introduce ``_TRUSTED_INSTALLER_OPS`` — a set of command prefixes
    (normalised to lowercase) that are pre-approved for autonomous execution.
    Commands whose stripped text starts with a trusted installer prefix bypass
    the ``high_risk_name_patterns`` scan and receive ALLOW directly.

    Security: trusted-installer promotion is only applied when:
      1. The operation is "command" or "install" (not write/type/click).
      2. The action dict carries the ``_trusted_installer`` flag set by
         ExecutionPlanner._validate_and_normalise_step() after its own
         dangerous-pattern scan has passed (see GAP #2 fix).
    This means the bypass is gated on TWO independent checks — the planner's
    dangerous-pattern scan and this engine's trusted-prefix check — so an
    attacker cannot reach the bypass without first clearing the planner's
    50+ dangerous-pattern regexes.
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
        "command",
        "install",
        "file_create",
    })

    # GAP #3 FIX: Trusted installer command prefixes.
    # Commands whose stripped text *starts with* one of these prefixes are
    # pre-approved for autonomous execution and bypass the sudo/delete/format
    # high_risk_name_patterns check.
    #
    # Design principles:
    #   - Prefix match (not substring) — "apt-get install" is trusted;
    #     "apt-get purge --allow-remove-essential" is NOT (starts with apt-get
    #     but the purge verb is not in this list).
    #   - All prefixes are package-manager install verbs or version commands.
    #   - Deliberately excludes: remove, purge, uninstall, autoremove, dist-upgrade,
    #     format, mkfs, dd, shred (all remain subject to human confirmation).
    #   - The ExecutionPlanner dangerous-pattern scan is a prerequisite — this
    #     list only promotes commands that have already cleared 50+ regex checks.
    _TRUSTED_INSTALLER_PREFIXES: FrozenSet[str] = frozenset({
        # Debian/Ubuntu
        "apt-get install",
        "apt install",
        "apt-get update",
        "apt update",
        # Red Hat / Fedora
        "dnf install",
        "yum install",
        # Arch
        "pacman -s",
        "pacman -sy",
        # macOS
        "brew install",
        "brew upgrade",
        # Node
        "npm install",
        "npm i",
        "npx",
        # Python
        "pip install",
        "pip3 install",
        "pip install --break-system-packages",
        "pip3 install --break-system-packages",
        # Version probe commands (safe, read-only)
        "node --version",
        "node -v",
        "npm --version",
        "npm -v",
        "python --version",
        "python3 --version",
        "pip --version",
        "pip3 --version",
        "java -version",
        "java --version",
        "go version",
        "rustc --version",
        "cargo --version",
        "git --version",
        "docker --version",
        "docker-compose --version",
        "which ",
        "command -v ",
    })

    # Default application allowlist — covers the most common desktop apps.
    # Operators extend this via policy.yaml allowed_apps list OR allow_app().
    _DEFAULT_ALLOWED_APPS: FrozenSet[str] = frozenset({
        # Warmup sentinel: permits actions before world-graph has observed focus
        "__unknown_app__",
        # Browsers
        "google-chrome", "firefox", "chromium", "chromium-browser",
        "brave-browser", "microsoft-edge",
        # Office / document editors
        "libreoffice", "soffice", "libreoffice-writer", "libreoffice-calc",
        "libreoffice-impress",
        # Text / code editors
        "gedit", "kate", "code", "code-oss", "sublime_text", "atom",
        "mousepad", "pluma",
        # Terminal emulators
        "gnome-terminal", "xterm", "konsole", "xfce4-terminal", "mate-terminal",
        "tilix", "alacritty", "terminal", "iterm", "iterm2", "hyper",
        # File managers
        "nautilus", "thunar", "nemo", "dolphin", "finder", "pcmanfm",
        # Media / utilities
        "evince", "eog", "gpicview", "totem", "vlc",
    })

    # Approval signal files for REQUIRE_HUMAN_CONFIRMATION flow
    _APPROVAL_SIGNAL_DIR: str = "/tmp"
    _APPROVAL_SIGNAL_PREFIX: str = "projectzeo_approve_"

    def __init__(self, allowed_apps: Optional[Set[str]] = None) -> None:
        # BUG-C2 FIX: Use a mutable set + lock instead of frozenset so that
        # allow_app() can add entries at runtime after AutonomousInstaller
        # completes a package installation.
        self._apps_lock = threading.RLock()
        if allowed_apps is not None:
            self._allowed_apps: Set[str] = {str(a).lower() for a in allowed_apps}
        else:
            self._allowed_apps = set(self._DEFAULT_ALLOWED_APPS)

        self.denied_roles: Set[str] = {"password text", "alert"}

        # GAP #3 FIX: "sudo" removed from high_risk_name_patterns.
        # Sudo on its own is NOT high-risk; ``sudo apt-get install nodejs`` is
        # a legitimate autonomous action.  The dangerous patterns that matter
        # (sudo su, sudo -i, sudo -s shell escalation) are already blocked by
        # the ExecutionPlanner's DANGEROUS_PATTERNS regex scan.
        # Retaining "delete", "remove", "format", "erase" here so destructive
        # operations that reach the policy engine still get human confirmation.
        self.high_risk_name_patterns = [
            re.compile(r"delete",  re.IGNORECASE),
            re.compile(r"remove",  re.IGNORECASE),
            re.compile(r"format",  re.IGNORECASE),
            re.compile(r"erase",   re.IGNORECASE),
        ]

        _logger.info(
            "PolicyEngine initialised. Allowed apps: %d entries.",
            len(self._allowed_apps),
        )

    # =========================================================================
    # TRUSTED INSTALLER HELPER (GAP #3 FIX)
    # =========================================================================

    def _is_trusted_installer_command(self, command: str) -> bool:
        """
        Return True if *command* starts with a known safe installer prefix.

        GAP #3 FIX: Pre-approved commands bypass the sudo/high-risk content
        check so the agent can autonomously install packages without requiring
        operator confirmation for each step.

        Security gate: this method is only called after validate_action_dict()
        has confirmed the action dict carries ``_trusted_installer=True``, which
        is set by ExecutionPlanner only after its own 50+ dangerous-pattern
        scan has cleared the command.  Two independent checks must both pass.
        """
        if not command:
            return False
        cmd_lower = command.strip().lower()
        for prefix in self._TRUSTED_INSTALLER_PREFIXES:
            if cmd_lower.startswith(prefix):
                return True
        return False

    # =========================================================================
    # DYNAMIC ALLOWLIST — BUG-C2 FIX
    # =========================================================================

    @property
    def allowed_apps(self) -> FrozenSet[str]:
        """Read-only snapshot of the current allowlist (thread-safe)."""
        with self._apps_lock:
            return frozenset(self._allowed_apps)

    def allow_app(self, app_name: str) -> None:
        """
        Dynamically add *app_name* to the allowlist.

        BUG-C2 FIX
        ----------
        AutonomousInstaller must call this after every successful tool
        installation so that subsequent actions on the newly installed
        application (e.g. ``focused_app = "nodejs"``) are permitted.

        Parameters
        ----------
        app_name:
            Process name as it appears in WorldGraph.focused_app.
            Normalised to lowercase automatically.
        """
        if not isinstance(app_name, str) or not app_name.strip():
            _logger.warning(
                "[PolicyEngine] allow_app: ignoring empty or non-string app_name %r.",
                app_name,
            )
            return

        normalised = app_name.strip().lower()
        with self._apps_lock:
            already = normalised in self._allowed_apps
            self._allowed_apps.add(normalised)

        if not already:
            _logger.info(
                "[PolicyEngine] allow_app: '%s' dynamically added to allowlist "
                "(total=%d). All future actions in this app will be ALLOW-ed.",
                normalised,
                len(self._allowed_apps),
            )
        else:
            _logger.debug(
                "[PolicyEngine] allow_app: '%s' already in allowlist — no-op.",
                normalised,
            )

    def allow_apps(self, app_names) -> None:
        """
        Bulk version of allow_app().  Accepts any iterable of app name strings.

        Useful when an installer step adds multiple related executables
        (e.g. ``["node", "npm", "npx"]`` from a single nodejs install).
        """
        for name in app_names:
            self.allow_app(name)

    def warn_if_unlisted(self, app_name: str) -> None:
        """
        Log a warning when *app_name* is not in the allowlist.

        Useful for pre-flight checks: operators can detect app gaps before
        the first action is denied, rather than only seeing DENY in logs.
        """
        if not app_name:
            return
        normalised = app_name.lower()
        with self._apps_lock:
            listed = normalised in self._allowed_apps
        if not listed:
            _logger.warning(
                "[PolicyEngine] warn_if_unlisted: active application %r is NOT "
                "in the allowlist.  All autonomous interactions will be DENIED "
                "until it is added.  Call allow_app(%r) or update policy.yaml.",
                app_name,
                app_name,
            )

    # =========================================================================
    # HUMAN APPROVAL SIGNAL-FILE SUPPORT
    # =========================================================================

    def approval_signal_path(self, action_key: str) -> str:
        """Return the filesystem path of the pending-approval signal file."""
        return os.path.join(
            self._APPROVAL_SIGNAL_DIR,
            f"{self._APPROVAL_SIGNAL_PREFIX}{action_key}.signal",
        )

    def check_human_approval(self, action_key: str) -> bool:
        """
        Return True if the human has approved the action by deleting its
        signal file.

        Design: the signal file's *absence* means approved.  The operator
        workflow is: see stderr notification → delete the file to approve →
        file absence detected → action proceeds.

        Fail-open: if os.path.exists() raises (e.g. filesystem error), the
        method returns True so a transient stat error cannot block execution
        indefinitely.

        Parameters
        ----------
        action_key:
            16-char hex key from ActionRanker.action_key().

        Returns
        -------
        bool
            True  — file absent → approved.
            False — file present → still pending.
        """
        path = self.approval_signal_path(action_key)
        try:
            approved = not os.path.exists(path)
        except OSError as exc:
            _logger.warning(
                "[PolicyEngine] check_human_approval: could not stat %r: %s "
                "— treating as approved (fail-open).",
                path, exc,
            )
            return True

        if approved:
            _logger.info(
                "[PolicyEngine] HUMAN_APPROVAL_GRANTED: action_key=%r signal=%r",
                action_key, path,
            )
        return approved

    # =========================================================================
    # PRIMARY ENTRY POINT — dict-based (no AT-SPI required)
    # =========================================================================

    def validate_action_dict(
        self,
        action: dict,
        *,
        focused_app: str = "__unknown_app__",
    ) -> Tuple[str, Optional[str]]:
        """
        Validate *action* dict against the current policy.

        Parameters
        ----------
        action:
            Action dict with at least an ``"operation"`` field.
        focused_app:
            Process name of the currently focused application as reported
            by WorldGraph.  Defaults to ``"__unknown_app__"`` (warmup sentinel).

        Returns
        -------
        (decision, reason)
            decision is one of ALLOW / DENY / REQUIRE_HUMAN_CONFIRMATION.
            reason is a human-readable explanation string, or None on ALLOW.
        """
        try:
            return self._validate_action_dict_inner(action, focused_app)
        except Exception as exc:
            _logger.error(
                "[PolicyEngine] validate_action_dict: unexpected error (fail-closed): %s",
                exc,
            )
            return self.DENY, f"Policy validation error (fail-closed): {exc}"

    def _validate_action_dict_inner(
        self,
        action: dict,
        focused_app: str,
    ) -> Tuple[str, Optional[str]]:
        if not isinstance(action, dict):
            return self.DENY, "Action must be a dict"

        op = str(action.get("operation") or "").lower().strip()
        if not op:
            return self.DENY, "Action has no 'operation' field"

        # DONE always succeeds — it terminates the task cleanly
        if op == "done":
            return self.ALLOW, None

        # ---- Application allowlist ----
        app = str(focused_app or "__unknown_app__").lower().strip()
        with self._apps_lock:
            app_allowed = app in self._allowed_apps
        if not app_allowed:
            reason = (
                f"Unauthorized application: {app!r}. "
                "Add to PolicyEngine.allowed_apps or call allow_app() "
                "after installation."
            )
            _logger.warning(
                "[PolicyEngine] DENY: op=%r app=%r — %s", op, app, reason
            )
            return self.DENY, reason

        # ---- Synthetic role check ----
        synthetic_role = self._OP_TO_SYNTHETIC_ROLE.get(op, op)
        if synthetic_role in self.denied_roles:
            reason = f"Forbidden operation role: {synthetic_role!r}"
            _logger.warning("[PolicyEngine] DENY: %s", reason)
            return self.DENY, reason

        # ---- Semantic type/write into non-text target ----
        if op in ("write", "type") and action.get("target_role"):
            target_role = str(action["target_role"]).lower()
            if "text" not in target_role and "entry" not in target_role:
                reason = (
                    f"Semantic violation: type/write into role {target_role!r}. "
                    "Only 'text' and 'entry' roles are writable."
                )
                _logger.warning("[PolicyEngine] DENY: %s", reason)
                return self.DENY, reason

        # ---- High-risk content check ----
        # GAP #3 FIX: Trusted installer commands bypass the high_risk_name_patterns
        # check so sudo apt-get install / pip install / npm install etc. do not
        # trigger REQUIRE_HUMAN_CONFIRMATION for every autonomous install step.
        #
        # The bypass is guarded by TWO conditions:
        #   1. The action dict carries ``_trusted_installer=True`` (set by
        #      ExecutionPlanner after its own 50+ dangerous-pattern scan passed).
        #   2. The command starts with a known safe installer prefix (e.g.
        #      "apt-get install", "pip install") checked by
        #      _is_trusted_installer_command().
        #
        # Both conditions must hold — a missing flag or unknown prefix falls
        # through to the normal high_risk_name_patterns scan below.
        _trusted_flag: bool = bool(action.get("_trusted_installer", False))
        content_to_check = self._extract_risk_content(op, action)

        if content_to_check.strip():
            # Check if this is a pre-approved installer command
            _bypass_high_risk = (
                _trusted_flag
                and op in ("command", "install")
                and self._is_trusted_installer_command(content_to_check)
            )
            if _bypass_high_risk:
                _logger.info(
                    "[PolicyEngine] GAP-3 TRUSTED_INSTALLER bypass: op=%r "
                    "command=%r — skipping high_risk_name_patterns scan.",
                    op, content_to_check[:80],
                )
            else:
                for pat in self.high_risk_name_patterns:
                    if pat.search(content_to_check):
                        reason = (
                            f"High-risk content detected (pattern={pat.pattern!r}): "
                            f"{content_to_check[:80]!r}"
                        )
                        _logger.warning(
                            "[PolicyEngine] REQUIRE_HUMAN_CONFIRMATION: op=%r "
                            "app=%r pattern=%r content=%r",
                            op, app, pat.pattern, content_to_check[:120],
                        )
                        return self.REQUIRE_HUMAN_CONFIRMATION, reason

        # ---- Unknown operation — DENY to fail closed ----
        if op not in self._OP_TO_SYNTHETIC_ROLE:
            reason = f"Unknown operation: {op!r}"
            _logger.warning("[PolicyEngine] DENY: %s", reason)
            return self.DENY, reason

        return self.ALLOW, None

    def _extract_risk_content(self, op: str, action: dict) -> str:
        """Extract the content string to scan for high-risk patterns."""
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
            return str(action.get("path") or "")
        return ""

    # =========================================================================
    # SECONDARY ENTRY POINT — AT-SPI node-based
    # =========================================================================

    def validate(self, node, action: str) -> Tuple[str, Optional[str]]:
        """
        Validate *action* against an AT-SPI accessibility *node*.

        Parameters
        ----------
        node:
            pyatspi Accessible node.
        action:
            Action string (e.g. ``"click"``, ``"type"``).

        Returns
        -------
        (decision, reason) — same semantics as validate_action_dict().
        """
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
            app = (
                app_obj.name.lower()
                if app_obj and app_obj.name
                else "unknown"
            )
        except Exception as exc:
            return self.DENY, f"Cannot read application identity: {exc}"

        if app == "unknown":
            return self.DENY, "Application identity unavailable"

        with self._apps_lock:
            app_allowed = app in self._allowed_apps
        if not app_allowed:
            return self.DENY, (
                f"Unauthorized application: {app!r}. "
                "Call allow_app() or update policy.yaml."
            )

        if role in self.denied_roles:
            return self.DENY, f"Forbidden role: {role}"

        if action == "type" and ("text" not in role and "entry" not in role):
            return self.DENY, f"Semantic violation: type into role '{role}'"

        for pat in self.high_risk_name_patterns:
            if pat.search(name):
                _logger.warning(
                    "[PolicyEngine] REQUIRE_HUMAN_CONFIRMATION (AT-SPI): "
                    "role=%r name=%r app=%r pattern=%r",
                    role, name, app, pat.pattern,
                )
                return (
                    self.REQUIRE_HUMAN_CONFIRMATION,
                    f"High-risk label detected: {pat.pattern}",
                )

        return self.ALLOW, None
