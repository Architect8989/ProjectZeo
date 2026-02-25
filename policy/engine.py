import re
import logging

_logger = logging.getLogger(__name__)


class PolicyViolationError(RuntimeError):
    pass


class PolicyEngine:

    ALLOW = "ALLOW"
    DENY = "DENY"
    REQUIRE_HUMAN_CONFIRMATION = "REQUIRE_HUMAN_CONFIRMATION"

    
    _OP_TO_SYNTHETIC_ROLE: dict = {
        "click":        "push button",
        "write":        "text",
        "type":         "text",
        "press":        "keyboard",
        "hotkey":       "keyboard",
        "key":          "keyboard",
        "command":      "terminal",
        "install":      "terminal",
        "file_create":  "file",
        "scroll":       "scroll",
        "verify":       "verify",
        "done":         "done",
    }

    # Operations that carry elevated risk and require confirmed environment
    # stability before execution. These map to 'high_risk=True' in operate.py's
    # authority evaluation.
    _HIGH_RISK_OPERATIONS: frozenset = frozenset({
        "command",
        "install",
        "file_create",
    })

    
    _DEFAULT_ALLOWED_APPS = frozenset({
        # Warmup sentinel — permits actions before world-graph has observed focus
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
        # System utilities
        "evince", "eog", "gpicview", "totem", "vlc",
    })

    def __init__(self, allowed_apps=None):
        self.denied_roles = {
            "password text",
            "alert",
        }

        self.high_risk_name_patterns = [
            re.compile(r"delete",  re.IGNORECASE),
            re.compile(r"remove",  re.IGNORECASE),
            re.compile(r"format",  re.IGNORECASE),
            re.compile(r"sudo",    re.IGNORECASE),
            re.compile(r"erase",   re.IGNORECASE),
        ]

        if allowed_apps is not None:
            self.allowed_apps = frozenset(str(a).lower() for a in allowed_apps)
        else:
            self.allowed_apps = self._DEFAULT_ALLOWED_APPS

        _logger.info(
            "PolicyEngine initialized. Allowed apps: %s.",
            sorted(self.allowed_apps),
        )

    # =========================================================================
    # PRIMARY ENTRY POINT (no AT-SPI required)
    # =========================================================================

    def validate_action_dict(
        self,
        action: dict,
        *,
        focused_app: str = "__unknown_app__",
    ) -> tuple:
        
        try:
            return self._validate_action_dict_inner(action, focused_app)
        except Exception as e:
            return self.DENY, f"Policy validation error (fail-closed): {e}"

    def _validate_action_dict_inner(self, action: dict, focused_app: str) -> tuple:
        if not isinstance(action, dict):
            return self.DENY, "Action must be a dict"

        op = str(action.get("operation") or "").lower().strip()
        if not op:
            return self.DENY, "Action has no 'operation' field"

        # DONE is always allowed — it terminates the task cleanly.
        if op == "done":
            return self.ALLOW, None

        # ---- Application allowlist ----
        app = str(focused_app or "__unknown_app__").lower().strip()
        if app not in self.allowed_apps:
            return self.DENY, (
                f"Unauthorized application: {app!r}. "
                f"Allowed: {sorted(self.allowed_apps)}. "
                "Add to PolicyEngine.allowed_apps or pass allowed_apps= at construction."
            )

        # ---- Synthetic role check ----
        synthetic_role = self._OP_TO_SYNTHETIC_ROLE.get(op, op)
        if synthetic_role in self.denied_roles:
            return self.DENY, f"Forbidden operation role: {synthetic_role!r}"

        # ---- Semantic misuse: type into non-text target ----
        # This check only applies to 'write'/'type' operations.
        if op in ("write", "type"):
            target_role = str(action.get("target_role") or "text").lower()
            if "text" not in target_role and "entry" not in target_role and target_role != "text":
                # Only block if a target_role was explicitly specified and is non-text.
                # If target_role is absent, assume text (conservative default).
                if action.get("target_role"):
                    return self.DENY, (
                        f"Semantic violation: type/write into role {target_role!r}. "
                        "Only 'text' and 'entry' roles are writable."
                    )

        # ---- High-risk content check ----
        # Inspect the operation's content/command fields for dangerous patterns.
        content_to_check = ""
        if op == "command":
            content_to_check = str(action.get("command") or "")
        elif op in ("write", "type"):
            content_to_check = str(action.get("content") or action.get("text") or "")
        elif op in ("press", "hotkey", "key"):
            keys = action.get("keys") or action.get("key") or []
            if isinstance(keys, str):
                keys = [keys]
            content_to_check = " ".join(str(k) for k in keys)
        elif op == "file_create":
            content_to_check = str(action.get("path") or "")

        for pat in self.high_risk_name_patterns:
            if pat.search(content_to_check):
                _logger.warning(
                    "POLICY_REQUIRES_CONFIRMATION: operation=%r pattern=%r content=%r app=%r",
                    op, pat.pattern, content_to_check[:120], app,
                )
                return (
                    self.REQUIRE_HUMAN_CONFIRMATION,
                    f"High-risk content detected (pattern={pat.pattern!r}): {content_to_check[:80]!r}",
                )

        # ---- Unknown operation — DENY to fail closed ----
        if op not in self._OP_TO_SYNTHETIC_ROLE:
            return self.DENY, f"Unknown operation: {op!r}"

        return self.ALLOW, None

    # =========================================================================
    # SECONDARY ENTRY POINT (AT-SPI node required)
    # =========================================================================

    def validate(self, node, action: str) -> tuple:
        

        # ---- NODE INTROSPECTION (FAIL-CLOSED, EXPLICIT) ----
        try:
            role = (node.getRoleName() or "unknown").lower()
        except Exception as e:
            return self.DENY, f"Cannot read node role: {e}"

        try:
            name = (node.name or "").lower()
        except Exception as e:
            return self.DENY, f"Cannot read node name: {e}"

        try:
            app_obj = node.getApplication()
            app = (
                app_obj.name.lower()
                if app_obj and app_obj.name
                else "unknown"
            )
        except Exception as e:
            return self.DENY, f"Cannot read application identity: {e}"

        # ---- POLICY RULES ----

        if app == "unknown":
            return self.DENY, "Application identity unavailable"

        if app not in self.allowed_apps:
            return self.DENY, (
                f"Unauthorized application: {app!r}. "
                f"Allowed: {sorted(self.allowed_apps)}. "
                "Add to PolicyEngine.allowed_apps to permit."
            )

        if role in self.denied_roles:
            return self.DENY, f"Forbidden role: {role}"

        if action == "type" and ("text" not in role and "entry" not in role):
            return self.DENY, f"Semantic violation: type into role '{role}'"

        for pat in self.high_risk_name_patterns:
            if pat.search(name):
                _logger.warning(
                    "POLICY_REQUIRES_CONFIRMATION: role=%r name=%r app=%r pattern=%r",
                    role, name, app, pat.pattern,
                )
                return (
                    self.REQUIRE_HUMAN_CONFIRMATION,
                    f"High-risk label detected: {pat.pattern}",
                )

        return self.ALLOW, None

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    def warn_if_unlisted(self, app_name: str) -> None:
        
        if app_name and app_name.lower() not in self.allowed_apps:
            _logger.warning(
                "PolicyEngine: active application %r is NOT in the allowlist %s. "
                "All autonomous interactions will be DENIED until it is added. "
                "Update PolicyEngine.allowed_apps or pass allowed_apps= at construction.",
                app_name,
                sorted(self.allowed_apps),
            )
