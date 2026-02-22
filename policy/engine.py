import re
import logging

_logger = logging.getLogger(__name__)


class PolicyViolationError(RuntimeError):
    
    pass


class PolicyEngine:
    

    ALLOW = "ALLOW"
    DENY = "DENY"
    REQUIRE_HUMAN_CONFIRMATION = "REQUIRE_HUMAN_CONFIRMATION"

    # HAR-04 / FIX-4 (SI-NEW-02): Default allowlist expanded to include common
    # terminal emulators, code editors, and file managers.
    #
    # Bug: The previous allowlist contained only {"google-chrome", "firefox",
    # "libreoffice", "gedit"}. Any task requiring gnome-terminal, code, kate,
    # nautilus, or any other unlisted application would receive DENY on every
    # action, burning all MAX_REPLANS=3 on consecutive DENY cycles and failing
    # with TASK_FAILED:max_replans_exceeded — with no operator-visible signal
    # unless policy logging was enabled.
    #
    # Fix: add the most commonly needed categories:
    #   - Terminal emulators: gnome-terminal, xterm, konsole, xfce4-terminal,
    #                         mate-terminal, tilix, alacritty, Terminal, iTerm
    #   - Code editors:       code (VSCode), kate, sublime_text, atom, gedit
    #   - File managers:      nautilus, thunar, nemo, dolphin, Finder
    #   - Office suite:       libreoffice (and sub-apps: writer, calc, impress)
    #   - Browsers:           google-chrome, firefox, chromium, brave-browser
    #
    # Operators who need a stricter allowlist should pass allowed_apps= at
    # construction or update policy.yaml (preferred — avoids source changes).
    # Operators who need additional apps should do the same rather than
    # modifying this constant directly.
    _DEFAULT_ALLOWED_APPS = frozenset({
        # Browsers
        "google-chrome",
        "firefox",
        "chromium",
        "chromium-browser",
        "brave-browser",
        "microsoft-edge",
        # Office / document editors
        "libreoffice",
        "soffice",
        "libreoffice-writer",
        "libreoffice-calc",
        "libreoffice-impress",
        # Text / code editors
        "gedit",
        "kate",
        "code",
        "code-oss",
        "sublime_text",
        "atom",
        "mousepad",
        "pluma",
        # Terminal emulators
        "gnome-terminal",
        "xterm",
        "konsole",
        "xfce4-terminal",
        "mate-terminal",
        "tilix",
        "alacritty",
        "terminal",       # macOS Terminal.app
        "iterm",          # macOS iTerm2
        "iterm2",
        "hyper",
        # File managers
        "nautilus",
        "thunar",
        "nemo",
        "dolphin",
        "finder",         # macOS Finder
        "pcmanfm",
        # System utilities (commonly needed by autonomous tasks)
        "evince",         # PDF viewer
        "eog",            # Image viewer
        "gpicview",
        "totem",          # Media player
        "vlc",
    })

    def __init__(self, allowed_apps=None):
        # Roles that should never be interacted with automatically
        self.denied_roles = {
            "terminal",
            "password text",
            "alert",
            "dialog",
        }

        # Names that imply destructive or privileged intent
        self.high_risk_name_patterns = [
            re.compile(r"delete", re.IGNORECASE),
            re.compile(r"remove", re.IGNORECASE),
            re.compile(r"format", re.IGNORECASE),
            re.compile(r"sudo", re.IGNORECASE),
            re.compile(r"erase", re.IGNORECASE),
        ]

        # HAR-04: Apps allowed for autonomous interaction.
        # Accepts a custom set for operator extensibility; falls back to the
        # conservative default when not provided.
        if allowed_apps is not None:
            self.allowed_apps = frozenset(str(a).lower() for a in allowed_apps)
        else:
            self.allowed_apps = self._DEFAULT_ALLOWED_APPS

        # HAR-04: Emit a startup warning so operators know the allowlist is
        # active.  Called once at construction rather than per validate() call
        # to avoid log spam during task execution.
        _logger.info(
            "PolicyEngine initialized. Autonomous interaction allowed for: %s. "
            "Applications outside this set will receive DENY. "
            "To extend, pass allowed_apps= to PolicyEngine or update "
            "PolicyEngine._DEFAULT_ALLOWED_APPS.",
            sorted(self.allowed_apps),
        )

    # -------------------------------------------------
    # HAR-04: Operator utility — emit structured warning for unlisted app.
    # -------------------------------------------------

    def warn_if_unlisted(self, app_name: str) -> None:
        """
        Emit a structured WARNING log if app_name is not in allowed_apps.

        Call this at task startup (before any validate() call) to surface
        silent DENY cases to operators who have not extended the allowlist.
        Without this, the system silently denies all actions on unlisted apps
        and the only signal is the DENY return value from validate().

        Example::

            policy.warn_if_unlisted(current_app_name)
        """
        if app_name and app_name.lower() not in self.allowed_apps:
            _logger.warning(
                "PolicyEngine: active application %r is NOT in the allowlist %s. "
                "All autonomous interactions will be DENIED until it is added. "
                "Update PolicyEngine.allowed_apps or pass allowed_apps= at construction.",
                app_name,
                sorted(self.allowed_apps),
            )

    # -------------------------------------------------

    def validate(self, node, action: str):
        """
        Returns:
        - (ALLOW, None)
        - (DENY, reason)
        - (REQUIRE_HUMAN_CONFIRMATION, reason)

        Policy MUST fail closed.
        """

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

        # Hard deny: cannot identify application
        if app == "unknown":
            return self.DENY, "Application identity unavailable"

        # Hard deny: app not allow-listed
        # HAR-04: This is the enforcement point for the allowlist.
        # An unlisted application causes DENY for every action, including
        # read-only ones. Operators must explicitly add apps to allowed_apps.
        if app not in self.allowed_apps:
            return self.DENY, (
                f"Unauthorized application: {app!r}. "
                f"Allowed: {sorted(self.allowed_apps)}. "
                "Add to PolicyEngine.allowed_apps to permit."
            )

        # Hard deny: forbidden UI role
        if role in self.denied_roles:
            return self.DENY, f"Forbidden role: {role}"

        # Semantic misuse: typing into non-text elements
        if action == "type" and ("text" not in role and "entry" not in role):
            return self.DENY, f"Semantic violation: type into role '{role}'"

        # High-risk intent → require human confirmation
        for pat in self.high_risk_name_patterns:
            if pat.search(name):
                return (
                    self.REQUIRE_HUMAN_CONFIRMATION,
                    f"High-risk label detected: {pat.pattern}",
                )

        # Otherwise safe
        return self.ALLOW, None
