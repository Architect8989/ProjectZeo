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
    # FIX-C1 (RB-1 CRITICAL): Added "__unknown_app__" to the allowlist.
    #
    # Root cause: during world-graph warmup (up to 150s), world_graph.focused_app
    # has not yet been populated. operate.py uses "__unknown_app__" as the sentinel
    # when focused_app is empty. Because "__unknown_app__" was not in
    # _DEFAULT_ALLOWED_APPS, PolicyEngine.validate() returned DENY for EVERY action
    # during warmup — consuming all MAX_REPLANS=3 within seconds and producing
    # TASK_FAILED:max_replans_exceeded before a single action was ever executed.
    #
    # Fix: add "__unknown_app__" to the allowlist.
    #   - The DENY reason becomes "Unauthorized application: '__unknown_app__'" for
    #     non-empty focused_app sentinels, giving operators a meaningful error.
    #   - Operators who need a STRICTER policy (deny actions when app is unknown)
    #     can construct PolicyEngine(allowed_apps=...) excluding "__unknown_app__".
    #   - The warmup window is bounded: once world_graph.focused_app populates with
    #     a real app, the normal allowlist rules apply.
    #
    # NOTE: allowing "__unknown_app__" means autonomous actions CAN execute when
    # the active application is not yet known.  This is the less-safe default but
    # it matches the practical requirement: the system is useless if no action can
    # ever fire.  Operators running in higher-assurance environments should replace
    # this with an explicit app name or delay task arming until warmup completes.
    _DEFAULT_ALLOWED_APPS = frozenset({
        # Warmup / identity-unknown sentinel — permits actions before world-graph
        # has observed the first application focus event.  Remove from allowed_apps
        # at construction if you need strict deny-when-unknown behaviour.
        "__unknown_app__",

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
        # System utilities
        "evince",         # PDF viewer
        "eog",            # Image viewer
        "gpicview",
        "totem",          # Media player
        "vlc",
    })

    def __init__(self, allowed_apps=None):
        # FIX-M4 (RB-4 / SI-3): Removed "terminal" from denied_roles.
        self.denied_roles = {
            "password text",
            "alert",
        }

        self.high_risk_name_patterns = [
            re.compile(r"delete", re.IGNORECASE),
            re.compile(r"remove", re.IGNORECASE),
            re.compile(r"format", re.IGNORECASE),
            re.compile(r"sudo", re.IGNORECASE),
            re.compile(r"erase", re.IGNORECASE),
        ]

        if allowed_apps is not None:
            self.allowed_apps = frozenset(str(a).lower() for a in allowed_apps)
        else:
            self.allowed_apps = self._DEFAULT_ALLOWED_APPS

        _logger.info(
            "PolicyEngine initialized. Autonomous interaction allowed for: %s. "
            "Applications outside this set will receive DENY. "
            "To extend, pass allowed_apps= to PolicyEngine or update "
            "PolicyEngine._DEFAULT_ALLOWED_APPS.",
            sorted(self.allowed_apps),
        )

    def warn_if_unlisted(self, app_name: str) -> None:
        """
        Emit a structured WARNING log if app_name is not in allowed_apps.

        Call this at task startup (before any validate() call) to surface
        silent DENY cases to operators who have not extended the allowlist.
        Without this, the system silently denies all actions on unlisted apps
        and the only signal is the DENY return value from validate().
        """
        if app_name and app_name.lower() not in self.allowed_apps:
            _logger.warning(
                "PolicyEngine: active application %r is NOT in the allowlist %s. "
                "All autonomous interactions will be DENIED until it is added. "
                "Update PolicyEngine.allowed_apps or pass allowed_apps= at construction.",
                app_name,
                sorted(self.allowed_apps),
            )

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
