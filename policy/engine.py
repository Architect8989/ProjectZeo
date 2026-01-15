import re


class PolicyEngine:
    """
    PURE POLICY ORACLE.
    Decides permission only. Never executes. Never prompts.
    """

    ALLOW = "ALLOW"
    DENY = "DENY"
    REQUIRE_HUMAN_CONFIRMATION = "REQUIRE_HUMAN_CONFIRMATION"

    def __init__(self):
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

        # Apps allowed for autonomous interaction
        self.allowed_apps = {
            "google-chrome",
            "firefox",
            "libreoffice",
            "gedit",
        }

    def validate(self, node, action: str):
        """
        Returns one of:
        - ALLOW
        - DENY
        - REQUIRE_HUMAN_CONFIRMATION
        """

        try:
            role = (node.getRoleName() or "unknown").lower()
            name = (node.name or "").lower()

            app_obj = node.getApplication()
            app = app_obj.name.lower() if app_obj and app_obj.name else "unknown"

            # Hard deny: cannot identify application
            if app == "unknown":
                return self.DENY, "Application identity unavailable"

            # Hard deny: app not allow-listed
            if app not in self.allowed_apps:
                return self.DENY, f"Unauthorized application: {app}"

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

        except Exception as e:
            # Policy must fail closed
            return self.DENY, f"Policy internal error: {e}"
