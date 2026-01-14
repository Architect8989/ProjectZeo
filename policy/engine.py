import re
import sys

class PolicyEngine:
    def __init__(self):
        # Authority Containment Layer v1 (Frozen)
        self.denied_roles = {"terminal", "password text", "alert", "dialog"}
        self.denied_names = [
            re.compile(r"delete", re.IGNORECASE),
            re.compile(r"format", re.IGNORECASE),
            re.compile(r"settings", re.IGNORECASE),
            re.compile(r"sudo", re.IGNORECASE),
            re.compile(r"remove", re.IGNORECASE)
        ]
        self.allowed_apps = {"google-chrome", "firefox", "libreoffice", "gedit"}

    def validate(self, node, action: str):
        """
        Policy v1: Deny-by-default, identity-required, search-based matching.
        """
        try:
            role = (node.getRoleName() or "unknown").lower()
            name = (node.name or "").lower()
            
            app_obj = node.getApplication()
            app = app_obj.name.lower() if app_obj else "unknown"

            # 1. Identity Requirement (Correction #2)
            if not app or app == "unknown":
                return False, "Application identity unavailable"

            # 2. Strict Application Containment
            if app not in self.allowed_apps:
                return False, f"Unauthorized Application: {app}"

            # 3. Role-based Blocking
            if role in self.denied_roles:
                return False, f"Unauthorized Role: {role}"

            # 4. Action-Role Synergy (Correction #3)
            if action == "type" and "text" not in role and "entry" not in role:
                return False, f"Semantic Abuse: Cannot 'type' into non-input role '{role}'"

            # 5. Global Search-based Denial (Correction #1)
            for pattern in self.denied_names:
                if pattern.search(name): # Fix: search() instead of match()
                    return False, f"Destructive Action Blocked: Label contains '{pattern.pattern}'"

            return True, None
        except Exception as e:
            return False, f"Policy Evaluation Internal Failure: {str(e)}"
