import re


class PolicyEngine:
    def __init__(self):
        self.denied_roles = {"terminal", "password text", "alert", "dialog"}
        self.denied_names = [
            re.compile(r"delete", re.IGNORECASE),
            re.compile(r"format", re.IGNORECASE),
            re.compile(r"sudo", re.IGNORECASE),
            re.compile(r"remove", re.IGNORECASE),
        ]
        self.allowed_apps = {"google-chrome", "firefox", "libreoffice", "gedit"}

    def validate(self, node, action: str):
        try:
            role = (node.getRoleName() or "unknown").lower()
            name = (node.name or "").lower()

            app_obj = node.getApplication()
            app = app_obj.name.lower() if app_obj else "unknown"

            if not app or app == "unknown":
                return False, "Application identity unavailable"

            if app not in self.allowed_apps:
                return False, f"Unauthorized Application: {app}"

            if role in self.denied_roles:
                return False, f"Unauthorized Role: {role}"

            if action == "type" and ("text" not in role and "entry" not in role):
                return False, f"Semantic abuse: type into role '{role}'"

            for pat in self.denied_names:
                if pat.search(name):
                    return False, f"Destructive label blocked: {pat.pattern}"

            return True, None
        except Exception as e:
            return False, f"Policy internal error: {e}"
