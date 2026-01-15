import pyatspi
import hashlib

class AccessibilityBackend:
    """
    STRICT EXECUTION INSTRUMENT.
    Zero sovereignty. Zero discovery during execution. Double-bind audit.
    """
    def __init__(self):
        self.registry = pyatspi.Registry

    def _get_stable_id(self, obj):
        """Deterministic ID generation for external tracking."""
        try:
            app = obj.getApplication()
            app_name = app.name if app else "system"
            raw = f"{app_name}|{obj.getRoleName()}|{obj.name}|{obj.getIndexInParent()}"
            return hashlib.sha256(raw.encode()).hexdigest()[:16]
        except Exception:
            raise RuntimeError("FAIL_CLOSED: ID_GENERATION_FAILURE")

    def get_nodes(self, max_depth=5):
        """
        Passive discovery only. 
        Sovereignty constraint: Used by controller to 'freeze' state before execution.
        """
        nodes = {}
        visited = set()
        desktop = self.registry.getDesktop(0)

        def walk(obj, depth):
            if obj is None or hash(obj) in visited or depth > max_depth:
                return
            visited.add(hash(obj))
            nid = self._get_stable_id(obj)
            nodes[nid] = obj
            for i in range(obj.getChildCount()):
                walk(obj.getChildAtIndex(i), depth + 1)

        walk(desktop, 0)
        return nodes

    def execute(self, mode, policy_engine, audit_callback, node, action_type, text=None):
        """
        HARD EXECUTION GATE.
        1. Mode check (Active vs Observer)
        2. Policy evaluation (bool, reason)
        3. Fail-hard audit (Intent -> Action -> Effect)
        4. Context lock: Operates only on the provided node reference.
        """
        # 1. Mode Guard
        if mode != "ACTIVE":
            raise PermissionError("NON_SOVEREIGN_VIOLATION: Execution blocked in OBSERVER mode.")

        # 2. Policy Guard: Contract aligned with (bool, reason)
        allowed, reason = policy_engine.validate(node, action_type)
        if not allowed:
            raise PermissionError(f"POLICY_VIOLATION: {reason}")

        # 3. Audit Phase 1: Intent
        # Must fail-hard; no boolean interpretation.
        audit_callback("INTENT", node, action_type)

        # 4. Hardware Execution: No retries, no stabilization.
        try:
            if action_type == "click":
                node.queryAction().doAction(0)
            elif action_type == "type":
                if text is None:
                    raise ValueError("Type action requires text input.")
                editable = node.queryEditableText()
                editable.insertText(editable.getCharacterCount(), text, len(text))
            else:
                raise NotImplementedError(f"Unsupported action: {action_type}")
                
        except Exception as e:
            # No recovery. Fail closed.
            raise RuntimeError(f"HARDWARE_EXECUTION_FAILURE: {str(e)}")

        # 5. Audit Phase 2: Effect
        # Final evidence-binding; must fail-hard.
        audit_callback("EFFECT", node, action_type)
