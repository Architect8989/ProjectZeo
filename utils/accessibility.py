import pyatspi
import hashlib
import time
import json

class AccessibilityBackend:
    def __init__(self):
        self.registry = pyatspi.Registry

    def _get_stable_id(self, obj):
        """Generates a non-volatile SHA-256 hash for auditability."""
        try:
            app = obj.getApplication()
            app_name = app.name if app else "system"
            # Stable attributes: Application, Role, Name, Index
            raw = f"{app_name}|{obj.getRoleName()}|{obj.name}|{obj.getIndexInParent()}"
            return hashlib.sha256(raw.encode()).hexdigest()[:16]
        except Exception as e:
            raise RuntimeError(f"ID Generation Failure: {e}")

    def _snapshot_signature(self):
        """Generates a state signature to verify UI stability."""
        nodes = self.get_nodes()
        sig_data = sorted([(nid, n.getRoleName(), n.name) for nid, n in nodes.items()])
        return hashlib.sha256(json.dumps(sig_data).encode()).hexdigest()

    def wait_for_ui_stabilization(self, timeout=3.0):
        """Enforces two identical snapshots to confirm UI is quiet."""
        start = time.time()
        last_sig = None
        while time.time() - start < timeout:
            current_sig = self._snapshot_signature()
            if last_sig is not None and current_sig == last_sig:
                return 
            last_sig = current_sig
            time.sleep(0.2)
        raise RuntimeError("CRITICAL: UI failed to stabilize")

    def get_nodes(self):
        nodes = {}
        visited = set()
        desktop = self.registry.getDesktop(0)

        def walk(obj):
            if obj is None or hash(obj) in visited: return
            visited.add(hash(obj))
            try:
                node_id = self._get_stable_id(obj)
                nodes[node_id] = obj
                for i in range(obj.getChildCount()):
                    walk(obj.getChildAtIndex(i))
            except Exception as e:
                raise RuntimeError(f"AT-SPI traversal failure: {e}")
        
        walk(desktop)
        return nodes

    def perform_action(self, node, action_type, text=None):
        if action_type == "click":
            action_iface = node.queryAction()
            if action_iface.nActions > 0:
                action_iface.doAction(0)
                return True
            raise RuntimeError("Node exposes no semantic actions")
        
        elif action_type == "type":
            editable = node.queryEditableText()
            pos = editable.getCharacterCount()
            editable.insertText(pos, text, len(text))
            return True
        
        raise RuntimeError(f"Unsupported action: {action_type}")
