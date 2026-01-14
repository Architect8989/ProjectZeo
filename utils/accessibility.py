import pyatspi
import hashlib
import time
import json


class AccessibilityBackend:
    def __init__(self):
        self.registry = pyatspi.Registry

    def _get_stable_id(self, obj):
        try:
            app = obj.getApplication()
            app_name = app.name if app else "system"
            raw = f"{app_name}|{obj.getRoleName()}|{obj.name}|{obj.getIndexInParent()}"
            return hashlib.sha256(raw.encode()).hexdigest()[:16]
        except Exception as e:
            raise RuntimeError(f"Stable ID generation failed: {e}")

    def _snapshot_signature(self):
        nodes = self.get_nodes()
        sig_data = sorted([(nid, n.getRoleName(), n.name) for nid, n in nodes.items()])
        return hashlib.sha256(json.dumps(sig_data).encode()).hexdigest()

    def wait_for_ui_stabilization(self, timeout=3.0):
        start = time.time()
        last = None
        while time.time() - start < timeout:
            cur = self._snapshot_signature()
            if last is not None and cur == last:
                return
            last = cur
            time.sleep(0.3)
        raise RuntimeError("UI failed to stabilize within timeout")

    def get_nodes(self):
        nodes = {}
        visited = set()
        desktop = self.registry.getDesktop(0)

        def walk(obj):
            if obj is None or hash(obj) in visited:
                return
            visited.add(hash(obj))
            nid = self._get_stable_id(obj)
            nodes[nid] = obj
            for i in range(obj.getChildCount()):
                walk(obj.getChildAtIndex(i))

        walk(desktop)
        return nodes

    def perform_action(self, node, action_type, text=None):
        try:
            if action_type == "click":
                node.queryAction().doAction(0)
            elif action_type == "type":
                editable = node.queryEditableText()
                editable.insertText(editable.getCharacterCount(), text or "", len(text or ""))
            else:
                raise RuntimeError(f"Unsupported action: {action_type}")
        except Exception as e:
            raise RuntimeError(f"Hardware execution failure: {e}")
