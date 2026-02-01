import pyatspi
import hashlib
from typing import Optional

from policy.engine import PolicyEngine


class AccessibilityBackend:
    """
    STRICT EXECUTION INSTRUMENT.

    ROLE:
    - Executes concrete UI actions on pre-selected nodes
    - Performs NO discovery during execution
    - Owns NO orchestration logic

    CONTRACT:
    - observer and screenpipe are OPTIONAL, late-bound references
    - If accessed without wiring → FAIL CLOSED
    """

    def __init__(self):
        self.registry = pyatspi.Registry

        # ---- Late-bound system interfaces (REQUIRED BY KERNEL PATHS) ----
        self.observer: Optional[object] = None
        self.screenpipe: Optional[object] = None

    # -------------------------------------------------
    # Internal helpers
    # -------------------------------------------------

    def _require_wired(self):
        """
        Fail-closed if system interfaces are accessed without wiring.
        """
        if self.observer is None or self.screenpipe is None:
            raise RuntimeError(
                "ACCESSIBILITY_BACKEND_NOT_WIRED: "
                "observer/screenpipe must be explicitly injected"
            )

    def _get_stable_id(self, obj):
        """Deterministic ID generation for external tracking."""
        try:
            app = obj.getApplication()
            app_name = app.name if app else "system"
            raw = f"{app_name}|{obj.getRoleName()}|{obj.name}|{obj.getIndexInParent()}"
            return hashlib.sha256(raw.encode()).hexdigest()[:16]
        except Exception:
            raise RuntimeError("FAIL_CLOSED: ID_GENERATION_FAILURE")

    # -------------------------------------------------
    # Passive Discovery (Observer-only)
    # -------------------------------------------------

    def get_nodes(self, max_depth=5):
        """
        Passive discovery only.

        Sovereignty constraint:
        - Must only be used in OBSERVER mode
        - No mutation
        - No execution
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

    # -------------------------------------------------
    # HARD EXECUTION GATE
    # -------------------------------------------------

    def execute(
        self,
        *,
        mode: str,
        policy_engine: PolicyEngine,
        audit_callback,
        node,
        action_type: str,
        text: Optional[str] = None,
    ):
        """
        HARD EXECUTION GATE.

        ENFORCEMENT ORDER:
        1. Mode check
        2. Policy validation
        3. Audit (intent)
        4. Hardware execution (fail-closed)
        5. Audit (effect)
        """

        # 1. Mode Guard
        if mode != "ACTIVE":
            raise PermissionError(
                "NON_SOVEREIGN_VIOLATION: Execution blocked in OBSERVER mode."
            )

        # 2. Policy Guard
        decision, reason = policy_engine.validate(node, action_type)
        if decision != PolicyEngine.ALLOW:
            raise PermissionError(f"POLICY_VIOLATION: {reason}")

        # 3. Audit Phase 1: Intent
        audit_callback("INTENT", node, action_type)

        # 4. Hardware Execution (FAIL CLOSED)
        try:
            if action_type == "click":
                node.queryAction().doAction(0)

            elif action_type == "type":
                if text is None:
                    raise ValueError("Type action requires text input.")
                editable = node.queryEditableText()
                editable.insertText(
                    editable.getCharacterCount(),
                    text,
                    len(text),
                )

            else:
                raise NotImplementedError(
                    f"Unsupported action: {action_type}"
                )

        except Exception as e:
            raise RuntimeError(
                f"HARDWARE_EXECUTION_FAILURE: {str(e)}"
            )

        # 5. Audit Phase 2: Effect
        audit_callback("EFFECT", node, action_type)
