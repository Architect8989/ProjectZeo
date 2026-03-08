"""
core/safety/verisafe_agent.py — VeriSafe Agent (VSA) Formal Verification

Layer 2 integration from the research report (§3, Research §10 Phase 2).

VeriSafe Agent works in two phases:

  Phase 1 — Autoformalization (at task start):
    Translates natural language intent into a formal logical specification
    (task_spec JSON). Example invariants:
      - "payment amount must be confirmed before submission"
      - "do not navigate away from the active form"
      - "file must be saved before closing"

  Phase 2 — Runtime Verification (before each action dispatch):
    Checks the planned action against task_spec. If the action would violate
    a safety invariant, VSA blocks it and records the violation reason so the
    planner can reformulate.

Implementation is a lightweight pure-Python approximation of the VSA logic
described in the MobiCom 2025 paper. The autoformalization step uses an LLM
call at task start; runtime verification is deterministic Python logic with
zero LLM cost.

Reference: "VeriSafe Agent: Safeguarding Mobile GUI Agent via Logic-based
Action Verification" — ACM MobiCom 2025, arxiv 2503.18492
"""
from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any, Callable, Dict, List, Optional

_logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Autoformalization system prompt
# ─────────────────────────────────────────────────────────────────────────────

_AUTOFORMALIZE_SYSTEM = """\
You are a formal safety specification generator for an autonomous computer agent.
Given a task description, produce a JSON object encoding the key safety invariants
that must hold throughout execution.

Format:
{
  "task_summary": "<one sentence>",
  "invariants": [
    {
      "id": "INV-1",
      "description": "<human-readable invariant>",
      "blocked_operations": ["<op1>", "<op2>"],
      "blocked_patterns": ["<regex_or_substring>"],
      "trigger_on_label_contains": ["<ui_label_fragment>"],
      "severity": "BLOCK" | "WARN"
    }
  ],
  "confirmation_required_for": ["<action_type_or_pattern>"],
  "max_allowed_irreversible_actions": <int or null>
}

Focus on:
- Payment/financial submission invariants
- Irreversible data deletion guards
- Form submission before confirmation guards
- External communication (email, post, publish) constraints
- Credential/password handling restrictions

Respond ONLY with the JSON. No markdown, no prose.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Runtime checker
# ─────────────────────────────────────────────────────────────────────────────

def _check_invariant(invariant: Dict[str, Any], action: Dict[str, Any]) -> Optional[str]:
    """
    Pure Python invariant check. Returns violation description or None.
    Zero LLM cost — deterministic.
    """
    op = str(action.get("operation", "")).lower()
    cmd = str(action.get("command", "")).lower()
    content = str(action.get("content", action.get("text", ""))).lower()
    label = str(action.get("target_label", action.get("label", ""))).lower()
    path = str(action.get("path", "")).lower()
    combined = f"{op} {cmd} {content} {label} {path}"

    # Check blocked operations
    blocked_ops: List[str] = invariant.get("blocked_operations", [])
    for b_op in blocked_ops:
        if op == b_op.lower():
            return (
                f"Invariant {invariant['id']}: operation '{op}' is blocked. "
                f"Constraint: {invariant['description']}"
            )

    # Check blocked patterns (substring or regex)
    blocked_patterns: List[str] = invariant.get("blocked_patterns", [])
    for pat in blocked_patterns:
        try:
            if re.search(pat, combined, re.IGNORECASE):
                return (
                    f"Invariant {invariant['id']}: pattern '{pat}' matched. "
                    f"Constraint: {invariant['description']}"
                )
        except re.error:
            if pat.lower() in combined:
                return (
                    f"Invariant {invariant['id']}: '{pat}' found in action. "
                    f"Constraint: {invariant['description']}"
                )

    # Check trigger_on_label_contains
    label_triggers: List[str] = invariant.get("trigger_on_label_contains", [])
    for fragment in label_triggers:
        if fragment.lower() in label or fragment.lower() in content:
            return (
                f"Invariant {invariant['id']}: high-risk label '{fragment}' detected. "
                f"Constraint: {invariant['description']}"
            )

    return None


# ─────────────────────────────────────────────────────────────────────────────
# VeriSafeAgent class
# ─────────────────────────────────────────────────────────────────────────────

class VeriSafeAgent:
    """
    VeriSafe Agent: formal verification for GUI agent actions.

    Lifecycle:
      1. Call start_task(objective, llm_callable) at task start to
         autoformalize the task spec.
      2. Call verify(action) before each action dispatch.
      3. Call last_violation_reason() to get feedback for the planner.
    """

    def __init__(self) -> None:
        self._task_spec: Optional[Dict[str, Any]] = None
        self._last_violation: str = ""
        self._invariants: List[Dict[str, Any]] = []
        self._irreversible_count: int = 0
        self._max_irreversible: Optional[int] = None
        self._lock = threading.Lock()

    def start_task(
        self,
        objective: str,
        llm_callable: Optional[Callable] = None,
        timeout_seconds: float = 60.0,
    ) -> bool:
        """
        Phase 1: Autoformalize the task into a formal spec.
        Returns True if spec was generated, False if autoformalization failed
        (in which case verify() will be permissive / fail-open).
        """
        if llm_callable is None:
            _logger.info("[VSA] No LLM callable — autoformalization skipped (fail-open).")
            return False

        result_holder: list = [None]
        error_holder: list = [None]

        def _call():
            try:
                raw = llm_callable(
                    messages=[
                        {"role": "system", "content": _AUTOFORMALIZE_SYSTEM},
                        {"role": "user", "content": f"TASK: {objective[:800]}"},
                    ],
                    objective=None,
                    session_id="vsa_autoformalize",
                )
                if isinstance(raw, list) and raw:
                    result_holder[0] = str(raw[0].get("content", "") if isinstance(raw[0], dict) else raw[0])
                elif isinstance(raw, str):
                    result_holder[0] = raw
            except Exception as e:
                error_holder[0] = e

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        t.join(timeout=timeout_seconds)

        if error_holder[0] or t.is_alive() or result_holder[0] is None:
            _logger.warning("[VSA] Autoformalization failed/timed-out — fail-open.")
            return False

        try:
            clean = re.sub(r"```(?:json)?", "", result_holder[0]).strip()
            spec = json.loads(clean)
            with self._lock:
                self._task_spec = spec
                self._invariants = spec.get("invariants", [])
                self._max_irreversible = spec.get("max_allowed_irreversible_actions")
                self._irreversible_count = 0
            _logger.info(
                "[VSA] Autoformalized spec: %d invariants. Summary: %s",
                len(self._invariants), spec.get("task_summary", ""),
            )
            return True
        except Exception as e:
            _logger.warning("[VSA] Spec parse error: %s — fail-open.", e)
            return False

    def verify(self, action: Dict[str, Any]) -> str:
        """
        Phase 2: Runtime verification. Returns "OK" or "VIOLATION".
        Zero LLM cost — pure Python invariant checks.
        """
        with self._lock:
            invariants = list(self._invariants)
            max_irrev = self._max_irreversible

        if not invariants and max_irrev is None:
            return "OK"  # No spec — fail-open

        # Check irreversible action budget
        if max_irrev is not None:
            op = str(action.get("operation", "")).lower()
            _IRREVERSIBLE_OPS = {"install", "delete", "rm", "format", "send", "submit", "publish", "deploy"}
            if op in _IRREVERSIBLE_OPS or action.get("_irreversible"):
                with self._lock:
                    self._irreversible_count += 1
                    if self._irreversible_count > max_irrev:
                        self._last_violation = (
                            f"Irreversible action budget exceeded: {self._irreversible_count} "
                            f"> max allowed {max_irrev}."
                        )
                        return "VIOLATION"

        # Check each invariant
        for inv in invariants:
            if inv.get("severity", "BLOCK") != "BLOCK":
                continue  # WARN-only invariants do not block
            violation = _check_invariant(inv, action)
            if violation:
                with self._lock:
                    self._last_violation = violation
                _logger.warning("[VSA] VIOLATION: %s", violation)
                return "VIOLATION"

        return "OK"

    def last_violation_reason(self) -> str:
        """Return the reason for the last violation (for planner feedback)."""
        with self._lock:
            return self._last_violation

    def reset(self) -> None:
        """Reset for a new task."""
        with self._lock:
            self._task_spec = None
            self._invariants = []
            self._last_violation = ""
            self._irreversible_count = 0
            self._max_irreversible = None
