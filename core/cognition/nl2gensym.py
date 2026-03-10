"""
core/cognition/nl2gensym.py — NL2GenSym: Natural Language → SOAR Operator Rules
==================================================================================
Blueprint §3.1 — Yuan et al., arXiv:2510.09355 (2025)

WHAT THIS IS
------------
NL2GenSym achieved >86% success rate generating SOAR symbolic operator
selection rules from natural language task descriptions.  An LLM dynamically
writes SOAR operator rules given a NL description of the task, so ProjectZeo
can generate context-specific SOAR operators for ANY new application type
without pre-coded rules.

HOW IT WORKS
------------
1. generate_operator_rules(objective, world_state, app_context) → List[OperatorRule]
   Calls the LLM to produce a structured JSON list of SOAR operator rules.
   Each rule has: name, preconditions, action_template, expected_outcome, priority.

2. rules_to_prompt_block(rules) → str
   Formats operator rules as a natural-language block for the OperatorCycle
   propose-operators step (injected into the LLM propose-system prompt).

3. cache: Generated rules are cached per (objective_hash, app_context) so the
   LLM is called once per task/app combination, not once per step.

4. Fallback: If LLM is unavailable or generation fails, returns a set of
   universal fallback rules that apply to any GUI task.

INTEGRATION
-----------
Called by OperatorCycle._propose_operators() to enrich the operator candidate
list with dynamically generated, task-specific operators.

REFERENCE
---------
Yuan et al. (2025) "NL2GenSym: Generating Symbolic Rules from Natural Language
for Soar Cognitive Architecture" — arXiv:2510.09355
Open source: https://soar.eecs.umich.edu (BSD) | Python bindings: SoarGroup/Python-SML-Clients
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

_logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Tunables
# ─────────────────────────────────────────────────────────────────────────────
_MAX_RULES_PER_TASK   = int(os.environ.get("PROJECTZEO_NL2GENSYM_MAX_RULES", "8"))
_CACHE_TTL_SECONDS    = int(os.environ.get("PROJECTZEO_NL2GENSYM_CACHE_TTL", "600"))
_NL2GENSYM_ENABLED    = os.environ.get("PROJECTZEO_NL2GENSYM_ENABLED", "1").strip() == "1"

_GENERATION_SYSTEM = """\
You are NL2GenSym: a SOAR operator rule generator for a GUI automation agent.
Given a task objective and the current application context, generate SOAR-style
operator selection rules in JSON format.

Each rule must have:
  - name: short snake_case operator name
  - precondition: what screen/world state makes this operator applicable (NL string)
  - action_template: the concrete GUI action dict template
    (keys: operation, and relevant args like command/text/xpath/coordinate)
  - expected_outcome: what the screen/world should look like after (NL string)
  - priority: integer 1 (low) to 10 (high)
  - when_to_prefer: brief guidance on when to select this over alternatives

Generate rules that are SPECIFIC to the objective, not generic.
Return ONLY a JSON array of rule objects. No prose. No markdown fences.
Maximum %d rules.

Example output:
[
  {
    "name": "open_file_menu",
    "precondition": "File menu is visible and closed",
    "action_template": {"operation": "click", "xpath": "//menu[@name='File']"},
    "expected_outcome": "File menu opens showing Save, Open, Close options",
    "priority": 7,
    "when_to_prefer": "When file operation is the next logical step"
  }
]
"""


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OperatorRule:
    """A SOAR-style operator rule generated from natural language."""
    name:              str
    precondition:      str
    action_template:   Dict[str, Any]
    expected_outcome:  str
    priority:          int = 5
    when_to_prefer:    str = ""
    generated_at:      float = field(default_factory=time.time)
    source:            str = "llm"      # "llm" | "fallback" | "cached"

    def to_prompt_fragment(self) -> str:
        """Format as a compact operator description for prompt injection."""
        at = json.dumps(self.action_template, separators=(",", ":"))
        lines = [
            f"Operator: {self.name} (priority={self.priority})",
            f"  When: {self.precondition}",
            f"  Action: {at}",
            f"  Expect: {self.expected_outcome}",
        ]
        if self.when_to_prefer:
            lines.append(f"  Prefer if: {self.when_to_prefer}")
        return "\n".join(lines)


@dataclass
class _CacheEntry:
    rules:      List[OperatorRule]
    created_at: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > _CACHE_TTL_SECONDS


# ─────────────────────────────────────────────────────────────────────────────
# NL2GenSym engine
# ─────────────────────────────────────────────────────────────────────────────

class NL2GenSym:
    """
    Dynamic SOAR operator rule generator.

    Generates task-specific SOAR operator selection rules from natural language
    using an LLM, enabling the agent to handle novel applications without
    pre-coded operator schemas.
    """

    def __init__(self, llm_caller: Optional[Callable[[str], str]] = None) -> None:
        self._llm   = llm_caller
        self._cache: Dict[str, _CacheEntry] = {}
        self._lock  = threading.Lock()

    def generate_operator_rules(
        self,
        objective: str,
        world_state: Optional[Dict[str, Any]] = None,
        app_context: str = "",
    ) -> List[OperatorRule]:
        """
        Generate SOAR operator rules for the given objective and context.

        Returns cached rules if available and fresh; generates new rules
        via LLM otherwise; falls back to universal rules if LLM unavailable.
        """
        if not _NL2GENSYM_ENABLED:
            return self._universal_fallback_rules(objective)

        cache_key = self._cache_key(objective, app_context)
        with self._lock:
            entry = self._cache.get(cache_key)
            if entry and not entry.is_expired():
                _logger.debug("[NL2GenSym] Cache hit for key=%s", cache_key[:12])
                return entry.rules

        rules = self._generate_via_llm(objective, world_state, app_context)
        if not rules:
            rules = self._universal_fallback_rules(objective)

        with self._lock:
            self._cache[cache_key] = _CacheEntry(rules=rules)
            # Prune stale entries
            stale = [k for k, v in self._cache.items() if v.is_expired()]
            for k in stale:
                del self._cache[k]

        return rules

    def rules_to_prompt_block(self, rules: List[OperatorRule]) -> str:
        """
        Format operator rules as a block for OperatorCycle prompt injection.
        Injected into the propose-operators system prompt.
        """
        if not rules:
            return ""
        lines = ["═══ NL2GenSym Operator Rules (task-specific) ═══"]
        for r in rules:
            lines.append(r.to_prompt_fragment())
            lines.append("")
        lines.append("═" * 47)
        return "\n".join(lines)

    def refresh_for_new_milestone(
        self,
        milestone: str,
        app_context: str = "",
        world_state: Optional[Dict[str, Any]] = None,
    ) -> List[OperatorRule]:
        """
        Regenerate rules for a specific milestone (called on milestone transition).
        Forces cache miss so rules are fresh for the new sub-goal.
        """
        cache_key = self._cache_key(milestone, app_context)
        with self._lock:
            # Evict to force regeneration
            self._cache.pop(cache_key, None)
        return self.generate_operator_rules(milestone, world_state, app_context)

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_via_llm(
        self,
        objective: str,
        world_state: Optional[Dict[str, Any]],
        app_context: str,
    ) -> List[OperatorRule]:
        """Call LLM to generate operator rules."""
        if self._llm is None:
            return []
        try:
            ws_summary = ""
            if world_state:
                entities = world_state.get("entities", [])
                apps = world_state.get("focused_app", "")
                ws_summary = (
                    f"Current app: {apps}. "
                    f"Visible elements: {', '.join(str(e) for e in entities[:8])}"
                )
            prompt = (
                f"{_GENERATION_SYSTEM % _MAX_RULES_PER_TASK}\n\n"
                f"Task objective: {objective[:200]}\n"
                f"Application context: {app_context or 'general desktop'}\n"
                f"Current world state: {ws_summary or 'not available'}\n\n"
                "Generate operator rules:"
            )
            raw = self._llm(prompt)
            if not raw:
                return []
            return self._parse_rules(raw)
        except Exception as exc:
            _logger.warning("[NL2GenSym] LLM generation failed: %s", exc)
            return []

    def _parse_rules(self, raw: str) -> List[OperatorRule]:
        """Parse LLM JSON output into OperatorRule list."""
        # Strip markdown fences
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        # Find JSON array
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
            if not isinstance(data, list):
                return []
            rules = []
            for item in data[:_MAX_RULES_PER_TASK]:
                if not isinstance(item, dict):
                    continue
                try:
                    rule = OperatorRule(
                        name             = str(item.get("name", "unnamed"))[:40],
                        precondition     = str(item.get("precondition", ""))[:200],
                        action_template  = item.get("action_template", {}),
                        expected_outcome = str(item.get("expected_outcome", ""))[:200],
                        priority         = int(item.get("priority", 5)),
                        when_to_prefer   = str(item.get("when_to_prefer", ""))[:100],
                        source           = "llm",
                    )
                    rules.append(rule)
                except Exception as parse_exc:
                    _logger.debug("[NL2GenSym] Rule parse error: %s", parse_exc)
            return rules
        except json.JSONDecodeError as je:
            _logger.debug("[NL2GenSym] JSON parse error: %s", je)
            return []

    def _universal_fallback_rules(self, objective: str) -> List[OperatorRule]:
        """
        Universal fallback rules applicable to any GUI task.
        Used when LLM is unavailable or generation fails.
        """
        obj_lower = objective.lower()
        rules = [
            OperatorRule(
                name="observe_screen",
                precondition="Uncertain about current state; need fresh observation",
                action_template={"operation": "screenshot"},
                expected_outcome="Updated view of current screen state",
                priority=6,
                when_to_prefer="Before attempting any action after a state change",
                source="fallback",
            ),
            OperatorRule(
                name="scroll_to_find",
                precondition="Target element not visible in current viewport",
                action_template={"operation": "scroll", "direction": "down", "amount": 3},
                expected_outcome="More of the page/document becomes visible",
                priority=5,
                when_to_prefer="When element is expected but not visible",
                source="fallback",
            ),
            OperatorRule(
                name="wait_for_load",
                precondition="Application is loading or processing (spinner/progress visible)",
                action_template={"operation": "wait", "duration": 2},
                expected_outcome="Loading completes; new content or state appears",
                priority=7,
                when_to_prefer="Whenever a loading indicator is visible",
                source="fallback",
            ),
            OperatorRule(
                name="dismiss_dialog",
                precondition="A modal dialog or alert is blocking interaction",
                action_template={"operation": "click", "description": "dialog OK or close button"},
                expected_outcome="Dialog dismissed; underlying content accessible",
                priority=8,
                when_to_prefer="When a dialog is blocking the main content",
                source="fallback",
            ),
        ]
        # Task-specific additions
        if any(w in obj_lower for w in ["save", "write", "create", "new"]):
            rules.append(OperatorRule(
                name="keyboard_save",
                precondition="Document or file has unsaved changes",
                action_template={"operation": "hotkey", "keys": "ctrl+s"},
                expected_outcome="File saved; title bar no longer shows unsaved indicator",
                priority=7,
                when_to_prefer="After making changes that need to persist",
                source="fallback",
            ))
        if any(w in obj_lower for w in ["search", "find", "look", "locate"]):
            rules.append(OperatorRule(
                name="use_search_box",
                precondition="Search field is visible and empty",
                action_template={"operation": "click", "description": "search input field"},
                expected_outcome="Search field is focused and ready for input",
                priority=7,
                when_to_prefer="When searching for content within the application",
                source="fallback",
            ))
        return rules

    def _cache_key(self, objective: str, app_context: str) -> str:
        raw = f"{objective[:100]}::{app_context[:50]}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_instance: Optional[NL2GenSym] = None
_instance_lock = threading.Lock()


def get_nl2gensym(llm_caller: Optional[Callable[[str], str]] = None) -> NL2GenSym:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = NL2GenSym(llm_caller=llm_caller)
    elif llm_caller and _instance._llm is None:
        _instance._llm = llm_caller
    return _instance
