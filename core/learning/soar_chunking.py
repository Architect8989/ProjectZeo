"""
core/learning/soar_chunking.py
================================
SOAR Chunking — Procedural Memory from Successful Operator Sequences.

Blueprint Reference: §3.5.2 (arXiv:2205.03854 SOAR Chunking)

When a subgoal succeeds, SOAR compresses the reasoning trace that produced
the success into a new production rule stored in procedural memory. On the
next identical impasse, the stored rule fires immediately — no subgoal
creation, no lookahead, no uncertainty.

For ProjectZeo:
  - "Successful operator sequence" = ordered list of action dicts that completed a task
  - "Production rule" = (goal_pattern, app_context) → operator_sequence
  - Storage: OpenMemoryStore procedural sector (persistent) + in-memory fast cache
  - Retrieval: called by OperatorCycle before LLM proposal to check for matching chunk

Chunking lifecycle:
  1. GIIController calls on_operator_success() after task completion
  2. SOARChunking.chunk() normalises the goal and compresses the operator sequence
  3. LLM extracts a generalised condition-action rule from the sequence
  4. Rule stored in OpenMemoryStore procedural sector with importance=0.85
  5. On next task with similar goal/app: OperatorCycle retrieves and fires the rule
  6. After N successful firings: rule is promoted (importance += 0.1)
  7. After M failed firings: rule is demoted or deprecated

Generalisation:
  Raw sequences like "click Button_47 → type 'hello' → click Button_Submit" are
  too specific to be reusable. The LLM generalises them to:
  "click [form submit trigger] → type [content] → click [submit button]"
  enabling cross-application transfer.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

_CHUNK_ENABLED        = os.environ.get("PROJECTZEO_CHUNKING", "1") == "1"
_MAX_CHUNK_ACTIONS    = int(os.environ.get("PROJECTZEO_CHUNK_MAX_ACTIONS", "20"))
_MIN_CHUNK_ACTIONS    = int(os.environ.get("PROJECTZEO_CHUNK_MIN_ACTIONS", "2"))
_GENERALISE_ENABLED   = os.environ.get("PROJECTZEO_CHUNK_GENERALISE", "1") == "1"
_GENERALISE_TIMEOUT   = float(os.environ.get("PROJECTZEO_CHUNK_TIMEOUT", "30"))
_PROMOTE_AFTER_N      = int(os.environ.get("PROJECTZEO_CHUNK_PROMOTE_N", "3"))
_DEMOTE_AFTER_M       = int(os.environ.get("PROJECTZEO_CHUNK_DEMOTE_M", "2"))
_CACHE_SIZE           = int(os.environ.get("PROJECTZEO_CHUNK_CACHE", "500"))

# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProductionRule:
    """
    A generalised SOAR production rule extracted from a successful operator sequence.

    IF:   goal_pattern matches current goal AND app_context matches
    THEN: execute operator_sequence (in order)
    """
    rule_id:          str
    goal_pattern:     str                     # Normalised/generalised goal description
    app_context:      str                     # Application (empty = cross-app)
    operator_sequence: List[Dict[str, Any]]  # Ordered list of action dicts
    generalised_desc: str = ""               # LLM-generalised description
    raw_goal:         str = ""               # Original specific goal
    success_count:    int = 1
    fire_count:       int = 0               # Times the rule has been recalled
    failure_count:    int = 0              # Times the rule was recalled but failed
    importance:       float = 0.85
    created_at:       float = field(default_factory=time.time)
    last_fired:       float = 0.0
    metadata:         Dict[str, Any] = field(default_factory=dict)

    @property
    def reliability(self) -> float:
        """Success rate when rule has been fired."""
        total = self.fire_count + self.failure_count
        if total == 0:
            return 1.0  # Untested: assume reliable
        return self.success_count / total

    @property
    def is_deprecated(self) -> bool:
        return (
            self.failure_count >= _DEMOTE_AFTER_M
            and self.reliability < 0.4
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id":          self.rule_id,
            "goal_pattern":     self.goal_pattern,
            "app_context":      self.app_context,
            "operator_count":   len(self.operator_sequence),
            "generalised_desc": self.generalised_desc,
            "success_count":    self.success_count,
            "fire_count":       self.fire_count,
            "failure_count":    self.failure_count,
            "importance":       round(self.importance, 3),
            "reliability":      round(self.reliability, 3),
            "is_deprecated":    self.is_deprecated,
        }


# ─────────────────────────────────────────────────────────────────────────────
# LLM prompts
# ─────────────────────────────────────────────────────────────────────────────

_GENERALISE_SYSTEM = """\
You are a SOAR Chunking Engine for a GUI automation agent.

Given a specific goal description and the exact sequence of actions that
successfully completed it, produce a GENERALISED production rule that will
transfer to similar goals in the same application.

GENERALISATION RULES:
1. Replace specific text values with semantic roles: 
   "john@example.com" → "[recipient email]"
   "Q3 Report.xlsx" → "[file name]"
   "Save" button → "[save trigger]"
2. Replace specific coordinates with element descriptions
3. Keep operation types (click, type, key) unchanged
4. Preserve ordering — order is semantically meaningful
5. Keep the rule compact: merge adjacent same-operation steps if appropriate
6. The generalised description must be a concise IF-THEN production rule

OUTPUT FORMAT (JSON, no markdown):
{
  "goal_pattern": "<generalised goal: e.g., 'save file in [app]'>",
  "generalised_sequence": [
    {"operation": "<op>", "target": "<generalised target>", "text": "<generalised text>"},
    ...
  ],
  "description": "IF <goal_pattern> THEN <sequence summary>",
  "transferability": <0.0-1.0, how reusable across tasks/apps>
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# SOARChunking
# ─────────────────────────────────────────────────────────────────────────────

class SOARChunking:
    """
    SOAR Chunking: converts successful operator sequences into generalised
    production rules stored in procedural memory.
    """

    def __init__(
        self,
        llm_call: Callable,
        openmemory_store: Optional[Any] = None,
    ) -> None:
        self._llm        = llm_call
        self._openmemory = openmemory_store
        self._lock       = threading.RLock()

        # In-memory rule cache (fast lookup, survives GII loop iterations)
        self._rules:  Dict[str, ProductionRule] = {}  # rule_id → rule
        self._index:  Dict[str, List[str]] = {}        # app_context → [rule_ids]

        # Load existing rules from OpenMemory at startup
        self._load_rules_from_memory()

        _logger.info(
            "[SOARChunking] Initialised. rules=%d openmemory=%s generalise=%s",
            len(self._rules), openmemory_store is not None, _GENERALISE_ENABLED
        )

    # =========================================================================
    # Chunking: store a successful sequence
    # =========================================================================

    def chunk(
        self,
        operator_sequence: List[Any],
        goal_description:  str,
        app_context:       str = "",
        *,
        success_reward:    float = 1.0,
    ) -> Optional[ProductionRule]:
        """
        Chunk a successful operator sequence into a production rule.

        Args:
            operator_sequence: List of action dicts (or Operator objects)
            goal_description:  The natural language goal that was achieved
            app_context:       Application context (empty = cross-app)
            success_reward:    Reward signal from ARPO (0.0-1.0)

        Returns:
            The stored ProductionRule, or None if chunking was skipped.
        """
        if not _CHUNK_ENABLED:
            return None

        # Normalise operator sequence to action dicts
        action_seq = self._normalise_sequence(operator_sequence)

        if len(action_seq) < _MIN_CHUNK_ACTIONS:
            _logger.debug(
                "[SOARChunking] Sequence too short (%d < %d) — skipping",
                len(action_seq), _MIN_CHUNK_ACTIONS
            )
            return None

        # Truncate if too long
        if len(action_seq) > _MAX_CHUNK_ACTIONS:
            action_seq = action_seq[:_MAX_CHUNK_ACTIONS]
            _logger.debug("[SOARChunking] Truncated sequence to %d actions", _MAX_CHUNK_ACTIONS)

        # Check for duplicate (same goal + app + first actions)
        existing = self._find_matching_rule(goal_description, app_context)
        if existing is not None:
            # Reinforce existing rule
            with self._lock:
                existing.success_count += 1
                if existing.success_count >= _PROMOTE_AFTER_N:
                    existing.importance = min(1.0, existing.importance + 0.05)
            _logger.info(
                "[SOARChunking] Reinforced existing rule %s (count=%d)",
                existing.rule_id, existing.success_count
            )
            self._persist_rule(existing)
            return existing

        # Generalise the sequence using LLM
        generalised_seq  = action_seq
        generalised_desc = f"Operator sequence for: {goal_description[:100]}"
        goal_pattern     = self._normalize_goal(goal_description)

        if _GENERALISE_ENABLED and self._llm is not None:
            try:
                gen_result = self._llm_generalise(
                    action_seq, goal_description, app_context
                )
                if gen_result:
                    if gen_result.get("generalised_sequence"):
                        generalised_seq = gen_result["generalised_sequence"]
                    if gen_result.get("goal_pattern"):
                        goal_pattern = gen_result["goal_pattern"]
                    if gen_result.get("description"):
                        generalised_desc = gen_result["description"]
            except Exception as exc:
                _logger.debug("[SOARChunking] LLM generalise failed: %s", exc)

        # Compute importance based on reward and transferability
        importance = 0.70 + 0.15 * success_reward

        rule = ProductionRule(
            rule_id           = f"rule_{uuid.uuid4().hex[:12]}",
            goal_pattern      = goal_pattern,
            app_context       = app_context.lower(),
            operator_sequence = generalised_seq,
            generalised_desc  = generalised_desc,
            raw_goal          = goal_description[:200],
            success_count     = 1,
            importance        = importance,
        )

        # Store in memory
        with self._lock:
            self._rules[rule.rule_id] = rule
            app_key = app_context.lower()
            self._index.setdefault(app_key, []).append(rule.rule_id)
            # Cross-app index
            self._index.setdefault("", []).append(rule.rule_id)
            # Bound cache
            if len(self._rules) > _CACHE_SIZE:
                self._evict()

        self._persist_rule(rule)

        _logger.info(
            "[SOARChunking] Stored rule %s: app=%r goal=%r ops=%d importance=%.2f",
            rule.rule_id, app_context[:30], goal_pattern[:60],
            len(generalised_seq), importance
        )
        return rule

    # =========================================================================
    # Recall: retrieve matching rule for current context
    # =========================================================================

    def recall(
        self,
        goal_description: str,
        app_context:      str = "",
        step_index:       int = 0,
    ) -> Optional[Dict[str, Any]]:
        """
        Recall a matching production rule's next action.

        Args:
            goal_description: Current active goal
            app_context:      Current application
            step_index:       Which step in the sequence to return

        Returns:
            Action dict for the next step, or None if no rule matches.
        """
        rule = self._find_matching_rule(goal_description, app_context)
        if rule is None:
            return None

        if rule.is_deprecated:
            _logger.debug("[SOARChunking] Rule %s is deprecated — skipping", rule.rule_id)
            return None

        if step_index >= len(rule.operator_sequence):
            return None

        action = dict(rule.operator_sequence[step_index])

        with self._lock:
            rule.fire_count += 1
            rule.last_fired = time.time()

        _logger.info(
            "[SOARChunking] Firing rule %s step %d/%d: op=%s",
            rule.rule_id, step_index + 1, len(rule.operator_sequence),
            action.get("operation", "?")
        )
        return action

    def on_rule_success(self, rule_id: str) -> None:
        """Called when a recalled rule's action succeeded."""
        with self._lock:
            rule = self._rules.get(rule_id)
            if rule:
                rule.success_count += 1
                if rule.success_count >= _PROMOTE_AFTER_N:
                    old_imp = rule.importance
                    rule.importance = min(1.0, rule.importance + 0.05)
                    _logger.info(
                        "[SOARChunking] Promoted rule %s: importance %.2f → %.2f",
                        rule_id, old_imp, rule.importance
                    )

    def on_rule_failure(self, rule_id: str) -> None:
        """Called when a recalled rule's action failed."""
        with self._lock:
            rule = self._rules.get(rule_id)
            if rule:
                rule.failure_count += 1
                rule.importance = max(0.10, rule.importance - 0.10)
                if rule.is_deprecated:
                    _logger.warning(
                        "[SOARChunking] Rule %s deprecated after %d failures",
                        rule_id, rule.failure_count
                    )

    # =========================================================================
    # Internal: rule matching
    # =========================================================================

    def _find_matching_rule(
        self,
        goal_description: str,
        app_context: str,
    ) -> Optional[ProductionRule]:
        """Find the best matching rule for the given goal and app context."""
        norm_goal = self._normalize_goal(goal_description)
        app_lower = app_context.lower()

        best_score = 0.0
        best_rule: Optional[ProductionRule] = None

        with self._lock:
            # Search app-specific rules first, then cross-app
            search_keys = []
            if app_lower:
                search_keys.append(app_lower)
            search_keys.append("")

            checked_ids: set = set()
            for key in search_keys:
                rule_ids = self._index.get(key, [])
                for rid in rule_ids:
                    if rid in checked_ids:
                        continue
                    checked_ids.add(rid)

                    rule = self._rules.get(rid)
                    if rule is None or rule.is_deprecated:
                        continue

                    score = self._match_score(norm_goal, rule, app_lower)
                    if score > best_score:
                        best_score = score
                        best_rule = rule

        # Require minimum match quality
        if best_score < 0.35:
            return None

        return best_rule

    def _match_score(
        self,
        norm_goal: str,
        rule: ProductionRule,
        app_lower: str,
    ) -> float:
        """Score how well a rule matches the current goal and app context."""
        # Goal similarity (Jaccard)
        goal_sim = self._goal_similarity(norm_goal, rule.goal_pattern)

        # App context similarity
        if rule.app_context == "" or app_lower == "":
            app_sim = 0.5   # Cross-app rule
        elif rule.app_context == app_lower:
            app_sim = 1.0
        elif rule.app_context in app_lower or app_lower in rule.app_context:
            app_sim = 0.7
        else:
            app_sim = 0.0

        # Reliability boost
        rel_boost = 0.8 + 0.2 * rule.reliability

        # Combined: 60% goal + 30% app + 10% importance
        score = (
            0.60 * goal_sim
            + 0.30 * app_sim
            + 0.10 * rule.importance
        ) * rel_boost

        return score

    def _goal_similarity(self, g1: str, g2: str) -> float:
        """Jaccard similarity between normalized goal strings."""
        if not g1 or not g2:
            return 0.0
        s1 = set(g1.split())
        s2 = set(g2.split())
        if not s1 or not s2:
            return 0.0
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        return intersection / union if union > 0 else 0.0

    # =========================================================================
    # Internal: LLM generalisation
    # =========================================================================

    def _llm_generalise(
        self,
        action_seq:       List[Dict[str, Any]],
        goal_description: str,
        app_context:      str,
    ) -> Optional[Dict[str, Any]]:
        """Use LLM to generalise the action sequence into a transferable rule."""
        # Build readable sequence for the prompt
        seq_text = "\n".join(
            f"{i+1}. {a.get('operation','?')} "
            f"target={a.get('target','')!r} "
            f"text={a.get('text','')!r} "
            f"keys={a.get('keys','')!r}"
            for i, a in enumerate(action_seq[:15])
        )

        messages = [
            {"role": "system", "content": _GENERALISE_SYSTEM},
            {"role": "user", "content": (
                f"GOAL: {goal_description}\n"
                f"APPLICATION: {app_context or '(any)'}\n\n"
                f"SUCCESSFUL ACTION SEQUENCE:\n{seq_text}\n\n"
                "Generalise this into a reusable production rule."
            )},
        ]

        result: Dict[str, Any] = {}
        exc_holder: List[Exception] = []

        def _call():
            try:
                raw = self._llm(messages, objective="soar_generalise")
                cleaned = re.sub(r"```(?:json)?", "", raw or "").strip()
                m = re.search(r"\{.*\}", cleaned, re.DOTALL)
                if m:
                    result.update(json.loads(m.group()))
            except Exception as exc:
                exc_holder.append(exc)

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        t.join(timeout=_GENERALISE_TIMEOUT)

        if exc_holder or not result:
            return None

        return result

    # =========================================================================
    # Internal: normalisation, persistence, eviction
    # =========================================================================

    def _normalize_goal(self, goal: str) -> str:
        """Normalize goal text for matching."""
        if not goal:
            return ""
        norm = re.sub(r"[^\w\s]", " ", goal.lower())
        norm = re.sub(r"\s+", " ", norm).strip()
        stops = {"a", "an", "the", "to", "in", "on", "at", "for",
                 "and", "or", "of", "by", "be", "do", "it"}
        words = [w for w in norm.split() if w not in stops and len(w) > 1]
        return " ".join(words[:25])

    def _normalise_sequence(self, operator_sequence: List[Any]) -> List[Dict[str, Any]]:
        """Convert any operator format to clean action dicts."""
        result: List[Dict[str, Any]] = []
        for op in operator_sequence:
            if isinstance(op, dict):
                action = op
            elif hasattr(op, "action"):
                action = getattr(op, "action", {})
            else:
                continue

            if not isinstance(action, dict):
                continue

            # Skip terminal operations
            if action.get("operation") in ("done", "require_human_confirmation"):
                continue

            # Strip large/irrelevant fields
            clean = {
                k: v for k, v in action.items()
                if k in ("operation", "target", "text", "keys", "url", "command")
                and v is not None and str(v).strip()
            }
            if clean.get("operation"):
                result.append(clean)

        return result

    def _persist_rule(self, rule: ProductionRule) -> None:
        """Persist rule to OpenMemoryStore procedural sector."""
        if self._openmemory is None:
            return
        try:
            content = json.dumps({
                "rule_id":          rule.rule_id,
                "goal_pattern":     rule.goal_pattern,
                "app_context":      rule.app_context,
                "generalised_desc": rule.generalised_desc,
                "raw_goal":         rule.raw_goal,
                "operator_sequence": rule.operator_sequence[:10],  # Store first 10 steps
                "success_count":    rule.success_count,
                "importance":       rule.importance,
            })
            self._openmemory.store_procedural(
                content=content,
                subject=rule.app_context or "cross_app",
                importance=rule.importance,
            )
        except Exception as exc:
            _logger.debug("[SOARChunking] Persist error: %s", exc)

    def _load_rules_from_memory(self) -> None:
        """Load existing rules from OpenMemoryStore at init."""
        if self._openmemory is None:
            return
        try:
            entries = self._openmemory.retrieve(
                query="production rule operator sequence",
                top_k=100,
                sector="procedural",
            )
            loaded = 0
            for entry in entries:
                try:
                    data = json.loads(entry.content)
                    if "rule_id" not in data or "goal_pattern" not in data:
                        continue
                    rule = ProductionRule(
                        rule_id           = data["rule_id"],
                        goal_pattern      = data["goal_pattern"],
                        app_context       = data.get("app_context", ""),
                        operator_sequence = data.get("operator_sequence", []),
                        generalised_desc  = data.get("generalised_desc", ""),
                        raw_goal          = data.get("raw_goal", ""),
                        success_count     = int(data.get("success_count", 1)),
                        importance        = float(data.get("importance", 0.85)),
                    )
                    self._rules[rule.rule_id] = rule
                    app_key = rule.app_context.lower()
                    self._index.setdefault(app_key, []).append(rule.rule_id)
                    self._index.setdefault("", []).append(rule.rule_id)
                    loaded += 1
                except Exception:
                    continue
            if loaded > 0:
                _logger.info(
                    "[SOARChunking] Loaded %d rules from procedural memory "
                    "(%d parse failures).",
                    loaded,
                    sum(1 for _ in entries) - loaded,
                )
            elif entries:
                _logger.warning(
                    "[SOARChunking] 0 rules loaded from %d procedural entries. "
                    "Check stored format vs ProductionRule schema.",
                    len(entries),
                )
        except Exception as exc:
            # DEFECT FIX: Was _logger.debug (silently hidden) — bumped to warning
            # so operators can see rule-load failures in production logs.
            _logger.warning(
                "[SOARChunking] Rule load from OpenMemory failed: %s. "
                "Starting with empty rule cache.", exc
            )

    def _evict(self) -> None:
        """Evict least important rules when cache is full."""
        with self._lock:
            sorted_rules = sorted(
                self._rules.values(),
                key=lambda r: r.importance * r.reliability
            )
            to_evict = sorted_rules[:max(1, len(sorted_rules) // 5)]
            for rule in to_evict:
                del self._rules[rule.rule_id]
                for key in self._index:
                    if rule.rule_id in self._index[key]:
                        self._index[key].remove(rule.rule_id)

    # =========================================================================
    # Diagnostics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            active = sum(1 for r in self._rules.values() if not r.is_deprecated)
            deprecated = sum(1 for r in self._rules.values() if r.is_deprecated)
            apps = set(r.app_context for r in self._rules.values() if r.app_context)
            return {
                "total_rules":      len(self._rules),
                "active_rules":     active,
                "deprecated_rules": deprecated,
                "apps_covered":     sorted(apps),
                "chunking_enabled": _CHUNK_ENABLED,
                "generalise_enabled": _GENERALISE_ENABLED,
            }

    def list_rules(self, app_context: str = "") -> List[Dict[str, Any]]:
        """List all rules, optionally filtered by app."""
        with self._lock:
            rules = list(self._rules.values())
        if app_context:
            rules = [r for r in rules if app_context.lower() in r.app_context.lower()
                     or r.app_context == ""]
        return [r.to_dict() for r in rules]
