"""
core/learning/dicp.py — Direct In-Context Policy (DICP)
=========================================================
Blueprint §9.1 — In-Context Self-Improvement

WHAT THIS IS
------------
DICP accumulates within-task failure/success patterns and injects them as a
"policy addendum" into every LLM prompt during operator selection.

It is the fine-grained complement to Algorithm Distillation (cross-session):
  Algorithm Distillation → cross-session, coarse-grained, task-type level
  DICP                   → intra-task, fine-grained, sub-goal/action level

HOW IT WORKS
------------
1. observe(operator, args, outcome, reward, note)
   Called after EVERY action dispatch.  Records what was tried and whether it
   worked.  On the same operator failing ≥ _FAILURE_THRESHOLD times, DICP
   auto-generates a constraint: "avoid this pattern".

2. add_constraint(kind, text, confidence)
   Explicit constraint injection from the GIILoop (policy deny, stagnation
   detection, BDI reconsideration signal).

3. get_policy_addendum(world_state) → str
   Returns a compact natural-language block injected into the LLM context
   window BEFORE operator selection.  Limits itself to the N most-relevant,
   highest-confidence constraints for the current world state.

4. ingest_ad_constraints(ad_context_string) → int
   Parses Algorithm Distillation context (cross-session lessons) into DICP
   constraints so in-context policy starts warm for known task patterns.

DESIGN DECISIONS
----------------
* Fully in-process: no network, no external DB.
* Thread-safe: a single threading.Lock guards all mutable state.
* LLM fallback: when an LLM caller is supplied, DICP can synthesize a
  natural-language constraint from multiple related failures (richer than
  rule matching).  Works without LLM (pure heuristic mode).
* Bounded memory: max _MAX_OBSERVATIONS observations; oldest evicted first.
* JSON-serialisable: entire state is a plain dict for checkpoint support.

REFERENCES
----------
* Laskin et al. (2022) — Algorithm Distillation (parent concept)
* DICP concept: "Direct In-Context Policy" as described in Blueprint §9.1
* OpenAI o1-style constraint injection patterns
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Tunables
# ─────────────────────────────────────────────────────────────────────────────
_MAX_OBSERVATIONS:   int   = int(os.environ.get("PROJECTZEO_DICP_MAX_OBS",     "200"))
_MAX_CONSTRAINTS:    int   = int(os.environ.get("PROJECTZEO_DICP_MAX_CONSTR",  "30"))
_MAX_ADDENDUM_ITEMS: int   = int(os.environ.get("PROJECTZEO_DICP_MAX_ITEMS",   "8"))
_FAILURE_THRESHOLD:  int   = int(os.environ.get("PROJECTZEO_DICP_FAIL_THRESH", "2"))
_CONFIDENCE_DECAY:   float = float(os.environ.get("PROJECTZEO_DICP_DECAY",     "0.02"))
_LLM_SYNTHESIS:      bool  = os.environ.get("PROJECTZEO_DICP_LLM_SYNTH", "1").strip() == "1"

_SYSTEM_PROMPT = """\
You are the DICP (Direct In-Context Policy) synthesis engine for a GUI agent.
You have observed a series of action failures or stagnation patterns during
the current task.  Your job: synthesise a CONCISE, ACTIONABLE constraint
statement (≤30 words) that the agent should inject into its planning context
to avoid repeating these failures.

Respond with ONLY the constraint text. No preamble, no bullet points.
Example good output: "Clicking the Submit button before filling all fields
causes a validation error — fill Name, Email, and Message fields first."
"""


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Observation:
    """Single (operator, args, outcome, reward) tuple."""
    obs_id:    str
    operator:  str
    args_key:  str          # Compact string key derived from args
    outcome:   str          # "success" | "failure" | "partial"
    reward:    float        # 1.0=success, 0.0=fail
    note:      str          # Short failure note
    timestamp: float = field(default_factory=time.time)


@dataclass
class Constraint:
    """A policy constraint derived from failure patterns."""
    constraint_id: str
    kind:       str         # "avoid" | "require" | "prefer" | "warn"
    text:       str         # Natural-language constraint
    confidence: float       # 0.0–1.0
    source:     str         # "auto_failure" | "policy_deny" | "stagnation" | "ad_ingest" | "explicit"
    created_at: float = field(default_factory=time.time)
    trigger_count: int = 0  # How many times this constraint was injected

    def decay(self, dt_seconds: float) -> None:
        """Confidence decays over time so stale constraints fade."""
        self.confidence = max(0.01, self.confidence - _CONFIDENCE_DECAY * (dt_seconds / 3600.0))

    def to_prompt_line(self) -> str:
        """Format as a single line for LLM prompt injection."""
        kind_prefix = {
            "avoid":   "⚠ AVOID",
            "require": "✓ REQUIRE",
            "prefer":  "→ PREFER",
            "warn":    "⚡ NOTE",
        }.get(self.kind, "→")
        return f"{kind_prefix}: {self.text}"


# ─────────────────────────────────────────────────────────────────────────────
# DICPEngine
# ─────────────────────────────────────────────────────────────────────────────

class DICPEngine:
    """
    Direct In-Context Policy Engine.

    Accumulates within-task failure/success patterns and injects them as a
    policy addendum into every LLM prompt during operator selection.
    """

    def __init__(
        self,
        objective: str,
        app_context: str = "",
        llm_caller: Optional[Callable[[str], str]] = None,
    ) -> None:
        self._objective    = objective
        self._app_context  = app_context
        self._llm          = llm_caller
        self._lock         = threading.Lock()
        self._observations: List[Observation] = []
        self._constraints:  List[Constraint]  = []
        # failure counter per (operator, args_key) tuple
        self._failure_counts: Dict[str, int] = defaultdict(int)
        # set of (operator, args_key) already synthesised into constraints
        self._synthesised: set = set()
        self._created_at = time.time()
        _logger.debug("[DICP] Engine created for objective=%r app=%r", objective[:60], app_context)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def observe(
        self,
        operator: str,
        args: Dict[str, Any],
        outcome: str,
        reward: float,
        note: str = "",
    ) -> None:
        """
        Record one action execution result.

        Called after every action dispatch by GIILoop._dicp_observe().
        Auto-generates constraints when the same operator+args fail repeatedly.
        """
        args_key = self._make_args_key(operator, args)
        obs = Observation(
            obs_id   = str(uuid.uuid4())[:8],
            operator = operator,
            args_key = args_key,
            outcome  = outcome,
            reward   = reward,
            note     = note[:80],
        )
        with self._lock:
            self._observations.append(obs)
            # Evict oldest if over limit
            if len(self._observations) > _MAX_OBSERVATIONS:
                self._observations = self._observations[-_MAX_OBSERVATIONS:]

            if outcome == "failure":
                fk = f"{operator}::{args_key}"
                self._failure_counts[fk] += 1
                if (
                    self._failure_counts[fk] >= _FAILURE_THRESHOLD
                    and fk not in self._synthesised
                ):
                    self._synthesised.add(fk)
                    # Schedule async synthesis (non-blocking)
                    self._auto_synthesise_constraint(operator, args, note)

    def add_constraint(
        self,
        kind: str,
        text: str,
        confidence: float = 0.8,
        source: str = "explicit",
    ) -> None:
        """
        Inject an explicit constraint from the GIILoop.

        Called for:
        - Policy DENY events (confidence=1.0)
        - Stagnation detection (confidence=0.9)
        - BDI reconsideration signals
        """
        if not text.strip():
            return
        constraint = Constraint(
            constraint_id = str(uuid.uuid4())[:8],
            kind          = kind,
            text          = text[:200],
            confidence    = min(1.0, max(0.01, confidence)),
            source        = source,
        )
        with self._lock:
            # Deduplicate: skip if very similar text already exists
            for existing in self._constraints:
                if self._text_similarity(existing.text, text) > 0.8:
                    # Update confidence to higher of the two
                    existing.confidence = max(existing.confidence, confidence)
                    _logger.debug("[DICP] Merged duplicate constraint: %r", text[:60])
                    return
            self._constraints.append(constraint)
            # Evict lowest-confidence constraints if over limit
            if len(self._constraints) > _MAX_CONSTRAINTS:
                self._constraints.sort(key=lambda c: c.confidence, reverse=True)
                self._constraints = self._constraints[:_MAX_CONSTRAINTS]
        _logger.debug(
            "[DICP] Constraint added: kind=%s conf=%.2f text=%r",
            kind, confidence, text[:60],
        )

    def get_policy_addendum(self, world_state: Optional[Dict[str, Any]] = None) -> str:
        """
        Return a compact natural-language block for LLM prompt injection.

        Selects the N most relevant, highest-confidence constraints for the
        current world state.  Applies confidence decay since last call.
        Returns empty string if no active constraints.
        """
        now = time.time()
        with self._lock:
            if not self._constraints:
                return ""

            # Decay all constraints
            for c in self._constraints:
                dt = now - c.created_at
                c.decay(dt_seconds=max(0, dt - 60))  # grace period: 60s no decay

            # Filter below minimum confidence
            active = [c for c in self._constraints if c.confidence >= 0.05]
            if not active:
                return ""

            # Score by relevance to current world state + confidence
            scored = []
            focused_app = (world_state or {}).get("focused_app", "")
            for c in active:
                relevance = self._relevance_score(c, world_state, focused_app)
                scored.append((relevance * c.confidence, c))

            scored.sort(key=lambda x: -x[0])
            top = scored[:_MAX_ADDENDUM_ITEMS]

            lines = [c.to_prompt_line() for _, c in top]
            for _, c in top:
                c.trigger_count += 1

        if not lines:
            return ""

        header = "─── IN-CONTEXT POLICY (learned this session) ───"
        return "\n".join([header] + lines + ["─" * 47])

    def ingest_ad_constraints(self, ad_context_string: str) -> int:
        """
        Parse Algorithm Distillation context string into DICP constraints.

        Called at task start so in-context policy begins warm with cross-session
        lessons.  Returns count of constraints ingested.
        """
        if not ad_context_string.strip():
            return 0
        ingested = 0
        # Extract lesson lines from AD context format
        lesson_patterns = [
            r"Lesson:\s*(.+)",
            r"LESSON:\s*(.+)",
            r"lesson:\s*(.+)",
            r"- Failed because:\s*(.+)",
            r"NOTE:\s*(.+)",
            r"AVOID:\s*(.+)",
            r"WARNING:\s*(.+)",
        ]
        for pattern in lesson_patterns:
            for match in re.finditer(pattern, ad_context_string, re.IGNORECASE):
                lesson = match.group(1).strip()
                if len(lesson) > 10:
                    kind = "avoid" if any(
                        w in lesson.lower() for w in ["avoid", "don't", "failed", "error", "fail"]
                    ) else "warn"
                    self.add_constraint(
                        kind=kind,
                        text=lesson[:200],
                        confidence=0.6,
                        source="ad_ingest",
                    )
                    ingested += 1
        _logger.debug("[DICP] Ingested %d constraints from AD context.", ingested)
        return ingested

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "observations":  len(self._observations),
                "constraints":   len(self._constraints),
                "synthesised":   len(self._synthesised),
                "failure_keys":  len(self._failure_counts),
                "objective":     self._objective[:60],
            }

    def to_dict(self) -> Dict[str, Any]:
        """Serialise state for checkpoint support."""
        with self._lock:
            return {
                "objective":    self._objective,
                "app_context":  self._app_context,
                "observations": [asdict(o) for o in self._observations[-50:]],
                "constraints":  [asdict(c) for c in self._constraints],
            }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        llm_caller: Optional[Callable[[str], str]] = None,
    ) -> "DICPEngine":
        engine = cls(
            objective   = data.get("objective", ""),
            app_context = data.get("app_context", ""),
            llm_caller  = llm_caller,
        )
        for o in data.get("observations", []):
            try:
                engine._observations.append(Observation(**o))
            except Exception:
                pass
        for c in data.get("constraints", []):
            try:
                engine._constraints.append(Constraint(**c))
            except Exception:
                pass
        return engine

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _make_args_key(self, operator: str, args: Dict[str, Any]) -> str:
        """Create a compact, stable key for (operator, args) deduplication."""
        key_fields = {}
        for k in ("command", "text", "xpath", "coordinate", "url", "path", "query"):
            if k in args:
                val = str(args[k])[:50]
                key_fields[k] = val
        if not key_fields:
            key_fields = {k: str(v)[:30] for k, v in list(args.items())[:3]}
        return json.dumps(key_fields, sort_keys=True)

    def _auto_synthesise_constraint(
        self,
        operator: str,
        args: Dict[str, Any],
        last_note: str,
    ) -> None:
        """
        Auto-generate a constraint from repeated failure.
        Runs in-line (non-blocking, fast path first, LLM fallback).
        """
        # Fast heuristic path
        heuristic = self._heuristic_constraint(operator, args, last_note)
        if heuristic:
            self.add_constraint(
                kind="avoid",
                text=heuristic,
                confidence=0.75,
                source="auto_failure",
            )
            return

        # LLM synthesis path (if available and enabled)
        if self._llm and _LLM_SYNTHESIS:
            t = threading.Thread(
                target=self._llm_synthesise_constraint,
                args=(operator, args, last_note),
                daemon=True,
            )
            t.start()

    def _heuristic_constraint(
        self,
        operator: str,
        args: Dict[str, Any],
        note: str,
    ) -> str:
        """Generate a rule-based constraint without LLM."""
        cmd = str(args.get("command", ""))
        text = str(args.get("text", ""))
        xpath = str(args.get("xpath", ""))
        coord = str(args.get("coordinate", ""))

        if operator == "command" and cmd:
            return (
                f"Command '{cmd[:40]}' failed repeatedly — "
                "try alternative command or check prerequisites."
            )
        if operator in ("click", "double_click") and (xpath or coord):
            target = xpath[:40] if xpath else coord[:20]
            return (
                f"Clicking '{target}' failed repeatedly — "
                "element may not be visible or interactable; try scrolling or waiting."
            )
        if operator in ("write", "type") and text:
            return (
                f"Typing '{text[:30]}' failed repeatedly — "
                "ensure the input field is focused and editable before typing."
            )
        if note:
            return f"Action '{operator}' failed: {note[:80]} — choose different approach."
        return ""

    def _llm_synthesise_constraint(
        self,
        operator: str,
        args: Dict[str, Any],
        note: str,
    ) -> None:
        """Use LLM to synthesize a rich constraint from failure context."""
        try:
            # Gather recent failures for context
            with self._lock:
                recent_fails = [
                    o for o in self._observations[-20:]
                    if o.operator == operator and o.outcome == "failure"
                ]
            context_lines = []
            for o in recent_fails[-5:]:
                context_lines.append(f"- {o.operator}({o.args_key[:60]}): {o.note}")
            context = "\n".join(context_lines) if context_lines else f"- {operator} failed: {note}"
            prompt = (
                f"{_SYSTEM_PROMPT}\n\n"
                f"Objective: {self._objective[:100]}\n"
                f"Failure history:\n{context}\n\n"
                f"Synthesise ONE actionable constraint (≤30 words):"
            )
            result = self._llm(prompt)
            if result and isinstance(result, str) and len(result.strip()) > 5:
                self.add_constraint(
                    kind="avoid",
                    text=result.strip()[:200],
                    confidence=0.80,
                    source="auto_failure",
                )
                _logger.debug("[DICP] LLM synthesised constraint for %r", operator)
        except Exception as exc:
            _logger.debug("[DICP] LLM synthesis failed: %s", exc)

    def _relevance_score(
        self,
        constraint: Constraint,
        world_state: Optional[Dict[str, Any]],
        focused_app: str,
    ) -> float:
        """Score constraint relevance to current context [0.0–1.0]."""
        if world_state is None:
            return 0.5
        score = 0.5
        text_lower = constraint.text.lower()
        # App match
        if focused_app and focused_app.lower() in text_lower:
            score += 0.3
        # Recency (constraints from last 5 min score higher)
        age_min = (time.time() - constraint.created_at) / 60.0
        if age_min < 5:
            score += 0.2
        elif age_min < 15:
            score += 0.1
        # Kind priority
        if constraint.kind == "avoid":
            score += 0.1
        # World state keyword match
        for key in ("focused_app", "window_title", "page_title"):
            ws_val = str(world_state.get(key, "")).lower()
            if ws_val and any(word in text_lower for word in ws_val.split()[:3] if len(word) > 3):
                score += 0.15
                break
        return min(1.0, score)

    def _text_similarity(self, a: str, b: str) -> float:
        """Quick Jaccard word-overlap similarity."""
        wa = set(a.lower().split())
        wb = set(b.lower().split())
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / len(wa | wb)


# ─────────────────────────────────────────────────────────────────────────────
# Singleton factory
# ─────────────────────────────────────────────────────────────────────────────

_instance: Optional[DICPEngine] = None
_instance_lock = threading.Lock()


def get_dicp_engine(
    objective: str = "",
    app_context: str = "",
    llm_caller: Optional[Callable[[str], str]] = None,
) -> DICPEngine:
    """
    Return the global singleton DICPEngine, creating it if necessary.

    Used by GIILoop when GIIController did not initialise a DICP engine
    (degraded startup path).
    """
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = DICPEngine(
                    objective   = objective,
                    app_context = app_context,
                    llm_caller  = llm_caller,
                )
    return _instance


def reset_dicp_singleton() -> None:
    """Reset the singleton — call between tasks for clean state."""
    global _instance
    with _instance_lock:
        _instance = None
