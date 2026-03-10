"""
core/cognition/tom_agent.py — Theory of Mind Agent (ToM-agent)
===============================================================
Blueprint §3.3 — arXiv:2501.15355 (2025)

WHAT THIS IS
------------
ToM-agent integrates Theory of Mind into BDI using counterfactual reflection.
The agent models not only its own Beliefs-Desires-Intentions, but also the
user's BDI — enabling it to predict what the user would want even when
instructions are ambiguous or incomplete.

Key insight: Standard BDI agents model their own mental states.
ToM-agent adds a second-order model: "What does the USER believe, desire, and
intend, given what they said and how they're reacting?"

HOW IT WORKS
------------
1. update_user_bdi(instruction, approval_events, world_state)
   Updates the agent's model of the USER's current BDI state based on:
   - The original instruction
   - Subsequent approval/denial events
   - Any explicit corrections

2. counterfactual_reflection(action, outcome) → CounterfactualResult
   "If the user had known this action would lead to this outcome, would they
   still have instructed me to take it?"  Used after task failure to infer
   what the user REALLY wanted vs. what they literally said.

3. generate_clarification_question(ambiguity_type) → str
   When user BDI is uncertain, generates the ONE most important clarifying
   question rather than asking multiple questions or proceeding with wrong
   assumptions.

4. infer_implicit_constraints() → List[str]
   Extracts constraints the user implicitly expects but didn't state
   (e.g., "don't close other windows", "preserve my work").

INTEGRATION
-----------
* GIIController._user_model → extended with ToM reasoning
* PerStepReasoner → receives implicit constraints as context
* BDIGate → enhanced reconsideration with user-intent drift detection

REFERENCE
---------
ToM-agent: arXiv:2501.15355 (2025)
"Theory of Mind in Language Agents: Integrating Belief-Desire-Intention
 Reasoning with Counterfactual Reflection"
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Tunables
# ─────────────────────────────────────────────────────────────────────────────
_TOM_ENABLED         = os.environ.get("PROJECTZEO_TOM_ENABLED", "1").strip() == "1"
_MAX_EVENT_HISTORY   = int(os.environ.get("PROJECTZEO_TOM_MAX_EVENTS", "50"))
_IMPLICIT_CONSTRAINTS_TTL = int(os.environ.get("PROJECTZEO_TOM_CONSTRAINT_TTL", "300"))

_TOM_SYSTEM_PROMPT = """\
You are a Theory of Mind reasoning engine for an autonomous GUI agent.
Your role: model what the USER believes, desires, and intends — not the agent.

Given:
- The user's original instruction
- Observation history (what the agent has tried and what happened)
- Any approval/denial events

Your tasks:
1. USER BELIEFS: What does the user believe about the current state?
2. USER DESIRES: What outcome does the user ultimately want?
3. USER INTENTIONS: What concrete steps does the user expect the agent to take?
4. IMPLICIT CONSTRAINTS: What constraints does the user assume without stating them?
5. AMBIGUITY: What is the single most important thing the agent should clarify?

Respond in JSON:
{
  "user_beliefs": "...",
  "user_desires": "...",
  "user_intentions": ["step1", "step2"],
  "implicit_constraints": ["constraint1", "constraint2"],
  "ambiguity": "...",
  "confidence": 0.0-1.0
}
"""

_COUNTERFACTUAL_PROMPT = """\
You are a counterfactual reasoning engine for a GUI agent.

The agent was instructed: {instruction}
The agent took action: {action}
The outcome was: {outcome}

Counterfactual question: "If the user had known this action would lead to
this outcome, would they still have given the same instruction?"

If NO — what did the user ACTUALLY want? What implicit constraint was violated?

Respond in JSON:
{
  "would_still_instruct": true/false,
  "violated_constraint": "...",
  "actual_user_intent": "...",
  "recommended_adjustment": "...",
  "confidence": 0.0-1.0
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# GII-FIX: Discrete VAD (Valence-Arousal-Dominance) emotion model
# Blueprint §3.3 — arXiv:2501.15355 recommends continuous emotion tracking
# alongside BDI to detect frustration/satisfaction signals that change intent.
# ─────────────────────────────────────────────────────────────────────────────

# VAD emotion vocabulary: (valence, arousal, dominance) ∈ [−1,+1]
_VAD_LEXICON: Dict[str, Tuple[float, float, float]] = {
    "frustration":   (-0.7,  0.5, -0.3),
    "anger":         (-0.8,  0.8, -0.1),
    "satisfaction":  ( 0.8, -0.2,  0.4),
    "approval":      ( 0.7, -0.1,  0.3),
    "confusion":     (-0.3,  0.3, -0.4),
    "urgency":       ( 0.0,  0.8,  0.1),
    "relief":        ( 0.8, -0.5,  0.2),
    "disappointment":(-0.6,  0.2, -0.2),
    "neutral":       ( 0.0,  0.0,  0.0),
}

def _infer_vad_from_events(events: List["_Event"]) -> Tuple[float, float, float]:
    """
    Heuristic VAD (Valence-Arousal-Dominance) inference from approval/denial events.
    Returns (valence, arousal, dominance) ∈ [−1,+1].
    """
    if not events:
        return (0.0, 0.0, 0.0)
    recent = events[-10:]
    denial_count = sum(1 for e in recent if e.event_type == "denial")
    approval_count = sum(1 for e in recent if e.event_type == "approval")
    failure_count = sum(1 for e in recent if e.event_type == "failure")
    total = len(recent)
    neg_ratio = (denial_count + failure_count) / max(total, 1)
    valence  = max(-1.0, min(1.0, 0.5 - neg_ratio * 1.2 + approval_count * 0.15))
    arousal  = max(-1.0, min(1.0, 0.3 + denial_count * 0.1 + failure_count * 0.08))
    dominance = max(-1.0, min(1.0, 0.1 - neg_ratio * 0.5))
    return (round(valence, 2), round(arousal, 2), round(dominance, 2))

def _emotion_label_from_vad(vad: Tuple[float, float, float]) -> str:
    """Map a VAD tuple to the closest emotion label in the lexicon."""
    v, a, d = vad
    best_label = "neutral"
    best_dist = float("inf")
    for label, (lv, la, ld) in _VAD_LEXICON.items():
        dist = (v - lv)**2 + (a - la)**2 + (d - ld)**2
        if dist < best_dist:
            best_dist = dist
            best_label = label
    return best_label


@dataclass
class UserBDI:
    """Model of the user's current Belief-Desire-Intention state with VAD emotion.""",
    beliefs:              str = ""           # What the user believes about current state
    desires:              str = ""           # Ultimate outcome the user wants
    intentions:           List[str] = field(default_factory=list)  # Expected agent steps
    implicit_constraints: List[str] = field(default_factory=list)  # Unstated assumptions
    ambiguity:            str = ""           # Most important open question
    confidence:           float = 0.5
    last_updated:         float = field(default_factory=time.time)
    # GII-FIX: VAD emotion model (Valence, Arousal, Dominance)
    emotion_vad:          Tuple[float, float, float] = field(default=(0.0, 0.0, 0.0))
    emotion_label:        str = "neutral"    # Closest emotion from _VAD_LEXICON

    def to_prompt_block(self) -> str:
        """Format for PerStepReasoner context injection (includes VAD emotion)."""
        if self.confidence < 0.3:
            return ""
        lines = ["── User Intent Model (ToM) ──"]
        if self.desires:
            lines.append(f"User wants: {self.desires}")
        if self.intentions:
            lines.append(f"Expected steps: {'; '.join(self.intentions[:3])}")
        if self.implicit_constraints:
            lines.append("Implicit constraints:")
            for c in self.implicit_constraints[:4]:
                lines.append(f"  • {c}")
        # GII-FIX: emit VAD emotion signal so the agent adapts to user state
        v, a, d = self.emotion_vad
        if abs(v) > 0.15 or abs(a) > 0.15:
            _EMOTION_HINTS = {
                "frustration": "User appears frustrated — slow down, confirm before acting.",
                "anger": "User appears angry — be cautious and confirm every step.",
                "disappointment": "User appears disappointed — acknowledge failure, pivot approach.",
                "satisfaction": "User appears satisfied — continue current approach.",
                "approval": "User approved — proceed confidently.",
                "relief": "User signal: relief — task on track.",
                "urgency": "User signal: urgency — prioritise speed.",
                "confusion": "User appears confused — add brief explanation to next action.",
            }
            hint = _EMOTION_HINTS.get(self.emotion_label, "")
            if hint:
                lines.append(f"⚡ Emotion [{self.emotion_label} V={v:+.1f} A={a:+.1f}]: {hint}")
        if self.ambiguity and self.confidence < 0.6:
            lines.append(f"⚠ Ambiguity: {self.ambiguity}")
        lines.append("─" * 30)
        return "\n".join(lines)


@dataclass
class CounterfactualResult:
    """Result of counterfactual reasoning about an action/outcome pair."""
    would_still_instruct:   bool  = True
    violated_constraint:    str   = ""
    actual_user_intent:     str   = ""
    recommended_adjustment: str   = ""
    confidence:             float = 0.5
    action_summary:         str   = ""
    outcome_summary:        str   = ""


@dataclass
class _Event:
    """An approval/denial/outcome event in the task history."""
    event_type:  str    # "approval" | "denial" | "success" | "failure" | "correction"
    description: str
    timestamp:   float = field(default_factory=time.time)


# ─────────────────────────────────────────────────────────────────────────────
# ToMAgent
# ─────────────────────────────────────────────────────────────────────────────

class ToMAgent:
    """
    Theory of Mind agent: models user BDI with counterfactual reflection.

    Wraps around UserModel to provide richer user-intent reasoning.
    """

    def __init__(
        self,
        original_instruction: str,
        llm_caller: Optional[Callable[[str], str]] = None,
    ) -> None:
        self._instruction = original_instruction
        self._llm         = llm_caller
        self._user_bdi    = UserBDI()
        self._events:      List[_Event] = []
        self._lock         = threading.Lock()
        self._implicit_constraints_cache: List[str] = []
        self._cache_ts:    float = 0.0

        # Initialise BDI from instruction alone (heuristic, no LLM)
        self._initialise_bdi_heuristic()
        _logger.debug("[ToMAgent] Created for instruction=%r", original_instruction[:60])

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def update_user_bdi(
        self,
        world_state: Optional[Dict[str, Any]] = None,
        new_events: Optional[List[Dict[str, Any]]] = None,
    ) -> UserBDI:
        """
        Update the user BDI model based on new observations and events.
        Uses LLM for rich inference if available; heuristic otherwise.
        """
        if new_events:
            with self._lock:
                for ev in new_events:
                    self._events.append(_Event(
                        event_type  = str(ev.get("type", "unknown")),
                        description = str(ev.get("description", ""))[:100],
                    ))
                if len(self._events) > _MAX_EVENT_HISTORY:
                    self._events = self._events[-_MAX_EVENT_HISTORY:]

        if self._llm and _TOM_ENABLED:
            updated = self._llm_update_bdi(world_state)
            if updated:
                return self._user_bdi

        # Heuristic update from events
        self._heuristic_update_bdi()
        return self._user_bdi

    def record_approval(self, action_desc: str) -> None:
        """Record a user approval event."""
        with self._lock:
            self._events.append(_Event("approval", action_desc[:80]))
            # User approved → they trust this direction; increase desire confidence
            self._user_bdi.confidence = min(1.0, self._user_bdi.confidence + 0.05)

    def record_denial(self, action_desc: str, reason: str = "") -> None:
        """Record a user denial event — triggers BDI confidence reduction."""
        with self._lock:
            self._events.append(_Event("denial", f"{action_desc[:60]}: {reason[:40]}"))
            # Denial is signal that our BDI model may be wrong
            self._user_bdi.confidence = max(0.1, self._user_bdi.confidence - 0.15)
            self._user_bdi.ambiguity = (
                f"User denied '{action_desc[:40]}'. "
                "Unclear if they want a different approach or different goal."
            )

    def counterfactual_reflection(
        self,
        action: Dict[str, Any],
        outcome: str,
    ) -> CounterfactualResult:
        """
        Counterfactual reasoning: would user still give same instruction
        knowing this action would lead to this outcome?
        """
        result = CounterfactualResult(
            action_summary  = f"{action.get('operation','?')}({str(action.get('command', action.get('text', '')))[:30]})",
            outcome_summary = outcome[:80],
        )
        if not _TOM_ENABLED or self._llm is None:
            return self._heuristic_counterfactual(action, outcome, result)

        try:
            prompt = _COUNTERFACTUAL_PROMPT.format(
                instruction = self._instruction[:150],
                action      = json.dumps(action, default=str)[:200],
                outcome     = outcome[:150],
            )
            raw = self._llm(prompt)
            if raw:
                parsed = self._parse_json_safe(raw)
                if parsed:
                    result.would_still_instruct   = bool(parsed.get("would_still_instruct", True))
                    result.violated_constraint    = str(parsed.get("violated_constraint", ""))[:200]
                    result.actual_user_intent     = str(parsed.get("actual_user_intent", ""))[:200]
                    result.recommended_adjustment = str(parsed.get("recommended_adjustment", ""))[:200]
                    result.confidence             = float(parsed.get("confidence", 0.5))
                    _logger.debug(
                        "[ToMAgent] Counterfactual: would_still=%s viol=%r",
                        result.would_still_instruct, result.violated_constraint[:40],
                    )
                    return result
        except Exception as exc:
            _logger.debug("[ToMAgent] Counterfactual LLM error: %s", exc)

        return self._heuristic_counterfactual(action, outcome, result)

    def generate_clarification_question(self) -> str:
        """
        Generate the ONE most important clarifying question.
        Returns empty string if intent is clear enough.
        """
        with self._lock:
            ambiguity = self._user_bdi.ambiguity
            confidence = self._user_bdi.confidence
        if confidence >= 0.7 or not ambiguity:
            return ""
        return ambiguity

    def infer_implicit_constraints(self) -> List[str]:
        """
        Return list of constraints the user implicitly expects.
        Cached for _IMPLICIT_CONSTRAINTS_TTL seconds.
        """
        now = time.time()
        if now - self._cache_ts < _IMPLICIT_CONSTRAINTS_TTL:
            return self._implicit_constraints_cache

        with self._lock:
            constraints = list(self._user_bdi.implicit_constraints)

        # Add universal implicit constraints
        universal = [
            "Do not close or modify files not related to the current task",
            "Preserve existing data unless explicitly instructed to delete it",
            "Do not submit forms or make purchases without explicit confirmation",
        ]
        for u in universal:
            if u not in constraints:
                constraints.append(u)

        # Task-specific implicit constraints
        obj_lower = self._instruction.lower()
        if any(w in obj_lower for w in ["install", "update", "upgrade"]):
            constraints.append("Do not restart the system without explicit confirmation")
        if any(w in obj_lower for w in ["email", "send", "message"]):
            constraints.append("Verify recipient and content before sending")
        if any(w in obj_lower for w in ["delete", "remove", "clean"]):
            constraints.append("Confirm scope of deletion matches exactly what was requested")

        self._implicit_constraints_cache = constraints[:8]
        self._cache_ts = now
        return self._implicit_constraints_cache

    def get_user_bdi(self) -> UserBDI:
        with self._lock:
            return self._user_bdi

    def get_context_for_psr(self) -> str:
        """Return formatted ToM context for PerStepReasoner injection."""
        bdi = self.get_user_bdi()
        implicit = self.infer_implicit_constraints()
        bdi.implicit_constraints = implicit
        return bdi.to_prompt_block()

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _initialise_bdi_heuristic(self) -> None:
        """Bootstrap BDI from instruction text without LLM."""
        obj = self._instruction
        desires = obj[:150]
        implicit = [
            "Preserve existing work and unsaved files",
            "Do not modify data outside the scope of this task",
        ]
        obj_lower = obj.lower()
        intentions = []
        if "open" in obj_lower or "launch" in obj_lower:
            intentions.append("Open the specified application or file")
        if "save" in obj_lower or "create" in obj_lower:
            intentions.append("Save changes when complete")
        if "close" in obj_lower or "exit" in obj_lower:
            intentions.append("Close the application cleanly after task")

        with self._lock:
            self._user_bdi = UserBDI(
                desires              = desires,
                intentions           = intentions,
                implicit_constraints = implicit,
                confidence           = 0.4,  # Low until confirmed by LLM/events
            )

    def _llm_update_bdi(self, world_state: Optional[Dict[str, Any]]) -> bool:
        """Use LLM to update BDI model."""
        try:
            with self._lock:
                recent_events = self._events[-10:]
            events_str = "\n".join(
                f"- [{e.event_type}] {e.description}"
                for e in recent_events
            )
            ws_summary = ""
            if world_state:
                ws_summary = f"App: {world_state.get('focused_app','?')}"
            prompt = (
                f"{_TOM_SYSTEM_PROMPT}\n\n"
                f"Original instruction: {self._instruction[:200]}\n"
                f"Current world: {ws_summary}\n"
                f"Recent events:\n{events_str or '(none)'}\n\n"
                "Infer user BDI:"
            )
            raw = self._llm(prompt)
            if not raw:
                return False
            parsed = self._parse_json_safe(raw)
            if not parsed:
                return False
            with self._lock:
                self._user_bdi.beliefs              = str(parsed.get("user_beliefs", ""))[:200]
                self._user_bdi.desires              = str(parsed.get("user_desires", ""))[:200]
                self._user_bdi.intentions           = [str(i)[:100] for i in parsed.get("user_intentions", [])[:5]]
                self._user_bdi.implicit_constraints = [str(c)[:100] for c in parsed.get("implicit_constraints", [])[:6]]
                self._user_bdi.ambiguity            = str(parsed.get("ambiguity", ""))[:200]
                self._user_bdi.confidence           = float(parsed.get("confidence", 0.5))
                self._user_bdi.last_updated         = time.time()
                # GII-FIX: Update VAD model after every LLM-based BDI update
                vad = _infer_vad_from_events(self._events)
                self._user_bdi.emotion_vad   = vad
                self._user_bdi.emotion_label = _emotion_label_from_vad(vad)
            # Invalidate implicit constraints cache
            self._cache_ts = 0.0
            return True
        except Exception as exc:
            _logger.debug("[ToMAgent] LLM BDI update failed: %s", exc)
            return False

    def _heuristic_update_bdi(self) -> None:
        """Update BDI heuristically from event history (includes VAD update)."""
        with self._lock:
            denial_count = sum(1 for e in self._events if e.event_type == "denial")
            approval_count = sum(1 for e in self._events if e.event_type == "approval")
            total = denial_count + approval_count
            if total > 0:
                approval_rate = approval_count / total
                self._user_bdi.confidence = min(0.9, 0.3 + approval_rate * 0.6)
            if denial_count >= 2:
                self._user_bdi.ambiguity = (
                    f"{denial_count} denials suggest the agent's approach doesn't match "
                    "user expectations — reconsider strategy."
                )
            # GII-FIX: Update VAD emotion model from event history
            vad = _infer_vad_from_events(self._events)
            self._user_bdi.emotion_vad = vad
            self._user_bdi.emotion_label = _emotion_label_from_vad(vad)

    def _heuristic_counterfactual(
        self,
        action: Dict[str, Any],
        outcome: str,
        result: CounterfactualResult,
    ) -> CounterfactualResult:
        """Heuristic counterfactual reasoning without LLM."""
        outcome_lower = outcome.lower()
        failure_signals = ["error", "fail", "denied", "exception", "crash", "timeout"]
        is_failure = any(s in outcome_lower for s in failure_signals)
        if is_failure:
            result.would_still_instruct = False
            result.violated_constraint = (
                "User likely expected the agent to handle this gracefully "
                "without causing errors."
            )
            result.recommended_adjustment = (
                "Try a safer, more conservative approach that avoids error conditions."
            )
        else:
            result.would_still_instruct = True
        result.confidence = 0.4
        return result

    def _parse_json_safe(self, text: str) -> Optional[Dict]:
        """Parse JSON from LLM output, tolerating markdown fences."""
        text = re.sub(r"```(?:json)?", "", text).strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_instance: Optional[ToMAgent] = None
_instance_lock = threading.Lock()


def get_tom_agent(
    instruction: str = "",
    llm_caller: Optional[Callable[[str], str]] = None,
) -> ToMAgent:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = ToMAgent(
                    original_instruction = instruction,
                    llm_caller           = llm_caller,
                )
    return _instance


def reset_tom_agent() -> None:
    """Reset per-task — call at task start with new instruction."""
    global _instance
    with _instance_lock:
        _instance = None
