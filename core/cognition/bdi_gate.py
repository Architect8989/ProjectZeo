"""
core/cognition/bdi_gate.py
===========================
BDI (Belief-Desire-Intention) Deliberation Gate.

Blueprint §3.3 — Bratman (1987); Rao & Georgeff (1995);
                  ToM-agent 2025 (arXiv:2501.15355)

Architecture:
    An agent should NOT spend every cycle re-deliberating the plan.
    It should commit to an intention and execute until the environment
    changes enough to warrant reconsidering.

    BDI Reconsideration Question:
        "Has my belief state diverged from what I expected enough to
         justify replanning?"

    Current ProjectZeo replans on every stagnation event and on
    MAX_REPLANS exhaustion. BDI reconsideration triggers on
    BELIEF STATE DIVERGENCE — not iteration count.

Jaccard Similarity threshold (from blueprint):
    def should_reconsider(expected_state, actual_state, threshold=0.3):
        expected_elements = set(expected_state.world_graph.nodes)
        actual_elements = set(actual_state.world_graph.nodes)
        intersection = len(expected_elements & actual_elements)
        union = len(expected_elements | actual_elements)
        jaccard = intersection / union if union > 0 else 1.0
        return jaccard < (1.0 - threshold)
    → Replan if <70% overlap between expected and actual world graph.

ToM Extension (Blueprint §17.1):
    Models user Beliefs-Desires-Intentions from conversation history.
    When user intent diverges from agent's current intention → reconsider.

Integration:
    - gii_loop.py → call bdi_gate.should_reconsider() before each milestone
    - gii_controller.py → call bdi_gate.update_expected_state() after plan
    - per_step_reasoner.py → call bdi_gate.update_actual_state() after observation
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

_logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Tunables
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_DIVERGENCE_THRESHOLD = float(
    os.environ.get("PROJECTZEO_BDI_THRESHOLD", "0.30")
)
_TOM_INTENT_DRIFT_THRESHOLD = float(
    os.environ.get("PROJECTZEO_TOM_THRESHOLD", "0.40")
)
_COMMITMENT_MIN_STEPS = int(
    os.environ.get("PROJECTZEO_BDI_COMMIT_STEPS", "3")
)


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BeliefStateSnapshot:
    """Lightweight snapshot of world state for divergence calculation."""
    timestamp:      float
    elements:       Set[str]       # UI element keys (app + type + label)
    focused_app:    str
    open_windows:   List[str]
    active_goal:    str
    context_hash:   str            # hash of screen description

    @classmethod
    def from_world_snapshot(cls, world_snapshot: Dict[str, Any], active_goal: str = "") -> "BeliefStateSnapshot":
        """Build from the world_snapshot dict used throughout ProjectZeo."""
        elements: Set[str] = set()
        for ent in world_snapshot.get("entities", []) or []:
            if isinstance(ent, dict):
                label = str(ent.get("label") or ent.get("name") or "")
                role  = str(ent.get("role") or ent.get("type") or "")
                app   = str(ent.get("app") or "")
                key   = f"{app}:{role}:{label[:30]}".lower().strip()
                if key:
                    elements.add(key)

        focused_app = str(world_snapshot.get("focused_app") or "")
        open_windows = [
            str(w) for w in (world_snapshot.get("open_windows") or [])
        ]
        screen_desc = str(world_snapshot.get("screen_description") or "")
        import hashlib
        ctx_hash = hashlib.md5(screen_desc.encode()).hexdigest()[:8]

        return cls(
            timestamp=time.time(),
            elements=elements,
            focused_app=focused_app,
            open_windows=open_windows,
            active_goal=active_goal,
            context_hash=ctx_hash,
        )

    def jaccard_similarity(self, other: "BeliefStateSnapshot") -> float:
        """Jaccard similarity of element sets."""
        union = self.elements | other.elements
        if not union:
            return 1.0
        intersection = self.elements & other.elements
        return len(intersection) / len(union)


@dataclass
class ReconsiderationResult:
    """Outcome of a BDI reconsideration check."""
    should_reconsider: bool
    reason:            str
    jaccard_similarity: float
    divergence_type:   str        # "world_state" | "user_intent" | "goal_drift" | "none"
    details:           Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserIntentModel:
    """Theory of Mind — models user Beliefs-Desires-Intentions (Blueprint §17.1)."""
    stated_goal:       str = ""
    implied_urgency:   float = 0.5   # 0=relaxed, 1=urgent
    estimated_patience: float = 0.7  # 0=impatient, 1=patient
    recent_utterances: List[str] = field(default_factory=list)
    preference_signals: Dict[str, float] = field(default_factory=dict)
    last_updated:      float = field(default_factory=time.time)

    def update_from_utterance(self, utterance: str) -> None:
        """Update user model from a new utterance (simple heuristics)."""
        u = utterance.lower()
        self.recent_utterances.append(utterance[:200])
        self.recent_utterances = self.recent_utterances[-5:]

        # Urgency signals
        if any(w in u for w in ["urgent", "asap", "now", "immediately", "hurry", "!"]):
            self.implied_urgency = min(1.0, self.implied_urgency + 0.2)
        # Patience signals
        if any(w in u for w in ["take your time", "no rush", "whenever"]):
            self.estimated_patience = min(1.0, self.estimated_patience + 0.2)
        # Impatience signals
        elif any(w in u for w in ["why", "still", "again", "already", "wrong", "stop"]):
            self.estimated_patience = max(0.0, self.estimated_patience - 0.15)

        self.last_updated = time.time()

    def intent_drift_from(self, original_goal: str) -> float:
        """
        Estimate how much user intent has drifted from original goal.
        Returns 0.0 (no drift) to 1.0 (complete drift).
        Simple token overlap heuristic; override with LLM in production.
        """
        import re
        orig_tokens = set(re.sub(r"[^\w]", " ", original_goal.lower()).split())
        current_tokens: Set[str] = set()
        for utt in self.recent_utterances:
            current_tokens |= set(re.sub(r"[^\w]", " ", utt.lower()).split())

        if not orig_tokens or not current_tokens:
            return 0.0
        overlap = len(orig_tokens & current_tokens) / len(orig_tokens | current_tokens)
        return 1.0 - overlap


# ─────────────────────────────────────────────────────────────────────────────
# BDIGate
# ─────────────────────────────────────────────────────────────────────────────

class BDIGate:
    """
    BDI deliberation gate — decides when to commit vs. replan.

    Commit conditions (CONTINUE executing current intention):
        1. Belief state Jaccard similarity >= (1.0 - threshold)
        2. At least min_commit_steps taken since last replan
        3. User intent model shows no significant drift

    Reconsider conditions (REPLAN):
        1. Belief state Jaccard similarity < (1.0 - threshold)  ← primary
        2. User intent has drifted significantly from stated goal
        3. Active goal is no longer achievable given current belief state
    """

    def __init__(
        self,
        *,
        divergence_threshold: float = _DEFAULT_DIVERGENCE_THRESHOLD,
        tom_threshold: float = _TOM_INTENT_DRIFT_THRESHOLD,
        min_commit_steps: int = _COMMITMENT_MIN_STEPS,
    ) -> None:
        self._div_threshold = divergence_threshold
        self._tom_threshold = tom_threshold
        self._min_commit_steps = min_commit_steps

        self._expected_state: Optional[BeliefStateSnapshot] = None
        self._last_reconsider_time: float = 0.0
        self._steps_since_replan: int = 0
        self._user_model = UserIntentModel()
        self._original_goal: str = ""

    # =========================================================================
    # Public API
    # =========================================================================

    def set_plan_context(
        self,
        expected_state: Dict[str, Any],
        goal: str,
    ) -> None:
        """
        Called when a new plan is created (HTN decomposition).
        Sets the expected world state to compare against.
        """
        self._expected_state = BeliefStateSnapshot.from_world_snapshot(
            expected_state, active_goal=goal
        )
        self._original_goal = goal
        self._steps_since_replan = 0
        self._last_reconsider_time = time.time()
        _logger.debug(
            "[BDIGate] Plan context set: goal=%r elements=%d",
            goal[:80], len(self._expected_state.elements),
        )

    def update_actual_state(self, world_snapshot: Dict[str, Any]) -> None:
        """Called after each observation — tracks actual world state."""
        self._steps_since_replan += 1

    def update_user_model(self, utterance: str) -> None:
        """Update Theory of Mind user model from new utterance."""
        self._user_model.update_from_utterance(utterance)

    def should_reconsider(
        self,
        actual_state: Dict[str, Any],
        active_goal: str = "",
    ) -> ReconsiderationResult:
        """
        BDI Reconsideration Check.

        Args:
            actual_state: Current world_snapshot dict
            active_goal: Current active milestone description

        Returns:
            ReconsiderationResult with should_reconsider=True if replanning needed
        """
        # Minimum commitment: don't reconsider if too few steps taken
        if self._steps_since_replan < self._min_commit_steps:
            return ReconsiderationResult(
                should_reconsider=False,
                reason=f"Commitment lock: only {self._steps_since_replan}/{self._min_commit_steps} steps taken",
                jaccard_similarity=1.0,
                divergence_type="none",
            )

        if self._expected_state is None:
            return ReconsiderationResult(
                should_reconsider=False,
                reason="No expected state set yet",
                jaccard_similarity=1.0,
                divergence_type="none",
            )

        # ── Check 1: World state divergence (Jaccard)
        actual_snapshot = BeliefStateSnapshot.from_world_snapshot(
            actual_state, active_goal=active_goal
        )
        jaccard = self._expected_state.jaccard_similarity(actual_snapshot)
        jaccard_threshold = 1.0 - self._div_threshold

        if jaccard < jaccard_threshold:
            reason = (
                f"World state diverged: Jaccard={jaccard:.2f} < threshold={jaccard_threshold:.2f}. "
                f"Expected {len(self._expected_state.elements)} elements, "
                f"got {len(actual_snapshot.elements)} ({len(self._expected_state.elements & actual_snapshot.elements)} overlap)."
            )
            _logger.info("[BDIGate] Reconsideration triggered: %s", reason)
            return ReconsiderationResult(
                should_reconsider=True,
                reason=reason,
                jaccard_similarity=jaccard,
                divergence_type="world_state",
                details={
                    "expected_elements": len(self._expected_state.elements),
                    "actual_elements": len(actual_snapshot.elements),
                    "overlap": len(self._expected_state.elements & actual_snapshot.elements),
                    "missing": list(
                        (self._expected_state.elements - actual_snapshot.elements)
                    )[:5],
                    "new_elements": list(
                        (actual_snapshot.elements - self._expected_state.elements)
                    )[:5],
                },
            )

        # ── Check 2: User intent drift (Theory of Mind)
        if self._original_goal and self._user_model.recent_utterances:
            drift = self._user_model.intent_drift_from(self._original_goal)
            if drift > self._tom_threshold:
                reason = (
                    f"User intent drift detected: drift={drift:.2f} > threshold={self._tom_threshold:.2f}. "
                    f"Original goal may no longer match user desire."
                )
                _logger.info("[BDIGate] ToM reconsideration triggered: %s", reason)
                return ReconsiderationResult(
                    should_reconsider=True,
                    reason=reason,
                    jaccard_similarity=jaccard,
                    divergence_type="user_intent",
                    details={"drift_score": drift},
                )

        # ── Check 3: Focused app changed unexpectedly
        if (
            actual_snapshot.focused_app
            and self._expected_state.focused_app
            and actual_snapshot.focused_app != self._expected_state.focused_app
        ):
            reason = (
                f"Unexpected app focus change: expected={self._expected_state.focused_app!r} "
                f"actual={actual_snapshot.focused_app!r}"
            )
            _logger.info("[BDIGate] App focus reconsideration: %s", reason)
            return ReconsiderationResult(
                should_reconsider=True,
                reason=reason,
                jaccard_similarity=jaccard,
                divergence_type="goal_drift",
                details={
                    "expected_app": self._expected_state.focused_app,
                    "actual_app":   actual_snapshot.focused_app,
                },
            )

        # No reconsideration needed
        return ReconsiderationResult(
            should_reconsider=False,
            reason=f"Belief state stable: Jaccard={jaccard:.2f} >= {jaccard_threshold:.2f}",
            jaccard_similarity=jaccard,
            divergence_type="none",
        )

    def reset_commitment(self) -> None:
        """Reset after a successful replan — restart commitment counter."""
        self._steps_since_replan = 0
        self._expected_state = None
        _logger.debug("[BDIGate] Commitment reset after replan.")

    def get_user_model(self) -> UserIntentModel:
        """Return the current Theory of Mind user model."""
        return self._user_model

    def adjust_behavior_for_user_state(self) -> Dict[str, Any]:
        """
        Blueprint §17.2 — return behavior adjustments based on user emotional state.
        Maps urgency/patience to operational adjustments.
        """
        u = self._user_model
        adjustments: Dict[str, Any] = {}

        if u.implied_urgency > 0.7:
            # Urgent: skip Self-Refine, reduce confirmation threshold
            adjustments["skip_self_refine"] = True
            adjustments["confirmation_threshold"] = "minimal"
            adjustments["verbosity"] = "concise"
        elif u.estimated_patience < 0.3:
            # Impatient: offer manual handoff
            adjustments["offer_handoff"] = True
            adjustments["verbose_explanations"] = True
            adjustments["verbosity"] = "detailed"
        else:
            adjustments["verbosity"] = "standard"

        return adjustments

    @property
    def steps_since_replan(self) -> int:
        return self._steps_since_replan
