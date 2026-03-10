"""
core/planner/agent_s2_planner.py — Agent S2 Planner
=====================================================
Blueprint §6.2 — Simular AI, arXiv:2504.00906, Best Paper ICLR 2025

WHAT THIS IS
------------
Agent S2 achieved SOTA performance on OSWorld (34.5%), AndroidWorld, and
WindowsAgentArena.  Four design principles that ProjectZeo can adopt:

  1. Proactive Hierarchical Planning
     Plans update after EACH SUBTASK completion — not just on failure.
     Milestone N+1 plan is refined based on what happened in milestone N.

  2. Mixture-of-Grounding (MoG)
     Multiple specialised grounding models combined by confidence weighting.
     Routes to the best available grounding model per element type.
     (In ProjectZeo: integrates with the 6-tier grounding stack.)

  3. Narrative Memory (Mn)
     Per-application narrative describing common patterns, learned through
     exploration.  "In Chrome, the address bar is at the top; clicking it
     selects all text — use ctrl+a then type to navigate."

  4. Episodic Memory (Me)
     Per-task trajectories organised by task type for Algorithm Distillation
     injection.  (Already partially in ProjectZeo — this module formalises it.)

HOW IT WORKS
------------
AgentS2Planner wraps HTNPlanner and adds:

  * after_milestone_complete(milestone, outcome, world_state)
    Called by GIILoop on EVERY milestone completion (success OR failure).
    Generates a "proactive update" that refines the NEXT milestone's plan
    based on what was learned.

  * get_narrative_context(app_name) → str
    Returns the narrative memory for an application.

  * update_narrative(app_name, observation)
    Updates narrative memory from task experience.

  * select_grounding_model(element_type, element_desc) → str
    Returns the recommended grounding model tier for this element type.

INTEGRATION
-----------
* GIILoop → call after_milestone_complete() on each milestone transition
* GIIController → exposes agent_s2 as a property
* PerStepReasoner → receives narrative context before each action decision
* grounding_stack.py → uses select_grounding_model() for tier selection

REFERENCE
---------
Liu et al. (2025) "Agent S2: A Compositional Generalist-Specialist Framework
for GUI Agents" — arXiv:2504.00906, ICLR 2025 Agentic AI Workshop Best Paper
https://github.com/simular-ai/Agent-S
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
_NARRATIVE_DIR        = os.path.expanduser(
    os.environ.get("PROJECTZEO_NARRATIVE_DIR", "~/.projectzeo/narrative_memory")
)
_MAX_NARRATIVE_WORDS  = int(os.environ.get("PROJECTZEO_NARRATIVE_MAX_WORDS", "200"))
_PROACTIVE_ENABLED    = os.environ.get("PROJECTZEO_AGENT_S2_PROACTIVE", "1").strip() == "1"
_MOG_ENABLED          = os.environ.get("PROJECTZEO_AGENT_S2_MOG", "1").strip() == "1"

_PROACTIVE_UPDATE_SYSTEM = """\
You are the Agent S2 proactive hierarchical planner for a GUI agent.

A subtask just completed.  Based on what happened, generate a PROACTIVE UPDATE
for the NEXT subtask plan — incorporating lessons learned.

Be specific.  If a button appeared in an unexpected location, note it.
If a workflow step was different than expected, document the actual sequence.
Return ONLY a brief JSON object:
{
  "next_milestone_guidance": "...",
  "watch_for": ["condition1", "condition2"],
  "avoid": ["pitfall1"],
  "estimated_steps": 1-10
}
"""

_NARRATIVE_UPDATE_SYSTEM = """\
You are updating the narrative memory for a GUI application.

Current narrative (may be empty for new apps):
{current_narrative}

New observation from task execution:
{observation}

Update the narrative to be a concise, factual description of how to work
effectively in this application.  Max {max_words} words.
Focus on: UI layout, common interaction patterns, gotchas, keyboard shortcuts.
Return ONLY the updated narrative text.  No JSON.  No markdown.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProactiveUpdate:
    """Guidance generated after a milestone completes, for the next milestone."""
    next_milestone_guidance: str  = ""
    watch_for:               List[str] = field(default_factory=list)
    avoid:                   List[str] = field(default_factory=list)
    estimated_steps:         int  = 5
    generated_at:            float = field(default_factory=time.time)

    def to_prompt_block(self) -> str:
        if not self.next_milestone_guidance:
            return ""
        lines = ["── Agent S2 Proactive Guidance ──"]
        lines.append(f"Next: {self.next_milestone_guidance}")
        if self.watch_for:
            lines.append(f"Watch for: {'; '.join(self.watch_for[:3])}")
        if self.avoid:
            lines.append(f"Avoid: {'; '.join(self.avoid[:2])}")
        lines.append(f"Est. steps: ~{self.estimated_steps}")
        lines.append("─" * 32)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Agent S2 Planner
# ─────────────────────────────────────────────────────────────────────────────

class AgentS2Planner:
    """
    Agent S2 proactive hierarchical planner with narrative memory and MoG.

    Wraps around HTNPlanner and adds proactive post-milestone updates,
    per-application narrative memory, and mixture-of-grounding routing.
    """

    def __init__(
        self,
        objective: str,
        llm_caller: Optional[Callable[[str], str]] = None,
        htn_planner: Optional[Any] = None,
    ) -> None:
        self._objective   = objective
        self._llm         = llm_caller
        self._htn         = htn_planner
        self._lock        = threading.Lock()
        self._narratives: Dict[str, str] = {}        # app_name → narrative text
        self._last_update: Optional[ProactiveUpdate] = None
        self._milestone_history: List[Dict[str, Any]] = []
        os.makedirs(_NARRATIVE_DIR, exist_ok=True)
        self._load_narratives()
        _logger.debug("[AgentS2] Planner created for objective=%r", objective[:60])

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Proactive Hierarchical Planning
    # ─────────────────────────────────────────────────────────────────────────

    def after_milestone_complete(
        self,
        milestone: str,
        outcome: str,
        world_state: Optional[Dict[str, Any]] = None,
        next_milestone: str = "",
    ) -> ProactiveUpdate:
        """
        Called after EVERY milestone completion (success OR failure).

        Generates a proactive update guiding how to approach the NEXT milestone,
        incorporating what was learned in this one.
        """
        with self._lock:
            self._milestone_history.append({
                "milestone": milestone,
                "outcome":   outcome,
                "ts":        time.time(),
            })

        if not _PROACTIVE_ENABLED:
            return ProactiveUpdate(next_milestone_guidance=next_milestone)

        update = self._generate_proactive_update(
            completed_milestone = milestone,
            outcome             = outcome,
            next_milestone      = next_milestone,
            world_state         = world_state,
        )
        with self._lock:
            self._last_update = update

        # Update narrative for the focused app
        if world_state:
            app = world_state.get("focused_app", "")
            if app:
                self._schedule_narrative_update(app, milestone, outcome, world_state)

        return update

    def get_last_proactive_update(self) -> Optional[ProactiveUpdate]:
        with self._lock:
            return self._last_update

    # ─────────────────────────────────────────────────────────────────────────
    # 2 + 3. Narrative Memory
    # ─────────────────────────────────────────────────────────────────────────

    def get_narrative_context(self, app_name: str) -> str:
        """
        Return the narrative memory for an application.

        This is a per-app accumulated description of UI patterns, shortcuts,
        and common workflows learned through exploration.
        """
        key = self._normalise_app(app_name)
        with self._lock:
            narrative = self._narratives.get(key, "")
        if not narrative:
            return ""
        return f"── App Knowledge: {app_name} ──\n{narrative}\n── end ──"

    def update_narrative(
        self,
        app_name: str,
        observation: str,
        world_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update narrative memory from task experience.

        Merges new observations into the app-specific narrative using
        LLM if available, otherwise simple append.
        """
        key = self._normalise_app(app_name)
        with self._lock:
            current = self._narratives.get(key, "")

        if self._llm:
            t = threading.Thread(
                target=self._llm_update_narrative,
                args=(key, current, observation),
                daemon=True,
            )
            t.start()
        else:
            # Heuristic: append new unique sentences
            new_sentences = [
                s.strip() for s in observation.split(".")
                if len(s.strip()) > 20 and s.strip() not in current
            ]
            if new_sentences:
                updated = current + " " + ". ".join(new_sentences[:3])
                words = updated.split()
                if len(words) > _MAX_NARRATIVE_WORDS:
                    updated = " ".join(words[-_MAX_NARRATIVE_WORDS:])
                with self._lock:
                    self._narratives[key] = updated.strip()
                self._save_narrative(key, updated.strip())

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Mixture-of-Grounding (MoG)
    # ─────────────────────────────────────────────────────────────────────────

    def select_grounding_model(
        self,
        element_type: str,
        element_desc: str = "",
        confidence_history: Optional[Dict[str, float]] = None,
    ) -> Tuple[str, int]:
        """
        Select the best grounding model for this element type.

        Returns (model_name, tier_number) for routing in grounding_stack.py.
        Uses confidence history to route to the model that historically
        performs best on this element category.

        Tier mapping:
          0: AT-SPI (text elements, deterministic)
          1: OmniParser V2 (standard UI elements)
          2: GUI-Actor + RC (novel/complex elements)
          3: UI-TARS-2 (SeeClick; mobile-style or dynamic)
          4: GroundingDINO + SAM2 (custom/novel elements)
          5: Cloud VLM (last resort)
        """
        if not _MOG_ENABLED:
            return ("OmniParser-V2", 1)

        el_lower = element_type.lower()
        desc_lower = element_desc.lower()

        # AT-SPI: text fields, labels, buttons with known text
        if el_lower in ("text", "label", "edittext", "input") and desc_lower:
            return ("AT-SPI", 0)

        # OmniParser V2: standard desktop UI elements
        standard = {"button", "menu", "toolbar", "tab", "checkbox", "radio", "combobox", "link"}
        if el_lower in standard:
            return ("OmniParser-V2", 1)

        # GUI-Actor: coordinate-free, best for custom widgets
        custom = {"canvas", "chart", "map", "slider", "scrollbar", "custom", "widget"}
        if el_lower in custom:
            return ("GUI-Actor+RC", 2)

        # UI-TARS-2: dynamic/animated elements, mobile-style
        dynamic = {"notification", "toast", "popup", "dropdown", "autocomplete", "suggestion"}
        if el_lower in dynamic:
            return ("UI-TARS-2", 3)

        # Grounding DINO + SAM2: open-set, described by NL
        if "icon" in desc_lower or "image" in desc_lower or len(desc_lower) > 30:
            return ("GroundingDINO+SAM2", 4)

        # Use confidence history if available
        if confidence_history:
            best_model = max(confidence_history, key=confidence_history.get)
            tier_map = {
                "AT-SPI": 0, "OmniParser-V2": 1, "GUI-Actor+RC": 2,
                "UI-TARS-2": 3, "GroundingDINO+SAM2": 4, "CloudVLM": 5,
            }
            return (best_model, tier_map.get(best_model, 1))

        # Default: OmniParser V2 for unknown elements
        return ("OmniParser-V2", 1)

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_proactive_update(
        self,
        completed_milestone: str,
        outcome: str,
        next_milestone: str,
        world_state: Optional[Dict[str, Any]],
    ) -> ProactiveUpdate:
        """Generate proactive guidance for next milestone."""
        if not self._llm:
            return self._heuristic_proactive_update(completed_milestone, outcome, next_milestone)
        try:
            with self._lock:
                hist = self._milestone_history[-5:]
            hist_str = "\n".join(
                f"- {h['milestone']}: {h['outcome']}" for h in hist
            )
            ws_app = (world_state or {}).get("focused_app", "")
            prompt = (
                f"{_PROACTIVE_UPDATE_SYSTEM}\n\n"
                f"Objective: {self._objective[:150]}\n"
                f"Completed: {completed_milestone[:100]}\n"
                f"Outcome: {outcome[:100]}\n"
                f"Next milestone: {next_milestone[:100]}\n"
                f"App: {ws_app}\n"
                f"History:\n{hist_str}\n\n"
                "Generate proactive update:"
            )
            raw = self._llm(prompt)
            if raw:
                raw = re.sub(r"```(?:json)?", "", raw).strip()
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                    return ProactiveUpdate(
                        next_milestone_guidance = str(data.get("next_milestone_guidance", ""))[:200],
                        watch_for               = [str(w)[:80] for w in data.get("watch_for", [])[:4]],
                        avoid                   = [str(a)[:80] for a in data.get("avoid", [])[:3]],
                        estimated_steps         = int(data.get("estimated_steps", 5)),
                    )
        except Exception as exc:
            _logger.debug("[AgentS2] Proactive update LLM error: %s", exc)

        return self._heuristic_proactive_update(completed_milestone, outcome, next_milestone)

    def _heuristic_proactive_update(
        self,
        completed: str,
        outcome: str,
        next_ms: str,
    ) -> ProactiveUpdate:
        """Heuristic proactive update without LLM."""
        is_failure = any(w in outcome.lower() for w in ["fail", "error", "denied", "timeout"])
        guidance = next_ms[:100] if next_ms else "Continue with next planned step"
        watch_for = []
        avoid = []
        if is_failure:
            watch_for.append("Check if prerequisites are met before proceeding")
            avoid.append(f"Repeating same approach that failed in: {completed[:40]}")
        return ProactiveUpdate(
            next_milestone_guidance = guidance,
            watch_for               = watch_for,
            avoid                   = avoid,
            estimated_steps         = 5,
        )

    def _schedule_narrative_update(
        self, app: str, milestone: str, outcome: str, world_state: Dict
    ) -> None:
        """Schedule async narrative update."""
        obs = (
            f"In milestone '{milestone[:60]}', outcome was '{outcome[:40]}'. "
            f"App: {app}. "
            f"Screen entities: {', '.join(str(e) for e in world_state.get('entities', [])[:5])}"
        )
        t = threading.Thread(
            target=self.update_narrative,
            args=(app, obs, world_state),
            daemon=True,
        )
        t.start()

    def _llm_update_narrative(
        self,
        key: str,
        current: str,
        observation: str,
    ) -> None:
        """LLM narrative update (runs in background thread)."""
        try:
            prompt = (
                _NARRATIVE_UPDATE_SYSTEM.format(
                    current_narrative = current[:500] if current else "(none)",
                    observation       = observation[:300],
                    max_words         = _MAX_NARRATIVE_WORDS,
                )
            )
            result = self._llm(prompt)
            if result and len(result.strip()) > 10:
                words = result.strip().split()
                if len(words) > _MAX_NARRATIVE_WORDS:
                    result = " ".join(words[:_MAX_NARRATIVE_WORDS])
                with self._lock:
                    self._narratives[key] = result.strip()
                self._save_narrative(key, result.strip())
        except Exception as exc:
            _logger.debug("[AgentS2] LLM narrative update failed: %s", exc)

    def _normalise_app(self, app_name: str) -> str:
        return re.sub(r"[^\w]", "_", app_name.lower())[:30]

    def _save_narrative(self, key: str, text: str) -> None:
        try:
            path = os.path.join(_NARRATIVE_DIR, f"{key}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as exc:
            _logger.debug("[AgentS2] Narrative save failed: %s", exc)

    def _load_narratives(self) -> None:
        """Load all existing narratives from disk."""
        try:
            if not os.path.isdir(_NARRATIVE_DIR):
                return
            for fname in os.listdir(_NARRATIVE_DIR):
                if fname.endswith(".txt"):
                    key = fname[:-4]
                    path = os.path.join(_NARRATIVE_DIR, fname)
                    try:
                        with open(path, encoding="utf-8") as f:
                            self._narratives[key] = f.read().strip()
                    except Exception:
                        pass
            _logger.debug("[AgentS2] Loaded %d narratives.", len(self._narratives))
        except Exception as exc:
            _logger.debug("[AgentS2] Narrative load error: %s", exc)
