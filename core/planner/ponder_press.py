"""
core/planner/ponder_press.py — Ponder & Press: Divide-and-Conquer Visual Control
==================================================================================
Blueprint §6.5

WHAT THIS IS
------------
Ponder & Press is a divide-and-conquer visual grounding approach that splits
complex UI interactions into two phases:

  PONDER: Reason about the task (no action taken).
    - Analyse the screenshot
    - Identify all relevant UI regions
    - Build a hierarchical action plan
    - Assign confidence scores to each approach

  PRESS: Execute the highest-confidence action.
    - Take the highest-scoring action from Ponder phase
    - If execution fails, back to Ponder with updated context
    - Iterative refinement without blind retries

WHY THIS MATTERS
----------------
Standard agents use a single "observe → act" cycle.  When uncertain, they
either (a) act blindly and fail, or (b) always request human confirmation.
Ponder & Press provides a third option: reason deeply before acting,
and only act when confident.

The "Divide" aspect: complex GUI tasks are split into independently-reasoned
sub-problems, each solved by the best available grounding method.

INTEGRATION
-----------
* Wraps around GIILoop's action selection
* Called by OperatorCycle._propose_operators() for complex/uncertain actions
* Provides a "ponder budget" — the number of reasoning steps before forced action
* Works with any grounding backend (OmniParser, UI-TARS, Aguvis, cloud VLM)

ALGORITHM
---------
1. ponder(screenshot, task, context) → PonderResult
   - Enumerate alternative approaches (up to K=3)
   - Score each by: feasibility, reversibility, confidence
   - Return ranked list with reasoning

2. press(ponder_result, screenshot) → Action
   - Select highest-scoring approach
   - Ground the target element
   - Return concrete action dict

3. ponder_and_press(screenshot, task, context) → Action
   - Combined call; handles full cycle
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
_PP_ENABLED          = os.environ.get("PROJECTZEO_PONDER_PRESS_ENABLED", "1").strip() == "1"
_PP_K_APPROACHES     = int(os.environ.get("PROJECTZEO_PP_K", "3"))
_PP_PONDER_BUDGET    = int(os.environ.get("PROJECTZEO_PP_PONDER_BUDGET", "2"))
_PP_MIN_CONFIDENCE   = float(os.environ.get("PROJECTZEO_PP_MIN_CONF", "0.5"))

_PONDER_SYSTEM = """\
You are Ponder: the reasoning phase of a divide-and-conquer GUI agent.

Screenshot and task given.  Do NOT take any action yet.
Instead, enumerate up to {k} approaches for: {task}

For each approach:
  - Describe the approach concisely
  - Identify the specific UI element to interact with
  - Estimate feasibility (0-1): is the element present and accessible?
  - Estimate reversibility (0-1): can this be undone if wrong?
  - Assign overall confidence (0-1)

Respond in JSON ONLY:
{{
  "approaches": [
    {{
      "approach_id": 1,
      "description": "...",
      "target_element": "...",
      "action": "click|type|scroll|hotkey|wait",
      "action_args": {{}},
      "feasibility": 0.0-1.0,
      "reversibility": 0.0-1.0,
      "confidence": 0.0-1.0,
      "reasoning": "..."
    }}
  ],
  "recommended_approach_id": 1,
  "uncertainty": "low|medium|high"
}}
"""

_PRESS_SYSTEM = """\
You are Press: the execution phase of a divide-and-conquer GUI agent.

Execute approach: {approach_description}
Target element: {target_element}
Action: {action}

Provide the exact action parameters.
Respond in JSON:
{{
  "operation": "click|type|scroll|hotkey|command|wait",
  "coordinate": [x_percent, y_percent],
  "text": "",
  "keys": "",
  "xpath": "",
  "thought": "..."
}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Approach:
    """A single candidate approach from the Ponder phase."""
    approach_id:    int
    description:    str
    target_element: str
    action:         str         # click|type|scroll|hotkey|wait
    action_args:    Dict[str, Any] = field(default_factory=dict)
    feasibility:    float = 0.5
    reversibility:  float = 0.5
    confidence:     float = 0.5
    reasoning:      str   = ""

    @property
    def score(self) -> float:
        """Combined score: confidence × (0.7*feasibility + 0.3*reversibility)."""
        return self.confidence * (0.7 * self.feasibility + 0.3 * self.reversibility)


@dataclass
class PonderResult:
    """Result of the Ponder reasoning phase."""
    approaches:               List[Approach] = field(default_factory=list)
    recommended_approach_id:  int = 1
    uncertainty:              str = "medium"   # low|medium|high
    ponder_duration_ms:       float = 0.0
    screenshot_hash:          str = ""

    @property
    def best_approach(self) -> Optional[Approach]:
        if not self.approaches:
            return None
        # Use recommended if valid, else highest scoring
        for a in self.approaches:
            if a.approach_id == self.recommended_approach_id:
                return a
        return max(self.approaches, key=lambda a: a.score)

    @property
    def is_confident(self) -> bool:
        best = self.best_approach
        if best is None:
            return False
        return best.confidence >= _PP_MIN_CONFIDENCE and self.uncertainty != "high"

    def to_prompt_block(self) -> str:
        """Format for operator selection prompt injection."""
        best = self.best_approach
        if not best:
            return ""
        lines = [
            "── Ponder Analysis ──",
            f"Best approach: {best.description}",
            f"Target: {best.target_element}",
            f"Confidence: {best.confidence:.2f} | Reversibility: {best.reversibility:.2f}",
            f"Uncertainty: {self.uncertainty}",
        ]
        if len(self.approaches) > 1:
            alt = [a for a in self.approaches if a.approach_id != best.approach_id]
            if alt:
                lines.append(f"Alternatives: {'; '.join(a.description[:40] for a in alt[:2])}")
        lines.append("─" * 20)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# PonderPress engine
# ─────────────────────────────────────────────────────────────────────────────

class PonderPress:
    """
    Divide-and-conquer GUI agent: reason deeply, then act precisely.

    Provides structured reasoning before action selection to reduce
    blind-retry loops and increase first-attempt accuracy.
    """

    def __init__(
        self,
        llm_caller: Optional[Callable[[str], str]] = None,
        vlm_caller: Optional[Callable[[str, str], str]] = None,
        screen_width:  int = 1920,
        screen_height: int = 1080,
    ) -> None:
        self._llm  = llm_caller
        self._vlm  = vlm_caller
        self._w    = screen_width
        self._h    = screen_height
        self._lock = threading.Lock()
        self._ponder_history: List[PonderResult] = []

    def ponder(
        self,
        screenshot_b64: str,
        task: str,
        context: str = "",
        world_state: Optional[Dict[str, Any]] = None,
    ) -> PonderResult:
        """
        Ponder phase: reason about the task, enumerate approaches.

        Returns a PonderResult with ranked approaches and confidence scores.
        Does NOT execute anything.
        """
        if not _PP_ENABLED:
            return PonderResult()

        t0 = time.time()
        result = PonderResult()

        if self._llm is None and self._vlm is None:
            result = self._heuristic_ponder(task, world_state)
        else:
            result = self._llm_ponder(screenshot_b64, task, context)

        result.ponder_duration_ms = (time.time() - t0) * 1000
        with self._lock:
            self._ponder_history.append(result)
            if len(self._ponder_history) > 20:
                self._ponder_history = self._ponder_history[-20:]

        _logger.debug(
            "[PonderPress] Ponder complete: %d approaches, best_conf=%.2f, uncertainty=%s",
            len(result.approaches),
            result.best_approach.confidence if result.best_approach else 0,
            result.uncertainty,
        )
        return result

    def press(
        self,
        ponder_result: PonderResult,
        screenshot_b64: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        Press phase: translate best approach into a concrete action dict.

        Returns an action dict compatible with operate.py's action format.
        Returns None if confidence is too low to act.
        """
        best = ponder_result.best_approach
        if best is None:
            return None
        if best.confidence < _PP_MIN_CONFIDENCE and ponder_result.uncertainty == "high":
            _logger.debug("[PonderPress] Confidence too low to press (%.2f < %.2f)", best.confidence, _PP_MIN_CONFIDENCE)
            return None

        # If approach already has full action_args, use them directly
        if best.action_args and "operation" in best.action_args:
            action = dict(best.action_args)
            action.setdefault("thought", f"[PonderPress] {best.description}")
            return action

        # Otherwise: use LLM to generate precise action args
        if self._llm or self._vlm:
            action = self._llm_press(ponder_result, best, screenshot_b64)
            if action:
                return action

        # Fallback: construct action from approach fields
        return self._construct_action_from_approach(best)

    def ponder_and_press(
        self,
        screenshot_b64: str,
        task: str,
        context: str = "",
        world_state: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Combined ponder + press call.

        Returns a concrete action dict, or None if unable to determine action.
        Iterates up to _PP_PONDER_BUDGET times on failure.
        """
        for attempt in range(_PP_PONDER_BUDGET):
            ponder = self.ponder(screenshot_b64, task, context, world_state)
            if not ponder.approaches:
                continue
            action = self.press(ponder, screenshot_b64)
            if action:
                action["_ponder_confidence"] = ponder.best_approach.confidence if ponder.best_approach else 0.0
                action["_ponder_uncertainty"] = ponder.uncertainty
                return action
            # If uncertainty is high, try again with more context
            if ponder.uncertainty == "high" and attempt < _PP_PONDER_BUDGET - 1:
                context += f" Previous analysis: {ponder.best_approach.reasoning if ponder.best_approach else ''}"
        return None

    def get_context_for_psr(self, world_state: Optional[Dict[str, Any]] = None) -> str:
        """Return last ponder result as context for PerStepReasoner."""
        with self._lock:
            if not self._ponder_history:
                return ""
            last = self._ponder_history[-1]
        return last.to_prompt_block()

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _llm_ponder(
        self,
        screenshot_b64: str,
        task: str,
        context: str,
    ) -> PonderResult:
        """LLM-based ponder phase."""
        try:
            prompt = _PONDER_SYSTEM.format(k=_PP_K_APPROACHES, task=task[:150])
            if context:
                prompt += f"\nContext: {context[:200]}"
            if self._vlm and screenshot_b64:
                raw = self._vlm(prompt, screenshot_b64)
            else:
                prompt += "\n(Reasoning without screenshot — use common UI patterns)"
                raw = self._llm(prompt)
            if not raw:
                return PonderResult()
            return self._parse_ponder_response(raw)
        except Exception as exc:
            _logger.debug("[PonderPress] LLM ponder error: %s", exc)
            return PonderResult()

    def _parse_ponder_response(self, raw: str) -> PonderResult:
        """Parse LLM ponder JSON response."""
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return PonderResult()
        try:
            data = json.loads(match.group(0))
            approaches = []
            for ap in data.get("approaches", []):
                try:
                    approaches.append(Approach(
                        approach_id    = int(ap.get("approach_id", 1)),
                        description    = str(ap.get("description", ""))[:150],
                        target_element = str(ap.get("target_element", ""))[:100],
                        action         = str(ap.get("action", "click")),
                        action_args    = ap.get("action_args", {}),
                        feasibility    = float(ap.get("feasibility", 0.5)),
                        reversibility  = float(ap.get("reversibility", 0.5)),
                        confidence     = float(ap.get("confidence", 0.5)),
                        reasoning      = str(ap.get("reasoning", ""))[:100],
                    ))
                except Exception:
                    pass
            return PonderResult(
                approaches              = approaches,
                recommended_approach_id = int(data.get("recommended_approach_id", 1)),
                uncertainty             = str(data.get("uncertainty", "medium")),
            )
        except json.JSONDecodeError:
            return PonderResult()

    def _llm_press(
        self,
        ponder_result: PonderResult,
        approach: Approach,
        screenshot_b64: str,
    ) -> Optional[Dict[str, Any]]:
        """LLM-based press phase."""
        try:
            prompt = _PRESS_SYSTEM.format(
                approach_description = approach.description[:150],
                target_element       = approach.target_element[:100],
                action               = approach.action,
            )
            if self._vlm and screenshot_b64:
                raw = self._vlm(prompt, screenshot_b64)
            else:
                raw = self._llm(prompt)
            if not raw:
                return None
            raw = re.sub(r"```(?:json)?", "", raw).strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                action = json.loads(match.group(0))
                if "operation" in action:
                    action.setdefault("thought", f"[PonderPress] {approach.description}")
                    return action
        except Exception as exc:
            _logger.debug("[PonderPress] LLM press error: %s", exc)
        return None

    def _construct_action_from_approach(self, approach: Approach) -> Dict[str, Any]:
        """Construct action dict from approach without LLM."""
        action: Dict[str, Any] = {
            "operation": approach.action or "click",
            "thought":   f"[PonderPress] {approach.description}",
        }
        if approach.action_args:
            action.update(approach.action_args)
        if approach.target_element and "text" not in action and approach.action == "type":
            action["text"] = approach.target_element
        return action

    def _heuristic_ponder(
        self,
        task: str,
        world_state: Optional[Dict[str, Any]],
    ) -> PonderResult:
        """Heuristic ponder without LLM."""
        approaches = []
        task_lower = task.lower()
        if "click" in task_lower or "press" in task_lower or "select" in task_lower:
            approaches.append(Approach(
                approach_id=1,
                description=f"Click the target element: {task[:60]}",
                target_element=task[:60],
                action="click",
                feasibility=0.6, reversibility=0.7, confidence=0.5,
                reasoning="Standard click approach for interaction tasks",
            ))
        if "type" in task_lower or "enter" in task_lower or "write" in task_lower:
            approaches.append(Approach(
                approach_id=2,
                description="Type text into focused field",
                target_element="text input field",
                action="type",
                feasibility=0.7, reversibility=0.8, confidence=0.55,
                reasoning="Type approach for text entry tasks",
            ))
        if not approaches:
            approaches.append(Approach(
                approach_id=1,
                description=f"Attempt task: {task[:60]}",
                target_element="primary UI element",
                action="click",
                feasibility=0.5, reversibility=0.6, confidence=0.4,
                reasoning="Default approach",
            ))
        return PonderResult(approaches=approaches, uncertainty="medium")


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_instance: Optional[PonderPress] = None
_instance_lock = threading.Lock()


def get_ponder_press(
    llm_caller:    Optional[Callable] = None,
    vlm_caller:    Optional[Callable] = None,
    screen_width:  int = 1920,
    screen_height: int = 1080,
) -> PonderPress:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = PonderPress(
                    llm_caller    = llm_caller,
                    vlm_caller    = vlm_caller,
                    screen_width  = screen_width,
                    screen_height = screen_height,
                )
    return _instance
