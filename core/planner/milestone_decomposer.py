from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

_MILESTONE_SYSTEM_PROMPT = """\
You are a goal decomposition engine for an autonomous computer agent.
Your task: decompose an objective into 3–7 milestone CONDITIONS.

A MILESTONE is NOT a list of steps. It is an OBSERVABLE WORLD STATE that
represents meaningful progress toward the objective.

RULES:
  - Each milestone is a CONDITION (something TRUE in the world), not an action
  - Milestones must be detectable from screen content or command output
  - Do NOT specify HOW to achieve the milestone — only WHAT state must exist
  - Order milestones logically: earlier milestones enable later ones
  - Number of milestones: 3 minimum, 7 maximum
  - If the objective is simple (< 3 natural milestones), return 2–3 milestones

OUTPUT FORMAT — respond ONLY with a valid JSON array, no markdown, no preamble:
[
  {
    "id": "milestone_1",
    "name": "short milestone name (< 40 chars)",
    "condition": "Observable condition that must be TRUE. E.g., 'Blender is open and the default workspace is visible'",
    "completion_signal": "What to look for in screen/VL output to verify this milestone is met. E.g., 'Blender title bar visible, Info editor shows default scene'",
    "next_milestone": "milestone_2",
    "retry_allowed": true,
    "estimated_actions": 5
  }
]

The last milestone MUST have "next_milestone": null (it is the terminal state).
"""

_MILESTONE_USER_TEMPLATE = """\
OBJECTIVE: {objective}

ENVIRONMENT:
{env_block}

Decompose this objective into 3–7 milestone conditions.
Return ONLY a JSON array of milestone objects.
"""


class MilestoneDecompositionError(RuntimeError):
    pass


class Milestone:
    """A single observable milestone condition in the goal graph."""

    __slots__ = (
        "id", "name", "condition", "completion_signal",
        "next_milestone", "retry_allowed", "estimated_actions",
        "achieved", "attempts",
    )

    def __init__(
        self,
        *,
        id: str,
        name: str,
        condition: str,
        completion_signal: str,
        next_milestone: Optional[str],
        retry_allowed: bool = True,
        estimated_actions: int = 5,
    ) -> None:
        self.id                = id
        self.name              = name
        self.condition         = condition
        self.completion_signal = completion_signal
        self.next_milestone    = next_milestone
        self.retry_allowed     = retry_allowed
        self.estimated_actions = max(1, int(estimated_actions))
        self.achieved          = False
        self.attempts          = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":                self.id,
            "name":              self.name,
            "condition":         self.condition,
            "completion_signal": self.completion_signal,
            "next_milestone":    self.next_milestone,
            "retry_allowed":     self.retry_allowed,
            "estimated_actions": self.estimated_actions,
            "achieved":          self.achieved,
            "attempts":          self.attempts,
        }

    def format_for_prompt(self) -> str:
        """Format this milestone for injection into PerStepReasoner prompt."""
        status = "✓ ACHIEVED" if self.achieved else "⬜ IN PROGRESS"
        return (
            f"[{status}] {self.name}\n"
            f"  Condition : {self.condition}\n"
            f"  Verify by : {self.completion_signal}"
        )


class MilestoneDecomposer:
    

    # Maximum number of milestones allowed (prevents runaway goal graphs)
    MAX_MILESTONES = 7
    # LLM call timeout for decomposition
    DEFAULT_TIMEOUT = 120.0

    def __init__(
        self,
        llm_callable: Callable,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT,
    ) -> None:
        if not callable(llm_callable):
            raise MilestoneDecompositionError("llm_callable must be callable")
        self._llm = llm_callable
        self._timeout = timeout_seconds

    def decompose(
        self,
        objective: str,
        environment: Optional[Dict[str, Any]] = None,
    ) -> List[Milestone]:
        
        if not isinstance(objective, str) or not objective.strip():
            raise MilestoneDecompositionError("Objective must be a non-empty string")

        env_lines = []
        for key in ("os", "architecture", "display_available", "tools"):
            val = (environment or {}).get(key)
            if val is not None:
                env_lines.append(f"  {key}: {val}")
        env_block = "\n".join(env_lines) or "  (unavailable)"

        prompt = _MILESTONE_USER_TEMPLATE.format(
            objective=objective.strip()[:600],
            env_block=env_block,
        )

        raw = self._call_llm(prompt)
        if raw is None:
            _logger.warning(
                "[MilestoneDecomposer] LLM returned no response — using fallback milestones."
            )
            return self._fallback_milestones(objective)

        try:
            milestones = self._parse_milestones(raw)
            if not milestones:
                _logger.warning(
                    "[MilestoneDecomposer] No valid milestones parsed — using fallback."
                )
                return self._fallback_milestones(objective)
            return milestones
        except MilestoneDecompositionError as exc:
            _logger.warning(
                "[MilestoneDecomposer] Parse failed (%s) — using fallback milestones.", exc
            )
            return self._fallback_milestones(objective)

    def _call_llm(self, prompt: str) -> Optional[str]:
        result_holder: List[Optional[str]] = [None]
        error_holder:  List[Optional[Exception]] = [None]

        def _call():
            try:
                raw = self._llm(
                    messages=[
                        {"role": "system", "content": _MILESTONE_SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    objective=None,
                    session_id="milestone_decomposition",
                )
                if isinstance(raw, list) and raw:
                    result_holder[0] = str(
                        raw[0].get("content", "") if isinstance(raw[0], dict) else raw[0]
                    )
                elif isinstance(raw, str):
                    result_holder[0] = raw
            except Exception as e:
                error_holder[0] = e

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        t.join(timeout=self._timeout)

        if error_holder[0]:
            _logger.warning("[MilestoneDecomposer] LLM call failed: %s", error_holder[0])
            return None
        if t.is_alive():
            _logger.warning("[MilestoneDecomposer] LLM call timed out after %.1fs.", self._timeout)
            return None

        return result_holder[0]

    def _parse_milestones(self, raw: str) -> List[Milestone]:
        clean = re.sub(r"```(?:json)?", "", raw).strip()

        # Try direct parse
        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            # Try extracting JSON array
            m = re.search(r"(\[[\s\S]*\])", clean)
            if not m:
                raise MilestoneDecompositionError(
                    f"No JSON array found in LLM response: {raw[:200]!r}"
                )
            try:
                data = json.loads(m.group(1))
            except json.JSONDecodeError as exc:
                raise MilestoneDecompositionError(
                    f"JSON parse error: {exc} | raw: {raw[:200]!r}"
                ) from exc

        if not isinstance(data, list) or not data:
            raise MilestoneDecompositionError("LLM returned empty or non-list milestone data")

        milestones: List[Milestone] = []
        for i, item in enumerate(data[:self.MAX_MILESTONES]):
            if not isinstance(item, dict):
                continue
            ms_id   = str(item.get("id") or f"milestone_{i+1}")
            name    = str(item.get("name") or f"Milestone {i+1}")[:60]
            cond    = str(item.get("condition") or "")
            signal  = str(item.get("completion_signal") or cond)
            next_ms = item.get("next_milestone")
            if next_ms is not None:
                next_ms = str(next_ms)
            retry   = bool(item.get("retry_allowed", True))
            est_act = int(item.get("estimated_actions") or 5)

            if not cond.strip():
                continue

            milestones.append(Milestone(
                id=ms_id,
                name=name,
                condition=cond[:300],
                completion_signal=signal[:300],
                next_milestone=next_ms,
                retry_allowed=retry,
                estimated_actions=est_act,
            ))

        if not milestones:
            raise MilestoneDecompositionError("No valid milestones after validation")

        # Ensure last milestone has no next
        milestones[-1].next_milestone = None

        return milestones

    def _fallback_milestones(self, objective: str) -> List[Milestone]:
        """
        Minimal 2-milestone fallback used when LLM decomposition fails.
        Provides just enough structure for the GII loop to start executing.
        """
        return [
            Milestone(
                id="milestone_1",
                name="Task initiated",
                condition=f"All prerequisites for '{objective[:60]}' are in place",
                completion_signal="Environment is ready and task can proceed",
                next_milestone="milestone_2",
                retry_allowed=True,
                estimated_actions=5,
            ),
            Milestone(
                id="milestone_2",
                name="Task complete",
                condition=f"Objective achieved: '{objective[:80]}'",
                completion_signal="Task result is visible/verifiable on screen or via command",
                next_milestone=None,
                retry_allowed=False,
                estimated_actions=20,
            ),
        ]

    @staticmethod
    def milestones_to_scaffold_steps(milestones: List[Milestone]) -> List[Dict[str, Any]]:
        
        steps = []
        for ms in milestones:
            steps.append({
                "description": (
                    f"[Milestone: {ms.name}] Achieve: {ms.condition[:150]}"
                ),
                "goal": ms.condition,
                "completion_signal": ms.completion_signal,
                "milestone_id": ms.id,
            })
        return steps

    @staticmethod
    def format_milestones_for_prompt(
        milestones: List[Milestone],
        current_milestone_id: Optional[str] = None,
    ) -> str:
        """Format milestone list for injection into PerStepReasoner system prompt."""
        lines = ["MILESTONES (goal conditions, not steps):"]
        for ms in milestones:
            is_current = (current_milestone_id == ms.id)
            prefix = "→ CURRENT:" if is_current else "  "
            status = "✓" if ms.achieved else "○"
            lines.append(
                f"  {status} {prefix} [{ms.id}] {ms.name}\n"
                f"       Condition : {ms.condition[:120]}\n"
                f"       Verify by : {ms.completion_signal[:80]}"
            )
        return "\n".join(lines)

    @staticmethod
    def advance_milestone(
        milestones: List[Milestone],
        current_idx: int,
    ) -> Tuple[int, Optional[Milestone]]:
        
        if milestones and 0 <= current_idx < len(milestones):
            milestones[current_idx].achieved = True
        next_idx = current_idx + 1
        next_milestone = milestones[next_idx] if next_idx < len(milestones) else None
        return next_idx, next_milestone

    @staticmethod
    def check_completion_signal(
        milestone: Milestone,
        world_state: Dict[str, Any],
    ) -> bool:
        
        signal = milestone.completion_signal.lower().strip()
        if not signal:
            return False

        # Split signal into meaningful words (>3 chars) to avoid stop-word noise
        signal_words = [w for w in signal.split() if len(w) > 3]
        if not signal_words:
            return False

        # Build a text corpus from the world state
        entities = world_state.get("entities", [])
        entity_text = " ".join(
            str(e.get("text", "") or e.get("label", "")).lower()
            for e in (entities if isinstance(entities, list) else [])[:50]
            if isinstance(e, dict)
        )
        focused_app = str(world_state.get("focused_app", "")).lower()
        last_output = str(world_state.get("_last_command_output", "")
                          or world_state.get("_gii_last_output", "")).lower()
        title = str(world_state.get("window_title", "")).lower()

        corpus = f"{entity_text} {focused_app} {last_output} {title}"

        # All key signal words must appear in corpus (AND logic, conservative)
        n_matches = sum(1 for w in signal_words[:5] if w in corpus)
        required = max(1, len(signal_words[:5]) // 2 + 1)  # majority
        return n_matches >= required
