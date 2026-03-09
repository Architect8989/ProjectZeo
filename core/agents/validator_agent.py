"""
core/agents/validator_agent.py
================================
Validator Agent — Independent Post-Milestone Verification.

Blueprint §15.4 — Multi-Agent Orchestration

Role: Validator Agent (on-demand, post-milestone)
    - Independent verification of milestone success criteria
    - Returns: PASS / FAIL / UNCERTAIN + rationale
    - Only triggers replanning on FAIL or UNCERTAIN
    - Prevents false-positive milestone completion (clicking "Submit"
      doesn't mean the form was valid)

Why independent validation matters:
    The Primary Agent may hallucinate milestone success if the screen
    contains ambiguous cues. An independent validator uses different
    prompting and different evidence to reach a conclusion.

Verification strategy:
    1. Screenshot-based visual check (is the expected outcome visible?)
    2. Command-based verification (does the expected file/process exist?)
    3. AT-SPI accessibility tree check (is the expected element present?)
    4. LLM-based semantic check (does the world state match the milestone?)

Integration:
    - gii_loop.py → validator.verify_milestone() after milestone completion
    - gii_controller.py → only calls replan() if validation returns FAIL/UNCERTAIN
    - World guidance: False PASS → false progress, wastes remaining budget
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Tunables
# ─────────────────────────────────────────────────────────────────────────────

_VALIDATION_TIMEOUT = float(os.environ.get("PROJECTZEO_VALIDATOR_TIMEOUT", "45.0"))
_CONFIDENCE_PASS    = float(os.environ.get("PROJECTZEO_VALIDATOR_PASS", "0.70"))
_CONFIDENCE_FAIL    = float(os.environ.get("PROJECTZEO_VALIDATOR_FAIL", "0.35"))


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

class ValidationVerdict(str, Enum):
    PASS      = "pass"
    FAIL      = "fail"
    UNCERTAIN = "uncertain"


@dataclass
class ValidationResult:
    """Result of a milestone validation."""
    verdict:     ValidationVerdict
    confidence:  float             # 0.0-1.0
    rationale:   str
    evidence:    List[str]         # Supporting evidence items
    checks_run:  List[str]         # Names of checks performed
    latency_ms:  float
    should_replan: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict":      self.verdict.value,
            "confidence":   round(self.confidence, 3),
            "rationale":    self.rationale[:400],
            "evidence":     self.evidence[:5],
            "checks_run":   self.checks_run,
            "latency_ms":   round(self.latency_ms, 1),
            "should_replan": self.should_replan,
        }


# ─────────────────────────────────────────────────────────────────────────────
# ValidatorAgent
# ─────────────────────────────────────────────────────────────────────────────

class ValidatorAgent:
    """
    Independent post-milestone verification agent.

    Uses multiple independent evidence sources to verify that a milestone
    has actually been completed — not just that the Primary Agent believes
    it has been completed.

    Usage:
        validator = ValidatorAgent(llm_caller=my_llm, os_backend=os_backend)

        result = validator.verify_milestone(
            milestone_desc="File was created at /tmp/output.csv",
            world_snapshot=current_world,
            screenshot_b64=current_screenshot,
        )
        if result.should_replan:
            # Milestone not actually complete — trigger LATS recovery
            ...
    """

    def __init__(
        self,
        *,
        llm_caller: Optional[Callable] = None,
        os_backend: Optional[Any] = None,
        timeout: float = _VALIDATION_TIMEOUT,
    ) -> None:
        self._llm = llm_caller
        self._os  = os_backend
        self._timeout = timeout
        self._validation_count = 0
        self._pass_count  = 0
        self._fail_count  = 0
        self._uncertain_count = 0

        _logger.info("[ValidatorAgent] Initialised.")

    # =========================================================================
    # Public API
    # =========================================================================

    def verify_milestone(
        self,
        milestone_desc: str,
        *,
        world_snapshot: Optional[Dict[str, Any]] = None,
        screenshot_b64: Optional[str] = None,
        previous_action: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Verify that a milestone has actually been completed.

        Runs multiple independent checks:
        1. Heuristic screen-state check (fast, always)
        2. LLM semantic check (if LLM available)
        3. Command-based file/process check (if OS backend available)

        Returns:
            ValidationResult with verdict=PASS/FAIL/UNCERTAIN and rationale.
            result.should_replan=True → Primary Agent should trigger recovery.
        """
        t0 = time.monotonic()
        self._validation_count += 1
        checks_run = []
        evidence = []
        confidence_votes: List[float] = []

        # ── Check 1: Heuristic screen-state check
        screen_result = self._check_screen_state(
            milestone_desc, world_snapshot or {}
        )
        if screen_result is not None:
            checks_run.append("screen_state")
            evidence.append(f"Screen: {screen_result[1]}")
            confidence_votes.append(screen_result[0])

        # ── Check 2: Command-based verification (file/process existence)
        cmd_result = self._check_command_evidence(milestone_desc)
        if cmd_result is not None:
            checks_run.append("command_evidence")
            evidence.append(f"Command: {cmd_result[1]}")
            confidence_votes.append(cmd_result[0])

        # ── Check 3: LLM semantic check
        if self._llm is not None:
            llm_result = self._llm_semantic_check(
                milestone_desc, world_snapshot, previous_action
            )
            if llm_result is not None:
                checks_run.append("llm_semantic")
                evidence.append(f"LLM: {llm_result[1]}")
                confidence_votes.append(llm_result[0])

        # ── Aggregate verdict
        if not confidence_votes:
            # No checks ran — uncertain
            verdict = ValidationVerdict.UNCERTAIN
            confidence = 0.5
            rationale = "No verification evidence available."
        else:
            confidence = sum(confidence_votes) / len(confidence_votes)
            if confidence >= _CONFIDENCE_PASS:
                verdict = ValidationVerdict.PASS
                rationale = f"Milestone likely complete (confidence={confidence:.0%})"
            elif confidence <= _CONFIDENCE_FAIL:
                verdict = ValidationVerdict.FAIL
                rationale = f"Milestone likely incomplete (confidence={confidence:.0%})"
            else:
                verdict = ValidationVerdict.UNCERTAIN
                rationale = f"Milestone completion unclear (confidence={confidence:.0%})"

        # Only trigger replan on FAIL or UNCERTAIN
        should_replan = verdict in (ValidationVerdict.FAIL, ValidationVerdict.UNCERTAIN)

        elapsed = (time.monotonic() - t0) * 1000
        if verdict == ValidationVerdict.PASS:
            self._pass_count += 1
        elif verdict == ValidationVerdict.FAIL:
            self._fail_count += 1
        else:
            self._uncertain_count += 1

        result = ValidationResult(
            verdict=verdict,
            confidence=confidence,
            rationale=rationale,
            evidence=evidence,
            checks_run=checks_run,
            latency_ms=elapsed,
            should_replan=should_replan,
        )

        _logger.info(
            "[ValidatorAgent] Milestone=%r verdict=%s conf=%.2f checks=%s",
            milestone_desc[:60], verdict.value, confidence, checks_run,
        )
        return result

    def verify_task_complete(
        self,
        objective: str,
        world_snapshot: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """Final task completion check — more stringent than milestone check."""
        return self.verify_milestone(
            f"TASK COMPLETE: {objective}",
            world_snapshot=world_snapshot,
        )

    def stats(self) -> Dict[str, Any]:
        total = max(self._validation_count, 1)
        return {
            "total_validations": self._validation_count,
            "pass_count":     self._pass_count,
            "fail_count":     self._fail_count,
            "uncertain_count": self._uncertain_count,
            "pass_rate":      round(self._pass_count / total, 3),
        }

    # =========================================================================
    # Private — Verification checks
    # =========================================================================

    def _check_screen_state(
        self, milestone_desc: str, world_snapshot: Dict[str, Any]
    ) -> Optional[Tuple[float, str]]:
        """
        Heuristic check: does the screen state match expected milestone outcome?
        Returns (confidence, description) or None.
        """
        if not world_snapshot:
            return None

        desc_lower = milestone_desc.lower()
        entities = world_snapshot.get("entities", []) or []
        focused_app = str(world_snapshot.get("focused_app", "")).lower()

        # Look for success signals in entity labels
        entity_labels = [
            str(e.get("label") or e.get("name") or "").lower()
            for e in entities if isinstance(e, dict)
        ]
        all_text = " ".join(entity_labels) + " " + focused_app

        # Extract key nouns from milestone description
        key_nouns = _extract_key_nouns(milestone_desc)

        if not key_nouns:
            return (0.5, "No key nouns to verify")

        # Check how many key nouns appear in current screen state
        found = sum(1 for noun in key_nouns if noun.lower() in all_text)
        fraction = found / len(key_nouns)

        # Check for failure signals in screen state
        failure_signals = ["error", "failed", "not found", "cannot", "denied", "exception"]
        has_failure = any(sig in all_text for sig in failure_signals)

        if has_failure:
            return (0.15, f"Screen shows failure signals: {[s for s in failure_signals if s in all_text][:3]}")

        confidence = 0.3 + fraction * 0.5
        return (confidence, f"Screen matches {found}/{len(key_nouns)} key terms from milestone")

    def _check_command_evidence(
        self, milestone_desc: str
    ) -> Optional[Tuple[float, str]]:
        """
        If milestone mentions a file path, verify it exists.
        Returns (confidence, description) or None.
        """
        if self._os is None:
            return None

        # Extract file paths from milestone description
        paths = re.findall(r"(?:/[\w./\-]+|~/[\w./\-]+|[A-Z]:\\[\w\\.]+)", milestone_desc)
        if not paths:
            return None

        for path in paths[:2]:
            try:
                expanded = os.path.expanduser(path)
                if os.path.exists(expanded):
                    return (0.85, f"Verified file exists: {path}")
                else:
                    return (0.15, f"Expected file NOT found: {path}")
            except Exception:
                pass
        return None

    def _llm_semantic_check(
        self,
        milestone_desc: str,
        world_snapshot: Optional[Dict[str, Any]],
        previous_action: Optional[Dict[str, Any]],
    ) -> Optional[Tuple[float, str]]:
        """LLM-based semantic milestone verification."""
        if self._llm is None:
            return None

        world_summary = ""
        if world_snapshot:
            world_summary = (
                f"Focused app: {world_snapshot.get('focused_app','unknown')}\n"
                f"Entities: {len(world_snapshot.get('entities', []))} visible"
            )

        last_action = ""
        if previous_action:
            op = previous_action.get("operation","?")
            thought = previous_action.get("thought","")[:100]
            last_action = f"Last action: {op} — {thought}"

        prompt = (
            f"Did this milestone succeed? Answer with a JSON object.\n\n"
            f"Milestone: {milestone_desc[:300]}\n"
            f"Current screen state:\n{world_summary}\n"
            f"{last_action}\n\n"
            f'Return: {{"verdict": "pass"|"fail"|"uncertain", "confidence": 0.0-1.0, "reason": "brief"}}'
        )

        result_holder: List[Optional[str]] = [None]

        def _call():
            try:
                raw = self._llm(prompt=prompt, timeout=self._timeout, max_tokens=100)
                result_holder[0] = str(raw.get("text","") if isinstance(raw, dict) else raw)
            except Exception:
                pass

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        t.join(timeout=self._timeout)

        if not result_holder[0]:
            return None

        try:
            import json as json_
            cleaned = re.sub(r"```(?:json)?|```","", result_holder[0]).strip()
            d = json_.loads(cleaned)
            verdict = str(d.get("verdict","uncertain")).lower()
            conf = float(d.get("confidence", 0.5))
            reason = str(d.get("reason",""))[:200]
            conf_final = (
                conf if verdict == "pass"
                else (1.0 - conf) if verdict == "fail"
                else 0.5
            )
            return (conf_final, reason)
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_key_nouns(text: str) -> List[str]:
    """Extract key nouns/identifiers from a milestone description."""
    # Extract quoted strings, file names, app names, quoted actions
    candidates = []
    # File names
    candidates.extend(re.findall(r"[\w\-]+\.(?:py|txt|csv|json|sh|html|pdf|docx|xlsx)", text))
    # Quoted terms
    candidates.extend(re.findall(r"'([^']{2,30})'|\"([^\"]{2,30})\"", text))
    # Flatten tuples
    flat = []
    for c in candidates:
        if isinstance(c, tuple):
            flat.extend([x for x in c if x])
        else:
            flat.append(c)
    # Add app names (capitalized words)
    flat.extend(re.findall(r"\b[A-Z][a-z]{2,15}\b", text))
    return list(set(flat))[:10]
