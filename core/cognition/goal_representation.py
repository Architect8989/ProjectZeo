from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Tunables (env-configurable)
# ─────────────────────────────────────────────────────────────────────────────
import os

_DECOMPOSE_MAX_CONDITIONS    = int(os.environ.get("PROJECTZEO_GOAL_MAX_CONDITIONS", "8"))
_EVAL_TIMEOUT_SEC            = float(os.environ.get("PROJECTZEO_GOAL_EVAL_TIMEOUT", "30"))
_STALL_THRESHOLD_EVALS       = int(os.environ.get("PROJECTZEO_GOAL_STALL_EVALS", "8"))
_MIN_CONFIDENCE_TO_COUNT     = float(os.environ.get("PROJECTZEO_GOAL_MIN_CONF", "0.5"))
_LLM_EVAL_ENABLED            = os.environ.get("PROJECTZEO_GOAL_LLM_EVAL", "1") == "1"
_COMPLETE_THRESHOLD          = float(os.environ.get("PROJECTZEO_GOAL_COMPLETE_THRESH", "0.90"))

# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

class ConditionStatus(str, Enum):
    PENDING     = "pending"
    SATISFIED   = "satisfied"
    FAILED      = "failed"
    UNKNOWN     = "unknown"


@dataclass
class ObservableCondition:
    
    condition_id:    str
    description:     str                     # Human-readable predicate
    screen_hint:     str                     # What to look for on screen
    weight:          float = 1.0             # Relative importance (1.0 = normal)
    status:          ConditionStatus = ConditionStatus.PENDING
    confidence:      float = 0.0            # 0.0-1.0 confidence of satisfaction
    satisfied_at:    Optional[float] = None
    eval_count:      int = 0
    last_eval_ts:    float = field(default_factory=time.time)
    failure_reason:  str = ""
    order_index:     int = 0                 # Expected satisfaction order

    @property
    def is_satisfied(self) -> bool:
        return (
            self.status == ConditionStatus.SATISFIED
            and self.confidence >= _MIN_CONFIDENCE_TO_COUNT
        )

    @property
    def effective_weight(self) -> float:
        """Weight adjusted by confidence when satisfied."""
        if self.is_satisfied:
            return self.weight * min(self.confidence, 1.0)
        return 0.0

    def mark_satisfied(self, confidence: float = 1.0) -> None:
        self.status = ConditionStatus.SATISFIED
        self.confidence = max(0.0, min(1.0, confidence))
        self.satisfied_at = time.time()
        self.eval_count += 1

    def mark_failed(self, reason: str = "", confidence: float = 0.9) -> None:
        self.status = ConditionStatus.FAILED
        self.confidence = max(0.0, min(1.0, confidence))
        self.failure_reason = reason
        self.eval_count += 1

    def mark_unknown(self) -> None:
        self.status = ConditionStatus.UNKNOWN
        self.eval_count += 1

    def reset(self) -> None:
        """Reset to pending (for re-evaluation after state change)."""
        self.status = ConditionStatus.PENDING
        self.confidence = 0.0
        self.satisfied_at = None
        self.failure_reason = ""


# ─────────────────────────────────────────────────────────────────────────────
# LLM prompts
# ─────────────────────────────────────────────────────────────────────────────

_DECOMPOSE_SYSTEM = """\
You are a Goal Decomposition Engine for a GUI automation agent.

Your task: decompose a natural language objective into 3-8 independently
verifiable screen conditions that together prove the task is complete.

RULES:
1. Each condition must be OBSERVABLE on screen (visible text, UI state, element presence)
2. Conditions must be ordered by expected satisfaction sequence
3. Be specific: "To field contains john@example.com" not "email is filled"
4. Include UI hints: what element/text/state to look for
5. Conditions must be MUTUALLY EXCLUSIVE in their scope
6. DO NOT require actions — only describe observable end-states

OUTPUT FORMAT (JSON only, no markdown):
{
  "conditions": [
    {
      "id": "c1",
      "description": "<verifiable predicate>",
      "screen_hint": "<what to look for on screen>",
      "weight": <0.5-2.0, default 1.0>,
      "order": <1-based integer>
    }
  ]
}

EXAMPLE for "send an email to john@example.com about Q3 report":
{
  "conditions": [
    {"id": "c1", "description": "Email compose window is open", "screen_hint": "New Message window or compose pane visible", "weight": 0.5, "order": 1},
    {"id": "c2", "description": "To field contains john@example.com", "screen_hint": "To: field shows john@example.com", "weight": 1.5, "order": 2},
    {"id": "c3", "description": "Subject field contains quarterly report reference", "screen_hint": "Subject: field shows Q3 or quarterly report", "weight": 1.0, "order": 3},
    {"id": "c4", "description": "Message body is non-empty with relevant content", "screen_hint": "Body area shows text content", "weight": 1.0, "order": 4},
    {"id": "c5", "description": "Email has been sent successfully", "screen_hint": "Sent confirmation, outbox cleared, or sent folder updated", "weight": 2.0, "order": 5}
  ]
}
"""

_EVAL_SYSTEM = """\
You are a Goal Condition Evaluator for a GUI automation agent.

Given a screen observation and a condition to check, determine if the condition
is SATISFIED, NOT_SATISFIED, or UNKNOWN (cannot determine from current screen).

OUTPUT FORMAT (JSON only, no markdown):
{
  "status": "SATISFIED" | "NOT_SATISFIED" | "UNKNOWN",
  "confidence": <0.0-1.0>,
  "evidence": "<what you saw that led to this conclusion>"
}

Be strict: SATISFIED only if there is clear visual evidence. UNKNOWN if the
relevant UI area is not visible. NOT_SATISFIED if you can see the area but
condition is not met.
"""


# ─────────────────────────────────────────────────────────────────────────────
# GoalRepresentation
# ─────────────────────────────────────────────────────────────────────────────

class GoalRepresentation:
    

    def __init__(
        self,
        objective: str,
        llm_call: Callable,
        *,
        max_conditions: int = _DECOMPOSE_MAX_CONDITIONS,
    ) -> None:
        self._objective     = objective
        self._llm           = llm_call
        self._max_conditions = max_conditions
        self._conditions: List[ObservableCondition] = []
        self._lock          = threading.RLock()
        self._eval_count    = 0
        self._stall_count   = 0
        self._last_progress = 0.0
        self._progress_ts   = time.time()
        self._created_at    = time.time()
        self._complete_ts: Optional[float] = None

        # Decompose on init (blocking, with timeout)
        self._decompose()

        _logger.info(
            "[GoalRepr] Initialised for objective=%r — %d conditions",
            objective[:80], len(self._conditions)
        )

    # =========================================================================
    # Decomposition
    # =========================================================================

    def _decompose(self) -> None:
        """Use LLM to decompose objective into verifiable conditions."""
        import threading as _th

        result: Dict[str, Any] = {}
        exc_holder: List[Exception] = []

        def _call():
            try:
                messages = [
                    {"role": "system", "content": _DECOMPOSE_SYSTEM},
                    {"role": "user", "content": (
                        f"OBJECTIVE: {self._objective[:600]}\n\n"
                        f"Decompose into {self._max_conditions} or fewer verifiable conditions."
                    )},
                ]
                raw = self._llm(messages, objective=self._objective)
                result["raw"] = raw
            except Exception as exc:
                exc_holder.append(exc)

        t = _th.Thread(target=_call, daemon=True)
        t.start()
        t.join(timeout=_EVAL_TIMEOUT_SEC)

        if exc_holder:
            _logger.warning("[GoalRepr] LLM decompose error: %s — using fallback", exc_holder[0])
            self._conditions = self._fallback_conditions()
            return

        if not result:
            _logger.warning("[GoalRepr] LLM decompose timed out — using fallback")
            self._conditions = self._fallback_conditions()
            return

        self._conditions = self._parse_conditions(result.get("raw", ""))
        if not self._conditions:
            _logger.warning("[GoalRepr] LLM returned no conditions — using fallback")
            self._conditions = self._fallback_conditions()

    def _parse_conditions(self, raw: str) -> List[ObservableCondition]:
        """Parse LLM JSON output into ObservableCondition list."""
        if not raw:
            return []

        # Strip markdown code fences
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()

        # Find JSON object
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return []

        try:
            data = json.loads(match.group())
        except json.JSONDecodeError as exc:
            _logger.debug("[GoalRepr] JSON parse error: %s", exc)
            return []

        conditions_data = data.get("conditions", [])
        if not isinstance(conditions_data, list):
            return []

        conditions: List[ObservableCondition] = []
        for item in conditions_data[: self._max_conditions]:
            if not isinstance(item, dict):
                continue
            desc = str(item.get("description", "")).strip()
            if not desc:
                continue
            cond_id = str(item.get("id", f"c{len(conditions)+1}"))
            hint    = str(item.get("screen_hint", desc))
            weight  = float(item.get("weight", 1.0))
            order   = int(item.get("order", len(conditions) + 1))

            conditions.append(ObservableCondition(
                condition_id = cond_id,
                description  = desc,
                screen_hint  = hint,
                weight       = max(0.1, min(3.0, weight)),
                order_index  = order,
            ))

        # Sort by order
        conditions.sort(key=lambda c: c.order_index)
        return conditions

    def _fallback_conditions(self) -> List[ObservableCondition]:
        """Single root condition used when LLM decomposition fails."""
        return [
            ObservableCondition(
                condition_id = "c_root",
                description  = f"Task complete: {self._objective[:200]}",
                screen_hint  = "Any visual indication that the task has been completed",
                weight       = 1.0,
                order_index  = 1,
            )
        ]

    # =========================================================================
    # Evaluation
    # =========================================================================

    def evaluate_from_screen(self, observation: Dict[str, Any]) -> None:
        
        with self._lock:
            self._eval_count += 1
            pending = [
                c for c in self._conditions
                if not c.is_satisfied
            ]
            if not pending:
                return  # All done

        if not _LLM_EVAL_ENABLED:
            # Heuristic-only evaluation for speed
            self._heuristic_evaluate(observation)
        else:
            # LLM-backed evaluation (parallel, with timeout)
            self._llm_evaluate_parallel(pending, observation)

        # Stall detection
        with self._lock:
            new_progress = self._compute_progress()
            if abs(new_progress - self._last_progress) < 0.01:
                self._stall_count += 1
            else:
                self._stall_count = 0
                self._last_progress = new_progress
                self._progress_ts = time.time()

            if new_progress >= _COMPLETE_THRESHOLD and self._complete_ts is None:
                self._complete_ts = time.time()
                _logger.info(
                    "[GoalRepr] COMPLETE at eval=%d progress=%.2f",
                    self._eval_count, new_progress
                )

    def _heuristic_evaluate(self, observation: Dict[str, Any]) -> None:
        """
        Fast heuristic evaluation using entity text matching.
        Used when LLM eval is disabled or as fallback.
        """
        entities = observation.get("entities", [])
        screen_text = observation.get("screen_description", "") or ""
        text_visible = observation.get("text_visible", "") or ""
        all_text = (screen_text + " " + text_visible).lower()

        # Collect all visible labels
        visible_labels = set()
        for e in entities:
            label = str(e.get("text", "") or e.get("label", "")).lower()
            if label:
                visible_labels.add(label)

        with self._lock:
            for cond in self._conditions:
                if cond.is_satisfied:
                    continue
                hint_lower = cond.screen_hint.lower()
                # Simple keyword overlap
                hint_words = set(re.findall(r"\w+", hint_lower))
                hint_words -= {"the", "a", "an", "is", "are", "has", "have",
                               "been", "to", "in", "on", "at", "for", "or", "and"}
                if len(hint_words) < 2:
                    continue  # Too generic to heuristic-match

                matched = sum(1 for w in hint_words if w in all_text)
                if matched / max(len(hint_words), 1) >= 0.6:
                    cond.mark_satisfied(confidence=0.65)
                    _logger.debug(
                        "[GoalRepr] Heuristic satisfied: %s (%.0f%% keyword match)",
                        cond.condition_id, 100 * matched / len(hint_words)
                    )

    def _llm_evaluate_parallel(
        self,
        conditions: List[ObservableCondition],
        observation: Dict[str, Any],
    ) -> None:
        
        import threading as _th

        screen_summary = self._build_screen_summary(observation)
        # Take a locked snapshot of condition references before spawning threads
        with self._lock:
            conditions_snap = list(conditions)

        def _eval_one(cond: ObservableCondition) -> None:
            try:
                status, confidence, evidence = self._llm_eval_condition(
                    cond, screen_summary
                )
                # Hold lock for the entire mutation sequence — no torn reads
                with self._lock:
                    if status == "SATISFIED":
                        cond.mark_satisfied(confidence)
                        _logger.debug(
                            "[GoalRepr] LLM satisfied %s (conf=%.2f): %s",
                            cond.condition_id, confidence, evidence[:100]
                        )
                    elif status == "NOT_SATISFIED":
                        # Atomically: mark failed then immediately reset to pending
                        cond.mark_failed(evidence[:200], confidence)
                        cond.status = ConditionStatus.PENDING   # conditions can recover
                        cond.failure_reason = evidence[:200]
                    else:  # UNKNOWN
                        cond.mark_unknown()
            except Exception as exc:
                _logger.debug("[GoalRepr] Eval thread error for %s: %s", cond.condition_id, exc)

        threads = [
            _th.Thread(target=_eval_one, args=(c,), daemon=True)
            for c in conditions_snap
        ]
        for t in threads:
            t.start()
        # Wait with per-condition timeout
        deadline = time.time() + _EVAL_TIMEOUT_SEC
        for t in threads:
            remaining = max(0.1, deadline - time.time())
            t.join(timeout=remaining)

    def _llm_eval_condition(
        self,
        cond: ObservableCondition,
        screen_summary: str,
    ) -> Tuple[str, float, str]:
        """
        Call LLM to evaluate a single condition against screen state.
        Returns (status, confidence, evidence).
        """
        messages = [
            {"role": "system", "content": _EVAL_SYSTEM},
            {"role": "user", "content": (
                f"CONDITION TO CHECK:\n{cond.description}\n\n"
                f"SCREEN HINT: {cond.screen_hint}\n\n"
                f"CURRENT SCREEN STATE:\n{screen_summary}\n\n"
                "Is this condition SATISFIED, NOT_SATISFIED, or UNKNOWN?"
            )},
        ]
        raw = self._llm(messages, objective=f"eval:{cond.condition_id}")

        # Parse response
        cleaned = re.sub(r"```(?:json)?", "", raw or "").strip()
        match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
        if not match:
            return "UNKNOWN", 0.5, "Could not parse LLM response"

        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return "UNKNOWN", 0.5, "JSON parse error"

        status_raw = str(data.get("status", "UNKNOWN")).upper()
        if "SATISFIED" in status_raw and "NOT" not in status_raw:
            status = "SATISFIED"
        elif "NOT" in status_raw or "NOT_SATISFIED" in status_raw:
            status = "NOT_SATISFIED"
        else:
            status = "UNKNOWN"

        confidence = float(data.get("confidence", 0.7))
        confidence = max(0.0, min(1.0, confidence))
        evidence   = str(data.get("evidence", ""))

        return status, confidence, evidence

    def _build_screen_summary(self, observation: Dict[str, Any]) -> str:
        """Compact screen state description for LLM prompts."""
        parts: List[str] = []

        focused_app = observation.get("focused_app", "unknown")
        parts.append(f"App in focus: {focused_app}")

        desc = observation.get("screen_description") or observation.get("description", "")
        if desc:
            parts.append(f"Screen description: {str(desc)[:300]}")

        entities = observation.get("entities", [])
        if entities:
            entity_strs = []
            for e in entities[:20]:
                label = e.get("text") or e.get("label") or e.get("type", "?")
                etype = e.get("type", "")
                entity_strs.append(f"[{etype}] {label}")
            parts.append("Visible elements: " + "; ".join(entity_strs))

        text_visible = observation.get("text_visible", "")
        if text_visible:
            parts.append(f"Visible text: {str(text_visible)[:400]}")

        return "\n".join(parts) if parts else "(no screen data)"

    # =========================================================================
    # Progress & completion
    # =========================================================================

    def _compute_progress(self) -> float:
        
        if not self._conditions:
            return 0.0
        total_weight = sum(c.weight for c in self._conditions)
        if total_weight <= 0:
            return 0.0
        satisfied_weight = sum(c.effective_weight for c in self._conditions)
        return min(1.0, satisfied_weight / total_weight)

    @property
    def progress(self) -> float:
        """Thread-safe progress score 0.0–1.0."""
        with self._lock:
            return self._compute_progress()

    @property
    def is_complete(self) -> bool:
        """True when progress meets or exceeds completion threshold."""
        return self.progress >= _COMPLETE_THRESHOLD

    @property
    def progress_summary(self) -> str:
        """Human-readable summary of which conditions are satisfied."""
        with self._lock:
            total = len(self._conditions)
            satisfied = [c for c in self._conditions if c.is_satisfied]
            pending   = [c for c in self._conditions if not c.is_satisfied]

            pct = self._compute_progress()
            sat_descs = [f"✓ {c.description[:60]}" for c in satisfied[:3]]
            pend_descs = [f"○ {c.description[:60]}" for c in pending[:2]]

            parts = [f"Progress: {pct:.0%} ({len(satisfied)}/{total} conditions)"]
            parts.extend(sat_descs)
            parts.extend(pend_descs)
            if len(pending) > 2:
                parts.append(f"  ... and {len(pending)-2} more pending")

            if self._stall_count >= _STALL_THRESHOLD_EVALS:
                parts.append(f"⚠ STALL: progress unchanged for {self._stall_count} evals")

            return "\n".join(parts)

    def next_pending(self) -> Optional[ObservableCondition]:
        
        with self._lock:
            pending = [c for c in self._conditions if not c.is_satisfied]
            if not pending:
                return None
            return min(pending, key=lambda c: c.order_index)

    @property
    def stall_detected(self) -> bool:
        """True if progress has not improved for N evaluations."""
        with self._lock:
            return self._stall_count >= _STALL_THRESHOLD_EVALS

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self._created_at

    @property
    def time_to_complete(self) -> Optional[float]:
        """Seconds taken to complete (None if not complete)."""
        if self._complete_ts is None:
            return None
        return self._complete_ts - self._created_at

    # =========================================================================
    # Manual overrides (for safety / operator confirmation paths)
    # =========================================================================

    def force_satisfy(self, condition_id: str, confidence: float = 1.0) -> bool:
        """Manually mark a condition as satisfied (e.g., after human confirmation)."""
        with self._lock:
            for cond in self._conditions:
                if cond.condition_id == condition_id:
                    cond.mark_satisfied(confidence)
                    _logger.info("[GoalRepr] Force-satisfied: %s", condition_id)
                    return True
        return False

    def force_complete(self) -> None:
        """Mark all conditions as satisfied (goal achieved externally)."""
        with self._lock:
            for cond in self._conditions:
                if not cond.is_satisfied:
                    cond.mark_satisfied(1.0)
            self._complete_ts = time.time()
        _logger.info("[GoalRepr] Force-complete: all %d conditions satisfied.", len(self._conditions))

    def inject_condition(
        self,
        description: str,
        screen_hint: str = "",
        weight: float = 1.0,
    ) -> ObservableCondition:
        
        with self._lock:
            max_order = max((c.order_index for c in self._conditions), default=0)
            new_id = f"c_dyn_{len(self._conditions)+1}"
            cond = ObservableCondition(
                condition_id = new_id,
                description  = description,
                screen_hint  = screen_hint or description,
                weight       = max(0.1, min(3.0, weight)),
                order_index  = max_order + 1,
            )
            self._conditions.append(cond)
            _logger.info("[GoalRepr] Injected dynamic condition: %s", description[:80])
            return cond

    # =========================================================================
    # Serialisation / diagnostics
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Snapshot of current goal state for logging/telemetry."""
        with self._lock:
            return {
                "objective":    self._objective[:100],
                "progress":     self._compute_progress(),
                "is_complete":  self.is_complete,
                "eval_count":   self._eval_count,
                "stall_count":  self._stall_count,
                "conditions": [
                    {
                        "id":          c.condition_id,
                        "description": c.description[:80],
                        "status":      c.status.value,
                        "confidence":  round(c.confidence, 3),
                        "weight":      c.weight,
                    }
                    for c in self._conditions
                ],
            }

    def __repr__(self) -> str:
        return (
            f"<GoalRepresentation progress={self.progress:.0%} "
            f"conditions={len(self._conditions)} "
            f"complete={self.is_complete}>"
        )
