from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

_logger = logging.getLogger(__name__)

_MAX_REFINE_ROUNDS: int = 2
_ACCEPT_THRESHOLD: float = 0.85
_REFINE_TIMEOUT_S: float = 30.0

_CRITIQUE_SYSTEM = """You are a rigorous action critic for a GUI automation agent.
Your role:
1. Evaluate the proposed action for correctness, safety, and goal alignment.
2. Rate your confidence that this action is optimal (0.0 = terrible, 1.0 = perfect).
3. If confidence < 0.85, list specific objections and output a refined action.

Always respond with ONLY a JSON object:
{
  "confidence": 0.0-1.0,
  "objections": ["..."],   // empty list if no objections
  "refined_action": {      // same schema as input action; IDENTICAL if no changes
    "operation": "...",
    "thought": "...",
    ... other fields ...
  }
}
No markdown. No explanations outside the JSON."""

@dataclass
class RefineResult:
    action: Dict[str, Any]
    original_action: Dict[str, Any]
    rounds_applied: int = 0
    final_confidence: float = 0.0
    objections: List[str] = field(default_factory=list)
    refined: bool = False

class SelfRefineEngine:

    def __init__(
        self,
        llm_callable: Callable,
        *,
        max_rounds: int = _MAX_REFINE_ROUNDS,
        accept_threshold: float = _ACCEPT_THRESHOLD,
        timeout_s: float = _REFINE_TIMEOUT_S,
    ) -> None:
        self._llm = llm_callable
        self._max_rounds = max_rounds
        self._accept_threshold = accept_threshold
        self._timeout_s = timeout_s
        self._lock = threading.Lock()

        self._total_calls: int = 0
        self._total_refinements: int = 0
        self._rounds_histogram: Dict[int, int] = {}

    def refine(
        self,
        action: Dict[str, Any],
        *,
        objective: str = "",
        context: str = "",
        world_state: Optional[Dict[str, Any]] = None,
    ) -> RefineResult:
        with self._lock:
            self._total_calls += 1

        original = dict(action)
        current = dict(action)
        rounds_applied = 0
        final_confidence = 0.0
        all_objections: List[str] = []

        for _round in range(self._max_rounds):
            result_holder: List[Optional[str]] = [None]

            def _call(_cur=current, _rh=result_holder):
                try:
                    action_json = json.dumps(
                        {k: v for k, v in _cur.items() if k != "thought"},
                        indent=2,
                    )
                    thought = _cur.get("thought", "")[:300]
                    world_ctx = ""
                    if world_state:
                        world_ctx = (
                            f"Focused app: {world_state.get('focused_app', '?')}\n"
                            f"Entities: {len(world_state.get('entities', []))} visible\n"
                        )
                    prompt = (
                        f"OBJECTIVE: {objective[:300]}\n\n"
                        f"{('WORLD STATE:\n' + world_ctx) if world_ctx else ''}"
                        f"{('CONTEXT:\n' + context[:400] + chr(10)) if context else ''}"
                        f"AGENT THOUGHT: {thought}\n\n"
                        f"PROPOSED ACTION:\n{action_json}\n\n"
                        "Critique this action. Is it correct, safe, and optimal?\n"
                        "Respond with JSON only (see system prompt format)."
                    )
                    raw = self._llm(
                        messages=[
                            {"role": "system", "content": _CRITIQUE_SYSTEM},
                            {"role": "user", "content": prompt},
                        ],
                        objective=objective,
                        session_id="self_refine_critique",
                    )
                    if isinstance(raw, list) and raw:
                        text = str(
                            raw[0].get("content", "") if isinstance(raw[0], dict) else raw[0]
                        )
                    elif isinstance(raw, str):
                        text = raw
                    else:
                        text = ""
                    _rh[0] = text
                except Exception as exc:
                    _logger.debug("[SelfRefine] LLM call failed (round %d): %s", _round + 1, exc)

            thread = threading.Thread(target=_call, daemon=True)
            thread.start()
            thread.join(timeout=self._timeout_s)

            raw_text = result_holder[0]
            if not raw_text:
                break

            critique = self._parse_critique(raw_text)
            if critique is None:
                break

            confidence = float(critique.get("confidence", 0.0))
            objections: List[str] = [
                str(o) for o in critique.get("objections", [])
            ]
            refined_action: Optional[Dict[str, Any]] = critique.get("refined_action")

            final_confidence = confidence
            all_objections.extend(objections)
            rounds_applied += 1

            if confidence >= self._accept_threshold or not objections:
                _logger.debug(
                    "[SelfRefine] Round %d: accepted (confidence=%.2f)",
                    _round + 1, confidence,
                )
                break

            if refined_action and isinstance(refined_action, dict):
                op = refined_action.get("operation")
                if op and op != current.get("operation", ""):
                    _logger.info(
                        "[SelfRefine] Round %d: refined %s→%s (confidence=%.2f, objections=%d)",
                        _round + 1, current.get("operation"), op, confidence, len(objections),
                    )
                elif op:
                    _logger.debug(
                        "[SelfRefine] Round %d: content refined (same op=%s, confidence=%.2f)",
                        _round + 1, op, confidence,
                    )
                if not refined_action.get("thought") and current.get("thought"):
                    refined_action["thought"] = current["thought"] + " [self-refined]"
                current = refined_action

        changed = current != original
        if changed:
            with self._lock:
                self._total_refinements += 1
        with self._lock:
            self._rounds_histogram[rounds_applied] = (
                self._rounds_histogram.get(rounds_applied, 0) + 1
            )

        _logger.debug(
            "[SelfRefine] Done: rounds=%d confidence=%.2f refined=%s objections=%d",
            rounds_applied, final_confidence, changed, len(all_objections),
        )

        return RefineResult(
            action=current,
            original_action=original,
            rounds_applied=rounds_applied,
            final_confidence=final_confidence,
            objections=all_objections,
            refined=changed,
        )

    def _parse_critique(self, raw: str) -> Optional[Dict[str, Any]]:
        try:
            clean = re.sub(r"```(?:json)?", "", raw).strip()
            m = re.search(r"\{.*\}", clean, re.DOTALL)
            if not m:
                return None
            parsed = json.loads(m.group(0))
            if not isinstance(parsed, dict):
                return None
            return parsed
        except (json.JSONDecodeError, Exception):
            return None

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_calls": self._total_calls,
                "total_refinements": self._total_refinements,
                "refinement_rate": (
                    round(self._total_refinements / max(self._total_calls, 1), 4)
                ),
                "rounds_histogram": dict(self._rounds_histogram),
            }

_global_engine: Optional[SelfRefineEngine] = None
_engine_lock = threading.Lock()

def get_global_self_refine_engine(
    llm_caller: Optional[Callable] = None,
) -> Optional[SelfRefineEngine]:
    global _global_engine
    with _engine_lock:
        if _global_engine is None and llm_caller is not None:
            _global_engine = SelfRefineEngine(llm_callable=llm_caller)
        return _global_engine
