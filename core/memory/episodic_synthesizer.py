from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional

_logger = logging.getLogger(__name__)

_SYNTHESIS_SYSTEM_PROMPT = """\
You are an AI agent that just completed a computer automation task.
Your job is to extract structured, reusable lessons from this execution history.

FOCUS ON:
1. What worked that wasn't obvious (clever shortcuts, unexpected UI behaviors)
2. Failures and exactly how they were recovered from
3. Application-specific quirks discovered (menu locations, keyboard shortcuts, dialog patterns)
4. Performance insights (what made some steps fast vs slow)
5. Error patterns and their solutions

OUTPUT FORMAT — respond ONLY with a valid JSON array, no markdown, no preamble:
[
  {
    "subject": "application_or_domain_name",
    "predicate": "lesson_category",
    "object": "specific actionable lesson text (max 200 chars)",
    "confidence": 0.7,
    "tags": ["tag1", "tag2"]
  }
]

LESSON CATEGORIES (use these for predicate field):
  - "keyboard_shortcut" — discovered shortcut that works
  - "menu_path" — menu navigation sequence that works
  - "dialog_pattern" — how a specific dialog type behaves
  - "install_method" — successful installation approach
  - "error_solution" — specific error and how it was resolved
  - "workflow" — multi-step sequence that works well
  - "quirk" — unexpected application behavior to know about
  - "avoid" — something that failed and should not be attempted
  - "performance" — what made execution faster or slower

Extract 3-7 lessons. Only extract lessons you are CONFIDENT about from the execution log.
Do not invent lessons. If the log shows nothing interesting, return an empty array: []
"""


class EpisodicSynthesizer:
    

    # Maximum characters of execution log to send to LLM
    MAX_LOG_CHARS = 8000
    # Minimum confidence for a lesson to be stored
    MIN_CONFIDENCE = 0.5
    # Maximum lessons to store per synthesis
    MAX_LESSONS = 10

    def __init__(
        self,
        llm_callable: Callable,
        *,
        timeout_seconds: float = 180.0,
    ) -> None:
        self._llm = llm_callable
        self._timeout = timeout_seconds
        self._total_syntheses = 0
        self._total_lessons_stored = 0
        self._lock = threading.Lock()

    def synthesize_and_store(
        self,
        *,
        execution_log: Dict[str, Any],
        objective: str,
        semantic_memory,
        focused_app: Optional[str] = None,
        application_memory=None,
        block: bool = False,
    ) -> None:
        
        if block:
            self._synthesize(execution_log, objective, semantic_memory, focused_app, application_memory)
        else:
            thread = threading.Thread(
                target=self._synthesize,
                args=(execution_log, objective, semantic_memory, focused_app, application_memory),
                daemon=True,
                name="episodic_synthesis",
            )
            thread.start()

    def _synthesize(
        self,
        execution_log: Dict[str, Any],
        objective: str,
        semantic_memory,
        focused_app: Optional[str],
        application_memory,
    ) -> None:
        t0 = time.monotonic()
        try:
            log_text = self._format_execution_log(execution_log)
            if not log_text.strip():
                _logger.debug("[EpisodicSynthesizer] Empty execution log — no lessons to extract.")
                return

            prompt = (
                f"TASK OBJECTIVE: {objective[:400]}\n"
                f"FOCUSED APPLICATION: {focused_app or 'multiple/unknown'}\n\n"
                f"EXECUTION LOG:\n{log_text}\n"
            )

            raw_response = self._call_llm(prompt)
            if raw_response is None:
                _logger.debug("[EpisodicSynthesizer] LLM returned no response.")
                return

            lessons = self._parse_lessons(raw_response)
            if not lessons:
                _logger.debug("[EpisodicSynthesizer] No valid lessons parsed.")
                return

            stored_count = 0
            for lesson in lessons[:self.MAX_LESSONS]:
                if not isinstance(lesson, dict):
                    continue
                subject = str(lesson.get("subject", focused_app or "task"))[:80]
                predicate = str(lesson.get("predicate", "lesson"))[:80]
                object_ = str(lesson.get("object", ""))[:300]
                confidence = float(lesson.get("confidence", 0.7))
                tags = lesson.get("tags", [])
                if not object_ or confidence < self.MIN_CONFIDENCE:
                    continue

                # Store in semantic memory
                try:
                    semantic_memory.store(
                        subject=subject,
                        predicate=predicate,
                        object_=object_,
                        category="episodic_lessons",
                        confidence=confidence,
                        source="episodic_synthesis",
                    )
                    stored_count += 1
                    _logger.info(
                        "[EpisodicSynthesizer] Stored: %s::%s = %s (conf=%.2f)",
                        subject, predicate, object_[:80], confidence,
                    )
                except Exception as store_exc:
                    _logger.warning("[EpisodicSynthesizer] Store failed: %s", store_exc)

                # Also store app-specific lessons in ApplicationMemory
                if application_memory and subject == focused_app:
                    try:
                        if predicate == "keyboard_shortcut":
                            application_memory.add_shortcut(focused_app, object_)
                        elif predicate == "menu_path":
                            application_memory.add_menu_path(focused_app, object_)
                        elif predicate in ("quirk", "avoid"):
                            application_memory.add_quirk(focused_app, object_)
                        elif predicate == "error_solution":
                            application_memory.add_error_solution(focused_app, object_)
                    except Exception:
                        pass  # ApplicationMemory integration is best-effort

            with self._lock:
                self._total_syntheses += 1
                self._total_lessons_stored += stored_count

            elapsed = (time.monotonic() - t0) * 1000
            _logger.info(
                "[EpisodicSynthesizer] Synthesis complete. %d lessons stored in %.1fms",
                stored_count, elapsed,
            )

        except Exception as exc:
            _logger.error("[EpisodicSynthesizer] Synthesis failed: %s", exc)

    def _format_execution_log(self, execution_log: Dict[str, Any]) -> str:
        """Format execution log into compact LLM-readable text."""
        lines = []
        for step_idx in sorted(execution_log.keys(), key=lambda k: int(k) if str(k).isdigit() else 0):
            step_data = execution_log.get(step_idx, {})
            if not isinstance(step_data, dict):
                continue
            for output_entry in step_data.get("outputs", []):
                if not isinstance(output_entry, dict):
                    continue
                op = str(output_entry.get("operation", "unknown"))
                success = output_entry.get("success", True)
                output = str(output_entry.get("output", ""))[:300]
                status = "✓" if success else "✗"
                lines.append(f"Step {step_idx}: {status} [{op}] {output}")
            # Also capture step description if present
            desc = str(step_data.get("description", ""))
            if desc:
                lines.append(f"Step {step_idx} desc: {desc[:100]}")

        full = "\n".join(lines)
        if len(full) > self.MAX_LOG_CHARS:
            # Keep beginning and end (most informative parts)
            half = self.MAX_LOG_CHARS // 2
            full = full[:half] + "\n... [truncated] ...\n" + full[-half:]
        return full

    def _call_llm(self, user_prompt: str) -> Optional[str]:
        """Call LLM with synthesis prompt. Returns raw text response or None."""
        result_holder: List[Optional[str]] = [None]

        def _call():
            try:
                raw = self._llm(
                    messages=[
                        {"role": "system", "content": _SYNTHESIS_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    objective=None,
                    session_id="episodic_synthesis",
                )
                if isinstance(raw, list) and raw:
                    result_holder[0] = str(
                        raw[0].get("content", "") if isinstance(raw[0], dict) else raw[0]
                    )
                elif isinstance(raw, str):
                    result_holder[0] = raw
            except Exception as exc:
                _logger.warning("[EpisodicSynthesizer] LLM call failed: %s", exc)

        thread = threading.Thread(target=_call, daemon=True)
        thread.start()
        thread.join(timeout=self._timeout)

        if thread.is_alive():
            _logger.warning("[EpisodicSynthesizer] LLM call timed out after %.1fs", self._timeout)
            return None

        return result_holder[0]

    def _parse_lessons(self, raw_text: str) -> List[Dict[str, Any]]:
        """Parse LLM response into list of lesson dicts."""
        try:
            # Strip markdown fences
            clean = re.sub(r"```(?:json)?", "", raw_text).strip()
            # Find JSON array
            match = re.search(r"\[.*\]", clean, re.DOTALL)
            if not match:
                return []
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, AttributeError) as exc:
            _logger.debug("[EpisodicSynthesizer] Parse error: %s", exc)
        return []

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "total_syntheses": self._total_syntheses,
                "total_lessons_stored": self._total_lessons_stored,
                "timeout_seconds": self._timeout,
            }
