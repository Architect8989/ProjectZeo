from __future__ import annotations

import logging
import os
import sys
import time
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)


class GIIMode:
    DISABLED = 0
    BASIC = 1
    FULL = 2


def get_gii_mode() -> int:
    """
    AUDIT-CRITICAL-1 FIX: Default changed from BASIC (1) to FULL (2).
    ConsequenceReasoner is the most important safety component. It must not be opt-in.
    """
    try:
        return int(os.environ.get("PROJECTZEO_GII_MODE", str(GIIMode.FULL)))
    except (ValueError, TypeError):
        return GIIMode.FULL


def _print_startup_safety_banner(gii_mode: int, consequence_active: bool) -> None:
    """AUDIT-REQUIRED: Print safety mode announcement at startup."""
    mode_names = {
        GIIMode.DISABLED: "DISABLED (scripted only, no GII)",
        GIIMode.BASIC:    "BASIC (per-step reasoning, consequence reasoning INACTIVE)",
        GIIMode.FULL:     "FULL (per-step reasoning + consequence reasoning ACTIVE)",
    }
    mode_label = mode_names.get(gii_mode, f"UNKNOWN ({gii_mode})")
    print(f"""
[SAFETY] GII_MODE: {mode_label}
[SAFETY] Consequence Reasoning (Tier2+3): {'ACTIVE' if consequence_active else 'INACTIVE — set PROJECTZEO_GII_MODE=2'}
[SAFETY] Restoration scope: cursor + window focus ONLY
[SAFETY] NOTE: Restoration does NOT preserve browser tabs, clipboard, unsaved docs, or terminal session.""",
          file=sys.stderr)
    if not consequence_active:
        print(
            "[SAFETY CRITICAL] Consequence reasoning is INACTIVE. "
            "Set PROJECTZEO_GII_MODE=2 for production use.",
            file=sys.stderr,
        )


class GIIController:

    def __init__(
        self,
        *,
        llm_callable: Callable,
        objective: str,
        scaffold_steps: Optional[List[Any]] = None,
        gii_mode: int = GIIMode.FULL,
        memory_dir: Optional[str] = None,
    ) -> None:
        self._llm = llm_callable
        self._objective = objective
        self._scaffold_steps = scaffold_steps or []
        self._gii_mode = gii_mode
        self._enabled = gii_mode > GIIMode.DISABLED
        self._per_step_reasoner = None
        self._consequence_reasoner = None
        self._semantic_memory = None
        self._application_memory = None
        self._task_start: float = time.time()
        self._lock = threading.Lock()
        # AUDIT-MEDIUM FIX: Track denied action keys to block plan-fallback too
        self._denied_action_keys: set = set()

        if self._enabled:
            self._initialise_components(memory_dir)

        consequence_active = self._consequence_reasoner is not None
        _print_startup_safety_banner(gii_mode, consequence_active)
        _logger.info(
            "[GIIController] Initialised. mode=%d enabled=%s consequence_active=%s",
            gii_mode, self._enabled, consequence_active,
        )

    @classmethod
    def create(
        cls,
        *,
        llm_callable: Callable,
        objective: str,
        scaffold_steps: Optional[List[Any]] = None,
        memory_dir: Optional[str] = None,
    ) -> "GIIController":
        gii_mode = get_gii_mode()
        return cls(
            llm_callable=llm_callable,
            objective=objective,
            scaffold_steps=scaffold_steps,
            gii_mode=gii_mode,
            memory_dir=memory_dir,
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def gii_mode(self) -> int:
        return self._gii_mode

    @property
    def consequence_reasoner(self):
        """Expose for wiring into operate.py main execution loop."""
        return self._consequence_reasoner

    def _initialise_components(self, memory_dir: Optional[str]) -> None:
        try:
            from core.memory.semantic_memory import SemanticMemory
            self._semantic_memory = SemanticMemory(memory_dir=memory_dir)
        except Exception as exc:
            _logger.warning("[GIIController] SemanticMemory init failed: %s", exc)

        try:
            from core.memory.application_memory import ApplicationMemory
            self._application_memory = ApplicationMemory(memory_dir=memory_dir)
        except Exception as exc:
            _logger.warning("[GIIController] ApplicationMemory init failed: %s", exc)

        # AUDIT-CRITICAL-1 FIX: ConsequenceReasoner now active for BOTH BASIC and FULL.
        # Tier2 (goal coherence) always active. Tier3 only in FULL mode.
        try:
            from core.safety.consequence_reasoner import ConsequenceReasoner
            enable_tier3 = self._gii_mode >= GIIMode.FULL
            self._consequence_reasoner = ConsequenceReasoner(
                llm_callable=self._llm,
                enable_tier2=True,
                enable_tier3=enable_tier3,
            )
            _logger.info(
                "[GIIController] ConsequenceReasoner active. tier2=True tier3=%s",
                enable_tier3,
            )
        except Exception as exc:
            _logger.warning("[GIIController] ConsequenceReasoner init failed: %s", exc)

        try:
            from core.cognition.per_step_reasoner import PerStepReasoner
            scaffold_dicts = []
            for step in self._scaffold_steps:
                if isinstance(step, dict):
                    scaffold_dicts.append(step)
                elif hasattr(step, "description"):
                    scaffold_dicts.append({
                        "description": getattr(step, "description", ""),
                        "type": str(getattr(step, "type", "")),
                    })
            self._per_step_reasoner = PerStepReasoner(
                llm_callable=self._llm,
                objective=self._objective,
                scaffold_steps=scaffold_dicts,
                application_memory=self._application_memory,
                semantic_memory=self._semantic_memory,
                consequence_reasoner=self._consequence_reasoner,
            )
        except Exception as exc:
            _logger.error("[GIIController] PerStepReasoner init failed: %s", exc)
            self._enabled = False

    def decide_next_action(
        self,
        world_state: Dict[str, Any],
        *,
        perception: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        if not self._enabled or self._per_step_reasoner is None:
            return None, "GII disabled"
        try:
            return self._per_step_reasoner.next_action(
                world_state, perception=perception
            )
        except Exception as exc:
            _logger.error("[GIIController] decide_next_action error: %s", exc)
            return None, f"GII reasoning error: {exc}"

    def is_action_denied(self, action_key: str) -> bool:
        """AUDIT-MEDIUM FIX: Check if action key was previously denied."""
        return action_key in self._denied_action_keys

    def record_denial(self, action_key: str) -> None:
        """Record denied action key to block plan-step fallback."""
        with self._lock:
            self._denied_action_keys.add(action_key)

    def record_outcome(
        self,
        action: Dict[str, Any],
        *,
        success: bool,
        output: str = "",
    ) -> None:
        if self._per_step_reasoner is not None:
            try:
                self._per_step_reasoner.record_outcome(
                    action, success=success, output=output
                )
            except Exception:
                pass

    def get_planning_context(self, focused_app: Optional[str] = None) -> str:
        if not self._enabled:
            return ""
        parts = []
        if self._application_memory and focused_app:
            try:
                app_context = self._application_memory.format_profile_for_prompt(focused_app)
                if app_context:
                    parts.append(app_context)
            except Exception:
                pass
        if self._semantic_memory:
            try:
                facts = self._semantic_memory.query(self._objective, max_results=8)
                sem_context = self._semantic_memory.format_for_prompt(facts)
                if sem_context:
                    parts.append(sem_context)
            except Exception:
                pass
        return "\n\n".join(parts)

    def on_task_complete(
        self,
        *,
        success: bool,
        focused_app: Optional[str] = None,
        execution_log: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._enabled:
            return
        if self._application_memory and focused_app:
            try:
                self._application_memory.increment_task_count(focused_app)
            except Exception:
                pass
        if self._semantic_memory and execution_log:
            try:
                self._extract_semantic_facts_from_log(execution_log, focused_app)
            except Exception as exc:
                _logger.debug("[GIIController] Regex fact extraction error: %s", exc)
            # TRANSFORMATION: LLM-synthesized episodic lessons
            try:
                self._synthesize_lessons_with_llm(execution_log, focused_app)
            except Exception as exc:
                _logger.debug("[GIIController] LLM lesson synthesis error: %s", exc)
        if self._semantic_memory:
            try:
                self._semantic_memory.save()
            except Exception:
                pass
        if self._application_memory:
            try:
                self._application_memory.save()
            except Exception:
                pass
        _logger.info(
            "[GIIController] Task complete. success=%s mode=%d",
            success, self._gii_mode,
        )

    def _extract_semantic_facts_from_log(
        self,
        execution_log: Dict[str, Any],
        focused_app: Optional[str],
    ) -> None:
        """Regex-based fact extraction — fast fallback for CPU deployments."""
        import re as _re
        install_success_re = _re.compile(
            r"(?:successfully installed|installation complete|already installed)", _re.IGNORECASE
        )
        version_re = _re.compile(r"(\w[\w\-]*)\s+(?:version\s+)?v?(\d+\.\d[\d.]*)", _re.IGNORECASE)
        error_re = _re.compile(r"(?:error|failed|not found|no such file)[\s:]+(.{10,120})", _re.IGNORECASE)

        for step_idx, step_data in execution_log.items():
            if not isinstance(step_data, dict):
                continue
            for output_entry in step_data.get("outputs", []):
                output_text = str(output_entry.get("output", ""))[:2000]
                if not output_text:
                    continue
                if install_success_re.search(output_text) and focused_app:
                    self._semantic_memory.store(
                        subject=focused_app, predicate="install_outcome",
                        object_=f"success (step {step_idx})", category="install_outcomes",
                        confidence=0.9, source="observed",
                    )
                for m in version_re.finditer(output_text):
                    tool = m.group(1).lower()
                    version = m.group(2)
                    if len(tool) > 2 and len(version) > 1:
                        self._semantic_memory.store(
                            subject=tool, predicate="version", object_=version,
                            category="application_facts", confidence=0.95, source="observed",
                        )
                if not output_entry.get("success", True):
                    for m in error_re.finditer(output_text):
                        error_snippet = m.group(1).strip()[:100]
                        if focused_app:
                            self._semantic_memory.store(
                                subject=focused_app, predicate="known_error",
                                object_=error_snippet, category="error_solutions",
                                confidence=0.6, source="observed",
                            )

    def _synthesize_lessons_with_llm(
        self,
        execution_log: Dict[str, Any],
        focused_app: Optional[str],
    ) -> None:
        """
        TRANSFORMATION STEP 5: LLM-synthesized lessons from execution log.
        Replaces regex extraction with genuine semantic understanding.
        Runs post-task with generous timeout (non-blocking on main execution).
        """
        if self._llm is None or self._semantic_memory is None:
            return

        summary_parts = []
        for step_idx, step_data in execution_log.items():
            if not isinstance(step_data, dict):
                continue
            for output_entry in step_data.get("outputs", []):
                op = str(output_entry.get("operation", "unknown"))
                success = output_entry.get("success", True)
                output_snippet = str(output_entry.get("output", ""))[:200]
                status = "✓" if success else "✗"
                summary_parts.append(f"  Step {step_idx}: {status} {op}: {output_snippet}")

        if not summary_parts:
            return

        execution_summary = "\n".join(summary_parts[:30])
        synthesis_prompt = (
            f"You are an AI agent that just completed a task. Extract reusable lessons.\n\n"
            f"TASK: {self._objective[:500]}\n"
            f"APP: {focused_app or 'unknown'}\n\n"
            f"EXECUTION:\n{execution_summary}\n\n"
            f"Extract 3-5 specific, reusable lessons. Focus on: what worked that wasn't obvious, "
            f"failures and resolutions, application quirks, shortcuts found.\n"
            f"Respond ONLY with a JSON array:\n"
            f'[{{"subject":"app","predicate":"lesson_type","object":"lesson text","confidence":0.8}}]'
        )

        result_holder: List[Optional[str]] = [None]

        def _call():
            try:
                raw = self._llm(
                    messages=[{"role": "user", "content": synthesis_prompt}],
                    objective=None,
                    session_id="lesson_synthesis",
                )
                if isinstance(raw, list) and raw:
                    result_holder[0] = str(raw[0].get("content", "") if isinstance(raw[0], dict) else raw[0])
                elif isinstance(raw, str):
                    result_holder[0] = raw
            except Exception as exc:
                _logger.debug("[GIIController] Lesson LLM call failed: %s", exc)

        thread = threading.Thread(target=_call, daemon=True)
        thread.start()
        thread.join(timeout=180.0)  # Generous — this runs after task completion

        if result_holder[0] is None:
            return

        try:
            import re as _re
            import json as _json
            clean = _re.sub(r"```(?:json)?", "", result_holder[0]).strip()
            match = _re.search(r"\[.*\]", clean, _re.DOTALL)
            if not match:
                return
            lessons = _json.loads(match.group(0))
            if not isinstance(lessons, list):
                return
            for lesson in lessons[:5]:
                if not isinstance(lesson, dict):
                    continue
                subject = str(lesson.get("subject", focused_app or "task"))[:80]
                predicate = str(lesson.get("predicate", "lesson"))[:80]
                object_ = str(lesson.get("object", ""))[:300]
                confidence = float(lesson.get("confidence", 0.7))
                if object_ and 0.0 < confidence <= 1.0:
                    self._semantic_memory.store(
                        subject=subject, predicate=predicate, object_=object_,
                        category="llm_synthesized_lessons",
                        confidence=confidence, source="llm_synthesis",
                    )
                    _logger.info(
                        "[GIIController] LLM lesson: %s::%s = %s (%.2f)",
                        subject, predicate, object_[:60], confidence,
                    )
        except Exception as parse_err:
            _logger.debug("[GIIController] Lesson parse failed: %s", parse_err)

    def get_stats(self) -> dict:
        stats = {
            "gii_mode": self._gii_mode,
            "enabled": self._enabled,
            "task_duration_seconds": round(time.time() - self._task_start, 1),
            "denied_action_keys": len(self._denied_action_keys),
        }
        if self._per_step_reasoner:
            stats["per_step_reasoner"] = self._per_step_reasoner.get_stats()
        if self._consequence_reasoner:
            stats["consequence_reasoner"] = self._consequence_reasoner.get_stats()
        if self._semantic_memory:
            stats["semantic_memory"] = self._semantic_memory.stats()
        if self._application_memory:
            stats["application_memory"] = self._application_memory.stats()
        return stats
