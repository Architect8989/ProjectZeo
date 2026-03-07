from __future__ import annotations

import logging
import os
import sys
import time
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)


_EPISODIC_CHECKPOINT_INTERVAL: int = max(
    1,
    int(os.environ.get("PROJECTZEO_EPISODIC_CHECKPOINT_INTERVAL", "50") or "50"),
)


_USE_MILESTONES: bool = (
    os.environ.get("PROJECTZEO_USE_MILESTONES", "0").strip() == "1"
)


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
        # MEM-2 FIX: Mem0Store as primary cross-session memory.
        # BeliefState (in operate.py) is preserved as the per-task working memory.
        # Mem0Store adds persistent memory that survives across task sessions.
        self._mem0_store = None
        # MEM-1 FIX: CogneeStore as primary knowledge-graph memory.
        # SemanticMemory is preserved as the regex-extraction fallback.
        self._cognee_store = None
        self._task_start: float = time.time()
        self._lock = threading.Lock()
        # AUDIT-MEDIUM FIX: Track denied action keys to block plan-fallback too
        self._denied_action_keys: set = set()

        # ── AUDIT MEDIUM-1 FIX: EpisodicSynthesizer for mid-task checkpoints ──
        # Initialised in _initialise_components(); used in record_outcome().
        self._episodic_synthesizer = None
        self._outcome_call_count: int = 0        # counts record_outcome() calls
        self._last_checkpoint_call: int = 0      # call# of last checkpoint synthesis

        # ── ARCH-1 FIX: MilestoneDecomposer (opt-in PROJECTZEO_USE_MILESTONES=1) ──
        self._milestone_decomposer = None
        self._milestones: list = []
        self._current_milestone_idx: int = 0
        self._milestones_active: bool = False

        if self._enabled:
            self._initialise_components(memory_dir)

        consequence_active = self._consequence_reasoner is not None
        _print_startup_safety_banner(gii_mode, consequence_active)
        _logger.info(
            "[GIIController] Initialised. mode=%d enabled=%s consequence_active=%s "
            "milestones_active=%s checkpoint_interval=%d",
            gii_mode, self._enabled, consequence_active,
            self._milestones_active, _EPISODIC_CHECKPOINT_INTERVAL,
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
        # ----------------------------------------------------------------
        # Layer 1: SemanticMemory — always initialised first as it is the
        # universal fallback.  Cognee/Mem0 layer is layered on top.
        # ----------------------------------------------------------------
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

        
        try:
            from core.memory.mem0_store import Mem0Store
            self._mem0_store = Mem0Store.get_instance()
            _logger.info(
                "[GIIController] Mem0Store initialised. "
                "available=%s cross-session memory active.",
                self._mem0_store._available,
            )
        except Exception as exc:
            _logger.warning(
                "[GIIController] Mem0Store init failed (non-fatal): %s. "
                "Cross-session memory will be unavailable this session.", exc
            )

        
        try:
            from core.memory.cognee_store import CogneeStore
            self._cognee_store = CogneeStore.get_instance()
            _logger.info(
                "[GIIController] CogneeStore initialised. "
                "available=%s knowledge-graph memory active.",
                self._cognee_store._available,
            )
        except Exception as exc:
            _logger.warning(
                "[GIIController] CogneeStore init failed (non-fatal): %s. "
                "SemanticMemory will be used as knowledge-graph fallback.", exc
            )

        
        try:
            from core.safety.consequence_reasoner import ConsequenceReasoner
            enable_tier3 = self._gii_mode >= GIIMode.FULL
            self._consequence_reasoner = ConsequenceReasoner(
                llm_callable=self._llm,
                enable_tier2=True,
                enable_tier3=enable_tier3,
                auto_wire_endpoints=True,   # AUDIT FIX: tiered model routing
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

        
        try:
            from core.memory.episodic_synthesizer import EpisodicSynthesizer
            self._episodic_synthesizer = EpisodicSynthesizer(
                llm_callable=self._llm,
            )
            _logger.info(
                "[GIIController] EpisodicSynthesizer initialised. "
                "Mid-task checkpoints every %d iterations.",
                _EPISODIC_CHECKPOINT_INTERVAL,
            )
        except Exception as exc:
            _logger.warning(
                "[GIIController] EpisodicSynthesizer init failed (non-fatal): %s. "
                "Post-task lesson synthesis will be unavailable.", exc,
            )

        
        if _USE_MILESTONES:
            try:
                from core.planner.milestone_decomposer import MilestoneDecomposer
                self._milestone_decomposer = MilestoneDecomposer(
                    llm_callable=self._llm,
                )
                self._milestones = self._milestone_decomposer.decompose(
                    objective=self._objective,
                )
                if self._milestones:
                    self._milestones_active = True
                    # Inject first milestone as the immediate sub-objective
                    self._inject_current_milestone()
                    _logger.info(
                        "[GIIController] MilestoneDecomposer active. "
                        "%d milestones: %s",
                        len(self._milestones),
                        [getattr(m, "name", str(m)) for m in self._milestones],
                    )
                else:
                    _logger.warning(
                        "[GIIController] MilestoneDecomposer returned empty list. "
                        "Falling back to scaffold execution."
                    )
            except Exception as exc:
                _logger.warning(
                    "[GIIController] MilestoneDecomposer init failed (non-fatal): %s. "
                    "Falling back to scaffold-based execution.", exc,
                )
                self._milestones_active = False

    def _inject_current_milestone(self) -> None:
        """Inject the current milestone condition as the PerStepReasoner sub-objective."""
        if not self._milestones_active or not self._milestones:
            return
        if self._current_milestone_idx >= len(self._milestones):
            return
        milestone = self._milestones[self._current_milestone_idx]
        # Milestone objects expose condition, completion_signal, name
        condition = getattr(milestone, "condition", str(milestone))
        signal = getattr(milestone, "completion_signal", "")
        name = getattr(milestone, "name", f"milestone_{self._current_milestone_idx + 1}")
        n_total = len(self._milestones)
        n_current = self._current_milestone_idx + 1

        sub_objective = (
            f"[Milestone {n_current}/{n_total}: {name}]\n"
            f"Achieve this observable condition: {condition}\n"
            f"Completion signal: {signal}\n"
            f"Full task context: {self._objective[:300]}"
        )
        if self._per_step_reasoner is not None and hasattr(
            self._per_step_reasoner, "update_objective"
        ):
            try:
                self._per_step_reasoner.update_objective(sub_objective)
                _logger.info(
                    "[GIIController] Milestone %d/%d injected: %r",
                    n_current, n_total, name,
                )
            except Exception as exc:
                _logger.debug("[GIIController] update_objective failed: %s", exc)

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
        execution_log: Optional[Dict[str, Any]] = None,
        focused_app: Optional[str] = None,
    ) -> None:
        """
        Record action outcome for per-step history.

        AUDIT MEDIUM-1 FIX: Also triggers EpisodicSynthesizer.store_checkpoint()
        every _EPISODIC_CHECKPOINT_INTERVAL calls when execution_log is provided.
        This ensures partial lessons are captured even if the task crashes
        before completion — previously zero lessons were stored on mid-task crashes.

        Parameters
        ----------
        action          : The dispatched action dict.
        success         : Whether the action succeeded.
        output          : Stdout/stderr from the action (truncated).
        execution_log   : The live execution_log dict from _execute_autonomous_loop.
                          When provided, triggers periodic episodic checkpoints.
        focused_app     : The focused application name, for memory tagging.
        """
        with self._lock:
            self._outcome_call_count += 1
            call_count = self._outcome_call_count

        if self._per_step_reasoner is not None:
            try:
                self._per_step_reasoner.record_outcome(
                    action, success=success, output=output
                )
            except Exception:
                pass

        # ── AUDIT MEDIUM-1 FIX: mid-task episodic checkpoint ─────────────────
        # store_checkpoint() is async (background thread, ~0ms latency here).
        # Only fires when:
        #   • EpisodicSynthesizer is initialised
        #   • execution_log is provided (caller must pass it)
        #   • call_count is a multiple of _EPISODIC_CHECKPOINT_INTERVAL
        #   • This checkpoint hasn't fired for the current call_count already
        _should_checkpoint = (
            self._episodic_synthesizer is not None
            and execution_log is not None
            and call_count > 0
            and call_count % _EPISODIC_CHECKPOINT_INTERVAL == 0
            and call_count != self._last_checkpoint_call
        )
        if _should_checkpoint:
            with self._lock:
                self._last_checkpoint_call = call_count
            _logger.info(
                "[GIIController] Triggering episodic checkpoint at call %d "
                "(interval=%d).",
                call_count, _EPISODIC_CHECKPOINT_INTERVAL,
            )
            try:
                self._episodic_synthesizer.store_checkpoint(
                    execution_log=execution_log,
                    objective=self._objective,
                    semantic_memory=self._semantic_memory,
                    focused_app=focused_app,
                    application_memory=self._application_memory,
                    iteration=call_count,
                )
            except Exception as cp_exc:
                _logger.warning(
                    "[GIIController] Episodic checkpoint failed (non-fatal): %s",
                    cp_exc,
                )

    def get_planning_context(self, focused_app: Optional[str] = None) -> str:
        """
        Build the memory context injected into every per-step reasoning prompt.

        Retrieval priority (highest to lowest):
          1. Mem0Store — cross-session working memory (agent experiences)
          2. CogneeStore — semantic knowledge graph (learned lessons)
          3. SemanticMemory — regex-extracted facts (CPU fallback)
          4. ApplicationMemory — per-app keyboard shortcuts / workflow patterns
        """
        if not self._enabled:
            return ""
        parts = []

        # --- Mem0: cross-session memory ---
        if self._mem0_store is not None:
            try:
                agent_id = getattr(self._mem0_store, 'make_agent_id',
                                   lambda x: x)(self._objective)
                memories = self._mem0_store.search_memory(
                    self._objective, agent_id, limit=6
                )
                if memories:
                    lines = ["[Cross-session memories]"]
                    for m in memories[:6]:
                        text = m.get("memory") or m.get("text") or str(m)
                        lines.append(f"  • {str(text)[:200]}")
                    parts.append("\n".join(lines))
                    _logger.debug(
                        "[GIIController] Mem0 provided %d cross-session memories.",
                        len(memories),
                    )
            except Exception as exc:
                _logger.debug("[GIIController] Mem0 context retrieval error: %s", exc)

        # --- CogneeStore: knowledge graph (primary) OR SemanticMemory (fallback) ---
        _knowledge_sourced = False
        if self._cognee_store is not None and self._cognee_store._available:
            try:
                cognee_results = self._cognee_store.search(self._objective, limit=8)
                if cognee_results:
                    lines = ["[Learned knowledge (Cognee)]"]
                    for r in cognee_results[:8]:
                        text = r.get("text") or r.get("object") or str(r)
                        lines.append(f"  • {str(text)[:200]}")
                    parts.append("\n".join(lines))
                    _knowledge_sourced = True
                    _logger.debug(
                        "[GIIController] CogneeStore provided %d knowledge results.",
                        len(cognee_results),
                    )
            except Exception as exc:
                _logger.debug("[GIIController] CogneeStore context retrieval error: %s", exc)

        if not _knowledge_sourced and self._semantic_memory:
            try:
                facts = self._semantic_memory.query(self._objective, max_results=8)
                sem_context = self._semantic_memory.format_for_prompt(facts)
                if sem_context:
                    parts.append(sem_context)
            except Exception:
                pass

        # --- ApplicationMemory: per-app keyboard shortcuts and workflow patterns ---
        if self._application_memory and focused_app:
            try:
                app_context = self._application_memory.format_profile_for_prompt(focused_app)
                if app_context:
                    parts.append(app_context)
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

        # --- Mem0: add task outcome to cross-session memory (background thread) ---
        if self._mem0_store is not None and execution_log is not None:
            def _store_mem0():
                try:
                    agent_id = getattr(self._mem0_store, 'make_agent_id',
                                       lambda x: x)(self._objective)
                    # Build a compact conversation for Mem0 extraction
                    log_str = str(execution_log)[:3000]
                    messages = [
                        {"role": "user", "content": f"Task: {self._objective}"},
                        {"role": "assistant", "content": (
                            f"Task {'completed successfully' if success else 'failed'}. "
                            f"Summary: {log_str}"
                        )},
                    ]
                    self._mem0_store.add_memory(
                        messages, agent_id,
                        metadata={"objective": self._objective[:200], "success": success}
                    )
                    _logger.info(
                        "[GIIController] Mem0 memory stored for agent_id=%s success=%s",
                        agent_id, success,
                    )
                except Exception as exc:
                    _logger.debug("[GIIController] Mem0 post-task storage error: %s", exc)
            import threading as _t
            _t.Thread(target=_store_mem0, daemon=True).start()

        # --- CogneeStore: ingest execution log into knowledge graph (background) ---
        if (self._cognee_store is not None
                and self._cognee_store._available
                and execution_log is not None):
            def _store_cognee():
                try:
                    log_text = (
                        f"Task: {self._objective}\n"
                        f"Outcome: {'success' if success else 'failure'}\n"
                        f"App: {focused_app or 'unknown'}\n"
                        f"Log: {str(execution_log)[:4000]}"
                    )
                    self._cognee_store.add(log_text)
                    _logger.info(
                        "[GIIController] CogneeStore knowledge graph updated "
                        "for task: %s", self._objective[:60],
                    )
                except Exception as exc:
                    _logger.debug("[GIIController] CogneeStore post-task error: %s", exc)
            import threading as _t2
            _t2.Thread(target=_store_cognee, daemon=True).start()
        if self._semantic_memory and execution_log:
            
            llm_synthesis_succeeded = False
            try:
                lessons_before = (
                    self._semantic_memory.stats().get("total_facts", 0)
                    if hasattr(self._semantic_memory, "stats") else 0
                )
                self._synthesize_lessons_with_llm(execution_log, focused_app)
                lessons_after = (
                    self._semantic_memory.stats().get("total_facts", 0)
                    if hasattr(self._semantic_memory, "stats") else 0
                )
                if lessons_after > lessons_before:
                    llm_synthesis_succeeded = True
                    _logger.info(
                        "[GIIController] LLM lesson synthesis stored %d new facts — "
                        "skipping regex fallback.",
                        lessons_after - lessons_before,
                    )
            except Exception as exc:
                _logger.warning("[GIIController] LLM lesson synthesis error: %s", exc)

            if not llm_synthesis_succeeded:
                # Regex fallback only when LLM synthesis produced nothing
                try:
                    self._extract_semantic_facts_from_log(execution_log, focused_app)
                    _logger.debug("[GIIController] Regex fact extraction used as LLM fallback.")
                except Exception as exc:
                    _logger.debug("[GIIController] Regex fact extraction error: %s", exc)
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
            # AUDIT MEDIUM-1: episodic checkpoint stats
            "outcome_call_count": self._outcome_call_count,
            "last_checkpoint_call": self._last_checkpoint_call,
            "episodic_checkpoint_interval": _EPISODIC_CHECKPOINT_INTERVAL,
            # ARCH-1: milestone stats
            "milestones_active": self._milestones_active,
            "milestones_total": len(self._milestones),
            "current_milestone_idx": self._current_milestone_idx,
        }
        if self._per_step_reasoner:
            stats["per_step_reasoner"] = self._per_step_reasoner.get_stats()
        if self._consequence_reasoner:
            stats["consequence_reasoner"] = self._consequence_reasoner.get_stats()
        if self._semantic_memory:
            stats["semantic_memory"] = self._semantic_memory.stats()
        if self._application_memory:
            stats["application_memory"] = self._application_memory.stats()
        if self._episodic_synthesizer:
            try:
                stats["episodic_synthesizer"] = self._episodic_synthesizer.get_stats()
            except Exception:
                pass
        return stats
