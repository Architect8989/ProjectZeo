from __future__ import annotations

import logging
import os
import time
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GII Mode
# ---------------------------------------------------------------------------

class GIIMode:
    """Enum-like GII mode constants."""
    DISABLED = 0    # Legacy scripted execution
    BASIC = 1       # Per-step reasoning only
    FULL = 2        # Per-step reasoning + consequence safety + memory


def get_gii_mode() -> int:
    """Read GII mode from environment variable. Default: DISABLED."""
    try:
        return int(os.environ.get("PROJECTZEO_GII_MODE", "0"))
    except (ValueError, TypeError):
        return GIIMode.DISABLED


# ---------------------------------------------------------------------------
# GIIController
# ---------------------------------------------------------------------------

class GIIController:
    

    def __init__(
        self,
        *,
        llm_callable: Callable,
        objective: str,
        scaffold_steps: Optional[List[Any]] = None,
        gii_mode: int = GIIMode.DISABLED,
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

        if self._enabled:
            self._initialise_components(memory_dir)

        _logger.info(
            "[GIIController] Initialised. mode=%d enabled=%s",
            gii_mode, self._enabled,
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

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def enabled(self) -> bool:
        """True if GII mode is active (mode ≥ BASIC)."""
        return self._enabled

    @property
    def gii_mode(self) -> int:
        return self._gii_mode

    # =========================================================================
    # Component initialisation
    # =========================================================================

    def _initialise_components(self, memory_dir: Optional[str]) -> None:
        """Initialise GII components. Fails gracefully on import errors."""

        # Semantic memory (always in BASIC mode)
        try:
            from core.memory.semantic_memory import SemanticMemory
            self._semantic_memory = SemanticMemory(memory_dir=memory_dir)
        except Exception as exc:
            _logger.warning("[GIIController] SemanticMemory init failed: %s", exc)

        # Application memory (always in BASIC mode)
        try:
            from core.memory.application_memory import ApplicationMemory
            self._application_memory = ApplicationMemory(memory_dir=memory_dir)
        except Exception as exc:
            _logger.warning("[GIIController] ApplicationMemory init failed: %s", exc)

        # Consequence reasoner (FULL mode only)
        if self._gii_mode >= GIIMode.FULL:
            try:
                from core.safety.consequence_reasoner import ConsequenceReasoner
                self._consequence_reasoner = ConsequenceReasoner(
                    llm_callable=self._llm,
                    enable_tier2=True,
                    enable_tier3=True,
                )
            except Exception as exc:
                _logger.warning("[GIIController] ConsequenceReasoner init failed: %s", exc)

        # Per-step reasoner (BASIC + FULL modes)
        try:
            from core.cognition.per_step_reasoner import PerStepReasoner

            # Convert ExecutionStep objects to plain dicts for scaffold
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

    # =========================================================================
    # Primary API
    # =========================================================================

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

    def record_outcome(
        self,
        action: Dict[str, Any],
        *,
        success: bool,
        output: str = "",
    ) -> None:
        """Record the outcome of a dispatched action for history tracking."""
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

        # Application memory
        if self._application_memory and focused_app:
            try:
                app_context = self._application_memory.format_profile_for_prompt(focused_app)
                if app_context:
                    parts.append(app_context)
            except Exception:
                pass

        # Semantic memory
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

        # Update application task count
        if self._application_memory and focused_app:
            try:
                self._application_memory.increment_task_count(focused_app)
            except Exception:
                pass

        # Extract semantic facts from execution log
        if self._semantic_memory and execution_log:
            try:
                self._extract_semantic_facts_from_log(execution_log, focused_app)
            except Exception as exc:
                _logger.debug("[GIIController] Fact extraction error: %s", exc)

        # Persist memories
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

    # =========================================================================
    # Knowledge extraction
    # =========================================================================

    def _extract_semantic_facts_from_log(
        self,
        execution_log: Dict[str, Any],
        focused_app: Optional[str],
    ) -> None:
        
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

                # Successful install
                if install_success_re.search(output_text) and focused_app:
                    self._semantic_memory.store(
                        subject=focused_app,
                        predicate="install_outcome",
                        object_=f"success (step {step_idx})",
                        category="install_outcomes",
                        confidence=0.9,
                        source="observed",
                    )

                # Version strings
                for m in version_re.finditer(output_text):
                    tool = m.group(1).lower()
                    version = m.group(2)
                    if len(tool) > 2 and len(version) > 1:
                        self._semantic_memory.store(
                            subject=tool,
                            predicate="version",
                            object_=version,
                            category="application_facts",
                            confidence=0.95,
                            source="observed",
                        )

                # Errors in failed steps
                if not output_entry.get("success", True):
                    for m in error_re.finditer(output_text):
                        error_snippet = m.group(1).strip()[:100]
                        if focused_app:
                            # Store error without solution for now — operator can
                            # later confirm solutions via record_error_solution()
                            self._semantic_memory.store(
                                subject=focused_app,
                                predicate="known_error",
                                object_=error_snippet,
                                category="error_solutions",
                                confidence=0.6,
                                source="observed",
                            )

    # =========================================================================
    # Diagnostics
    # =========================================================================

    def get_stats(self) -> dict:
        stats = {
            "gii_mode": self._gii_mode,
            "enabled": self._enabled,
            "task_duration_seconds": round(time.time() - self._task_start, 1),
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
