"""
core/cognition/session_reflector.py
=====================================
Session Reflector — Daily Plan Reflection at Session Start.

Blueprint §10 — Episodic Memory + §14 — Self-Improvement Flywheel

Role: At the start of each agent session, query episodic and semantic memory
for relevant past experience with the current objective, generate a "session
plan" that incorporates prior lessons learned, known failure modes, and
verified successful workflows.

This is the "morning brief" for the agent — analogous to how a human
professional reviews notes before starting a task they've done before.

Why this matters (GII gap):
    Without session reflection, the agent restarts every task from scratch,
    ignoring rich cross-session experience accumulated in episodic/semantic
    memory. A GII system must actively use its memory, not just store it.

Output:
    - session_plan: structured JSON with approach, known pitfalls, shortcuts
    - Injected into per_step_reasoner as a "session context" hint
    - Stored as a new episodic note for the current session

Integration:
    - gii_controller._initialise_phase3_components() spawns this as a daemon thread
    - per_step_reasoner.set_session_context() consumes the output
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

_logger = logging.getLogger(__name__)

_REFLECTION_TIMEOUT = float(os.environ.get("PROJECTZEO_SESSION_REFLECT_TIMEOUT", "30.0"))
_MAX_MEMORIES = int(os.environ.get("PROJECTZEO_SESSION_REFLECT_MEMORIES", "5"))
_ENABLED = os.environ.get("PROJECTZEO_SESSION_REFLECT_ENABLED", "1").strip() != "0"


@dataclass
class SessionPlan:
    """Structured output of session-start reflection."""
    objective:          str
    approach:           str           # Recommended high-level approach
    known_pitfalls:     List[str] = field(default_factory=list)
    successful_patterns: List[str] = field(default_factory=list)
    app_quirks:         List[str] = field(default_factory=list)
    estimated_steps:    int = 0
    confidence:         float = 0.0   # 0.0 = no prior experience, 1.0 = well-practised
    generated_at:       float = field(default_factory=time.time)
    source_memory_ids:  List[str] = field(default_factory=list)

    def to_prompt_block(self) -> str:
        """Format for injection into per_step_reasoner system prompt."""
        lines = [
            "=== SESSION REFLECTION (prior experience) ===",
            f"Objective: {self.objective[:150]}",
            f"Recommended approach: {self.approach}",
        ]
        if self.successful_patterns:
            lines.append("Verified successful patterns:")
            for p in self.successful_patterns[:3]:
                lines.append(f"  ✓ {p}")
        if self.known_pitfalls:
            lines.append("Known pitfalls to avoid:")
            for p in self.known_pitfalls[:3]:
                lines.append(f"  ✗ {p}")
        if self.app_quirks:
            lines.append("App-specific quirks:")
            for q in self.app_quirks[:2]:
                lines.append(f"  ⚠ {q}")
        if self.estimated_steps:
            lines.append(f"Estimated steps: {self.estimated_steps}")
        lines.append(f"Prior experience confidence: {self.confidence:.0%}")
        lines.append("=== END SESSION REFLECTION ===")
        return "\n".join(lines)


_SYSTEM_PROMPT = """\
You are a session planner for an autonomous GUI agent.
Given the agent's current objective and relevant past memories,
generate a structured session plan in JSON.

MEMORY RETRIEVAL:
The memories below are retrieved from prior sessions where this or similar
objectives were attempted. Use them to identify: proven approaches,
common failure modes, application-specific quirks.

OUTPUT FORMAT (JSON only, no markdown):
{
  "approach": "<recommended high-level approach in 1-2 sentences>",
  "known_pitfalls": ["<pitfall 1>", "<pitfall 2>"],
  "successful_patterns": ["<pattern 1>", "<pattern 2>"],
  "app_quirks": ["<quirk 1>"],
  "estimated_steps": <integer or 0 if unknown>,
  "confidence": <float 0.0-1.0 based on how much relevant memory exists>
}

Rules:
- Be specific and actionable — no generic advice
- Only include items supported by the provided memories
- If no relevant memories: confidence=0.0, approach="No prior experience — proceed carefully"
- Do NOT hallucinate patterns not present in memories
"""


class SessionReflector:
    """
    Generates a session-start plan from episodic + semantic memory retrieval.

    Thread-safe. Designed to run asynchronously so it does not block the
    agent's startup sequence.
    """

    def __init__(
        self,
        *,
        llm_call: Callable,
        objective: str,
        episodic_synthesizer=None,
        semantic_memory=None,
        knowledge_vault=None,
        application_memory=None,
    ) -> None:
        self._llm           = llm_call
        self._objective     = objective
        self._episodic      = episodic_synthesizer
        self._semantic      = semantic_memory
        self._vault         = knowledge_vault
        self._app_memory    = application_memory

        self._plan: Optional[SessionPlan] = None
        self._plan_ready = threading.Event()
        self._lock = threading.Lock()

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def reflect_on_session_start(self) -> Optional[SessionPlan]:
        """
        Run session-start reflection. Blocking — call from a daemon thread.
        Sets self._plan and signals _plan_ready when complete.
        """
        if not _ENABLED:
            return None

        _start = time.time()
        try:
            memories = self._retrieve_relevant_memories()
            plan = self._generate_plan(memories)
            with self._lock:
                self._plan = plan
            self._plan_ready.set()
            _logger.info(
                "[SessionReflector] Plan ready in %.1fs: confidence=%.2f pitfalls=%d patterns=%d",
                time.time() - _start,
                plan.confidence,
                len(plan.known_pitfalls),
                len(plan.successful_patterns),
            )
            return plan
        except Exception as exc:
            _logger.warning("[SessionReflector] Reflection failed (non-fatal): %s", exc)
            self._plan_ready.set()
            return None

    def get_plan(self, wait: bool = False, timeout: float = 5.0) -> Optional[SessionPlan]:
        """Get the session plan. If not ready, optionally wait."""
        if wait:
            self._plan_ready.wait(timeout=timeout)
        with self._lock:
            return self._plan

    def get_plan_as_prompt_block(
        self,
        wait: bool = True,
        timeout: float = 5.0,
    ) -> str:
        """Return plan formatted for prompt injection. Empty string if unavailable."""
        plan = self.get_plan(wait=wait, timeout=timeout)
        if plan is None:
            return ""
        return plan.to_prompt_block()

    # ─────────────────────────────────────────────────────────────────────────
    # Memory retrieval
    # ─────────────────────────────────────────────────────────────────────────

    def _retrieve_relevant_memories(self) -> List[str]:
        """Retrieve relevant memories from all available memory systems."""
        memories: List[str] = []

        # 1. Episodic synthesizer — cross-session retrieval
        if self._episodic is not None:
            try:
                results = self._episodic.retrieve(
                    query=self._objective,
                    max_results=_MAX_MEMORIES,
                )
                for r in results or []:
                    content = getattr(r, "content", None) or str(r)
                    if content:
                        memories.append(f"[EPISODE] {content[:400]}")
            except Exception as exc:
                _logger.debug("[SessionReflector] Episodic retrieve error: %s", exc)

        # 2. Semantic memory — ACT-R fact retrieval
        if self._semantic is not None:
            try:
                facts = self._semantic.query(
                    self._objective,
                    max_results=_MAX_MEMORIES,
                )
                for fact in facts or []:
                    s = getattr(fact, "subject", "")
                    p = getattr(fact, "predicate", "")
                    o = getattr(fact, "object_", "")
                    if s or p or o:
                        memories.append(f"[SEMANTIC] {s} {p} {o}".strip()[:300])
            except Exception as exc:
                _logger.debug("[SessionReflector] Semantic memory error: %s", exc)

        # 3. Knowledge vault — curated lessons
        if self._vault is not None:
            try:
                entries = self._vault.query_relevant(
                    self._objective,
                    max_results=_MAX_MEMORIES,
                )
                for entry in entries or []:
                    content = getattr(entry, "content", None) or str(entry)
                    if content:
                        memories.append(f"[VAULT] {content[:300]}")
            except Exception as exc:
                _logger.debug("[SessionReflector] Vault error: %s", exc)

        # 4. Application memory — app-specific profiles
        if self._app_memory is not None:
            try:
                # Extract app name hint from objective
                app_hint = self._objective.split()[-1].lower() if self._objective else ""
                if app_hint:
                    profile = self._app_memory.get_profile(app_hint)
                    if profile and profile.is_known():
                        if profile.quirks:
                            memories.append(f"[APP_QUIRK] {'; '.join(profile.quirks[:3])}")
                        if profile.error_solutions:
                            for err, sol in list(profile.error_solutions.items())[:2]:
                                memories.append(f"[APP_ERROR_SOL] {err}: {sol}")
            except Exception as exc:
                _logger.debug("[SessionReflector] AppMemory error: %s", exc)

        return memories

    # ─────────────────────────────────────────────────────────────────────────
    # Plan generation
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_plan(self, memories: List[str]) -> SessionPlan:
        """Call LLM to synthesise a session plan from retrieved memories."""
        if not memories:
            return SessionPlan(
                objective=self._objective,
                approach="No prior experience — proceed carefully, observe before acting.",
                confidence=0.0,
            )

        memory_block = "\n".join(memories[:15])  # Cap to avoid token overflow
        user_msg = (
            f"OBJECTIVE: {self._objective[:300]}\n\n"
            f"RELEVANT MEMORIES ({len(memories)} retrieved):\n"
            f"{memory_block}\n\n"
            "Generate the session plan JSON."
        )

        try:
            raw = self._llm(
                system=_SYSTEM_PROMPT,
                user=user_msg,
                max_tokens=600,
                timeout=_REFLECTION_TIMEOUT,
            )
            if isinstance(raw, dict):
                text = raw.get("content") or raw.get("text") or ""
            else:
                text = str(raw)

            # Strip markdown fences
            text = text.strip()
            if text.startswith("```"):
                text = text[text.find("\n") + 1:]
            if text.endswith("```"):
                text = text[: text.rfind("```")]
            text = text.strip()

            data = json.loads(text)

            return SessionPlan(
                objective=self._objective,
                approach=str(data.get("approach", "Proceed step by step."))[:500],
                known_pitfalls=[str(p)[:200] for p in (data.get("known_pitfalls") or [])[:5]],
                successful_patterns=[str(p)[:200] for p in (data.get("successful_patterns") or [])[:5]],
                app_quirks=[str(q)[:200] for q in (data.get("app_quirks") or [])[:3]],
                estimated_steps=int(data.get("estimated_steps") or 0),
                confidence=float(data.get("confidence") or 0.0),
                source_memory_ids=[f"mem_{i}" for i in range(len(memories))],
            )

        except Exception as exc:
            _logger.warning("[SessionReflector] LLM plan generation failed: %s", exc)
            # Graceful degradation — return bare plan with raw memories as pitfalls
            return SessionPlan(
                objective=self._objective,
                approach="Review memories and proceed carefully.",
                known_pitfalls=[m[:150] for m in memories[:3]],
                confidence=0.1,
            )
