"""
core/cognition/dual_mode_reasoner.py
=====================================
Dual-Mode Reasoning — Blueprint §4.3 / SGLang Integration

WHY THIS FILE EXISTS
--------------------
The Blueprint specifies a tiered LLM routing strategy:
- **Fast / Instruct mode**: Qwen3-32B (SGLang Tier 1) for reversible actions,
  routine decisions, SOAR operator selection.  Low latency, high throughput.
- **Deep / Thinking mode**: Qwen3-235B-Thinking (SGLang Tier 2) for
  irreversible actions, consequence simulation, complex planning decisions,
  new-app exploration (Blender Test scenarios).

Without this module, ALL decisions went to the same model at the same depth,
wasting inference budget on simple reversible actions while potentially
under-thinking genuinely dangerous irreversible ones.

ROUTING LOGIC
-------------
The DualModeReasoner wraps the llm_callable and transparently routes:

    Reversibility.REVERSIBLE  + low_complexity  →  fast/instruct mode
    Reversibility.CAUTION     + any             →  thinking mode
    Reversibility.IRREVERSIBLE                  →  thinking mode (ALWAYS)
    unknown_app = True                          →  thinking mode (Blender Test)
    stagnant_count >= threshold                  →  thinking mode (deeper recovery)

The mode selection is transparent — the caller sees the same interface.
The router logs its decision so it's auditable.

INTEGRATION
-----------
    from core.cognition.dual_mode_reasoner import DualModeReasoner
    reasoner = DualModeReasoner(llm_callable=base_fn)

    # In per_step_reasoner.py, replace self._llm(messages, ...) with:
    action, reason = reasoner.decide(
        messages=messages,
        action_candidates=candidates,
        reversibility=Reversibility.CAUTION,
        stagnant_count=3,
        known_app=True,
    )
"""
from __future__ import annotations

import logging
import os
import time
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Reversibility enum (matches core/safety/consequence_reasoner.py)
# ─────────────────────────────────────────────────────────────────────────────

class Reversibility(str, Enum):
    REVERSIBLE   = "REVERSIBLE"
    CAUTION      = "CAUTION"
    IRREVERSIBLE = "IRREVERSIBLE"
    UNKNOWN      = "UNKNOWN"


# ─────────────────────────────────────────────────────────────────────────────
# Mode selection thresholds
# ─────────────────────────────────────────────────────────────────────────────

# Use thinking mode when stagnant_count exceeds this
_STAGNANT_THINKING_THRESHOLD = int(
    os.environ.get("PROJECTZEO_THINKING_STAGNANT_THRESH", "3")
)

# Model names (override via env if needed)
_FAST_MODEL   = os.environ.get("PROJECTZEO_FAST_MODEL",    "sglang/fast")
_DEEP_MODEL   = os.environ.get("PROJECTZEO_DEEP_MODEL",    "sglang/deep")
_VISION_MODEL = os.environ.get("PROJECTZEO_VISION_MODEL",  "sglang/vision")
_CODER_MODEL  = os.environ.get("PROJECTZEO_CODER_MODEL",   "sglang/coder")

# Operations that ALWAYS use coder model
_CODER_OPERATIONS = frozenset({
    "command", "exec", "script", "install", "terminal", "bash", "python",
    "run", "execute", "shell",
})

# Operations that ALWAYS use vision model
_VISION_OPERATIONS = frozenset({
    "click", "navigate", "tap", "drag", "scroll", "screenshot", "grounding",
})


class ReasoningMode(str, Enum):
    FAST      = "fast"      # Qwen3-32B instruct — low-latency
    DEEP      = "deep"      # Qwen3-235B-Thinking — deep reasoning
    VISION    = "vision"    # UI-TARS-2 — GUI grounding
    CODER     = "coder"     # Qwen3-Coder-480B — shell/script generation
    FALLBACK  = "fallback"  # Base llm_callable


@dataclass
class ReasoningDecision:
    mode:         ReasoningMode
    model:        str
    rationale:    str
    latency_ms:   float = 0.0
    thinking_tokens: int = 0


class DualModeReasoner:
    """
    Transparently routes LLM calls to the appropriate model tier based on
    action reversibility, complexity, and operational context.

    Falls back gracefully to the base llm_callable when SGLang is unavailable.
    """

    def __init__(
        self,
        llm_callable: Callable,
        *,
        adapter_factory=None,
    ) -> None:
        self._base_llm = llm_callable
        self._factory  = adapter_factory
        self._lock     = threading.Lock()

        # Metrics
        self._call_counts: Dict[ReasoningMode, int] = {m: 0 for m in ReasoningMode}
        self._total_latency_ms: Dict[ReasoningMode, float] = {m: 0.0 for m in ReasoningMode}

        # Check if SGLang tiers are available
        self._sglang_available = self._probe_sglang()

        if self._sglang_available:
            _logger.info(
                "[DualMode] SGLang tiers available. "
                "Fast=%s | Deep=%s | Vision=%s | Coder=%s",
                _FAST_MODEL, _DEEP_MODEL, _VISION_MODEL, _CODER_MODEL,
            )
        else:
            _logger.info(
                "[DualMode] SGLang unavailable — all calls route to base llm_callable. "
                "Set PROJECTZEO_USE_SGLANG=1 and start SGLang server to enable tiered routing."
            )

    # ─────────────────────────────────────────────────────────────────────────
    # MODE SELECTION
    # ─────────────────────────────────────────────────────────────────────────

    def select_mode(
        self,
        *,
        operation: str = "",
        reversibility: Reversibility = Reversibility.UNKNOWN,
        stagnant_count: int = 0,
        known_app: bool = True,
        is_script: bool = False,
        force_mode: Optional[ReasoningMode] = None,
    ) -> ReasoningDecision:
        """
        Determine the optimal reasoning mode for a given action context.
        Returns a ReasoningDecision with mode, model, and rationale.
        """
        if force_mode is not None:
            model = self._mode_to_model(force_mode)
            return ReasoningDecision(
                mode=force_mode,
                model=model,
                rationale=f"Forced mode: {force_mode.value}",
            )

        op = operation.lower().strip()

        # Coder model for shell/script operations
        if op in _CODER_OPERATIONS or is_script:
            return ReasoningDecision(
                mode=ReasoningMode.CODER,
                model=_CODER_MODEL,
                rationale=f"Coder model: operation={op!r} is a shell/script action",
            )

        # Vision model for GUI grounding operations
        if op in _VISION_OPERATIONS:
            return ReasoningDecision(
                mode=ReasoningMode.VISION,
                model=_VISION_MODEL,
                rationale=f"Vision model: operation={op!r} requires pixel-level grounding",
            )

        # Unknown app → always deep (Blender Test)
        if not known_app:
            return ReasoningDecision(
                mode=ReasoningMode.DEEP,
                model=_DEEP_MODEL,
                rationale="Deep mode: unknown app — Blender Test scenario requires generalization",
            )

        # Irreversible action → always deep
        if reversibility == Reversibility.IRREVERSIBLE:
            return ReasoningDecision(
                mode=ReasoningMode.DEEP,
                model=_DEEP_MODEL,
                rationale="Deep mode: IRREVERSIBLE action requires careful consequence reasoning",
            )

        # Stagnation recovery → deep
        if stagnant_count >= _STAGNANT_THINKING_THRESHOLD:
            return ReasoningDecision(
                mode=ReasoningMode.DEEP,
                model=_DEEP_MODEL,
                rationale=f"Deep mode: stagnant_count={stagnant_count} >= threshold={_STAGNANT_THINKING_THRESHOLD}",
            )

        # Caution actions → deep
        if reversibility == Reversibility.CAUTION:
            return ReasoningDecision(
                mode=ReasoningMode.DEEP,
                model=_DEEP_MODEL,
                rationale="Deep mode: CAUTION-level action",
            )

        # Default: fast/instruct for reversible routine actions
        return ReasoningDecision(
            mode=ReasoningMode.FAST,
            model=_FAST_MODEL,
            rationale="Fast mode: REVERSIBLE action, known app, not stagnant",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # CALL DISPATCH
    # ─────────────────────────────────────────────────────────────────────────

    def call(
        self,
        messages: List[Dict[str, Any]],
        *,
        mode_decision: Optional[ReasoningDecision] = None,
        operation: str = "",
        reversibility: Reversibility = Reversibility.UNKNOWN,
        stagnant_count: int = 0,
        known_app: bool = True,
        is_script: bool = False,
        objective: str = "",
        session_id: str = "dual_mode",
        **kwargs,
    ) -> Any:
        """
        Call the appropriate model tier.

        If SGLang is unavailable, falls back transparently to base llm_callable.
        """
        if mode_decision is None:
            mode_decision = self.select_mode(
                operation=operation,
                reversibility=reversibility,
                stagnant_count=stagnant_count,
                known_app=known_app,
                is_script=is_script,
            )

        _logger.debug(
            "[DualMode] mode=%s model=%s | %s",
            mode_decision.mode.value, mode_decision.model, mode_decision.rationale[:80],
        )

        start_ts = time.time()

        try:
            if self._sglang_available and self._factory is not None:
                result = self._call_via_factory(
                    mode_decision, messages, objective=objective,
                    session_id=session_id, **kwargs
                )
            else:
                # Fallback to base callable
                result = self._base_llm(
                    messages=messages,
                    objective=objective,
                    session_id=session_id,
                    **kwargs,
                )
                mode_decision = ReasoningDecision(
                    mode=ReasoningMode.FALLBACK,
                    model="base",
                    rationale="SGLang unavailable — base llm_callable",
                )
        except Exception as e:
            _logger.warning(
                "[DualMode] %s failed (%s) — falling back to base callable.",
                mode_decision.mode.value, e,
            )
            result = self._base_llm(
                messages=messages,
                objective=objective,
                session_id=session_id,
                **kwargs,
            )
            mode_decision = ReasoningDecision(
                mode=ReasoningMode.FALLBACK,
                model="base",
                rationale=f"Tier error — fallback: {e}",
            )

        latency_ms = (time.time() - start_ts) * 1000
        mode_decision.latency_ms = latency_ms

        with self._lock:
            self._call_counts[mode_decision.mode] = (
                self._call_counts.get(mode_decision.mode, 0) + 1
            )
            self._total_latency_ms[mode_decision.mode] = (
                self._total_latency_ms.get(mode_decision.mode, 0.0) + latency_ms
            )

        _logger.debug(
            "[DualMode] Call complete: mode=%s latency=%.0fms",
            mode_decision.mode.value, latency_ms,
        )
        return result

    def _call_via_factory(
        self,
        mode_decision: ReasoningDecision,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> Any:
        """Route call through adapter factory to the correct SGLang tier."""
        adapter = self._factory.get_adapter(mode_decision.model)
        if adapter is None:
            raise RuntimeError(f"No adapter for model {mode_decision.model!r}")
        return adapter(messages=messages, **kwargs)

    # ─────────────────────────────────────────────────────────────────────────
    # SGLANG AVAILABILITY CHECK
    # ─────────────────────────────────────────────────────────────────────────

    def _probe_sglang(self) -> bool:
        """Check if SGLang server is reachable."""
        if os.environ.get("PROJECTZEO_USE_SGLANG", "0").strip() not in ("1", "true", "yes"):
            return False
        try:
            from config.model_config import get_fast_endpoint
            ep = get_fast_endpoint()
            import urllib.request
            req = urllib.request.Request(
                f"{ep.base_url}/health",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    @staticmethod
    def _mode_to_model(mode: ReasoningMode) -> str:
        return {
            ReasoningMode.FAST:     _FAST_MODEL,
            ReasoningMode.DEEP:     _DEEP_MODEL,
            ReasoningMode.VISION:   _VISION_MODEL,
            ReasoningMode.CODER:    _CODER_MODEL,
            ReasoningMode.FALLBACK: "base",
        }.get(mode, "base")

    # ─────────────────────────────────────────────────────────────────────────
    # METRICS
    # ─────────────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            stats = {}
            for mode in ReasoningMode:
                count = self._call_counts.get(mode, 0)
                total_ms = self._total_latency_ms.get(mode, 0.0)
                stats[mode.value] = {
                    "calls": count,
                    "avg_latency_ms": total_ms / count if count > 0 else 0.0,
                    "total_latency_ms": total_ms,
                }
            stats["sglang_available"] = self._sglang_available
        return stats


# ─────────────────────────────────────────────────────────────────────────────
# Singleton accessor
# ─────────────────────────────────────────────────────────────────────────────
_GLOBAL_REASONER: Optional[DualModeReasoner] = None
_GLOBAL_LOCK = threading.Lock()


def get_dual_mode_reasoner(
    llm_callable: Optional[Callable] = None,
    *,
    adapter_factory=None,
) -> Optional[DualModeReasoner]:
    """Get or create the global DualModeReasoner singleton."""
    global _GLOBAL_REASONER
    with _GLOBAL_LOCK:
        if _GLOBAL_REASONER is None:
            if llm_callable is None:
                return None
            _GLOBAL_REASONER = DualModeReasoner(
                llm_callable=llm_callable,
                adapter_factory=adapter_factory,
            )
        return _GLOBAL_REASONER
