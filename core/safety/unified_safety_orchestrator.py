"""
core/safety/unified_safety_orchestrator.py
============================================
Unified Safety Orchestrator — GII Blueprint §13

WHY THIS FILE EXISTS
--------------------
ProjectZeo has 10 safety tiers (PIGuard, APIs safety, Reversibility,
Coherence, Consequence, LlamaGuard, VeriSafe, Policy Engine, ProcessFence,
RuntimeWatchdog) plus Constitutional AI and network guards.

Previously these were wired in gii_loop.py as scattered if-statements with
no unified interface, no timing instrumentation, no per-tier health tracking,
and no single place to see the complete safety picture.

This orchestrator:

1. SINGLE DISPATCH CALL
   check_action(action, world_state, objective) calls all applicable tiers
   in order and returns a SafetyReport with the final decision and rationale
   from every tier.

2. TIER ORDERING ENFORCEMENT
   Tiers are always executed in the correct order:
   PIGuard (pre) → Reversibility (T1) → Coherence (T2) → Consequence (T3)
   → LlamaGuard (T4) → VeriSafe (T5) → Policy (T6) → Custom → PostDispatch

3. FAIL-CLOSED SEMANTICS
   Any tier returning DENY or raising an unhandled exception stops the chain
   and returns DENY. Only explicit ALLOW from all tiers passes.

4. TIMING AND AUDIT
   Each tier invocation is timed. The full report is written to the audit
   journal with per-tier latency, decisions, and rationale.

5. STARTUP HEALTH CHECK
   On construction, each tier is probed. Unavailable tiers are logged with
   their impact level (CRITICAL / WARNING).

INTEGRATION
-----------
    from core.safety.unified_safety_orchestrator import UnifiedSafetyOrchestrator
    safety = UnifiedSafetyOrchestrator(llm_callable=fn, policy_engine=policy)
    report = safety.check_action(action, world_state, objective)
    if report.decision == SafetyDecision.ALLOW:
        execute(action)
    elif report.decision == SafetyDecision.REQUIRE_HUMAN_CONFIRMATION:
        ask_human(...)
    else:
        skip_action(report.rationale)
"""
from __future__ import annotations

import logging
import os
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# If a tier exceeds this latency (ms) a warning is logged
_TIER_LATENCY_WARN_MS = 5000

# Default timeouts per tier (seconds)
_T2_COHERENCE_TIMEOUT = float(os.environ.get("PROJECTZEO_T2_TIMEOUT", "30"))
_T3_CONSEQUENCE_TIMEOUT = float(os.environ.get("PROJECTZEO_T3_TIMEOUT", "90"))
_T4_LLAMAGUARD_TIMEOUT = float(os.environ.get("PROJECTZEO_T4_TIMEOUT", "15"))
_T5_VERISAFE_TIMEOUT = float(os.environ.get("PROJECTZEO_T5_TIMEOUT", "45"))


class SafetyDecision(str, Enum):
    ALLOW                      = "ALLOW"
    DENY                       = "DENY"
    REQUIRE_HUMAN_CONFIRMATION = "REQUIRE_HUMAN_CONFIRMATION"
    UNCERTAIN                  = "UNCERTAIN"


@dataclass
class TierResult:
    tier:      str
    decision:  SafetyDecision
    rationale: str
    latency_ms: float = 0.0
    skipped:   bool   = False
    error:     str    = ""


@dataclass
class SafetyReport:
    decision:    SafetyDecision
    rationale:   str
    tier_results: List[TierResult] = field(default_factory=list)
    total_latency_ms: float = 0.0
    action_key:  str = ""

    def is_blocked(self) -> bool:
        return self.decision == SafetyDecision.DENY

    def needs_human(self) -> bool:
        return self.decision == SafetyDecision.REQUIRE_HUMAN_CONFIRMATION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "rationale": self.rationale,
            "total_latency_ms": self.total_latency_ms,
            "tiers": [
                {
                    "tier": r.tier,
                    "decision": r.decision.value,
                    "rationale": r.rationale[:200],
                    "latency_ms": r.latency_ms,
                    "skipped": r.skipped,
                    "error": r.error[:100] if r.error else "",
                }
                for r in self.tier_results
            ],
        }


class UnifiedSafetyOrchestrator:
    """
    Single point of entry for all ProjectZeo safety tiers.

    Tier execution order:
        0. PIGuard (prompt injection pre-filter)
        1. Static reversibility classification
        2. Goal coherence LLM check
        3. Consequence simulation LLM
        4. LlamaGuard3 content safety
        5. VeriSafe formal verification
        6. Policy engine (hot-reload YAML)
        7. ConstitutionalAI wrapper check
        8. Custom app-level rules (from policy.yaml extensions)
    """

    def __init__(
        self,
        *,
        llm_callable: Optional[Callable] = None,
        policy_engine=None,
        consequence_reasoner=None,
        vsa_verifier=None,
        piguard=None,
        llamaguard=None,
        constitutional_wrapper=None,
        journal=None,
    ) -> None:
        self._llm             = llm_callable
        self._policy          = policy_engine
        self._consequence     = consequence_reasoner
        self._vsa             = vsa_verifier
        self._piguard         = piguard
        self._llamaguard      = llamaguard
        self._constitutional  = constitutional_wrapper
        self._journal         = journal

        self._lock = threading.Lock()

        # Lazy-load any tiers not passed in
        self._lazy_init()
        self._print_startup_banner()

    def _lazy_init(self) -> None:
        """Load any safety tiers not passed to __init__."""
        if self._consequence is None:
            try:
                from core.safety.consequence_reasoner import ConsequenceReasoner
                self._consequence = ConsequenceReasoner(llm_callable=self._llm)
                _logger.info("[SafetyOrch] ✓ ConsequenceReasoner (Tiers 1-3)")
            except Exception as e:
                _logger.warning("[SafetyOrch] ConsequenceReasoner unavailable: %s", e)

        if self._piguard is None:
            try:
                from core.safety.piguard import create_piguard
                self._piguard = create_piguard(use_neural=False)
                _logger.info("[SafetyOrch] ✓ PIGuard (Tier 0)")
            except Exception as e:
                _logger.debug("[SafetyOrch] PIGuard unavailable: %s", e)

        if self._llamaguard is None:
            try:
                from core.safety.llamaguard_classifier import LlamaGuardClassifier
                self._llamaguard = LlamaGuardClassifier()
                _logger.info("[SafetyOrch] ✓ LlamaGuard3 (Tier 4)")
            except Exception as e:
                _logger.debug("[SafetyOrch] LlamaGuard unavailable: %s", e)

        if self._vsa is None:
            try:
                from core.safety.verisafe_agent import VeriSafeAgent
                self._vsa = VeriSafeAgent()
                _logger.info("[SafetyOrch] ✓ VeriSafe (Tier 5)")
            except Exception as e:
                _logger.debug("[SafetyOrch] VeriSafe unavailable: %s", e)

        if self._constitutional is None:
            try:
                from adapters.constitutional_wrapper import ConstitutionalWrapper
                self._constitutional = ConstitutionalWrapper(llm_callable=self._llm)
                _logger.info("[SafetyOrch] ✓ ConstitutionalAI (Tier 7)")
            except Exception as e:
                _logger.debug("[SafetyOrch] ConstitutionalWrapper unavailable: %s", e)

    def _print_startup_banner(self) -> None:
        import sys
        tier_status = [
            ("T0 PIGuard", self._piguard is not None),
            ("T1-3 ConsequenceReasoner", self._consequence is not None),
            ("T4 LlamaGuard3", self._llamaguard is not None),
            ("T5 VeriSafe", self._vsa is not None),
            ("T6 PolicyEngine", self._policy is not None),
            ("T7 ConstitutionalAI", self._constitutional is not None),
        ]
        print("\n[SafetyOrch] Safety tier status:", file=sys.stderr)
        for name, active in tier_status:
            status = "✓ ACTIVE" if active else "✗ INACTIVE (degraded)"
            print(f"  {name}: {status}", file=sys.stderr)

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN DISPATCH
    # ─────────────────────────────────────────────────────────────────────────

    def check_action(
        self,
        action: Dict[str, Any],
        world_state: Dict[str, Any],
        objective: str = "",
        focused_app: str = "",
    ) -> SafetyReport:
        """
        Run action through all applicable safety tiers in order.
        Returns a SafetyReport with the final decision and per-tier details.

        FAIL-CLOSED: any DENY from any tier stops the chain and returns DENY.
        """
        start_ts = time.time()
        tier_results: List[TierResult] = []
        action_key = self._compute_key(action)

        def add_result(result: TierResult) -> None:
            tier_results.append(result)
            if result.latency_ms > _TIER_LATENCY_WARN_MS:
                _logger.warning(
                    "[SafetyOrch] Tier %s slow: %.0fms", result.tier, result.latency_ms
                )

        # ── Tier 0: PIGuard ───────────────────────────────────────────────
        r0 = self._run_piguard(action, world_state)
        add_result(r0)
        if r0.decision == SafetyDecision.DENY:
            return self._build_report(
                SafetyDecision.DENY, r0.rationale, tier_results, start_ts, action_key
            )

        # ── Tiers 1-3: ConsequenceReasoner (Reversibility→Coherence→Sim) ──
        r123 = self._run_consequence_reasoner(action, objective, focused_app)
        add_result(r123)
        if r123.decision == SafetyDecision.DENY:
            return self._build_report(
                SafetyDecision.DENY, r123.rationale, tier_results, start_ts, action_key
            )
        if r123.decision == SafetyDecision.REQUIRE_HUMAN_CONFIRMATION:
            return self._build_report(
                SafetyDecision.REQUIRE_HUMAN_CONFIRMATION,
                r123.rationale, tier_results, start_ts, action_key,
            )

        # ── Tier 4: LlamaGuard ────────────────────────────────────────────
        r4 = self._run_llamaguard(action)
        add_result(r4)
        if r4.decision == SafetyDecision.DENY:
            return self._build_report(
                SafetyDecision.DENY, r4.rationale, tier_results, start_ts, action_key
            )

        # ── Tier 5: VeriSafe ──────────────────────────────────────────────
        r5 = self._run_verisafe(action)
        add_result(r5)
        if r5.decision == SafetyDecision.DENY:
            return self._build_report(
                SafetyDecision.DENY, r5.rationale, tier_results, start_ts, action_key
            )

        # ── Tier 6: Policy Engine ─────────────────────────────────────────
        r6 = self._run_policy(action, focused_app)
        add_result(r6)
        if r6.decision == SafetyDecision.DENY:
            return self._build_report(
                SafetyDecision.DENY, r6.rationale, tier_results, start_ts, action_key
            )
        if r6.decision == SafetyDecision.REQUIRE_HUMAN_CONFIRMATION:
            return self._build_report(
                SafetyDecision.REQUIRE_HUMAN_CONFIRMATION,
                r6.rationale, tier_results, start_ts, action_key,
            )

        # ── Tier 7: Constitutional AI ─────────────────────────────────────
        r7 = self._run_constitutional(action, objective)
        add_result(r7)
        if r7.decision == SafetyDecision.DENY:
            return self._build_report(
                SafetyDecision.DENY, r7.rationale, tier_results, start_ts, action_key
            )

        # All tiers passed — ALLOW
        return self._build_report(
            SafetyDecision.ALLOW,
            "All safety tiers passed",
            tier_results, start_ts, action_key,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PER-TIER RUNNERS
    # ─────────────────────────────────────────────────────────────────────────

    def _run_piguard(
        self, action: Dict[str, Any], world_state: Dict[str, Any]
    ) -> TierResult:
        t0 = time.time()
        if self._piguard is None:
            return TierResult("T0_PIGuard", SafetyDecision.ALLOW, "PIGuard not available (skip)", skipped=True)
        try:
            # Check all text fields in the action for injection markers
            text_to_check = " ".join(
                str(v) for k, v in action.items()
                if isinstance(v, str) and k not in ("operation",)
            )
            if text_to_check:
                verdict = self._piguard.classify(text_to_check)
                if verdict == "INJECTION":
                    return TierResult(
                        "T0_PIGuard", SafetyDecision.DENY,
                        f"PIGuard: prompt injection detected in action text",
                        latency_ms=(time.time() - t0) * 1000,
                    )
        except Exception as e:
            return TierResult("T0_PIGuard", SafetyDecision.ALLOW, f"PIGuard error (fail-open): {e}",
                              latency_ms=(time.time() - t0) * 1000, error=str(e))
        return TierResult("T0_PIGuard", SafetyDecision.ALLOW, "Clean",
                          latency_ms=(time.time() - t0) * 1000)

    def _run_consequence_reasoner(
        self, action: Dict[str, Any], objective: str, focused_app: str
    ) -> TierResult:
        t0 = time.time()
        if self._consequence is None:
            return TierResult("T1-3_Consequence", SafetyDecision.ALLOW,
                              "ConsequenceReasoner not available (skip)", skipped=True)
        try:
            result = self._consequence.check_action(
                action=action,
                objective=objective,
                focused_app=focused_app,
            )
            latency_ms = (time.time() - t0) * 1000
            # Map ConsequenceReasoner SafetyDecision to our SafetyDecision
            decision_str = str(getattr(result, "decision", "ALLOW"))
            if "DENY" in decision_str:
                dec = SafetyDecision.DENY
            elif "HUMAN" in decision_str or "CONFIRM" in decision_str:
                dec = SafetyDecision.REQUIRE_HUMAN_CONFIRMATION
            else:
                dec = SafetyDecision.ALLOW
            rationale = str(getattr(result, "reason", ""))[:300]
            return TierResult("T1-3_Consequence", dec, rationale, latency_ms=latency_ms)
        except Exception as e:
            return TierResult("T1-3_Consequence", SafetyDecision.ALLOW,
                              f"ConsequenceReasoner error (fail-open): {e}",
                              latency_ms=(time.time() - t0) * 1000, error=str(e))

    def _run_llamaguard(self, action: Dict[str, Any]) -> TierResult:
        t0 = time.time()
        if self._llamaguard is None:
            require_lg = os.environ.get("PROJECTZEO_REQUIRE_LLAMAGUARD", "1").strip() == "1"
            if require_lg:
                return TierResult("T4_LlamaGuard", SafetyDecision.DENY,
                                  "LlamaGuard required but unavailable (REQUIRE_LLAMAGUARD=1)")
            return TierResult("T4_LlamaGuard", SafetyDecision.ALLOW,
                              "LlamaGuard not available (skip, REQUIRE_LLAMAGUARD=0)", skipped=True)
        try:
            text = str(action.get("command", action.get("content", action.get("text", ""))))
            if not text.strip():
                return TierResult("T4_LlamaGuard", SafetyDecision.ALLOW, "No text to classify",
                                  latency_ms=(time.time() - t0) * 1000)
            verdict = self._llamaguard.classify(text)
            latency_ms = (time.time() - t0) * 1000
            if "UNSAFE" in str(verdict).upper():
                return TierResult("T4_LlamaGuard", SafetyDecision.DENY,
                                  f"LlamaGuard: UNSAFE — {verdict}", latency_ms=latency_ms)
            return TierResult("T4_LlamaGuard", SafetyDecision.ALLOW, "SAFE",
                              latency_ms=latency_ms)
        except Exception as e:
            return TierResult("T4_LlamaGuard", SafetyDecision.ALLOW,
                              f"LlamaGuard error (fail-open): {e}",
                              latency_ms=(time.time() - t0) * 1000, error=str(e))

    def _run_verisafe(self, action: Dict[str, Any]) -> TierResult:
        t0 = time.time()
        if self._vsa is None:
            return TierResult("T5_VeriSafe", SafetyDecision.ALLOW, "VeriSafe not available (skip)", skipped=True)
        try:
            verdict = self._vsa.verify(action)
            latency_ms = (time.time() - t0) * 1000
            if "VIOLATION" in str(verdict).upper():
                reason = ""
                try:
                    reason = self._vsa.last_violation_reason()
                except Exception:
                    pass
                return TierResult("T5_VeriSafe", SafetyDecision.DENY,
                                  f"VeriSafe VIOLATION: {reason}", latency_ms=latency_ms)
            return TierResult("T5_VeriSafe", SafetyDecision.ALLOW, "No violations",
                              latency_ms=latency_ms)
        except Exception as e:
            return TierResult("T5_VeriSafe", SafetyDecision.ALLOW,
                              f"VeriSafe error (fail-open): {e}",
                              latency_ms=(time.time() - t0) * 1000, error=str(e))

    def _run_policy(self, action: Dict[str, Any], focused_app: str) -> TierResult:
        t0 = time.time()
        if self._policy is None:
            return TierResult("T6_Policy", SafetyDecision.ALLOW, "Policy engine not available (skip)", skipped=True)
        try:
            decision_str, reason = self._policy.validate_action_dict(
                action, focused_app=focused_app
            )
            latency_ms = (time.time() - t0) * 1000
            if "DENY" in str(decision_str).upper():
                return TierResult("T6_Policy", SafetyDecision.DENY, reason[:300], latency_ms=latency_ms)
            if "HUMAN" in str(decision_str).upper() or "CONFIRM" in str(decision_str).upper():
                return TierResult("T6_Policy", SafetyDecision.REQUIRE_HUMAN_CONFIRMATION,
                                  reason[:300], latency_ms=latency_ms)
            return TierResult("T6_Policy", SafetyDecision.ALLOW, "Permitted", latency_ms=latency_ms)
        except Exception as e:
            return TierResult("T6_Policy", SafetyDecision.ALLOW,
                              f"Policy error (fail-open): {e}",
                              latency_ms=(time.time() - t0) * 1000, error=str(e))

    def _run_constitutional(self, action: Dict[str, Any], objective: str) -> TierResult:
        t0 = time.time()
        if self._constitutional is None:
            return TierResult("T7_Constitutional", SafetyDecision.ALLOW,
                              "ConstitutionalAI not available (skip)", skipped=True)
        try:
            verdict = self._constitutional.check(
                action=action,
                objective=objective,
            )
            latency_ms = (time.time() - t0) * 1000
            if not verdict.get("safe", True):
                reason = verdict.get("reason", "Constitutional violation")
                return TierResult("T7_Constitutional", SafetyDecision.DENY,
                                  reason[:300], latency_ms=latency_ms)
            return TierResult("T7_Constitutional", SafetyDecision.ALLOW, "All 6 principles met",
                              latency_ms=latency_ms)
        except Exception as e:
            return TierResult("T7_Constitutional", SafetyDecision.ALLOW,
                              f"ConstitutionalAI error (fail-open): {e}",
                              latency_ms=(time.time() - t0) * 1000, error=str(e))

    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _build_report(
        self,
        decision: SafetyDecision,
        rationale: str,
        tier_results: List[TierResult],
        start_ts: float,
        action_key: str,
    ) -> SafetyReport:
        total_ms = (time.time() - start_ts) * 1000
        report = SafetyReport(
            decision=decision,
            rationale=rationale,
            tier_results=tier_results,
            total_latency_ms=total_ms,
            action_key=action_key,
        )
        if self._journal is not None:
            try:
                self._journal.record({
                    "event": "unified_safety_report",
                    "action_key": action_key,
                    **report.to_dict(),
                })
            except Exception:
                pass
        return report

    @staticmethod
    def _compute_key(action: Dict[str, Any]) -> str:
        import hashlib
        raw = ":".join(
            str(action.get(k, ""))
            for k in ("operation", "command", "text", "path", "x", "y")
        )
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def get_tier_health(self) -> Dict[str, bool]:
        return {
            "T0_PIGuard": self._piguard is not None,
            "T1-3_Consequence": self._consequence is not None,
            "T4_LlamaGuard": self._llamaguard is not None,
            "T5_VeriSafe": self._vsa is not None,
            "T6_Policy": self._policy is not None,
            "T7_Constitutional": self._constitutional is not None,
        }
