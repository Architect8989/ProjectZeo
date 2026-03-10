from __future__ import annotations

"""
gii_loop.py — GII Goal-Directed Execution Loop

PATCH HISTORY
─────────────────────────────────────────────────────────────────────────────
March 2026 — Audit fixes applied:
  CRITICAL  : REQUIRE_HUMAN_CONFIRMATION handler added with _wait_for_human_approval()
  HIGH      : Hard iteration ceiling (500)
  HIGH      : Pre-dispatch screen diff

March 2026 — Research integration patch:
  GAP-1     : SHA-256 nibble compare → imagehash.phash() perceptual hashing
  LAYER-2   : VeriSafe Agent pre-action formal verification (Research §3)
  LAYER-5   : PIGuard external-content injection filter (Research §6.2)
  LAYER-5   : AT-SPI window registry for popup attack detection (Research §6.3)
  LAYER-7   : ProcessFence context manager wraps execute block (Research §8.1)

March 2026 — GII completeness patch (this file):
  CRITICAL-BUG : Fixed return-inside-while bug. Line 690 (old) had
                 `return self._result(False, "Maximum iterations reached")`
                 INSIDE the while loop body under on_action_executed callback.
                 This caused the loop to return after the FIRST successful action
                 execution instead of continuing to the next iteration.
                 Fix: moved the max-iterations return OUTSIDE the while loop.

  WIRE-1    : Active Inference (FEP) candidate generation wired.
              _generate_ai_candidates() builds candidate actions from world
              state BEFORE PSR reasoning. ActiveInferenceAgent.select_action()
              ranks them by EFE and top candidate injected as
              world_state["_active_inference_top_action"].

  WIRE-2    : GlobalWorkspace (GWT) cycle wired into main loop.
              Each iteration calls _global_workspace.run_cycle() with current
              world state. Broadcast winner injected into world state so PSR
              sees memory, perception, and reflection competition results.

  WIRE-3    : SNN Event Processor wired to AT-SPI interrupt preprocessing.
              SNNEventProcessor.process_event() compresses burst events before
              queue insertion, reducing spurious re-observe cycles.

  WIRE-4    : UserModel ToM updated on every approval/denial event.
              BDI gate now has up-to-date user intent model for reconsideration.

  WIRE-5    : Active Inference update_from_outcome() called after each
              execution. Generative model A/B matrices learn from transitions.

  WIRE-6    : ActiveInferenceAgent.adapt_precision() called periodically
              based on stagnant_count / iteration ratio (success rate proxy).
"""

import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Set

_logger = logging.getLogger(__name__)

_DONE_OP = "done"

_APPROVAL_WAIT_POLL_INTERVAL = 2.0
_APPROVAL_WAIT_TIMEOUT       = 300.0
_HARD_ITERATION_CEILING      = 500

_PHASH_CHANGE_THRESHOLD: int = int(
    os.environ.get("PROJECTZEO_PHASH_THRESHOLD", "10")
)

_POPUP_SUSPICIOUS_WINDOW_AGE_SEC: float = float(
    os.environ.get("PROJECTZEO_POPUP_AGE_SEC", "3.0")
)

# How many iterations between Active Inference precision adaptation calls
_AI_PRECISION_ADAPT_INTERVAL: int = int(
    os.environ.get("PROJECTZEO_AI_ADAPT_INTERVAL", "10")
)

# How many candidate actions to generate for Active Inference ranking
_AI_CANDIDATE_COUNT: int = int(
    os.environ.get("PROJECTZEO_AI_CANDIDATES", "4")
)


def _compute_phash(img) -> Optional[int]:
    try:
        import imagehash
        return int(imagehash.phash(img))
    except ImportError:
        return None
    except Exception:
        return None

def _phash_distance(h1: Optional[int], h2: Optional[int]) -> int:
    if h1 is None or h2 is None:
        return 0
    return bin(h1 ^ h2).count("1")


class GIIGoalDirectedLoop:

    MAX_ITERATIONS_DEFAULT    = 500
    MAX_STAGNANT_DEFAULT      = 25
    MIN_LOOP_INTERVAL_SECONDS = 0.1

    def __init__(
        self,
        *,
        gii_controller,
        os_backend,
        world_graph,
        policy_engine,
        journal,
        objective: str,
        max_wallclock_seconds: int = 3600,
        max_iterations: int = MAX_ITERATIONS_DEFAULT,
        max_stagnant: int = MAX_STAGNANT_DEFAULT,
        watchdog=None,
        execute_decision_fn: Optional[Callable] = None,
        on_action_executed: Optional[Callable] = None,
        vsa_verifier=None,
        piguard=None,
        atspi_window_registry: Optional[Set[int]] = None,
        use_process_fence: bool = True,
    ) -> None:
        self._gii = gii_controller
        self._os_backend = os_backend
        self._world_graph = world_graph
        self._policy = policy_engine
        self._journal = journal
        self._objective = objective
        self._max_wallclock = max_wallclock_seconds
        self._max_iterations = min(max_iterations, _HARD_ITERATION_CEILING)
        self._max_stagnant = max_stagnant
        self._watchdog = watchdog
        self._execute_decision_fn = execute_decision_fn
        self._on_action_executed = on_action_executed

        self._vsa = vsa_verifier
        self._piguard = piguard
        self._atspi_window_registry: Set[int] = atspi_window_registry or set()
        self._use_process_fence = use_process_fence

        self._iteration = 0
        self._stagnant_count = 0
        self._visited_keys: Dict[str, int] = {}
        self._permanent_deny: set = set()

        self._last_reasoning_phash: Optional[int] = None

        self._goal_repr = getattr(gii_controller, "_goal_repr", None)
        self._operator_cycle = getattr(gii_controller, "_operator_cycle", None)
        self._algorithm_distiller = getattr(gii_controller, "_algorithm_distiller", None)
        self._executed_operators: list = []

        # ── Active Inference (FEP) — WIRE-1 ───────────────────────────────────
        # Pull active_inference from gii_controller so we can call
        # select_action() to rank candidates by EFE before PSR reasoning.
        self._active_inference = getattr(gii_controller, "_active_inference", None)
        self._prev_world_state_for_ai: Optional[Dict[str, Any]] = None

        # ── Global Workspace (GWT) — WIRE-2 ───────────────────────────────────
        self._global_workspace = getattr(gii_controller, "_global_workspace", None)

        # ── SNN Event Processor — WIRE-3 ──────────────────────────────────────
        self._snn_processor = None
        try:
            from core.perception.snn_event_processor import SNNEventProcessor
            self._snn_processor = SNNEventProcessor()
            _logger.info("[GIILoop] SNN Event Processor active.")
        except Exception as _snn_exc:
            _logger.debug("[GIILoop] SNN processor unavailable (non-fatal): %s", _snn_exc)

        try:
            from core.learning.reflexion_engine import get_global_reflexion_engine
            _llm_for_reflexion = getattr(gii_controller, "_llm_callable", None)
            self._reflexion = get_global_reflexion_engine(llm_caller=_llm_for_reflexion)
        except Exception:
            self._reflexion = None

        try:
            from core.cognition.bdi_gate import BDIGate
            self._bdi_gate = BDIGate()
        except Exception:
            self._bdi_gate = None

        try:
            from core.planner.lats_planner import LATSPlanner
            _cr = getattr(gii_controller, "consequence_reasoner", None)
            _llm_lats = getattr(gii_controller, "_llm_callable", None)
            self._lats = LATSPlanner(llm_caller=_llm_lats, consequence_reasoner=_cr)
        except Exception:
            self._lats = None

        try:
            from core.memory.knowledge_vault import get_global_knowledge_vault
            self._vault = get_global_knowledge_vault()
        except Exception:
            self._vault = None

        try:
            from core.agents.monitor_agent import MonitorAgent
            _atspi = getattr(gii_controller, "_atspi_bridge", None)
            self._monitor = MonitorAgent(atspi_bridge=_atspi)
            self._monitor.start()
        except Exception:
            self._monitor = None

        try:
            from core.agents.safety_agent import get_global_safety_agent
            _llm_safety = getattr(gii_controller, "_llm_callable", None)
            self._safety_agent = get_global_safety_agent(llm_caller=_llm_safety)
        except Exception:
            self._safety_agent = None

        try:
            from core.agents.validator_agent import ValidatorAgent
            _llm_validator = getattr(gii_controller, "_llm_callable", None)
            self._validator = ValidatorAgent(llm_caller=_llm_validator, os_backend=os_backend)
        except Exception:
            self._validator = None

        try:
            from core.learning.grounding_trainer import get_global_grounding_trainer
            self._grounding_trainer = get_global_grounding_trainer()
        except Exception:
            self._grounding_trainer = None

        self._current_milestone: Optional[str] = None
        self._milestone_trajectory: List[Dict[str, Any]] = []
        self._milestone_start_world: Optional[Dict[str, Any]] = None
        self._current_app: str = ""

        _logger.info(
            "[GIILoop] Components wired: reflexion=%s bdi=%s lats=%s vault=%s "
            "monitor=%s safety=%s validator=%s grounding=%s "
            "active_inference=%s global_workspace=%s snn=%s",
            self._reflexion is not None, self._bdi_gate is not None,
            self._lats is not None, self._vault is not None,
            self._monitor is not None, self._safety_agent is not None,
            self._validator is not None, self._grounding_trainer is not None,
            self._active_inference is not None, self._global_workspace is not None,
            self._snn_processor is not None,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Active Inference candidate generation (WIRE-1)
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_ai_candidates(
        self, world_state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate lightweight candidate actions from world state for Active
        Inference EFE ranking.  These are heuristic candidates derived from
        visible entities — the ranked winner is injected into world_state so
        PSR can use it as a starting-point hint.

        Candidate generation strategy:
          1. For each visible entity, propose a click action.
          2. Always include a verify and a done candidate.
          3. Include LATS recovery action if present in world_state.
        """
        candidates: List[Dict[str, Any]] = []

        entities = world_state.get("entities", []) or []
        for ent in entities[:_AI_CANDIDATE_COUNT]:
            if isinstance(ent, dict):
                x = float(ent.get("x", 0.5))
                y = float(ent.get("y", 0.5))
                label = str(ent.get("label") or ent.get("name") or "")
                candidates.append({
                    "operation": "click",
                    "x": x,
                    "y": y,
                    "thought": f"Click {label!r} to progress toward goal",
                    "_ai_candidate": True,
                })

        candidates.append({
            "operation": "verify",
            "thought": "Verify current state matches objective",
            "_ai_candidate": True,
        })
        candidates.append({
            "operation": "done",
            "summary": "Objective achieved",
            "thought": "Check if objective is complete",
            "_ai_candidate": True,
        })

        # Include LATS recovery action if available
        lats_action = world_state.get("_lats_recovery_action")
        if isinstance(lats_action, dict):
            candidates.append(dict(lats_action, _ai_candidate=True))

        return candidates

    def _run_active_inference(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run Active Inference EFE ranking and inject top candidate.
        Returns modified world_state dict (copy if modified).
        """
        if self._active_inference is None:
            return world_state

        try:
            candidates = self._generate_ai_candidates(world_state)
            if not candidates:
                return world_state

            ranked = self._active_inference.select_action(
                candidate_actions=candidates,
                goal_description=self._objective,
                world_state=world_state,
            )

            if ranked:
                ws = dict(world_state)
                top = ranked[0]
                ws["_active_inference_top_action"] = top.action
                ws["_active_inference_efe"] = top.efe
                ws["_active_inference_ambiguity"] = top.ambiguity
                ws["_active_inference_risk"] = top.risk
                _logger.debug(
                    "[GIILoop] AIF top action: op=%s efe=%.3f ambiguity=%.3f risk=%.3f",
                    top.action.get("operation"), top.efe, top.ambiguity, top.risk,
                )
                return ws
        except Exception as ai_exc:
            _logger.debug("[GIILoop] Active Inference error (non-fatal): %s", ai_exc)

        return world_state

    def _adapt_ai_precision(self) -> None:
        """Periodically adapt Active Inference precision from success rate."""
        if self._active_inference is None:
            return
        if self._iteration % _AI_PRECISION_ADAPT_INTERVAL != 0:
            return
        if self._iteration == 0:
            return
        success_rate = max(0.0, 1.0 - self._stagnant_count / max(1, self._iteration))
        try:
            self._active_inference.adapt_precision(success_rate)
        except Exception:
            pass

    def _update_ai_from_outcome(
        self,
        prev_world: Optional[Dict[str, Any]],
        action: Dict[str, Any],
        next_world: Dict[str, Any],
        success: bool,
    ) -> None:
        """Update AIF generative model from executed transition."""
        if self._active_inference is None or prev_world is None:
            return
        try:
            self._active_inference.update_from_outcome(
                prev_state=prev_world,
                action=action,
                next_state=next_world,
                success=success,
            )
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # Global Workspace cycle (WIRE-2)
    # ─────────────────────────────────────────────────────────────────────────

    def _run_global_workspace_cycle(
        self, world_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run one GWT competition cycle and inject broadcast winner into
        world_state so PSR and SOAR cycle can observe it.

        Winner types and their effects:
          REFLECTION winner w/ is_complete=True → inject done signal
          MEMORY winner → inject memories into _gwt_memories
          PERCEPTION winner → inject updated entity count
        """
        if self._global_workspace is None:
            return world_state

        try:
            from core.cognition.global_workspace import ModuleType

            broadcast = self._global_workspace.run_cycle(
                external_state={
                    "objective":    self._objective,
                    "entity_count": len(world_state.get("entities", [])),
                    "focused_app":  world_state.get("focused_app", ""),
                    "iteration":    self._iteration,
                }
            )

            if broadcast is None:
                return world_state

            ws = dict(world_state)
            ws["_gwt_cycle"] = broadcast.cycle
            ws["_gwt_winner"] = broadcast.winner.module_type.value
            ws["_gwt_winner_activation"] = broadcast.winner.activation

            if broadcast.winner.module_type == ModuleType.REFLECTION:
                content = broadcast.winner.content
                ws["_gwt_goal_progress"] = content.get("goal_progress", 0.0)
                if content.get("is_complete", False):
                    ws["_gwt_goal_complete"] = True
                    _logger.info(
                        "[GIILoop] GWT Reflection winner: goal complete at iter=%d",
                        self._iteration,
                    )

            elif broadcast.winner.module_type == ModuleType.MEMORY:
                memories = broadcast.winner.content.get("memories", [])
                if memories:
                    ws["_gwt_memories"] = "; ".join(str(m)[:100] for m in memories[:3])
                    _logger.debug(
                        "[GIILoop] GWT Memory winner: %d memories injected",
                        len(memories),
                    )

            elif broadcast.winner.module_type == ModuleType.PERCEPTION:
                obs = broadcast.winner.content.get("observation", {})
                if obs:
                    ws["_gwt_entity_count"] = broadcast.winner.content.get("entity_count", 0)

            return ws

        except Exception as gws_exc:
            _logger.debug("[GIILoop] GWT cycle error (non-fatal): %s", gws_exc)
            return world_state

    # ─────────────────────────────────────────────────────────────────────────
    # Main execution loop
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, start_ts: Optional[float] = None) -> Dict[str, Any]:
        start_ts = start_ts or time.time()
        last_loop_ts = 0.0

        self._journal.record({
            "event": "gii_goal_directed_loop_start",
            "objective": self._objective[:200],
            "max_wallclock": self._max_wallclock,
            "max_iterations": self._max_iterations,
            "hard_ceiling": _HARD_ITERATION_CEILING,
            "phash_threshold": _PHASH_CHANGE_THRESHOLD,
            "vsa_active": self._vsa is not None,
            "piguard_active": self._piguard is not None,
            "active_inference_active": self._active_inference is not None,
            "global_workspace_active": self._global_workspace is not None,
        })

        _logger.info(
            "[GIILoop] Starting. objective=%r max_iter=%d vsa=%s piguard=%s "
            "fence=%s aif=%s gwt=%s",
            self._objective[:80], self._max_iterations,
            self._vsa is not None, self._piguard is not None,
            self._use_process_fence,
            self._active_inference is not None,
            self._global_workspace is not None,
        )

        self._current_milestone = self._objective
        self._milestone_trajectory = []

        while self._iteration < self._max_iterations:

            elapsed = time.time() - start_ts
            if elapsed > self._max_wallclock:
                self._journal.record({"event": "gii_loop_timeout", "elapsed": elapsed})
                return self._result(False, "Wall-clock timeout exceeded")

            if self._watchdog is not None:
                try:
                    self._watchdog.check()
                except Exception as wd_exc:
                    self._journal.record({"event": "gii_loop_watchdog_violation", "reason": str(wd_exc)})
                    return self._result(False, f"Watchdog violation: {wd_exc}")

            now = time.time()
            if now - last_loop_ts < self.MIN_LOOP_INTERVAL_SECONDS:
                time.sleep(self.MIN_LOOP_INTERVAL_SECONDS - (now - last_loop_ts))
            last_loop_ts = time.time()

            self._iteration += 1

            # ── AT-SPI interrupt drain ─────────────────────────────────────
            _atspi_bridge = getattr(self._gii, "_atspi_bridge", None)
            if _atspi_bridge is not None:
                try:
                    _interrupts = _atspi_bridge.drain_interrupts()
                    if _interrupts:
                        # WIRE-3: Pass interrupts through SNN processor first
                        if self._snn_processor is not None:
                            try:
                                for interrupt in _interrupts:
                                    self._snn_processor.process_event(
                                        event_type=interrupt.get("event_type", ""),
                                        source_id=interrupt.get("source_id", 0),
                                        timestamp=interrupt.get("timestamp", time.time()),
                                    )
                                # Only re-observe if SNN fires (burst detected)
                                if not self._snn_processor.should_interrupt():
                                    _logger.debug(
                                        "[GIILoop] SNN suppressed AT-SPI burst "
                                        "(%d events below threshold)", len(_interrupts),
                                    )
                                    continue
                            except Exception as _snn_exc:
                                _logger.debug("[GIILoop] SNN error (non-fatal): %s", _snn_exc)

                        _logger.info(
                            "[GIILoop] AT-SPI interrupt(s) received (%d) — forcing re-observe (iter=%d)",
                            len(_interrupts), self._iteration,
                        )
                        self._journal.record({
                            "event": "atspi_direct_interrupt",
                            "iteration": self._iteration,
                            "interrupt_count": len(_interrupts),
                            "events": [i.get("event_type") for i in _interrupts[:5]],
                        })
                        self._last_reasoning_phash = None
                        self._iteration -= 1
                        continue
                except Exception as _iq_exc:
                    _logger.debug("[GIILoop] Interrupt queue drain error (non-fatal): %s", _iq_exc)

            try:
                world_state = self._get_world_state()
            except Exception as obs_exc:
                _logger.warning("[GIILoop] World state observation failed: %s", obs_exc)
                self._stagnant_count += 1
                if self._stagnant_count >= self._max_stagnant:
                    return self._result(False, "Stagnation limit reached (observation failures)")
                continue

            _, self._last_reasoning_phash = self._capture_phash()

            # ── WIRE-2: Run Global Workspace cycle ─────────────────────────
            world_state = self._run_global_workspace_cycle(world_state)

            # Check if GWT Reflection module declared goal complete
            if world_state.get("_gwt_goal_complete"):
                self._journal.record({
                    "event": "gii_loop_gwt_goal_complete",
                    "iteration": self._iteration,
                    "progress": world_state.get("_gwt_goal_progress", 0.0),
                })
                if self._monitor is not None:
                    try: self._monitor.stop()
                    except Exception: pass
                return self._result(True, "GWT Reflection: all goal conditions satisfied")

            # ── WIRE-1: Run Active Inference EFE ranking ───────────────────
            _prev_ws = dict(world_state)  # save for AIF outcome update
            world_state = self._run_active_inference(world_state)

            # ── Adapt AIF precision periodically ──────────────────────────
            self._adapt_ai_precision()

            try:
                _screenshot_b64 = None
                try:
                    _, _img = self._capture_phash()
                    if _img is not None:
                        import io as _io, base64 as _b64
                        _buf = _io.BytesIO()
                        _img.save(_buf, format="PNG")
                        _screenshot_b64 = _b64.b64encode(_buf.getvalue()).decode()
                except Exception:
                    pass

                _operator_cycle_active = getattr(self._gii, "_operator_cycle", None) is not None
                if _operator_cycle_active and hasattr(self._gii, "decide_next_action_operator_cycle"):
                    action, reason = self._gii.decide_next_action_operator_cycle(
                        world_state, screenshot=_screenshot_b64
                    )
                    if action is None:
                        _logger.debug(
                            "[GIILoop] SOAR impasse (%s) — PSR fallback.", reason[:80]
                        )
                        action, reason = self._gii.decide_next_action(world_state)
                    else:
                        _logger.debug(
                            "[GIILoop] SOAR→op=%s | %s", action.get("operation"), reason[:60]
                        )
                        if isinstance(action, dict) and action.get("operation") not in ("done", "wait"):
                            self._executed_operators.append(action)
                else:
                    action, reason = self._gii.decide_next_action(world_state)
            except Exception as gii_exc:
                _logger.warning("[GIILoop] GII decide error: %s", gii_exc)
                action, reason = None, str(gii_exc)

            if action is None:
                self._stagnant_count += 1
                if self._stagnant_count >= self._max_stagnant:
                    return self._result(False, f"Stagnation: GII returned no action. Last: {reason}")
                continue

            if action.get("operation") == _DONE_OP:
                summary = action.get("summary", "Task complete")
                self._journal.record({"event": "gii_goal_complete", "iteration": self._iteration, "summary": summary})
                _logger.info("[GIILoop] Goal complete: %s (iter=%d)", summary, self._iteration)

                # ── ValidatorAgent post-milestone verification ─────────────
                # Independent validation prevents false-positive goal completion.
                # If validator returns FAIL, we re-enter the loop with a hint.
                if self._validator is not None:
                    try:
                        val_result = self._validator.verify_milestone(
                            milestone_desc=self._current_milestone or self._objective,
                            world_state=self._get_world_state(),
                            objective=self._objective,
                            iteration=self._iteration,
                        )
                        verdict = getattr(val_result, "verdict", None)
                        verdict_str = verdict.value if hasattr(verdict, "value") else str(verdict)
                        self._journal.record({
                            "event": "validator_post_milestone",
                            "iteration": self._iteration,
                            "verdict": verdict_str,
                            "confidence": getattr(val_result, "confidence", 0.0),
                            "rationale": getattr(val_result, "rationale", "")[:200],
                        })
                        if verdict_str in ("fail", "uncertain"):
                            _logger.warning(
                                "[GIILoop] Validator says %s (conf=%.2f) — NOT marking complete. "
                                "Re-entering loop with validation feedback.",
                                verdict_str,
                                getattr(val_result, "confidence", 0.0),
                            )
                            # Inject validation feedback and continue
                            world_state["_validator_feedback"] = (
                                f"VALIDATION {verdict_str.upper()}: "
                                f"{getattr(val_result, 'rationale', 'Milestone not confirmed')[:300]}. "
                                "The task is NOT yet complete. Continue working."
                            )
                            self._stagnant_count = 0  # Reset stagnant count
                            self._iteration -= 1      # Don't count this as a wasted iteration
                            continue
                        _logger.info("[GIILoop] Validator confirmed PASS (conf=%.2f).", getattr(val_result, "confidence", 0.0))
                    except Exception as val_exc:
                        _logger.debug("[GIILoop] ValidatorAgent error (non-fatal): %s", val_exc)

                if self._monitor is not None:
                    try: self._monitor.stop()
                    except Exception: pass
                if self._grounding_trainer is not None:
                    try: self._grounding_trainer.on_episode_complete(success=True)
                    except Exception: pass

                # ── AgentQ task-completion hook ────────────────────────────
                _gii_ctrl = self._gii
                if hasattr(_gii_ctrl, "_on_task_complete"):
                    try:
                        _gii_ctrl._on_task_complete(
                            success=True,
                            objective=self._objective[:200],
                            app_context=self._current_app or "",
                        )
                    except Exception:
                        pass

                return self._result(True, summary)


            if self._piguard is not None and action.get("_external_content_source"):
                external_content = str(action.get("content") or action.get("text") or "")
                if external_content and self._piguard_check(external_content) == "INJECTION":
                    _logger.warning("[GIILoop] PIGuard: injection blocked in external content (iter=%d)", self._iteration)
                    self._journal.record({"event": "gii_loop_piguard_injection_blocked", "iteration": self._iteration})
                    self._stagnant_count += 1
                    if self._stagnant_count >= self._max_stagnant:
                        return self._result(False, "Stagnation: repeated PIGuard blocks")
                    continue

            if self._is_suspicious_popup(action, world_state):
                _logger.warning("[GIILoop] SUSPICIOUS POPUP (iter=%d) — escalating", self._iteration)
                self._journal.record({"event": "gii_loop_suspicious_popup", "iteration": self._iteration})
                approved = self._wait_for_human_approval(
                    action,
                    "SUSPICIOUS: dialog appeared without AT-SPI window:create event (possible pop-up attack)"
                )
                if not approved:
                    self._permanent_deny.add(self._compute_action_key(action))
                    self._stagnant_count += 1
                    if self._stagnant_count >= self._max_stagnant:
                        return self._result(False, "Stagnation: repeated suspicious popup blocks")
                    continue

            action_key = self._compute_action_key(action)
            if action_key in self._permanent_deny:
                self._stagnant_count += 1
                if self._stagnant_count >= self._max_stagnant:
                    return self._result(False, "Stagnation: permanently denied action repeated")
                continue

            visit_count = self._visited_keys.get(action_key, 0)
            self._visited_keys[action_key] = visit_count + 1
            if visit_count >= 3:
                self._stagnant_count += 1
                if self._stagnant_count >= self._max_stagnant:
                    return self._result(False, "Stagnation: same action repeated too many times")
                world_state["_gii_loop_note"] = (
                    f"WARNING: action {action_key!r} repeated {visit_count + 1} times. Choose DIFFERENT approach."
                )
                continue

            focused_app = world_state.get("focused_app", "__unknown_app__")
            self._current_app = focused_app

            if self._monitor is not None:
                try:
                    self._monitor.update_world_context(world_state)
                    monitor_alerts = self._monitor.summarize_for_primary_agent()
                    if monitor_alerts:
                        world_state["_monitor_alerts"] = monitor_alerts
                        _logger.info("[GIILoop] Monitor alerts injected at iter=%d", self._iteration)
                except Exception as mon_exc:
                    _logger.debug("[GIILoop] Monitor update error: %s", mon_exc)

            if self._bdi_gate is not None and self._milestone_start_world is not None:
                try:
                    bdi_result = self._bdi_gate.should_reconsider(
                        world_state, active_goal=self._current_milestone or self._objective
                    )
                    if bdi_result.should_reconsider:
                        _logger.info(
                            "[GIILoop] BDI reconsideration: %s (iter=%d)",
                            bdi_result.reason[:80], self._iteration,
                        )
                        self._journal.record({
                            "event": "bdi_reconsider",
                            "iteration": self._iteration,
                            "reason": bdi_result.reason,
                            "jaccard": bdi_result.jaccard_similarity,
                            "divergence_type": bdi_result.divergence_type,
                        })
                        self._milestone_start_world = world_state
                        if self._bdi_gate is not None:
                            self._bdi_gate.reset_commitment()
                        world_state["_bdi_replan_signal"] = bdi_result.reason
                    else:
                        self._bdi_gate.update_actual_state(world_state)
                except Exception as bdi_exc:
                    _logger.debug("[GIILoop] BDI gate error: %s", bdi_exc)

            psr = getattr(self._gii, "_per_step_reasoner", None)
            if psr is not None:
                try:
                    if self._reflexion is not None and self._current_milestone:
                        reflex_ctx = self._reflexion.inject_context(self._current_milestone)
                        psr.set_reflexion_context(reflex_ctx)

                    if self._vault is not None:
                        vault_entries = self._vault.query_relevant(
                            f"{self._current_milestone or self._objective} {focused_app}",
                            max_results=3,
                            subject_filter=focused_app.lower() if focused_app else None,
                        )
                        if vault_entries:
                            vault_ctx = self._vault.format_for_prompt(vault_entries)
                            psr.set_vault_context(vault_ctx)
                except Exception as psr_exc:
                    _logger.debug("[GIILoop] PSR context injection error: %s", psr_exc)

            policy_decision, policy_reason = self._policy.validate_action_dict(action, focused_app=focused_app)

            if policy_decision == "DENY":
                self._permanent_deny.add(action_key)
                self._gii.record_denial(action_key)
                self._journal.record({"event": "gii_loop_policy_deny", "iteration": self._iteration, "action_key": action_key, "reason": policy_reason})
                _logger.warning("[GIILoop] Policy DENY: %s | %s", action_key, policy_reason)
                self._stagnant_count += 1
                if self._stagnant_count >= self._max_stagnant:
                    return self._result(False, "Stagnation: repeated policy denials")
                continue

            elif policy_decision == "REQUIRE_HUMAN_CONFIRMATION":
                self._journal.record({"event": "gii_loop_require_human_confirmation", "iteration": self._iteration, "action_key": action_key, "reason": policy_reason})
                _logger.warning("[GIILoop] REQUIRE_HUMAN_CONFIRMATION: %s | %s", action_key, policy_reason)

                # WIRE-4: Update BDI user model on confirmation event
                if self._bdi_gate is not None:
                    try:
                        self._bdi_gate.update_user_model(
                            f"Human confirmation requested for {action.get('operation','?')} operation"
                        )
                    except Exception:
                        pass

                approved = self._wait_for_human_approval(action, policy_reason)
                if not approved:
                    _logger.warning("[GIILoop] Human confirmation timeout for %s — denying.", action_key)
                    self._journal.record({"event": "gii_loop_human_confirmation_timeout", "iteration": self._iteration, "action_key": action_key})
                    self._permanent_deny.add(action_key)
                    self._stagnant_count += 1
                    if self._stagnant_count >= self._max_stagnant:
                        return self._result(False, "Stagnation: repeated unapproved confirmation requests")
                    continue
                self._journal.record({"event": "gii_loop_human_confirmation_approved", "iteration": self._iteration, "action_key": action_key})
                _logger.info("[GIILoop] Human confirmation approved for %s.", action_key)

            if self._vsa is not None:
                vsa_result = self._vsa_verify(action)
                if vsa_result == "VIOLATION":
                    vsa_feedback = ""
                    try:
                        vsa_feedback = self._vsa.last_violation_reason()
                    except Exception:
                        pass
                    _logger.warning("[GIILoop] VSA VIOLATION (iter=%d): %s", self._iteration, vsa_feedback)
                    self._journal.record({"event": "gii_loop_vsa_violation", "iteration": self._iteration, "feedback": vsa_feedback})
                    world_state["_vsa_violation"] = (
                        f"FORMAL SPEC VIOLATION: {vsa_feedback}. "
                        "Your action violates a safety invariant from the task spec. "
                        "Choose a different action."
                    )
                    self._stagnant_count += 1
                    if self._stagnant_count >= self._max_stagnant:
                        return self._result(False, f"Stagnation: VSA violations. Last: {vsa_feedback}")
                    continue

            if self._last_reasoning_phash is not None:
                _, current_phash = self._capture_phash()
                dist = _phash_distance(self._last_reasoning_phash, current_phash)
                if dist > _PHASH_CHANGE_THRESHOLD:
                    _logger.info("[GIILoop] phash diff=%d > threshold=%d — re-observing.", dist, _PHASH_CHANGE_THRESHOLD)
                    self._journal.record({"event": "gii_loop_screen_diff_skip", "iteration": self._iteration, "phash_distance": dist})
                    continue

            # ── Scaffold Audit pre-dispatch check (Blueprint §16 wire) ──────────
            # ScaffoldAudit was initialised and armed but never called from
            # the live execution path. Now wired here: BLOCK before dispatch.
            if self._gii_controller is not None:
                try:
                    _audit = self._gii_controller.scaffold_audit
                    if _audit is not None:
                        _audit_ok = self._gii_controller.check_action_with_scaffold_audit(action)
                        if not _audit_ok:
                            _logger.critical(
                                "[GIILoop] ScaffoldAudit BLOCKED op=%s iter=%d",
                                action.get("operation"), self._iteration,
                            )
                            self._journal.record({
                                "event": "gii_loop_scaffold_audit_block",
                                "iteration": self._iteration,
                                "action_key": action_key,
                                "operation": action.get("operation"),
                            })
                            self._stagnant_count += 1
                            if self._stagnant_count >= self._max_stagnant:
                                return self._result(False, "Stagnation: repeated scaffold-audit blocks")
                            continue
                except Exception as _sa_exc:
                    _logger.debug("[GIILoop] ScaffoldAudit pre-check error: %s", _sa_exc)

            exec_success = False
            exec_output = ""
            try:
                if self._use_process_fence:
                    try:
                        from core.safety.process_fence import process_fence
                        with process_fence():
                            exec_success, exec_output = self._dispatch_action(action, world_state)
                    except ImportError:
                        exec_success, exec_output = self._dispatch_action(action, world_state)
                else:
                    exec_success, exec_output = self._dispatch_action(action, world_state)

                self._stagnant_count = 0
                self._journal.record({
                    "event": "gii_loop_action_executed",
                    "iteration": self._iteration,
                    "action_key": action_key,
                    "operation": action.get("operation"),
                    "success": exec_success,
                    "output_snippet": exec_output[:200],
                })

                if self._monitor is not None and self._monitor.interrupt_event.is_set():
                    self._monitor.clear_interrupt()
                    self._last_reasoning_phash = None
                    self._journal.record({
                        "event": "monitor_interrupt_post_action",
                        "iteration": self._iteration,
                        "action_key": action_key,
                        "operation": action.get("operation"),
                        "exec_success": exec_success,
                    })
                    _logger.info(
                        "[GIILoop] Monitor interrupt fired during action — forcing re-observe (iter=%d)",
                        self._iteration,
                    )
                    self._iteration -= 1
                    continue

                if self._current_milestone:
                    self._milestone_trajectory.append({
                        "operation": action.get("operation"),
                        "thought": action.get("thought", ""),
                        "outcome": "success" if exec_success else "failure",
                        "output": exec_output[:100],
                    })

                # UI-AGILE Grounding Data Recording (Blueprint §9.5)
                if self._grounding_trainer is not None:
                    _op = str(action.get("operation", "")).lower()
                    if _op in ("click", "navigate", "tap"):
                        try:
                            self._grounding_trainer.record_click(
                                instruction=str(
                                    action.get("thought", "")
                                    or action.get("description", "")
                                    or action.get("target_label", "")
                                    or self._current_milestone
                                    or self._objective
                                )[:300],
                                predicted_x=float(action.get("x", 0.5)),
                                predicted_y=float(action.get("y", 0.5)),
                                outcome="success" if exec_success else "failure",
                                app_name=self._current_app or "",
                                task_objective=self._objective[:200],
                            )
                        except Exception as _gt_exc:
                            _logger.debug("[GIILoop] Grounding record failed (non-fatal): %s", _gt_exc)

                # WIRE-5: Update AIF generative model from outcome
                _next_ws = self._get_world_state()
                self._update_ai_from_outcome(_prev_ws, action, _next_ws, exec_success)
                self._prev_world_state_for_ai = _next_ws

            except Exception as exec_exc:
                exec_success = False
                exec_output = str(exec_exc)
                _logger.warning("[GIILoop] Action execution failed: %s | %s", action_key, exec_exc)

                if (
                    self._reflexion is not None
                    and self._current_milestone
                    and self._stagnant_count >= 2
                ):
                    try:
                        traj_summary = "; ".join(
                            f"{s.get('operation','?')}: {s.get('thought','')[:60]}"
                            for s in self._milestone_trajectory[-5:]
                        )
                        self._reflexion.reflect_on_failure(
                            milestone_desc=self._current_milestone,
                            trajectory_summary=traj_summary or "No trajectory recorded",
                            failure_reason=exec_output[:200],
                            belief_state_summary=str(world_state.get("focused_app","")) + " - " + str(self._iteration),
                            app_name=self._current_app or None,
                        )
                        if self._vault is not None:
                            self._vault.store_failure_pattern(
                                f"In {self._current_app}: '{self._current_milestone[:100]}' failed — {exec_output[:150]}",
                                subject=self._current_app.lower() if self._current_app else "general",
                                importance=0.7,
                            )
                    except Exception as rfx_exc:
                        _logger.debug("[GIILoop] Reflexion store failed: %s", rfx_exc)

                if (
                    self._lats is not None
                    and self._stagnant_count >= 3
                    and self._current_milestone
                ):
                    try:
                        reflex_ctx = ""
                        if self._reflexion and self._current_milestone:
                            reflex_ctx = self._reflexion.inject_context(self._current_milestone)

                        lats_result = self._lats.recover(
                            milestone_desc=self._current_milestone,
                            world_snapshot=world_state,
                            objective=self._objective,
                            reflection_context=reflex_ctx,
                            previous_trajectory=self._milestone_trajectory[-5:],
                        )
                        if lats_result.success and lats_result.best_action:
                            _logger.info(
                                "[GIILoop] LATS recovery found action (score=%.2f) at iter=%d",
                                lats_result.best_prm_score, self._iteration,
                            )
                            self._journal.record({
                                "event": "lats_recovery_activated",
                                "iteration": self._iteration,
                                "milestone": self._current_milestone[:80],
                                "prm_score": lats_result.best_prm_score,
                                "elapsed_ms": lats_result.elapsed_ms,
                            })
                            world_state["_lats_recovery_action"] = lats_result.best_action
                            world_state["_lats_recovery_thought"] = lats_result.best_thought
                    except Exception as lats_exc:
                        _logger.debug("[GIILoop] LATS recovery failed: %s", lats_exc)

                self._stagnant_count += 1
                if self._stagnant_count >= self._max_stagnant:
                    return self._result(False, f"Stagnation: repeated execution failures. Last: {exec_exc}")

            try:
                self._gii.record_outcome(action, success=exec_success, output=exec_output)
            except Exception:
                pass

            if self._algorithm_distiller is not None:
                try:
                    focused_app = world_state.get("focused_app", "unknown")
                    obs_summary = (
                        f"App: {focused_app} | "
                        f"Entities: {len(world_state.get('entities', []))}"
                    )
                    ad_episode = getattr(self, "_ad_current_episode", None)
                    if ad_episode is None:
                        ad_episode = self._algorithm_distiller.create_episode(
                            task_type=self._objective[:80],
                            app_context=focused_app,
                        )
                        self._ad_current_episode = ad_episode
                    ad_episode.add_step(
                        observation=obs_summary,
                        action=action,
                        reward=1.0 if exec_success else 0.0,
                        outcome="success" if exec_success else "failure",
                    )

                    _psr = getattr(self._gii, "_per_step_reasoner", None)
                    if _psr is not None and hasattr(_psr, "set_algorithm_distillation_context"):
                        try:
                            _ad_context = self._algorithm_distiller.get_prompt_injection(
                                task_type=self._objective[:80],
                                app_context=focused_app,
                            )
                            if _ad_context:
                                _psr.set_algorithm_distillation_context(_ad_context)
                        except Exception as _ad_inj_exc:
                            _logger.debug(
                                "[GIILoop] AD prompt injection failed (non-fatal): %s",
                                _ad_inj_exc,
                            )
                except Exception:
                    pass

            if self._goal_repr is not None:
                try:
                    self._goal_repr.evaluate_from_screen(world_state)
                    if self._goal_repr.is_complete:
                        _logger.info("[GIILoop] GoalRepresentation: all conditions satisfied.")
                        self._journal.record({
                            "event": "gii_loop_goal_complete_via_goal_repr",
                            "iteration": self._iteration,
                            "progress": self._goal_repr.progress,
                        })
                        ad_ep = getattr(self, "_ad_current_episode", None)
                        if ad_ep and self._algorithm_distiller:
                            self._algorithm_distiller.finalize_episode(ad_ep, success=True)
                        return self._result(True, "All goal sub-conditions satisfied")
                except Exception:
                    pass

            if self._on_action_executed is not None:
                try:
                    self._on_action_executed(action, exec_success, exec_output, self._iteration)
                except Exception:
                    pass

            # ── IMPORTANT: continue to next iteration — DO NOT return here ──
            # The loop continues until: goal complete, timeout, stagnation,
            # or max_iterations reached (handled BELOW, outside the while loop).
            # BUG FIX: The original code had `return self._result(...)` here
            # INSIDE the while loop, causing exit after the first successful
            # action. This is now correctly placed outside the loop.

        # ── Loop exhausted (max_iterations reached) ────────────────────────
        # This is the ONLY place max-iterations is returned.
        # All other termination paths are returns INSIDE the loop above.
        _logger.info(
            "[GIILoop] Max iterations (%d) reached after %d stagnant cycles.",
            self._max_iterations, self._stagnant_count,
        )
        return self._result(False, f"Maximum iterations ({self._max_iterations}) reached")

    def _dispatch_action(self, action: Dict[str, Any], world_state: Dict[str, Any]):
        if self._execute_decision_fn is not None:
            result = self._execute_decision_fn(action, world_state)
            if isinstance(result, dict):
                return result.get("success", True), str(result.get("output", ""))
            return bool(result), ""
        return self._minimal_dispatch(action.get("operation", ""), action), ""

    def _vsa_verify(self, action: Dict[str, Any]) -> str:
        try:
            return self._vsa.verify(action)
        except Exception as exc:
            _logger.warning("[GIILoop] VSA error (fail-open): %s", exc)
            return "OK"

    def _piguard_check(self, text: str) -> str:
        try:
            return self._piguard.classify(text)
        except Exception:
            return "SAFE"

    def _is_suspicious_popup(self, action: Dict[str, Any], world_state: Dict[str, Any]) -> bool:
        if not self._atspi_window_registry:
            return False
        target_win_id = action.get("_target_window_id")
        if target_win_id is None:
            return False
        if target_win_id in self._atspi_window_registry:
            return False
        win_age = action.get("_target_window_age_sec", 999)
        return isinstance(win_age, (int, float)) and win_age < _POPUP_SUSPICIOUS_WINDOW_AGE_SEC

    def _capture_phash(self):
        try:
            try:
                import pyautogui as _pya
                img = _pya.screenshot().resize((320, 180))
            except Exception:
                import tempfile as _tf
                from operate.utils.screenshot import capture_screen_with_cursor as _cap
                from PIL import Image as _PILImage
                tmp = _tf.NamedTemporaryFile(suffix=".png", delete=False)
                tmp_name = tmp.name; tmp.close()
                try:
                    _cap(tmp_name)
                    img = _PILImage.open(tmp_name).resize((320, 180))
                finally:
                    try: os.unlink(tmp_name)
                    except OSError: pass
            return img, _compute_phash(img)
        except Exception as exc:
            _logger.debug("[GIILoop] phash capture failed (non-fatal): %s", exc)
            return None, None

    def _wait_for_human_approval(self, action: Dict[str, Any], policy_reason: str) -> bool:
        import json as _json, secrets as _secrets, tempfile as _tempfile
        try:
            signal_dir = self._policy._APPROVAL_SIGNAL_DIR
        except AttributeError:
            signal_dir = _tempfile.gettempdir()
        token = _secrets.token_hex(8)
        signal_path = os.path.join(signal_dir, f"gii_approve_{token}.signal")
        approve_path = signal_path + ".APPROVE"
        op = str(action.get("operation", "?"))
        cmd_snippet = str(action.get("command", action.get("content", action.get("text", ""))))[:80]
        try:
            with open(signal_path, "w", encoding="utf-8") as sf:
                _json.dump({
                    "iteration": self._iteration,
                    "operation": op,
                    "command_snippet": cmd_snippet,
                    "reason": policy_reason[:300],
                    "objective": self._objective[:200],
                    "approve_by_creating": approve_path,
                }, sf, indent=2)
        except OSError as e:
            _logger.warning("[GIILoop] Cannot write signal file: %s — denying.", e)
            return False
        print(
            f"\n[GIILoop] ⚠  HUMAN APPROVAL REQUIRED (iter {self._iteration})\n"
            f"  Op: {op}\n  Cmd: {cmd_snippet!r}\n"
            f"  Reason: {policy_reason[:120]}\n"
            f"  Approve: CREATE {approve_path}\n",
            file=sys.stderr,
        )
        deadline = time.time() + _APPROVAL_WAIT_TIMEOUT
        approved = False
        try:
            while time.time() < deadline:
                if os.path.exists(approve_path):
                    approved = True; break
                time.sleep(_APPROVAL_WAIT_POLL_INTERVAL)
        finally:
            for p in (signal_path, approve_path):
                try: os.unlink(p)
                except OSError: pass
        return approved

    def _get_world_state(self) -> Dict[str, Any]:
        if self._world_graph is not None:
            try:
                snap = self._world_graph.snapshot()
                if isinstance(snap, dict):
                    return snap
            except Exception as exc:
                _logger.debug("[GIILoop] WorldGraph snapshot error: %s", exc)
        return {}

    def _compute_action_key(self, action: Dict[str, Any]) -> str:
        import hashlib
        raw = ":".join(str(action.get(k, "")) for k in ("operation", "command", "text", "path", "x", "y"))
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def _minimal_dispatch(self, op: str, action: Dict[str, Any]) -> bool:
        if self._os_backend is None:
            return False
        try:
            if op == "click":
                self._os_backend.click(float(action.get("x", 0.5)), float(action.get("y", 0.5)))
            elif op in ("write", "type"):
                self._os_backend.write(str(action.get("content", "")))
            elif op == "press":
                self._os_backend.press(action.get("keys", []))
            elif op == "command":
                r = self._os_backend.exec(str(action.get("command", "")))
                return r.returncode == 0 if hasattr(r, "returncode") else True
            return True
        except Exception as exc:
            _logger.warning("[GIILoop] Minimal dispatch error: %s", exc)
            return False

    def _result(self, success: bool, reason: str) -> Dict[str, Any]:
        ad_ep = getattr(self, "_ad_current_episode", None)
        if ad_ep is not None and self._algorithm_distiller is not None:
            try:
                self._algorithm_distiller.finalize_episode(ad_ep, success=success)
            except Exception:
                pass

        if success and hasattr(self._gii, "on_operator_success"):
            try:
                focused_app = self._get_world_state().get("focused_app", "")
                self._gii.on_operator_success(
                    executed_operators=self._executed_operators,
                    focused_app=focused_app,
                )
            except Exception:
                pass

        if self._monitor is not None:
            try:
                self._monitor.stop()
            except Exception:
                pass

        if self._grounding_trainer is not None:
            try:
                self._grounding_trainer.on_episode_complete(success=success)
            except Exception:
                pass

        return {
            "success": success,
            "reason": reason,
            "iterations": self._iteration,
            "stagnant_count": self._stagnant_count,
            "final_world_state": self._get_world_state(),
            "goal_progress": getattr(self._goal_repr, "progress", None),
            "active_inference_calls": getattr(self._active_inference, "_calls", 0) if self._active_inference else 0,
            "gwt_cycles": getattr(self._global_workspace, "_cycle", 0) if self._global_workspace else 0,
        }
