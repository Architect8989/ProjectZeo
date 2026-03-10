from __future__ import annotations

"""
core/cognition/global_workspace.py
====================================
Global Workspace Theory (GWT) implementation.

Blueprint §3.4 — Baars (1988); Goertzel (2023 LLM adaptation)

Architecture:
  A central "broadcast medium" where competing specialist modules place
  proposals for global attention. The highest-activation proposal wins and
  its content is broadcast to ALL modules simultaneously.

  This replaces ad-hoc inter-module communication (constructor injection,
  direct method calls) with a unified cognitive broadcast bus.

Modules registered (this patch adds PlanningModule and SafetyModule):
  PerceptionModule  — Qwen3-VL screen observations (activation: 0.65-0.85)
  MemoryModule      — OpenMemory cross-session retrieval (activation: 0.60)
  ReflectionModule  — GoalRepresentation progress monitor (activation: 0.40-0.99)
  PlanningModule    — HTNPlanner milestone status (activation: 0.55-0.90)  [NEW]
  SafetyModule      — ConsequenceReasoner alert injection (activation: 0.70-0.95) [NEW]

CHANGES (March 2026):
  - Added PlanningModule: surfaces next pending HTN milestone as broadcast.
    When a milestone completes, activation=0.90 forces it to win the cycle,
    notifying all modules of the plan advancement.
  - Added SafetyModule: If ConsequenceReasoner has cached a DENY/CONFIRM result
    from a recent evaluation, it re-broadcasts with activation=0.95 to ensure
    the safety signal reaches PSR before the next action dispatch.
  - GlobalWorkspace.get_context_for_psr(): new method returns a human-readable
    context string for PSR prompt injection (GWT → per-step reasoning).
  - ModuleType enum extended with PLANNING and SAFETY values.
  - run_cycle() now updates internal workspace_state with winner content for
    subsequent cycles (GWT sliding state).
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

_logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Module types
# ─────────────────────────────────────────────────────────────────────────────

class ModuleType(str, Enum):
    PERCEPTION  = "perception"
    PLANNING    = "planning"
    MEMORY      = "memory"
    SAFETY      = "safety"
    MOTOR       = "motor"
    REFLECTION  = "reflection"


# ─────────────────────────────────────────────────────────────────────────────
# Broadcast proposal / global broadcast
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BroadcastProposal:
    """A module's bid to broadcast to the global workspace."""
    module_type:  ModuleType
    content:      Dict[str, Any]          # What the module wants to broadcast
    activation:   float                   # 0.0-1.0; highest wins
    timestamp:    float = field(default_factory=time.time)
    metadata:     Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: "BroadcastProposal") -> bool:
        return self.activation < other.activation


@dataclass
class GlobalBroadcast:
    """Content broadcast to all modules after competition."""
    winner:       BroadcastProposal
    all_proposals: List[BroadcastProposal]
    cycle:        int
    broadcast_ts: float = field(default_factory=time.time)


# ─────────────────────────────────────────────────────────────────────────────
# Module base class
# ─────────────────────────────────────────────────────────────────────────────

class WorkspaceModule:

    module_type: ModuleType = ModuleType.PERCEPTION

    def propose(self, workspace_state: Dict[str, Any]) -> Optional[BroadcastProposal]:
        """Generate a proposal for the global workspace. Return None to abstain."""
        return None

    def receive(self, broadcast: GlobalBroadcast) -> None:
        """Process a broadcast from the global workspace."""
        pass

    def shutdown(self) -> None:
        """Clean up resources."""
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Concrete modules
# ─────────────────────────────────────────────────────────────────────────────

class PerceptionModule(WorkspaceModule):
    """Qwen3-VL perception: generates screen observations."""

    module_type = ModuleType.PERCEPTION

    def __init__(self, vision_runtime=None) -> None:
        self._vision = vision_runtime
        self._last_observation: Optional[Dict] = None
        self._screenshot = None

    def update_screenshot(self, screenshot) -> None:
        self._screenshot = screenshot

    def propose(self, workspace_state: Dict[str, Any]) -> Optional[BroadcastProposal]:
        if self._vision is None and self._screenshot is None:
            return None

        try:
            obs = {}
            if self._vision is not None and self._screenshot is not None:
                obs = self._vision.capture_and_analyze(self._screenshot)
            elif self._vision is not None:
                obs = self._vision.last_observation() if hasattr(self._vision, "last_observation") else {}

            if not obs:
                return None

            self._last_observation = obs
            prev_count = workspace_state.get("entity_count", 0)
            curr_count = len(obs.get("entities", []))
            activation = 0.85 if curr_count != prev_count else 0.65

            return BroadcastProposal(
                module_type=ModuleType.PERCEPTION,
                content={"observation": obs, "entity_count": curr_count},
                activation=activation,
            )
        except Exception as exc:
            _logger.debug("[GWS:Perception] propose error: %s", exc)
            return None

    def receive(self, broadcast: GlobalBroadcast) -> None:
        pass


class MemoryModule(WorkspaceModule):
    """OpenMemory retrieval: surfaces relevant memories for current context."""

    module_type = ModuleType.MEMORY

    def __init__(self, openmemory_store=None) -> None:
        self._memory = openmemory_store

    def propose(self, workspace_state: Dict[str, Any]) -> Optional[BroadcastProposal]:
        if self._memory is None:
            return None

        objective = workspace_state.get("objective", "")
        focused_app = workspace_state.get("focused_app", "")

        try:
            query = f"{objective} {focused_app}".strip()
            memories = self._memory.retrieve(query=query, top_k=5)
            if not memories:
                return None
            return BroadcastProposal(
                module_type=ModuleType.MEMORY,
                content={"memories": [m.content for m in memories[:5]]},
                activation=0.60,
            )
        except Exception as exc:
            _logger.debug("[GWS:Memory] propose error: %s", exc)
            return None

    def receive(self, broadcast: GlobalBroadcast) -> None:
        if broadcast.winner.module_type == ModuleType.MOTOR:
            action = broadcast.winner.content.get("action", {})
            if action and self._memory:
                try:
                    self._memory.store_episodic(
                        content=f"Executed action: {action}",
                        subject=broadcast.winner.metadata.get("app", "unknown"),
                        importance=0.5,
                    )
                except Exception:
                    pass


class ReflectionModule(WorkspaceModule):
    """GoalRepresentation evaluator: monitors goal progress."""

    module_type = ModuleType.REFLECTION

    def __init__(self, goal_repr=None) -> None:
        self._goal = goal_repr

    def propose(self, workspace_state: Dict[str, Any]) -> Optional[BroadcastProposal]:
        if self._goal is None:
            return None

        try:
            progress = self._goal.progress
            is_complete = self._goal.is_complete

            if is_complete:
                activation = 0.99   # highest — goal done, must broadcast
            elif progress > 0.0:
                activation = 0.55
            else:
                activation = 0.40

            next_pending_desc = ""
            try:
                next_cond = self._goal.next_pending()
                next_pending_desc = getattr(next_cond, "description", "")
            except Exception:
                pass

            return BroadcastProposal(
                module_type=ModuleType.REFLECTION,
                content={
                    "goal_progress":    progress,
                    "is_complete":      is_complete,
                    "progress_summary": getattr(self._goal, "progress_summary", ""),
                    "next_pending":     next_pending_desc,
                },
                activation=activation,
            )
        except Exception as exc:
            _logger.debug("[GWS:Reflection] propose error: %s", exc)
            return None

    def receive(self, broadcast: GlobalBroadcast) -> None:
        if broadcast.winner.module_type == ModuleType.PERCEPTION and self._goal:
            obs = broadcast.winner.content.get("observation", {})
            if obs:
                try:
                    self._goal.evaluate_from_screen(obs)
                except Exception:
                    pass


class PlanningModule(WorkspaceModule):
    """
    HTN milestone status monitor.

    NEW: Broadcasts milestone transitions with high activation (0.90) so
    that all other modules are notified when the plan advances.
    During normal execution emits at activation=0.55 with current milestone
    context for PSR to use as planning hint.
    """

    module_type = ModuleType.PLANNING

    def __init__(self, htn_planner=None, gii_controller=None) -> None:
        self._htn = htn_planner
        self._gii = gii_controller
        self._last_milestone_idx: int = -1
        self._last_emitted_milestone: str = ""

    def propose(self, workspace_state: Dict[str, Any]) -> Optional[BroadcastProposal]:
        gii = self._gii
        if gii is None:
            return None

        try:
            milestones = getattr(gii, "_milestones", [])
            idx = getattr(gii, "_current_milestone_idx", 0)
            active = milestones[idx] if milestones and idx < len(milestones) else None
            if active is None:
                return None

            milestone_name = getattr(active, "name", str(active))
            milestone_desc = getattr(active, "condition", str(active))

            # High activation on milestone transition
            milestone_changed = (idx != self._last_milestone_idx)
            activation = 0.90 if milestone_changed else 0.55

            if milestone_changed:
                _logger.info(
                    "[GWS:Planning] Milestone transition: idx=%d → %r",
                    idx, milestone_name,
                )
                self._last_milestone_idx = idx
                self._last_emitted_milestone = milestone_name

            return BroadcastProposal(
                module_type=ModuleType.PLANNING,
                content={
                    "current_milestone": milestone_name,
                    "milestone_condition": milestone_desc,
                    "milestone_idx": idx,
                    "total_milestones": len(milestones),
                    "milestone_changed": milestone_changed,
                },
                activation=activation,
            )
        except Exception as exc:
            _logger.debug("[GWS:Planning] propose error: %s", exc)
            return None

    def receive(self, broadcast: GlobalBroadcast) -> None:
        # If reflection says goal complete and we have milestones, try to advance
        if broadcast.winner.module_type == ModuleType.REFLECTION:
            if broadcast.winner.content.get("is_complete") and self._gii is not None:
                try:
                    advanced = self._gii.advance_milestone()
                    if advanced:
                        _logger.info("[GWS:Planning] Milestone auto-advanced on Reflection win.")
                        self._last_milestone_idx = -1  # force transition broadcast next cycle
                except Exception:
                    pass


class SafetyModule(WorkspaceModule):
    """
    ConsequenceReasoner alert broadcaster.

    NEW: If ConsequenceReasoner has a pending DENY or REQUIRE_HUMAN_CONFIRMATION
    result cached from a recent pre-dispatch evaluation, SafetyModule re-broadcasts
    it with activation=0.95 so PSR receives the safety context before the next
    reasoning cycle begins.

    This closes the gap where safety evaluations happened AFTER PSR reasoning
    but the feedback was not injected BEFORE the NEXT reasoning step.
    """

    module_type = ModuleType.SAFETY

    def __init__(self, consequence_reasoner=None) -> None:
        self._cr = consequence_reasoner
        self._last_alert_ts: float = 0.0
        self._alert_ttl: float = 5.0   # seconds before alert expires

    def propose(self, workspace_state: Dict[str, Any]) -> Optional[BroadcastProposal]:
        if self._cr is None:
            return None

        try:
            # Pull cached result from consequence reasoner if available
            last_result = getattr(self._cr, "_last_result", None)
            if last_result is None:
                return None

            from core.safety.consequence_reasoner import SafetyDecision
            decision = getattr(last_result, "decision", None)
            if decision not in (SafetyDecision.DENY, SafetyDecision.REQUIRE_HUMAN_CONFIRMATION):
                return None

            # Check TTL — don't re-broadcast stale alerts
            last_ts = getattr(last_result, "_evaluated_at", 0.0)
            if time.time() - last_ts > self._alert_ttl:
                return None

            activation = 0.95 if decision == SafetyDecision.DENY else 0.75

            return BroadcastProposal(
                module_type=ModuleType.SAFETY,
                content={
                    "safety_decision": decision.value,
                    "safety_reason":   getattr(last_result, "reason", "")[:200],
                    "tier_reached":    getattr(last_result, "tier_reached", 0),
                    "reversibility":   getattr(last_result, "reversibility", "UNKNOWN"),
                    "numeric_score":   getattr(last_result, "numeric_score", 0.5),
                },
                activation=activation,
            )
        except Exception as exc:
            _logger.debug("[GWS:Safety] propose error: %s", exc)
            return None

    def receive(self, broadcast: GlobalBroadcast) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# GlobalWorkspace
# ─────────────────────────────────────────────────────────────────────────────

_COLLECTION_WINDOW_SECONDS = 0.15   # time to collect proposals before arbitrating
_MAX_CYCLE_RATE_HZ         = 10     # cap at 10 arbitration cycles per second


class GlobalWorkspace:

    def __init__(self, objective: str = "") -> None:
        self._objective = objective
        self._modules:  List[WorkspaceModule] = []
        self._cycle:    int = 0
        self._lock:     threading.RLock = threading.RLock()
        self._last_broadcast: Optional[GlobalBroadcast] = None
        self._last_cycle_ts:  float = 0.0
        self._workspace_state: Dict[str, Any] = {"objective": objective}

        _logger.info("[GlobalWorkspace] Initialized. objective=%r", objective[:60])

    def register(self, module: WorkspaceModule) -> None:
        """Register a specialist module."""
        with self._lock:
            self._modules.append(module)
            _logger.debug("[GlobalWorkspace] Registered module: %s", module.module_type)

    def unregister(self, module_type: ModuleType) -> None:
        with self._lock:
            self._modules = [m for m in self._modules
                             if m.module_type != module_type]

    def update_workspace_state(self, updates: Dict[str, Any]) -> None:
        """Inject external state into workspace (e.g., world_state from GII loop)."""
        with self._lock:
            self._workspace_state.update(updates)

    def run_cycle(
        self,
        external_state: Optional[Dict[str, Any]] = None,
    ) -> Optional[GlobalBroadcast]:
        # Rate limiting
        now = time.time()
        min_interval = 1.0 / _MAX_CYCLE_RATE_HZ
        elapsed = now - self._last_cycle_ts
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        with self._lock:
            self._cycle += 1
            cycle = self._cycle
            state = dict(self._workspace_state)
            if external_state:
                state.update(external_state)
            modules = list(self._modules)

        if not modules:
            return None

        # Collect proposals in parallel
        proposals: List[Optional[BroadcastProposal]] = [None] * len(modules)
        threads = []

        def _propose(i: int, module: WorkspaceModule) -> None:
            try:
                proposals[i] = module.propose(state)
            except Exception as exc:
                _logger.debug("[GlobalWorkspace] Module %s propose error: %s",
                              module.module_type, exc)

        for i, module in enumerate(modules):
            t = threading.Thread(target=_propose, args=(i, module), daemon=True)
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=_COLLECTION_WINDOW_SECONDS)

        valid = [p for p in proposals if p is not None]
        if not valid:
            _logger.debug("[GlobalWorkspace] Cycle %d: no proposals.", cycle)
            return None

        # Arbitrate: highest activation wins
        winner = max(valid, key=lambda p: p.activation)

        broadcast = GlobalBroadcast(
            winner=winner,
            all_proposals=valid,
            cycle=cycle,
        )
        self._last_cycle_ts = time.time()

        # Update sliding workspace state with winner content
        with self._lock:
            self._workspace_state.update({
                f"_gwt_last_{winner.module_type.value}": winner.content,
                "_gwt_last_winner": winner.module_type.value,
            })

        _logger.debug(
            "[GlobalWorkspace] Cycle %d: winner=%s activation=%.2f proposals=%d",
            cycle, winner.module_type, winner.activation, len(valid),
        )

        # Broadcast to all modules in parallel
        def _recv(module: WorkspaceModule) -> None:
            try:
                module.receive(broadcast)
            except Exception as exc:
                _logger.debug("[GlobalWorkspace] Module %s receive error: %s",
                              module.module_type, exc)

        recv_threads = [
            threading.Thread(target=_recv, args=(m,), daemon=True)
            for m in modules
        ]
        for t in recv_threads:
            t.start()
        for t in recv_threads:
            t.join(timeout=_COLLECTION_WINDOW_SECONDS)

        with self._lock:
            self._last_broadcast = broadcast

        return broadcast

    def get_last_broadcast(self) -> Optional[GlobalBroadcast]:
        with self._lock:
            return self._last_broadcast

    def get_context_for_psr(self) -> str:
        """
        Returns a human-readable context string for PSR prompt injection.
        Summarizes the last broadcast winner content in plain text.
        """
        with self._lock:
            b = self._last_broadcast
        if b is None:
            return ""

        parts = []
        w = b.winner
        if w.module_type == ModuleType.PLANNING:
            c = w.content
            parts.append(
                f"[GWT/Planning] Milestone {c.get('milestone_idx',0)+1}/"
                f"{c.get('total_milestones',1)}: {c.get('current_milestone','')}"
                + (" [NEW]" if c.get("milestone_changed") else "")
            )
        elif w.module_type == ModuleType.SAFETY:
            c = w.content
            parts.append(
                f"[GWT/Safety] {c.get('safety_decision','')}: {c.get('safety_reason','')[:100]}"
            )
        elif w.module_type == ModuleType.MEMORY:
            mems = w.content.get("memories", [])
            if mems:
                parts.append(f"[GWT/Memory] Relevant: {'; '.join(str(m)[:80] for m in mems[:2])}")
        elif w.module_type == ModuleType.REFLECTION:
            c = w.content
            parts.append(
                f"[GWT/Reflection] Goal progress: {c.get('goal_progress',0):.0%}. "
                f"Next: {c.get('next_pending','')[:80]}"
            )

        # Include safety alert even if not winner
        for prop in b.all_proposals:
            if prop.module_type == ModuleType.SAFETY and prop != w:
                c = prop.content
                parts.append(
                    f"[GWT/Safety-alert] {c.get('safety_decision','')}: "
                    f"{c.get('safety_reason','')[:80]}"
                )
                break

        return "\n".join(parts)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "cycle":           self._cycle,
                "modules":         [m.module_type.value for m in self._modules],
                "objective":       self._objective[:60],
                "last_winner":     (
                    self._last_broadcast.winner.module_type.value
                    if self._last_broadcast else None
                ),
            }

    def shutdown(self) -> None:
        with self._lock:
            for m in self._modules:
                try:
                    m.shutdown()
                except Exception:
                    pass
            self._modules.clear()
