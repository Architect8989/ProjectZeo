"""
core/gii/gii_controller.py — Central GII orchestrator.

GII UPGRADES (v3 — March 2026):
  ✅ PROJECTZEO_USE_MILESTONES now defaults to 1 (enabled).
     Milestone decomposition is the key difference between scripted and GII.
     The original default of 0 (disabled) meant the system never used its own
     goal decomposer. Fixed.

  ✅ WorldModel integration — persistent world model built from perception
     and injected into PerStepReasoner for every reasoning call.

  ✅ SelfModel integration — agent tracks its own capabilities, error rates,
     and learns what it does/doesn't know.

  ✅ Continuous vision during execution — after every action dispatch,
     PerStepReasoner.push_screenshot() is called with a fresh screen capture
     so the VL model always reasons on current visual state.

  ✅ PerStepReasoner failure escalation — if PSR fails to initialize,
     operator gets a LOUD error (not a silent degraded-mode warning).

  ✅ Mid-task episodic checkpoints every 50 iterations (configurable).
  ✅ LLM lesson synthesis post-task. 
  ✅ Denied action key tracking.
  ✅ Full Mem0 + Cognee memory integration.
"""
from __future__ import annotations

import logging
import os
import sys
import time
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPISODIC_CHECKPOINT_INTERVAL: int = max(
    1,
    int(os.environ.get("PROJECTZEO_EPISODIC_CHECKPOINT_INTERVAL", "50") or "50"),
)

# ARCH FIX: Milestones NOW DEFAULT TO ENABLED.
# Set PROJECTZEO_USE_MILESTONES=0 to disable.
_USE_MILESTONES: bool = (
    os.environ.get("PROJECTZEO_USE_MILESTONES", "1").strip() != "0"
)

# World model: persist facts about the environment across actions
_USE_WORLD_MODEL: bool = (
    os.environ.get("PROJECTZEO_USE_WORLD_MODEL", "1").strip() != "0"
)

# Self model: agent tracks its own capabilities
_USE_SELF_MODEL: bool = (
    os.environ.get("PROJECTZEO_USE_SELF_MODEL", "1").strip() != "0"
)


class GIIMode:
    DISABLED = 0
    BASIC    = 1
    FULL     = 2


def get_gii_mode() -> int:
    """Default is FULL (2). Consequence reasoning is not opt-in."""
    try:
        return int(os.environ.get("PROJECTZEO_GII_MODE", str(GIIMode.FULL)))
    except (ValueError, TypeError):
        return GIIMode.FULL


def _print_startup_safety_banner(
    gii_mode: int,
    consequence_active: bool,
    milestones_active: bool,
    world_model_active: bool,
    self_model_active: bool,
) -> None:
    mode_names = {
        GIIMode.DISABLED: "DISABLED (scripted only — no GII)",
        GIIMode.BASIC:    "BASIC (per-step reasoning, consequence reasoning INACTIVE)",
        GIIMode.FULL:     "FULL (per-step + consequence + milestone + world-model)",
    }
    mode_label = mode_names.get(gii_mode, f"UNKNOWN ({gii_mode})")
    print(
        f"\n[SAFETY] GII_MODE: {mode_label}\n"
        f"[SAFETY] Consequence Reasoning (Tier2+3): {'ACTIVE' if consequence_active else 'INACTIVE'}\n"
        f"[SAFETY] Milestone decomposition: {'ACTIVE' if milestones_active else 'INACTIVE'}\n"
        f"[SAFETY] Persistent world model: {'ACTIVE' if world_model_active else 'INACTIVE'}\n"
        f"[SAFETY] Agent self-model: {'ACTIVE' if self_model_active else 'INACTIVE'}\n"
        "[SAFETY] Restoration scope: cursor + window focus ONLY\n"
        "[SAFETY] NOTE: Restoration does NOT preserve browser tabs, clipboard,\n"
        "         unsaved docs, or terminal session.",
        file=sys.stderr,
    )
    if not consequence_active:
        print(
            "[SAFETY CRITICAL] Consequence reasoning INACTIVE. "
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
        self._llm       = llm_callable
        self._objective = objective
        self._scaffold_steps = scaffold_steps or []
        self._gii_mode  = gii_mode
        self._enabled   = gii_mode > GIIMode.DISABLED

        # Core reasoning components
        self._per_step_reasoner    = None
        self._consequence_reasoner = None

        # Memory tiers
        self._semantic_memory    = None
        self._application_memory = None
        self._mem0_store         = None
        self._cognee_store       = None
        self._episodic_synthesizer = None

        # NEW GII modules
        self._world_model        = None   # Persistent world model
        self._self_model         = None   # Agent self-model

        # Phase 1: SOAR Cognitive Loop (blueprint §3.3)
        self._operator_cycle     = None   # SOAR operator-selection cycle
        self._goal_repr          = None   # Structured goal representation
        self._global_workspace   = None   # GWT broadcaster (§2.3.4)

        # Phase 1: HTN Planning (blueprint §2.4.1)
        self._htn_planner        = None

        # Phase 3: Self-improving (blueprint §3.1, §3.2)
        self._trajectory_flywheel = None  # GUI-Owl self-evolution
        self._algorithm_distiller = None  # Algorithm Distillation (Laskin 2023)
        self._soar_chunker        = None  # SOAR chunking

        # Phase 0: OpenMemory 5-sector store (blueprint §2.5.2)
        self._openmemory_store   = None

        # Planning
        self._milestone_decomposer  = None
        self._milestones: list       = []
        self._current_milestone_idx: int = 0
        self._milestones_active: bool    = False

        # Telemetry
        self._task_start: float = time.time()
        self._lock = threading.Lock()
        self._denied_action_keys: set = set()
        self._outcome_call_count: int = 0
        self._last_checkpoint_call: int = 0

        if self._enabled:
            self._initialise_components(memory_dir)

        _print_startup_safety_banner(
            gii_mode,
            consequence_active=self._consequence_reasoner is not None,
            milestones_active=self._milestones_active,
            world_model_active=self._world_model is not None,
            self_model_active=self._self_model is not None,
        )
        # Phase 1 and Phase 3 components (non-blocking: failures are warnings)
        if self._enabled:
            self._initialise_phase1_components()
            self._initialise_phase3_components()
        _logger.info(
            "[GIIController] Initialised. mode=%d enabled=%s consequence=%s "
            "milestones=%s world_model=%s self_model=%s checkpoint_interval=%d",
            gii_mode, self._enabled,
            self._consequence_reasoner is not None,
            self._milestones_active,
            self._world_model is not None,
            self._self_model is not None,
            _EPISODIC_CHECKPOINT_INTERVAL,
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
        return self._enabled

    @property
    def gii_mode(self) -> int:
        return self._gii_mode

    @property
    def consequence_reasoner(self):
        return self._consequence_reasoner

    @property
    def world_model(self):
        return self._world_model

    @property
    def self_model(self):
        return self._self_model

    # =========================================================================
    # Initialisation
    # =========================================================================

    def _initialise_components(self, memory_dir: Optional[str]) -> None:

        # ── 1. SemanticMemory (universal fallback) ───────────────────────────
        try:
            from core.memory.semantic_memory import SemanticMemory
            self._semantic_memory = SemanticMemory(memory_dir=memory_dir)
        except Exception as exc:
            _logger.warning("[GIIController] SemanticMemory init failed: %s", exc)

        # ── 2. ApplicationMemory ─────────────────────────────────────────────
        try:
            from core.memory.application_memory import ApplicationMemory
            self._application_memory = ApplicationMemory(memory_dir=memory_dir)
        except Exception as exc:
            _logger.warning("[GIIController] ApplicationMemory init failed: %s", exc)

        # ── 3. Mem0Store — cross-session working memory ──────────────────────
        try:
            from core.memory.mem0_store import Mem0Store
            self._mem0_store = Mem0Store.get_instance()
            _logger.info(
                "[GIIController] Mem0Store: available=%s", self._mem0_store._available
            )
        except Exception as exc:
            _logger.warning("[GIIController] Mem0Store init failed: %s", exc)

        # ── 4. CogneeStore — knowledge-graph memory ──────────────────────────
        try:
            from core.memory.cognee_store import CogneeStore
            self._cognee_store = CogneeStore.get_instance()
            _logger.info(
                "[GIIController] CogneeStore: available=%s", self._cognee_store._available
            )
        except Exception as exc:
            _logger.warning("[GIIController] CogneeStore init failed: %s", exc)

        # ── 5. ConsequenceReasoner ───────────────────────────────────────────
        try:
            from core.safety.consequence_reasoner import ConsequenceReasoner
            enable_tier3 = self._gii_mode >= GIIMode.FULL
            self._consequence_reasoner = ConsequenceReasoner(
                llm_callable=self._llm,
                enable_tier2=True,
                enable_tier3=enable_tier3,
                auto_wire_endpoints=True,
            )
            _logger.info(
                "[GIIController] ConsequenceReasoner: tier2=True tier3=%s", enable_tier3
            )
        except Exception as exc:
            _logger.warning("[GIIController] ConsequenceReasoner init failed: %s", exc)

        # ── 6. WorldModel (NEW) ──────────────────────────────────────────────
        if _USE_WORLD_MODEL:
            try:
                from core.vision.world_graph import WorldGraph
                self._world_model = WorldGraph()
                _logger.info("[GIIController] WorldModel (WorldGraph) active.")
            except Exception as exc:
                _logger.warning("[GIIController] WorldModel init failed: %s", exc)

        # ── 7. SelfModel (NEW) ───────────────────────────────────────────────
        if _USE_SELF_MODEL:
            try:
                from core.cognition.self_model import SelfModel
                self._self_model = SelfModel(
                    agent_id=f"projectzeo_{os.getpid()}",
                    memory_dir=memory_dir,
                )
                _logger.info("[GIIController] SelfModel active.")
            except Exception as exc:
                _logger.warning("[GIIController] SelfModel init failed (non-fatal): %s", exc)

        # ── 8. PerStepReasoner ───────────────────────────────────────────────
        scaffold_dicts = []
        for step in self._scaffold_steps:
            if isinstance(step, dict):
                scaffold_dicts.append(step)
            elif hasattr(step, "description"):
                scaffold_dicts.append({
                    "description": getattr(step, "description", ""),
                    "type":        str(getattr(step, "type", "")),
                })
        try:
            from core.cognition.per_step_reasoner import PerStepReasoner
            self._per_step_reasoner = PerStepReasoner(
                llm_callable=self._llm,
                objective=self._objective,
                scaffold_steps=scaffold_dicts,
                application_memory=self._application_memory,
                semantic_memory=self._semantic_memory,
                consequence_reasoner=self._consequence_reasoner,
                world_model=self._world_model,
                self_model=self._self_model,
            )
            _logger.info("[GIIController] PerStepReasoner active.")
        except Exception as exc:
            # ARCH FIX: PSR failure is now a LOUD error, not a silent degradation.
            _logger.error(
                "[GIIController] CRITICAL: PerStepReasoner init FAILED: %s. "
                "GII will be DISABLED for this session. Check LLM adapter configuration.",
                exc,
            )
            print(
                f"[GIIController] CRITICAL: PerStepReasoner could not be initialised: {exc}\n"
                "GII is DISABLED. The system will fall back to scripted scaffold execution.\n"
                "This means consequence reasoning, memory, and adaptive replanning are ALL INACTIVE.\n"
                "Fix the LLM adapter and restart.",
                file=sys.stderr,
            )
            self._enabled = False
            return  # Stop initialisation — no point continuing

        # ── 9. EpisodicSynthesizer ───────────────────────────────────────────
        try:
            from core.memory.episodic_synthesizer import EpisodicSynthesizer
            self._episodic_synthesizer = EpisodicSynthesizer(llm_callable=self._llm)
            _logger.info(
                "[GIIController] EpisodicSynthesizer: checkpoint every %d iterations.",
                _EPISODIC_CHECKPOINT_INTERVAL,
            )
        except Exception as exc:
            _logger.warning("[GIIController] EpisodicSynthesizer init failed: %s", exc)

        # ── 10. MilestoneDecomposer (NOW DEFAULT-ON) ─────────────────────────
        if _USE_MILESTONES:
            try:
                from core.planner.milestone_decomposer import MilestoneDecomposer
                self._milestone_decomposer = MilestoneDecomposer(llm_callable=self._llm)
                self._milestones = self._milestone_decomposer.decompose(
                    objective=self._objective
                )
                if self._milestones:
                    self._milestones_active = True
                    self._inject_current_milestone()
                    _logger.info(
                        "[GIIController] MilestoneDecomposer: %d milestones: %s",
                        len(self._milestones),
                        [getattr(m, "name", str(m)) for m in self._milestones],
                    )
                else:
                    _logger.warning(
                        "[GIIController] MilestoneDecomposer returned 0 milestones — "
                        "falling back to scaffold execution."
                    )
            except Exception as exc:
                _logger.warning(
                    "[GIIController] MilestoneDecomposer init failed: %s. "
                    "Falling back to scaffold-based execution.", exc
                )
                self._milestones_active = False
        else:
            _logger.info(
                "[GIIController] MilestoneDecomposer disabled "
                "(PROJECTZEO_USE_MILESTONES=0)."
            )

    def _inject_current_milestone(self) -> None:
        if not self._milestones_active or not self._milestones:
            return
        if self._current_milestone_idx >= len(self._milestones):
            return
        milestone  = self._milestones[self._current_milestone_idx]
        condition  = getattr(milestone, "condition", str(milestone))
        signal     = getattr(milestone, "completion_signal", "")
        name       = getattr(milestone, "name", f"milestone_{self._current_milestone_idx + 1}")
        n_total    = len(self._milestones)
        n_current  = self._current_milestone_idx + 1

        sub_objective = (
            f"[Milestone {n_current}/{n_total}: {name}]\n"
            f"Achieve this observable condition: {condition}\n"
            f"Completion signal: {signal}\n"
            f"Full task context: {self._objective[:300]}"
        )
        if self._per_step_reasoner is not None:
            try:
                self._per_step_reasoner.update_objective(sub_objective)
                _logger.info(
                    "[GIIController] Milestone %d/%d injected: %r",
                    n_current, n_total, name,
                )
            except Exception as exc:
                _logger.debug("[GIIController] update_objective failed: %s", exc)

    def advance_milestone(self) -> bool:
        """
        Advance to the next milestone. Returns True if there are more
        milestones, False if all are complete.
        Called by operate.py when a 'done' action is received while
        milestones are still pending.
        """
        if not self._milestones_active:
            return False
        with self._lock:
            self._current_milestone_idx += 1
            if self._current_milestone_idx >= len(self._milestones):
                _logger.info("[GIIController] All %d milestones complete.", len(self._milestones))
                return False
        self._inject_current_milestone()
        _logger.info(
            "[GIIController] Advanced to milestone %d/%d.",
            self._current_milestone_idx + 1, len(self._milestones),
        )
        return True

    # =========================================================================
    # Phase 1: SOAR Cognitive Loop + GoalAct + GlobalWorkspace
    # =========================================================================

    def _initialise_phase1_components(self) -> None:
        """
        Initialise Phase 1 GII components:
          - GoalRepresentation (GoalAct structured goals)
          - OperatorCycle (SOAR operator selection)
          - GlobalWorkspace (GWT broadcaster)
          - HTNPlanner (hierarchical task network)
          - OpenMemoryStore (5-sector memory)
        """
        # OpenMemory 5-sector store (Phase 0.3)
        try:
            from core.memory.openmemory_store import OpenMemoryStore
            self._openmemory_store = OpenMemoryStore()
            _logger.info("[GIIController] OpenMemoryStore (5-sector) active.")
        except Exception as exc:
            _logger.warning("[GIIController] OpenMemoryStore init failed: %s", exc)

        # GoalRepresentation — structured, verifiable goal decomposition
        try:
            from core.cognition.goal_representation import GoalRepresentation
            self._goal_repr = GoalRepresentation(
                objective=self._objective,
                llm_call=self._llm,
            )
            _logger.info("[GIIController] GoalRepresentation active: %d sub-conditions.",
                         len(self._goal_repr._conditions))
        except Exception as exc:
            _logger.warning("[GIIController] GoalRepresentation init failed: %s", exc)

        # SOAR OperatorCycle — replaces scripted plan with per-step reasoning
        try:
            from core.cognition.operator_cycle import OperatorCycle
            # Try Qwen3-VL as TSWM if available
            tswm = None
            try:
                from adapters.qwen3_vl_adapter import get_qwen3_vl, is_qwen3_vl_preferred
                if is_qwen3_vl_preferred():
                    tswm = get_qwen3_vl()
                    _logger.info("[GIIController] OperatorCycle: Qwen3-VL as TSWM.")
            except Exception:
                pass

            # Try GUI-Actor for grounding
            gui_actor = None
            try:
                from adapters.gui_actor_adapter import get_gui_actor
                gui_actor = get_gui_actor()
                _logger.info("[GIIController] OperatorCycle: GUI-Actor grounding active.")
            except Exception:
                pass

            self._operator_cycle = OperatorCycle(
                llm_call=self._llm,
                tswm=tswm,
                gui_actor=gui_actor,
                openmemory=self._openmemory_store,
            )
            _logger.info("[GIIController] OperatorCycle (SOAR) active.")
        except Exception as exc:
            _logger.warning("[GIIController] OperatorCycle init failed: %s", exc)

        # HTN Planner
        try:
            from core.planner.htn_planner import HTNPlanner
            self._htn_planner = HTNPlanner(
                llm_call=self._llm,
                objective=self._objective,
            )
            _logger.info("[GIIController] HTNPlanner active.")
        except Exception as exc:
            _logger.warning("[GIIController] HTNPlanner init failed: %s", exc)

        # Global Workspace Theory broadcaster
        try:
            from core.cognition.global_workspace import (
                GlobalWorkspace, PerceptionModule, MemoryModule, ReflectionModule
            )
            self._global_workspace = GlobalWorkspace(objective=self._objective)
            self._global_workspace.register(MemoryModule(self._openmemory_store))
            self._global_workspace.register(ReflectionModule(self._goal_repr))
            _logger.info("[GIIController] GlobalWorkspace (GWT) active.")
        except Exception as exc:
            _logger.warning("[GIIController] GlobalWorkspace init failed: %s", exc)

    # =========================================================================
    # Phase 3: Self-improving Flywheel + Algorithm Distillation + SOAR Chunking
    # =========================================================================

    def _initialise_phase3_components(self) -> None:
        """
        Initialise Phase 3 self-improvement components:
          - TrajectoryFlywheel (GUI-Owl self-evolution)
          - AlgorithmDistiller (Laskin 2023 in-context RL)
          - SOARChunking (procedural memory from success)
        """
        # TrajectoryFlywheel
        try:
            from core.learning.trajectory_flywheel import TrajectoryFlywheel
            self._trajectory_flywheel = TrajectoryFlywheel(
                self._llm,
                openmemory=self._openmemory_store,
            )
            _logger.info("[GIIController] TrajectoryFlywheel active.")
        except Exception as exc:
            _logger.warning("[GIIController] TrajectoryFlywheel init failed: %s", exc)

        # Algorithm Distillation
        try:
            from core.learning.algorithm_distillation import AlgorithmDistiller
            self._algorithm_distiller = AlgorithmDistiller(llm_call=self._llm)
            _logger.info("[GIIController] AlgorithmDistiller (in-context RL) active.")
        except Exception as exc:
            _logger.warning("[GIIController] AlgorithmDistiller init failed: %s", exc)

        # SOAR Chunking
        try:
            from core.learning.soar_chunking import SOARChunking as SOARChunker
            self._soar_chunker = SOARChunker(
                self._llm,
                self._openmemory_store,
            )
            _logger.info("[GIIController] SOARChunker active.")
        except Exception as exc:
            _logger.warning("[GIIController] SOARChunker init failed: %s", exc)

    # =========================================================================
    # Phase 1: Operator-based action decision
    # =========================================================================

    def decide_next_action_operator_cycle(
        self,
        world_state: Dict[str, Any],
        *,
        screenshot=None,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        SOAR operator-selection cycle action decision.
        Used when OperatorCycle is active (Phase 1+).

        Returns the selected operator's action dict.
        """
        if self._operator_cycle is None or self._goal_repr is None:
            return None, "OperatorCycle not initialised"

        from core.cognition.operator_cycle import WorkingMemory

        # Build WorkingMemory from world_state
        wm = WorkingMemory(
            entities=world_state.get("entities", []),
            focused_app=world_state.get("focused_app", "unknown"),
            screen_desc=world_state.get("screen_description", ""),
            goal=self._goal_repr,
            active_milestones=(
                [getattr(m, "name", str(m)) for m in self._milestones[:3]]
                if self._milestones else []
            ),
        )

        # Update GlobalWorkspace with current observation
        if self._global_workspace is not None:
            try:
                gws_broadcast = self._global_workspace.run_cycle(
                    external_state={"entity_count": len(wm.entities),
                                    "focused_app": wm.focused_app,
                                    "objective": self._objective}
                )
                # Inject GWT reflection insights into world_state
                if gws_broadcast and gws_broadcast.winner:
                    from core.cognition.global_workspace import ModuleType
                    if gws_broadcast.winner.module_type == ModuleType.REFLECTION:
                        prog = gws_broadcast.winner.content.get("goal_progress", 0.0)
                        if gws_broadcast.winner.content.get("is_complete", False):
                            return (
                                {"operation": "done",
                                 "summary": f"Goal complete ({prog:.0%})"},
                                "GWT reflection: goal complete"
                            )
            except Exception as gws_exc:
                _logger.debug("[GIIController] GWT cycle error: %s", gws_exc)

        # Run SOAR operator cycle
        operator, impasse = self._operator_cycle.step(
            wm, self._goal_repr, screenshot=screenshot
        )

        if impasse is not None:
            # Resolve impasse — may create subgoal or require human confirmation
            resolution = self._operator_cycle.resolve_impasse(impasse, wm)
            if resolution:
                if resolution.get("operation") == "require_human_confirmation":
                    return None, f"Impasse REQUIRE_HUMAN: {impasse.description}"
                return resolution, f"Impasse resolved: {impasse.impasse_type}"
            return None, f"Impasse unresolved: {impasse.description}"

        if operator is None:
            return None, "OperatorCycle returned no operator"

        # GoalAct check: update goal progress
        if self._goal_repr is not None:
            try:
                self._goal_repr.evaluate_from_screen(world_state)
                if self._goal_repr.is_complete:
                    return (
                        {"operation": "done",
                         "summary": f"Goal complete: {self._goal_repr.progress_summary}"},
                        "GoalAct: all conditions satisfied"
                    )
            except Exception:
                pass

        # HTN GoalAct anti-stall check
        if self._htn_planner is not None:
            try:
                goalact = self._htn_planner.goalact_check(world_state)
                if goalact.get("stall_detected") and goalact.get("recommendation") == "replan":
                    _logger.info("[GIIController] GoalAct stall detected — triggering replan.")
                    world_state["_gii_loop_note"] = (
                        f"GoalAct: stall detected — {goalact.get('reason', '')}"
                    )
            except Exception:
                pass

        return operator.action, f"OperatorCycle: {operator.description[:100]}"

    def on_operator_success(
        self,
        executed_operators: list,
        focused_app: str = "",
    ) -> None:
        """
        Called after successful task completion to trigger SOAR chunking
        and Algorithm Distillation episode recording.
        """
        # SOAR chunking — store operator sequence as procedural memory
        if self._operator_cycle and executed_operators:
            try:
                self._operator_cycle.on_success(
                    successful_operators=executed_operators,
                    goal_description=self._objective,
                    app_context=focused_app,
                )
                _logger.info("[GIIController] SOAR chunk stored: %d operators.",
                             len(executed_operators))
            except Exception as exc:
                _logger.debug("[GIIController] SOAR chunking error: %s", exc)

        # Force goal complete
        if self._goal_repr is not None:
            try:
                self._goal_repr.force_complete()
            except Exception:
                pass

    # =========================================================================
    # Primary action decision
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

    def push_screenshot_for_grounding(self, screenshot_b64: str) -> None:
        """
        NEW: Push a fresh screenshot into the multi-frame grounding buffer.
        Called by operate.py after every action dispatch.
        """
        if self._per_step_reasoner is not None:
            try:
                self._per_step_reasoner.push_screenshot(screenshot_b64)
            except Exception:
                pass

    # =========================================================================
    # Denied action tracking
    # =========================================================================

    def is_action_denied(self, action_key: str) -> bool:
        return action_key in self._denied_action_keys

    def record_denial(self, action_key: str) -> None:
        with self._lock:
            self._denied_action_keys.add(action_key)

    # =========================================================================
    # Outcome recording + episodic checkpoints
    # =========================================================================

    def record_outcome(
        self,
        action: Dict[str, Any],
        *,
        success: bool,
        output: str = "",
        execution_log: Optional[Dict[str, Any]] = None,
        focused_app: Optional[str] = None,
    ) -> None:
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

        # Self-model update
        if self._self_model is not None:
            try:
                self._self_model.record_action_result(
                    action=action, success=success, output=output
                )
            except Exception:
                pass

        # Episodic mid-task checkpoint
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
                "[GIIController] Episodic checkpoint at call %d.", call_count
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
                _logger.warning("[GIIController] Checkpoint failed: %s", cp_exc)

    # =========================================================================
    # Planning context for prompt injection
    # =========================================================================

    def get_planning_context(self, focused_app: Optional[str] = None) -> str:
        if not self._enabled:
            return ""
        parts = []

        if self._mem0_store is not None:
            try:
                agent_id = getattr(self._mem0_store, "make_agent_id",
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
            except Exception:
                pass

        _knowledge_sourced = False
        if self._cognee_store is not None and self._cognee_store._available:
            try:
                results = self._cognee_store.search(self._objective, limit=8)
                if results:
                    lines = ["[Learned knowledge (Cognee)]"]
                    for r in results[:8]:
                        text = r.get("text") or r.get("object") or str(r)
                        lines.append(f"  • {str(text)[:200]}")
                    parts.append("\n".join(lines))
                    _knowledge_sourced = True
            except Exception:
                pass

        if not _knowledge_sourced and self._semantic_memory:
            try:
                facts = self._semantic_memory.query(self._objective, max_results=8)
                ctx = self._semantic_memory.format_for_prompt(facts)
                if ctx:
                    parts.append(ctx)
            except Exception:
                pass

        if self._application_memory and focused_app:
            try:
                ctx = self._application_memory.format_profile_for_prompt(focused_app)
                if ctx:
                    parts.append(ctx)
            except Exception:
                pass

        # World model context
        if self._world_model is not None:
            try:
                wm_ctx = self._world_model.get_context_for_objective(self._objective)
                if wm_ctx:
                    parts.append(f"[World Model Context]\n{wm_ctx[:600]}")
            except Exception:
                pass

        # Self model context
        if self._self_model is not None:
            try:
                sm_ctx = self._self_model.format_context()
                if sm_ctx:
                    parts.append(f"[Agent Self-Model]\n{sm_ctx[:400]}")
            except Exception:
                pass

        return "\n\n".join(parts)

    # =========================================================================
    # Task completion
    # =========================================================================

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

        # Self-model: record task outcome
        if self._self_model is not None:
            try:
                self._self_model.record_task_outcome(
                    objective=self._objective,
                    success=success,
                    focused_app=focused_app,
                )
            except Exception:
                pass

        # Mem0 post-task storage (background)
        if self._mem0_store is not None and execution_log is not None:
            def _store_mem0():
                try:
                    agent_id = getattr(
                        self._mem0_store, "make_agent_id", lambda x: x
                    )(self._objective)
                    messages = [
                        {"role": "user", "content": f"Task: {self._objective}"},
                        {"role": "assistant", "content": (
                            f"Task {'completed successfully' if success else 'failed'}. "
                            f"Summary: {str(execution_log)[:3000]}"
                        )},
                    ]
                    self._mem0_store.add_memory(
                        messages, agent_id,
                        metadata={"objective": self._objective[:200], "success": success}
                    )
                except Exception as exc:
                    _logger.debug("[GIIController] Mem0 post-task error: %s", exc)
            threading.Thread(target=_store_mem0, daemon=True).start()

        # CogneeStore knowledge graph update (background)
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
                except Exception as exc:
                    _logger.debug("[GIIController] CogneeStore post-task error: %s", exc)
            threading.Thread(target=_store_cognee, daemon=True).start()

        # LLM lesson synthesis → semantic memory
        if self._semantic_memory and execution_log:
            llm_succeeded = False
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
                    llm_succeeded = True
            except Exception as exc:
                _logger.warning("[GIIController] LLM lesson synthesis error: %s", exc)

            if not llm_succeeded:
                try:
                    self._extract_semantic_facts_from_log(execution_log, focused_app)
                except Exception:
                    pass

        # Persist memories
        for mem in (self._semantic_memory, self._application_memory):
            if mem:
                try:
                    mem.save()
                except Exception:
                    pass

        _logger.info("[GIIController] Task complete. success=%s mode=%d", success, self._gii_mode)

    def _extract_semantic_facts_from_log(
        self, execution_log: Dict[str, Any], focused_app: Optional[str]
    ) -> None:
        import re as _re
        install_re = _re.compile(
            r"(?:successfully installed|installation complete|already installed)", _re.IGNORECASE
        )
        version_re = _re.compile(r"(\w[\w\-]*)\s+(?:version\s+)?v?(\d+\.\d[\d.]*)", _re.IGNORECASE)
        error_re   = _re.compile(r"(?:error|failed|not found)[:\s]+(.{10,120})", _re.IGNORECASE)

        for step_idx, step_data in execution_log.items():
            if not isinstance(step_data, dict):
                continue
            for output_entry in step_data.get("outputs", []):
                output_text = str(output_entry.get("output", ""))[:2000]
                if not output_text:
                    continue
                if install_re.search(output_text) and focused_app:
                    self._semantic_memory.store(
                        subject=focused_app, predicate="install_outcome",
                        object_=f"success (step {step_idx})", category="install_outcomes",
                        confidence=0.9, source="observed",
                    )
                for m in version_re.finditer(output_text):
                    tool, version = m.group(1).lower(), m.group(2)
                    if len(tool) > 2 and len(version) > 1:
                        self._semantic_memory.store(
                            subject=tool, predicate="version", object_=version,
                            category="application_facts", confidence=0.95, source="observed",
                        )
                if not output_entry.get("success", True):
                    for m in error_re.finditer(output_text):
                        err = m.group(1).strip()[:100]
                        if focused_app:
                            self._semantic_memory.store(
                                subject=focused_app, predicate="known_error",
                                object_=err, category="error_solutions",
                                confidence=0.6, source="observed",
                            )

    def _synthesize_lessons_with_llm(
        self, execution_log: Dict[str, Any], focused_app: Optional[str]
    ) -> None:
        if self._llm is None or self._semantic_memory is None:
            return

        summary_parts = []
        for step_idx, step_data in execution_log.items():
            if not isinstance(step_data, dict):
                continue
            for output_entry in step_data.get("outputs", []):
                op       = str(output_entry.get("operation", "unknown"))
                success  = output_entry.get("success", True)
                snippet  = str(output_entry.get("output", ""))[:200]
                status   = "✓" if success else "✗"
                summary_parts.append(f"  Step {step_idx}: {status} {op}: {snippet}")

        if not summary_parts:
            return

        execution_summary = "\n".join(summary_parts[:30])
        prompt = (
            f"You just completed a task. Extract reusable lessons.\n\n"
            f"TASK: {self._objective[:500]}\n"
            f"APP: {focused_app or 'unknown'}\n\n"
            f"EXECUTION:\n{execution_summary}\n\n"
            "Extract 3-5 specific, reusable lessons about: what worked, failures/resolutions, "
            "app quirks, shortcuts discovered.\n"
            'Respond ONLY with a JSON array:\n'
            '[{"subject":"app","predicate":"lesson_type","object":"lesson text","confidence":0.8}]'
        )

        result_holder: List[Optional[str]] = [None]

        def _call():
            try:
                raw = self._llm(
                    messages=[{"role": "user", "content": prompt}],
                    objective=None,
                    session_id="lesson_synthesis",
                )
                if isinstance(raw, list) and raw:
                    result_holder[0] = str(
                        raw[0].get("content", "") if isinstance(raw[0], dict) else raw[0]
                    )
                elif isinstance(raw, str):
                    result_holder[0] = raw
            except Exception as exc:
                _logger.debug("[GIIController] Lesson LLM call failed: %s", exc)

        thread = threading.Thread(target=_call, daemon=True)
        thread.start()
        thread.join(timeout=180.0)

        if result_holder[0] is None:
            return

        try:
            import re as _re, json as _json
            clean = _re.sub(r"```(?:json)?", "", result_holder[0]).strip()
            m = _re.search(r"\[.*\]", clean, _re.DOTALL)
            if not m:
                return
            lessons = _json.loads(m.group(0))
            if not isinstance(lessons, list):
                return
            for lesson in lessons[:5]:
                if not isinstance(lesson, dict):
                    continue
                subject    = str(lesson.get("subject", focused_app or "task"))[:80]
                predicate  = str(lesson.get("predicate", "lesson"))[:80]
                object_    = str(lesson.get("object", ""))[:300]
                confidence = float(lesson.get("confidence", 0.7))
                if object_ and 0.0 < confidence <= 1.0:
                    self._semantic_memory.store(
                        subject=subject, predicate=predicate, object_=object_,
                        category="llm_synthesized_lessons",
                        confidence=confidence, source="llm_synthesis",
                    )
        except Exception as parse_err:
            _logger.debug("[GIIController] Lesson parse failed: %s", parse_err)

    # =========================================================================
    # Stats
    # =========================================================================

    def get_stats(self) -> dict:
        stats = {
            "gii_mode":                self._gii_mode,
            "enabled":                 self._enabled,
            "task_duration_seconds":   round(time.time() - self._task_start, 1),
            "denied_action_keys":      len(self._denied_action_keys),
            "outcome_call_count":      self._outcome_call_count,
            "last_checkpoint_call":    self._last_checkpoint_call,
            "episodic_checkpoint_interval": _EPISODIC_CHECKPOINT_INTERVAL,
            "milestones_active":       self._milestones_active,
            "milestones_total":        len(self._milestones),
            "current_milestone_idx":   self._current_milestone_idx,
            "world_model_active":      self._world_model is not None,
            "self_model_active":       self._self_model is not None,
        }
        if self._per_step_reasoner:
            stats["per_step_reasoner"] = self._per_step_reasoner.get_stats()
        if self._consequence_reasoner:
            stats["consequence_reasoner"] = self._consequence_reasoner.get_stats()
        if self._semantic_memory:
            stats["semantic_memory"] = self._semantic_memory.stats()
        if self._application_memory:
            stats["application_memory"] = self._application_memory.stats()
        if self._world_model and hasattr(self._world_model, "stats"):
            stats["world_model"] = self._world_model.stats()
        if self._self_model and hasattr(self._self_model, "get_stats"):
            stats["self_model"] = self._self_model.get_stats()
        if self._operator_cycle is not None:
            stats["operator_cycle_active"] = True
        if self._goal_repr is not None:
            stats["goal_progress"] = self._goal_repr.progress
            stats["goal_complete"] = self._goal_repr.is_complete
        if self._global_workspace is not None:
            stats["global_workspace"] = self._global_workspace.get_stats()
        if self._htn_planner is not None:
            stats["htn_active"] = True
        if self._algorithm_distiller is not None:
            stats["algorithm_distillation_active"] = True
        if self._openmemory_store is not None:
            stats["openmemory_active"] = True
        return stats
