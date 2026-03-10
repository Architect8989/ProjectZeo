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
    os.environ.get("PROJECTZEO_USE_MILESTONES", "1").strip() != "0"
)

_USE_WORLD_MODEL: bool = (
    os.environ.get("PROJECTZEO_USE_WORLD_MODEL", "1").strip() != "0"
)

_USE_SELF_MODEL: bool = (
    os.environ.get("PROJECTZEO_USE_SELF_MODEL", "1").strip() != "0"
)

class GIIMode:
    DISABLED = 0
    BASIC    = 1
    FULL     = 2

def get_gii_mode() -> int:
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
        "[SAFETY] Restoration scope: 5-TIER (CRIU process snapshot → BTRFS/rsync filesystem\n"
        "         → Playwright browser CDP state → wmctrl/xdotool window geometry → cursor focus)\n"
        "[SAFETY] NOTE: CRIU requires sudo/CAP_SYS_PTRACE. Browser snapshot requires Playwright.\n"
        "         Clipboard contents are NOT preserved across CRIU restore boundaries.",
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

        self._per_step_reasoner    = None
        self._consequence_reasoner = None

        self._semantic_memory    = None
        self._application_memory = None
        self._mem0_store         = None
        self._cognee_store       = None
        self._episodic_synthesizer = None
        self._memory_manager     = None

        self._world_model        = None
        self._self_model         = None

        self._operator_cycle     = None
        self._goal_repr          = None
        self._global_workspace   = None

        self._htn_planner        = None

        # GII-WIRE: VJEPAWorldModel initialised in _initialise_phase1_components
        # and wired to both ConsequenceReasoner and HTNPlanner
        self._vjepa_world_model  = None

        self._trajectory_flywheel = None
        self._algorithm_distiller = None
        self._soar_chunker        = None

        # ── DICP: Direct In-Context Policy (Blueprint §9.1) ───────────────────
        self._dicp_engine        = None  # initialised in _initialise_phase3_components

        self._openmemory_store   = None

        self._active_inference   = None

        self._user_model         = None

        self._atspi_bridge       = None

        self._milestone_decomposer  = None
        self._milestones: list       = []
        self._current_milestone_idx: int = 0
        self._milestones_active: bool    = False

        self._task_start: float = time.time()
        self._lock = threading.Lock()
        self._denied_action_keys: set = set()
        self._outcome_call_count: int = 0
        self._last_checkpoint_call: int = 0
        # WIRE: LATS active flag — prevents Self-Refine from running concurrently
        # Blueprint §9.2 conflict table: LATS internally generates and critiques
        # its own search nodes; running Self-Refine during LATS rollouts adds
        # latency with no quality gain (both critique the same action space).
        self._lats_recovery_active: bool = False

        if self._enabled:
            self._initialise_components(memory_dir)

        _print_startup_safety_banner(
            gii_mode,
            consequence_active=self._consequence_reasoner is not None,
            milestones_active=self._milestones_active,
            world_model_active=self._world_model is not None,
            self_model_active=self._self_model is not None,
        )
        if self._enabled:
            self._initialise_phase1_components()
            self._initialise_phase3_components()
        _logger.info(
            "[GIIController] Initialised. mode=%d enabled=%s consequence=%s "
            "milestones=%s world_model=%s self_model=%s checkpoint_interval=%d "
            "dicp=%s",
            gii_mode, self._enabled,
            self._consequence_reasoner is not None,
            self._milestones_active,
            self._world_model is not None,
            self._self_model is not None,
            _EPISODIC_CHECKPOINT_INTERVAL,
            self._dicp_engine is not None,
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
        return self._consequence_reasoner

    @property
    def world_model(self):
        return self._world_model

    @property
    def self_model(self):
        return self._self_model

    @property
    def grounding_stack(self):
        """Six-tier grounding dispatcher (Blueprint §6.7)."""
        return getattr(self, "_grounding_stack", None)

    @property
    def sppo_trainer(self):
        """SPPO Self-Play trainer (Blueprint §12.1)."""
        return getattr(self, "_sppo_trainer", None)

    @property
    def scaffold_audit(self):
        """ScaffoldAudit for live action path checking."""
        return getattr(self, "_scaffold_audit", None)

    @property
    def dicp_engine(self):
        """DICP in-context policy engine (Blueprint §9.1)."""
        return self._dicp_engine

    @property
    def nl2gensym(self):
        """NL2GenSym dynamic SOAR rule generator (Blueprint §3.1)."""
        return getattr(self, "_nl2gensym", None)

    @property
    def tom_agent(self):
        """Theory of Mind agent with counterfactual reflection (Blueprint §3.3)."""
        return getattr(self, "_tom_agent", None)

    @property
    def agent_s2(self):
        """Agent S2 proactive planner with narrative memory (Blueprint §6.2)."""
        return getattr(self, "_agent_s2", None)

    @property
    def aguvis(self):
        """Aguvis pure-vision grounding adapter (Blueprint §6.4)."""
        return getattr(self, "_aguvis", None)

    @property
    def ponder_press(self):
        """Ponder & Press divide-and-conquer planner (Blueprint §6.5)."""
        return getattr(self, "_ponder_press", None)

    # GII-FIX: _initialise_components was called in __init__ but the def
    # declaration was missing — the entire method body was unreachable orphan
    # code (came after property definitions with no def header). This meant
    # ALL memory stores, ConsequenceReasoner, PerStepReasoner, WorldModel,
    # SelfModel, and MilestoneDecomposer were NEVER initialised despite the
    # elaborate try/except blocks below. Fixed by restoring the def.
    def _initialise_components(self, memory_dir: Optional[str] = None) -> None:

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
                "[GIIController] Mem0Store: available=%s", self._mem0_store._available
            )
        except Exception as exc:
            _logger.warning("[GIIController] Mem0Store init failed: %s", exc)

        try:
            from core.memory.cognee_store import CogneeStore
            self._cognee_store = CogneeStore.get_instance()
            _logger.info(
                "[GIIController] CogneeStore: available=%s", self._cognee_store._available
            )
        except Exception as exc:
            _logger.warning("[GIIController] CogneeStore init failed: %s", exc)

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
            # Late-wire VJEPAWorldModel if it was already initialised
            # (phase1 runs after _initialise_components, so this handles the
            # case where phase1 runs first in a different init order)
            if self._vjepa_world_model is not None:
                try:
                    self._consequence_reasoner.set_vjepa_world_model(self._vjepa_world_model)
                    self._vjepa_world_model.set_consequence_reasoner(self._consequence_reasoner)
                    _logger.info("[GIIController] V-JEPA ↔ CR late-wired in _initialise_components.")
                except Exception as _lw_exc:
                    _logger.debug("[GIIController] V-JEPA late-wire failed: %s", _lw_exc)
        except Exception as exc:
            _logger.warning("[GIIController] ConsequenceReasoner init failed: %s", exc)

        if _USE_WORLD_MODEL:
            try:
                from core.vision.world_graph import WorldGraph
                self._world_model = WorldGraph()
                _logger.info("[GIIController] WorldModel (WorldGraph) active.")
            except Exception as exc:
                _logger.warning("[GIIController] WorldModel init failed: %s", exc)

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
            return

        try:
            from core.memory.episodic_synthesizer import EpisodicSynthesizer
            self._episodic_synthesizer = EpisodicSynthesizer(llm_callable=self._llm)
            _logger.info(
                "[GIIController] EpisodicSynthesizer: checkpoint every %d iterations.",
                _EPISODIC_CHECKPOINT_INTERVAL,
            )
        except Exception as exc:
            _logger.warning("[GIIController] EpisodicSynthesizer init failed: %s", exc)

        try:
            from core.memory.memory_manager import MemoryManager
            self._memory_manager = MemoryManager(
                memory_dir=memory_dir,
                episodic_synthesizer=self._episodic_synthesizer,
                semantic_memory=self._semantic_memory,
            )
            _logger.info(
                "[GIIController] MemoryManager (MemGPT tiers) active: %s",
                self._memory_manager,
            )
        except Exception as exc:
            _logger.warning("[GIIController] MemoryManager init failed: %s", exc)

        # GII-FIX: Wire MemoryReconciler into MemoryManager so cross-store
        # conflicts are resolved before results reach the LLM context window.
        try:
            from core.memory.memory_reconciler import get_memory_reconciler
            _reconciler = get_memory_reconciler(llm_caller=self._llm)
            if self._memory_manager is not None:
                self._memory_manager._reconciler = _reconciler
                # Late-bind LLM so LLM arbitration is available
                _reconciler._llm = self._llm
                _reconciler._enable_llm = True
            _logger.info("[GIIController] MemoryReconciler wired into MemoryManager.")
        except Exception as _rec_exc:
            _logger.debug("[GIIController] MemoryReconciler wire failed (non-fatal): %s", _rec_exc)

        try:
            from core.cognition.active_inference import ActiveInferenceAgent
            self._active_inference = ActiveInferenceAgent(n_states=16, n_obs=32)
            _logger.info("[GIIController] ActiveInferenceAgent (FEP) active.")
        except Exception as exc:
            _logger.warning("[GIIController] ActiveInferenceAgent init failed: %s", exc)

        try:
            from core.cognition.user_model import UserModel
            # GII-FIX: Old code called UserModel(memory_dir=memory_dir) but
            # UserModel.__init__ only accepted state_path.  UserModel is now
            # patched to accept memory_dir= as well, deriving state_path from it.
            self._user_model = UserModel(memory_dir=memory_dir)
            # WIRE: Notify UserModel of new objective for urgency/frustration detection
            # Blueprint §12 — urgency signal skips expensive Self-Refine critique
            # when user says quickly / asap / hurry etc. in task description.
            try:
                self._user_model.on_objective_received(self._objective)
            except Exception:
                pass
            _logger.info("[GIIController] UserModel (ToM + urgency-adapt) active.")
        except Exception as exc:
            _logger.warning("[GIIController] UserModel init failed: %s", exc)

        try:
            from core.perception.atspi_bridge import ATSPIBridge
            self._atspi_bridge = ATSPIBridge()
            _logger.info("[GIIController] ATSPIBridge active.")
        except Exception as exc:
            _logger.debug("[GIIController] ATSPIBridge unavailable (non-fatal): %s", exc)

        if _USE_MILESTONES:
            try:
                from core.planner.milestone_decomposer import MilestoneDecomposer
                self._milestone_decomposer = MilestoneDecomposer(llm_callable=self._llm)
                partial = self._assess_partial_completion(self._objective)
                self._milestones = self._milestone_decomposer.decompose(
                    objective=self._objective,
                    partial_completion=partial,
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

    def _initialise_phase1_components(self) -> None:
        try:
            from core.memory.openmemory_store import OpenMemoryStore
            self._openmemory_store = OpenMemoryStore()
            _logger.info("[GIIController] OpenMemoryStore (5-sector) active.")
        except Exception as exc:
            _logger.warning("[GIIController] OpenMemoryStore init failed: %s", exc)

        # GII-WIRE: VJEPAWorldModel (Blueprint §13.2)
        # Initialise BEFORE ConsequenceReasoner and HTNPlanner so it can be
        # wired into both.  Also sets back-reference for notification flow.
        try:
            from adapters.vjepa_adapter import get_vjepa_world_model
            self._vjepa_world_model = get_vjepa_world_model(llm_callable=self._llm)
            _logger.info(
                "[GIIController] VJEPAWorldModel active. mode=%s",
                self._vjepa_world_model._mode,
            )
            # Wire to ConsequenceReasoner if already initialised
            if self._consequence_reasoner is not None:
                try:
                    self._consequence_reasoner.set_vjepa_world_model(self._vjepa_world_model)
                    self._vjepa_world_model.set_consequence_reasoner(self._consequence_reasoner)
                    _logger.info("[GIIController] V-JEPA ↔ ConsequenceReasoner cross-wired.")
                except Exception as _w_exc:
                    _logger.debug("[GIIController] V-JEPA→CR wire failed: %s", _w_exc)
        except Exception as exc:
            _logger.warning("[GIIController] VJEPAWorldModel init failed: %s", exc)
            self._vjepa_world_model = None

        # GII-WIRE: Wire SemanticMemory ACT-R active goal to root objective
        # so every memory query during planning gets spreading activation boost.
        if self._semantic_memory is not None:
            try:
                self._semantic_memory.set_active_goal(self._objective)
                _logger.debug("[GIIController] SemanticMemory.set_active_goal seeded at init.")
            except Exception as _sam_exc:
                _logger.debug("[GIIController] SemanticMemory.set_active_goal failed: %s", _sam_exc)

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

        try:
            from core.cognition.operator_cycle import OperatorCycle
            tswm = None
            try:
                from adapters.qwen3_vl_adapter import get_qwen3_vl, is_qwen3_vl_preferred
                if is_qwen3_vl_preferred():
                    tswm = get_qwen3_vl()
                    _logger.info("[GIIController] OperatorCycle: Qwen3-VL as TSWM.")
            except Exception:
                pass

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

        try:
            from core.planner.htn_planner import HTNPlanner
            self._htn_planner = HTNPlanner(
                llm_call=self._llm,
                objective=self._objective,
                consequence_reasoner=self._consequence_reasoner,
                # GII-WIRE: ACT-R spreading activation — HTNPlanner calls
                # semantic_memory.set_active_goal() for each sub-task during
                # decompose() so memory queries auto-boost task-relevant facts.
                semantic_memory=self._semantic_memory,
            )
            _logger.info("[GIIController] HTNPlanner active (with ACT-R SemanticMemory wiring).")
        except Exception as exc:
            _logger.warning("[GIIController] HTNPlanner init failed: %s", exc)

        try:
            from core.cognition.global_workspace import (
                GlobalWorkspace, PerceptionModule, MemoryModule, ReflectionModule,
                PlanningModule, SafetyModule,
            )
            self._global_workspace = GlobalWorkspace(objective=self._objective)
            _vision_rt = None
            try:
                from core.vision.vision_runtime import get_vision_runtime
                _vision_rt = get_vision_runtime()
            except Exception as _vr_exc:
                _logger.debug(
                    "[GIIController] Could not acquire vision_runtime for "
                    "PerceptionModule (non-fatal): %s", _vr_exc
                )
            self._global_workspace.register(PerceptionModule(_vision_rt))
            self._global_workspace.register(MemoryModule(self._openmemory_store))
            self._global_workspace.register(ReflectionModule(self._goal_repr))

            try:
                self._global_workspace.register(PlanningModule(self))
                _logger.debug("[GIIController] PlanningModule registered in GWT.")
            except Exception as _pm_exc:
                _logger.warning("[GIIController] PlanningModule registration failed (non-fatal): %s", _pm_exc)

            try:
                _cr = getattr(self, "_consequence_reasoner", None)
                self._global_workspace.register(SafetyModule(_cr))
                _logger.debug("[GIIController] SafetyModule registered in GWT (cr=%s).", _cr is not None)
            except Exception as _sm_exc:
                _logger.warning("[GIIController] SafetyModule registration failed (non-fatal): %s", _sm_exc)

            _logger.info(
                "[GIIController] GlobalWorkspace (GWT) active. "
                "Modules: PerceptionModule vision=%s, MemoryModule, ReflectionModule, "
                "PlanningModule, SafetyModule",
                _vision_rt is not None,
            )
        except Exception as exc:
            _logger.warning("[GIIController] GlobalWorkspace init failed: %s", exc)

    def _initialise_phase3_components(self) -> None:
        try:
            from core.learning.trajectory_flywheel import TrajectoryFlywheel
            self._trajectory_flywheel = TrajectoryFlywheel(
                self._llm,
                openmemory=self._openmemory_store,
            )
            _logger.info("[GIIController] TrajectoryFlywheel active.")
        except Exception as exc:
            _logger.warning("[GIIController] TrajectoryFlywheel init failed: %s", exc)

        try:
            from core.learning.algorithm_distillation import AlgorithmDistiller
            self._algorithm_distiller = AlgorithmDistiller(llm_call=self._llm)
            _logger.info("[GIIController] AlgorithmDistiller (in-context RL) active.")
        except Exception as exc:
            _logger.warning("[GIIController] AlgorithmDistiller init failed: %s", exc)

        # ── DICP: Direct In-Context Policy (Blueprint §9.1) ───────────────────
        # Accumulates within-task failure/success patterns and injects them as
        # a "policy addendum" into LLM prompts during operator selection.
        # Complements Algorithm Distillation: AD is cross-session/coarse-grained;
        # DICP is intra-task/fine-grained (sub-goal resolution level).
        self._dicp_engine = None
        try:
            from core.learning.dicp import DICPEngine
            self._dicp_engine = DICPEngine(
                objective=self._objective,
                app_context="",
                llm_caller=self._llm,
            )
            # Ingest any cross-session constraint patterns from AlgorithmDistiller
            if self._algorithm_distiller is not None:
                try:
                    _ad_ctx = self._algorithm_distiller.get_context_string(
                        self._objective, ""
                    )
                    n_ingested = self._dicp_engine.ingest_ad_constraints(_ad_ctx)
                    if n_ingested:
                        _logger.info(
                            "[GIIController] DICP ingested %d cross-session AD constraints.",
                            n_ingested,
                        )
                except Exception as _ad_exc:
                    _logger.debug("[GIIController] DICP AD ingest skipped: %s", _ad_exc)
            _logger.info("[GIIController] DICP Engine (in-context policy) active.")
        except Exception as exc:
            _logger.warning("[GIIController] DICP init failed: %s", exc)

        try:
            from core.learning.soar_chunking import SOARChunking as SOARChunker
            self._soar_chunker = SOARChunker(
                self._llm,
                self._openmemory_store,
            )
            _logger.info("[GIIController] SOARChunker active.")
        except Exception as exc:
            _logger.warning("[GIIController] SOARChunker init failed: %s", exc)

        # ── AgentQ: MCTS trajectory collection + DPO pair generation ─────────
        self._agent_q = None
        self._agent_q_task_count: int = 0
        self._agent_q_trigger_interval: int = int(
            os.environ.get("PROJECTZEO_AGENT_Q_TRIGGER_INTERVAL", "50")
        )
        try:
            from core.learning.agent_q import AgentQStore
            self._agent_q = AgentQStore()
            _logger.info(
                "[GIIController] AgentQ active. Auto-trigger every %d tasks.",
                self._agent_q_trigger_interval,
            )
        except Exception as exc:
            _logger.warning("[GIIController] AgentQ init failed: %s", exc)

        # ── Session Reflector ─────────────────────────────────────────────────
        self._session_reflector = None
        try:
            from core.cognition.session_reflector import SessionReflector
            self._session_reflector = SessionReflector(
                llm_call=self._llm,
                objective=self._objective,
                episodic_synthesizer=self._episodic_synthesizer,
                semantic_memory=self._semantic_memory,
            )
            import threading as _thr
            _thr.Thread(
                target=self._session_reflector.reflect_on_session_start,
                daemon=True,
                name="session-reflector",
            ).start()
            _logger.info("[GIIController] SessionReflector active (async session-start).")
        except Exception as exc:
            _logger.warning("[GIIController] SessionReflector init failed: %s", exc)

        # ── Progressive Neural Network (PNN) ──────────────────────────────────
        self._pnn = None
        try:
            from core.learning.progressive_nn import ProgressiveNeuralNetwork
            self._pnn = ProgressiveNeuralNetwork()
            _logger.info("[GIIController] ProgressiveNeuralNetwork (PNN) active.")
        except Exception as exc:
            _logger.debug("[GIIController] PNN not available (non-fatal): %s", exc)

        # ── SPPO Trainer: Self-Play Policy Optimization (Blueprint §12.1) ─────
        self._sppo_trainer = None
        try:
            from core.learning.sppo_trainer import get_sppo_trainer
            self._sppo_trainer = get_sppo_trainer(llm_callable=self._llm)
            _logger.info("[GIIController] SPPO Trainer (Self-Play PO) active.")
        except Exception as exc:
            _logger.warning("[GIIController] SPPO Trainer init failed: %s", exc)

        # ── Six-Tier Grounding Stack (Blueprint §6.7) ─────────────────────────
        self._grounding_stack = None
        try:
            from core.perception.grounding_stack import get_grounding_stack
            self._grounding_stack = get_grounding_stack(
                atspi_bridge=self._atspi_bridge,
                omniparser=None,
                uitars_runtime=None,
                llm_callable=self._llm,
            )
            _logger.info("[GIIController] GroundingStack (6-tier) active.")
        except Exception as exc:
            _logger.warning("[GIIController] GroundingStack init failed: %s", exc)

        # ── Scaffold Audit ─────────────────────────────────────────────────────
        self._scaffold_audit = None
        try:
            from core.safety.scaffold_audit import ScaffoldAudit
            self._scaffold_audit = ScaffoldAudit(journal=None)
            self._scaffold_audit.arm()
            _logger.info("[GIIController] ScaffoldAudit ARMED.")
        except Exception as exc:
            _logger.warning("[GIIController] ScaffoldAudit init failed: %s", exc)

        # ── SICA: Self-Improving Consequence Analysis (Blueprint §13.3) ────────
        # Wire LLM caller into the SICA singleton so it can propose policy rules
        # for UNCERTAIN consequence verdicts during this task.
        self._sica_proposer = None
        try:
            from core.safety.sica_policy_proposer import get_sica_proposer
            self._sica_proposer = get_sica_proposer(llm_caller=self._llm)
            _logger.info("[GIIController] SICA policy proposer active.")
        except Exception as exc:
            _logger.debug("[GIIController] SICA init failed (non-fatal): %s", exc)

        # ── NL2GenSym: Dynamic SOAR Rule Generation (Blueprint §3.1) ──────────
        self._nl2gensym = None
        try:
            from core.cognition.nl2gensym import NL2GenSym
            self._nl2gensym = NL2GenSym(llm_caller=self._llm)
            # WIRE: Pre-warm rule generation at task start (async, non-blocking)
            # This caches rules for the objective+app so the first SOAR step
            # doesn't incur rule-generation latency.
            try:
                import threading as _thr
                _thr.Thread(
                    target=self._nl2gensym.generate_operator_rules,
                    kwargs=dict(objective=objective[:200], app_context=""),
                    daemon=True,
                    name="nl2gensym-prewarm",
                ).start()
            except Exception:
                pass
            _logger.info("[GIIController] NL2GenSym (dynamic SOAR rules from NL) active.")
        except Exception as exc:
            _logger.warning("[GIIController] NL2GenSym init failed: %s", exc)

        # ── ToM-Agent: Theory of Mind (Blueprint §3.3) ────────────────────────
        self._tom_agent = None
        try:
            from core.cognition.tom_agent import ToMAgent, reset_tom_agent
            reset_tom_agent()  # Ensure fresh instance per task
            from core.cognition.tom_agent import ToMAgent
            self._tom_agent = ToMAgent(
                original_instruction=self._objective,
                llm_caller=self._llm,
            )
            _logger.info("[GIIController] ToMAgent (Theory of Mind + counterfactual) active.")
        except Exception as exc:
            _logger.warning("[GIIController] ToMAgent init failed: %s", exc)

        # ── Agent S2 Planner (Blueprint §6.2) ─────────────────────────────────
        self._agent_s2 = None
        try:
            from core.planner.agent_s2_planner import AgentS2Planner
            self._agent_s2 = AgentS2Planner(
                objective=self._objective,
                llm_caller=self._llm,
                htn_planner=self._htn_planner,
            )
            _logger.info("[GIIController] AgentS2Planner (proactive + narrative + MoG) active.")
        except Exception as exc:
            _logger.warning("[GIIController] AgentS2Planner init failed: %s", exc)

        # ── Aguvis: Pure-Vision Fallback Grounding (Blueprint §6.4) ───────────
        self._aguvis = None
        try:
            from core.vision.aguvis_adapter import AguvisAdapter
            self._aguvis = AguvisAdapter(llm_caller=self._llm)
            _logger.info("[GIIController] AguvisAdapter (pure-vision AT-SPI fallback) active.")
        except Exception as exc:
            _logger.warning("[GIIController] AguvisAdapter init failed: %s", exc)

        # ── Ponder & Press: Divide-and-Conquer Planning (Blueprint §6.5) ──────
        self._ponder_press = None
        try:
            from core.planner.ponder_press import PonderPress
            self._ponder_press = PonderPress(llm_caller=self._llm)
            _logger.info("[GIIController] PonderPress (divide-and-conquer reasoning) active.")
        except Exception as exc:
            _logger.warning("[GIIController] PonderPress init failed: %s", exc)

        # ── UnifiedMemoryOrchestrator (Blueprint §10 — NEW) ───────────────────
        # Coordinates all memory backends with startup reconciliation,
        # write fanout, RRF read fusion, and per-backend health monitoring.
        # Solves the split-brain bug when FAISS data exists but Qdrant starts.
        self._unified_memory = None
        try:
            from core.memory.unified_memory_orchestrator import UnifiedMemoryOrchestrator
            self._unified_memory = UnifiedMemoryOrchestrator(
                memory_dir=memory_dir,
                llm_callable=self._llm,
            )
            # Non-blocking startup reconciliation (FAISS→Qdrant migration)
            import threading as _um_thr
            _um_thr.Thread(
                target=self._unified_memory.startup_reconciliation,
                daemon=True,
                name="unified-memory-reconciliation",
            ).start()
            _logger.info("[GIIController] UnifiedMemoryOrchestrator active (startup reconciliation async).")
        except Exception as exc:
            _logger.warning("[GIIController] UnifiedMemoryOrchestrator init failed: %s", exc)

        # ── DualModeReasoner (Blueprint §4.3 — NEW) ───────────────────────────
        # Routes LLM calls to fast/instruct or deep/thinking tier based on
        # action reversibility, stagnation, and known-app status.
        self._dual_mode_reasoner = None
        try:
            from core.cognition.dual_mode_reasoner import DualModeReasoner
            _factory = None
            try:
                from adapters.factory import get_adapter_factory
                _factory = get_adapter_factory()
            except Exception:
                pass
            self._dual_mode_reasoner = DualModeReasoner(
                llm_callable=self._llm,
                adapter_factory=_factory,
            )
            _logger.info("[GIIController] DualModeReasoner (fast/deep tier routing) active.")
        except Exception as exc:
            _logger.warning("[GIIController] DualModeReasoner init failed: %s", exc)

        # ── ReasoningEngine (Blueprint §3 stagnation recovery — FIXED) ────────
        # Previously never triggered (dead code). Now activated at 50% of
        # max_stagnant threshold via GIILoop.should_activate() gate.
        # Wired here so GIILoop can find it via getattr(gii_controller, "_reasoning_engine").
        self._reasoning_engine = None
        try:
            from core.cognition.reasoning_engine import ReasoningEngine
            self._reasoning_engine = ReasoningEngine(
                llm_callable=self._llm,
            )
            _logger.info("[GIIController] ReasoningEngine (stagnation recovery) active.")
        except Exception as exc:
            _logger.warning("[GIIController] ReasoningEngine init failed: %s", exc)

        # ── UnifiedSafetyOrchestrator (Blueprint §13 — NEW) ──────────────────
        # Single dispatch for all 8 safety tiers with per-tier health tracking,
        # timing instrumentation, and fail-closed semantics.
        self._unified_safety = None
        try:
            from core.safety.unified_safety_orchestrator import UnifiedSafetyOrchestrator
            self._unified_safety = UnifiedSafetyOrchestrator(
                llm_callable=self._llm,
                policy_engine=None,  # wired in GIILoop via policy_engine param
                consequence_reasoner=self._consequence_reasoner,
                journal=None,
            )
            _logger.info("[GIIController] UnifiedSafetyOrchestrator active.")
        except Exception as exc:
            _logger.warning("[GIIController] UnifiedSafetyOrchestrator init failed: %s", exc)

    def _on_task_complete(self, success: bool, objective: str, app_context: str = "") -> None:
        """
        Called after every completed task. Triggers AgentQ auto-collection,
        PNN update, and DICP flush.
        """
        self._agent_q_task_count += 1

        # Reset LATS active flag on task boundary
        self._lats_recovery_active = False

        if (
            self._agent_q is not None
            and self._agent_q_task_count % self._agent_q_trigger_interval == 0
        ):
            try:
                from core.learning.lats_planner import LATSPlanner
                lats = LATSPlanner(llm_caller=self._llm)
                pairs = lats.get_dpo_pairs()
                if pairs:
                    self._agent_q.ingest_dpo_pairs(pairs, app_name=app_context)
                    _logger.info(
                        "[GIIController] AgentQ auto-triggered at task_count=%d: "
                        "%d DPO pairs ingested from LATS.",
                        self._agent_q_task_count, len(pairs),
                    )
            except Exception as exc:
                _logger.debug("[GIIController] AgentQ auto-trigger error: %s", exc)

        if self._pnn is not None and success:
            try:
                self._pnn.register_task_completion(
                    task_description=objective[:200],
                    app_context=app_context,
                )
            except Exception:
                pass

        # ── SICA flush: write pending policy proposals for human review ─────
        if self._sica_proposer is not None:
            try:
                n_flushed = self._sica_proposer.flush_pending_to_file()
                if n_flushed:
                    _logger.info(
                        "[GIIController] SICA flushed %d pending rules to disk.", n_flushed
                    )
            except Exception as _sica_flush_exc:
                _logger.debug("[GIIController] SICA flush error: %s", _sica_flush_exc)

        # ── DICP flush: reset intra-task policy accumulator on task boundary ──
        # Cross-session AD constraints are preserved; only within-task evidence
        # (failure patterns, stagnation counters, discovered constraints) resets.
        if self._dicp_engine is not None:
            try:
                self._dicp_engine.flush()
                _logger.debug("[GIIController] DICP engine flushed for new task.")
            except Exception as _dicp_flush_exc:
                _logger.debug("[GIIController] DICP flush error: %s", _dicp_flush_exc)

        # ── UnifiedMemory task lifecycle hook ─────────────────────────────────
        # Stores task outcome in ALL backends and triggers HippoRAG/Graphiti sync
        if hasattr(self, "_unified_memory") and self._unified_memory is not None:
            try:
                self._unified_memory.on_task_complete(
                    objective=objective,
                    success=success,
                    app_name=app_context,
                    duration_sec=time.time() - self._task_start,
                )
            except Exception as _um_exc:
                _logger.debug("[GIIController] UnifiedMemory task hook error: %s", _um_exc)

        # ── Reset ReasoningEngine activation state for new task ───────────────
        if hasattr(self, "_reasoning_engine") and self._reasoning_engine is not None:
            try:
                self._reasoning_engine.reset_for_new_task()
            except Exception:
                pass

    def decide_next_action_operator_cycle(
        self,
        world_state: Dict[str, Any],
        *,
        screenshot=None,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        if self._operator_cycle is None or self._goal_repr is None:
            return None, "OperatorCycle not initialised"

        from core.cognition.operator_cycle import WorkingMemory

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

        if self._global_workspace is not None:
            try:
                gws_broadcast = self._global_workspace.run_cycle(
                    external_state={"entity_count": len(wm.entities),
                                    "focused_app": wm.focused_app,
                                    "objective": self._objective}
                )
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

        operator, impasse = self._operator_cycle.step(
            wm, self._goal_repr, screenshot=screenshot
        )

        if impasse is not None:
            resolution = self._operator_cycle.resolve_impasse(impasse, wm)
            if resolution:
                if resolution.get("operation") == "require_human_confirmation":
                    return None, f"Impasse REQUIRE_HUMAN: {impasse.description}"
                return resolution, f"Impasse resolved: {impasse.impasse_type}"
            return None, f"Impasse unresolved: {impasse.description}"

        if operator is None:
            return None, "OperatorCycle returned no operator"

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

        if self._goal_repr is not None:
            try:
                self._goal_repr.force_complete()
            except Exception:
                pass

    def _assess_partial_completion(self, objective: str) -> Optional[Dict[str, Any]]:
        if not objective:
            return None
        partial: Dict[str, Any] = {}
        if self._semantic_memory is not None:
            try:
                facts = self._semantic_memory.query(objective, max_results=5)
                if facts:
                    partial["semantic_context"] = [
                        {"s": f.subject, "p": f.predicate, "o": f.object_}
                        for f in facts[:5]
                        if hasattr(f, "subject")
                    ]
            except Exception:
                pass
        if self._mem0_store is not None:
            try:
                agent_id = getattr(self._mem0_store, "make_agent_id", lambda x: x)(objective)
                memories = self._mem0_store.search_memory(objective, agent_id, limit=3)
                if memories:
                    partial["cross_session_memories"] = [
                        str(m.get("memory") or m.get("text", ""))[:200]
                        for m in memories[:3]
                    ]
            except Exception:
                pass

        try:
            from core.perception.atbridge import get_atbridge
            _bridge = get_atbridge()
            if _bridge is not None:
                _snap = _bridge.get_screen_state()
                if _snap:
                    partial["screen_state"] = {
                        "focused_app": _snap.get("focused_app", ""),
                        "visible_text_sample": str(_snap.get("text", ""))[:300],
                        "entities_count": len(_snap.get("entities", [])),
                    }
        except Exception:
            pass

        return partial if partial else None

    def decide_next_action(
        self,
        world_state: Dict[str, Any],
        *,
        perception: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        if not self._enabled or self._per_step_reasoner is None:
            return None, "GII disabled"

        if self._pnn is not None:
            try:
                focused_app = str(world_state.get("focused_app", ""))
                lateral = self._pnn.get_lateral_context(
                    task_description=self._objective[:200],
                    app_context=focused_app,
                )
                if lateral and lateral.transfer_hints:
                    block = "\n".join(lateral.transfer_hints[:5])
                    self._per_step_reasoner.set_pnn_context(block)
            except Exception as _pnn_exc:
                _logger.debug("[GIIController] PNN lateral inject failed: %s", _pnn_exc)

        if self._active_inference is not None:
            try:
                self._active_inference.update_belief(world_state)
                _candidates: List[Dict[str, Any]] = world_state.get("_candidate_actions", [])
                if _candidates:
                    _ranked = self._active_inference.select_action(
                        _candidates,
                        world_state=world_state,
                        goal_description=self._objective,
                    )
                    if _ranked:
                        world_state = dict(world_state)
                        world_state["_active_inference_top_action"] = _ranked[0].action
                        world_state["_active_inference_efe"] = _ranked[0].efe
                        world_state["_active_inference_softmax"] = _ranked[0].softmax_prob
            except Exception as ai_exc:
                _logger.debug("[GIIController] ActiveInference error (non-fatal): %s", ai_exc)

        if self._user_model is not None:
            try:
                user_ctx = self._user_model.format_context()
                if user_ctx:
                    world_state = dict(world_state) if not isinstance(world_state, dict) else world_state
                    world_state.setdefault("_user_model_context", user_ctx[:400])
            except Exception:
                pass

        if self._global_workspace is not None:
            try:
                gwt_ctx = self._global_workspace.get_context_for_psr()
                if gwt_ctx:
                    world_state = dict(world_state) if not isinstance(world_state, dict) else world_state
                    world_state["_gwt_context"] = gwt_ctx
            except Exception as _gwt_exc:
                _logger.debug("[GIIController] GWT context injection failed (non-fatal): %s", _gwt_exc)

        # ── DICP policy addendum injection into world_state ───────────────────
        # The DICP addendum (failure patterns, stagnation warnings, discovered
        # constraints) is injected here so PSR sees it via world_state keys.
        # The loop also injects it directly before each decide call via
        # world_state["_dicp_policy_addendum"].
        if self._dicp_engine is not None:
            try:
                _dicp_addendum = self._dicp_engine.get_policy_addendum(
                    context={"world_state": world_state, "goal": self._objective},
                )
                if _dicp_addendum and "_dicp_policy_addendum" not in world_state:
                    world_state = dict(world_state) if not isinstance(world_state, dict) else world_state
                    world_state["_dicp_policy_addendum"] = _dicp_addendum
            except Exception as _dicp_exc:
                _logger.debug("[GIIController] DICP addendum inject failed: %s", _dicp_exc)

        try:
            return self._per_step_reasoner.next_action(
                world_state, perception=perception
            )
        except Exception as exc:
            _logger.error("[GIIController] decide_next_action error: %s", exc)
            return None, f"GII reasoning error: {exc}"

    def push_screenshot_for_grounding(self, screenshot_b64: str) -> None:
        if self._per_step_reasoner is not None:
            try:
                self._per_step_reasoner.push_screenshot(screenshot_b64)
            except Exception:
                pass

    def is_action_denied(self, action_key: str) -> bool:
        return action_key in self._denied_action_keys

    def record_denial(self, action_key: str) -> None:
        with self._lock:
            self._denied_action_keys.add(action_key)

    def check_action_with_scaffold_audit(self, action: Dict[str, Any]) -> bool:
        if self._scaffold_audit is None:
            return True
        try:
            from core.safety.scaffold_audit import AuditResult
            result = self._scaffold_audit.check_action(action)
            if result.decision == "BLOCK":
                _logger.critical(
                    "[GIIController] ScaffoldAudit BLOCKED action: op=%s reason=%s",
                    action.get("operation"), result.reason,
                )
                return False
        except Exception as _sa_exc:
            _logger.debug("[GIIController] ScaffoldAudit check error: %s", _sa_exc)
        return True

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

        if self._self_model is not None:
            try:
                self._self_model.record_action_result(
                    action=action, success=success, output=output
                )
            except Exception:
                pass

        if self._user_model is not None:
            try:
                # Update UserModel on approval/denial based on action outcome
                # on_approval / on_denial track per-app confidence for auto-approve
                op_str = str(action.get("operation", "unknown"))
                app_str = focused_app or ""
                if success:
                    self._user_model.on_approval(app_str, op_str)
                else:
                    self._user_model.on_denial(app_str, op_str)
            except Exception:
                pass

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

        if self._world_model is not None:
            try:
                wm_ctx = self._world_model.get_context_for_objective(self._objective)
                if wm_ctx:
                    parts.append(f"[World Model Context]\n{wm_ctx[:600]}")
            except Exception:
                pass

        if self._self_model is not None:
            try:
                sm_ctx = self._self_model.format_context()
                if sm_ctx:
                    parts.append(f"[Agent Self-Model]\n{sm_ctx[:400]}")
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
                lesson = ""
                if execution_log and isinstance(execution_log, dict):
                    outcomes = []
                    for _step_data in list(execution_log.values())[:5]:
                        if isinstance(_step_data, dict):
                            for _out in _step_data.get("outputs", [])[:2]:
                                _op = _out.get("operation", "")
                                _ok = _out.get("success", True)
                                outcomes.append(f"{'✓' if _ok else '✗'} {_op}")
                    lesson = "; ".join(outcomes[:5])
                profile = self._application_memory.get_profile(focused_app)
                if profile is not None:
                    profile.add_attempt(
                        objective=self._objective,
                        success=success,
                        lesson=lesson,
                        outcome_summary="success" if success else "failed",
                    )
            except Exception:
                pass

        if self._self_model is not None:
            try:
                self._self_model.record_task_outcome(
                    objective=self._objective,
                    success=success,
                    focused_app=focused_app,
                )
            except Exception:
                pass

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

        for mem in (self._semantic_memory, self._application_memory):
            if mem:
                try:
                    mem.save()
                except Exception:
                    pass

        # ── SPPO trajectory recording ─────────────────────────────────────────
        if self._sppo_trainer is not None:
            try:
                self._sppo_trainer.record_trajectory(
                    objective=self._objective,
                    app_context=focused_app or "",
                    execution_log=execution_log or {},
                    success=success,
                    duration_s=time.time() - self._task_start,
                )
            except Exception as _sppo_exc:
                _logger.debug("[GIIController] SPPO record failed: %s", _sppo_exc)

        # ── GRPO EWC sync ─────────────────────────────────────────────────────
        try:
            from core.learning.grpo_trainer import get_grpo_trainer
            _grpo = get_grpo_trainer()
            _grpo.sync_ewc_from_arpo()
        except Exception as _grpo_exc:
            _logger.debug("[GIIController] GRPO EWC sync failed (non-fatal): %s", _grpo_exc)

        if self._grounding_stack is not None:
            try:
                self._grounding_stack.update_llm(self._llm)
            except Exception:
                pass

        # ── DICP flush via _on_task_complete ──────────────────────────────────
        # Flush DICP intra-task evidence so the next task starts clean.
        # Cross-session AD constraints survive the flush.
        self._on_task_complete(
            success=success,
            objective=self._objective,
            app_context=focused_app or "",
        )

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
        if getattr(self, "_sppo_trainer", None) is not None:
            stats["sppo_active"] = True
            stats["sppo"] = self._sppo_trainer.get_stats()
        if getattr(self, "_grounding_stack", None) is not None:
            stats["grounding_stack_active"] = True
            stats["grounding_stack"] = self._grounding_stack.get_stats()
        if getattr(self, "_scaffold_audit", None) is not None:
            stats["scaffold_audit"] = self._scaffold_audit.get_stats()
        if getattr(self, "_pnn", None) is not None:
            stats["pnn_active"] = True
        # ── DICP stats ────────────────────────────────────────────────────────
        if self._dicp_engine is not None:
            stats["dicp_active"] = True
            stats["dicp"] = self._dicp_engine.get_stats()
        return stats
