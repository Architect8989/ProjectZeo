import time
import json
from typing import Any, Dict, Optional, List

from authority.authority_policy import AuthorityDecision
from authority.input_arbitrator import InputArbitrator

from core.safety.action_timeout import action_timeout
from core.telemetry.logger import log_warn

from operate.utils.operating_system import OperatingSystem
from utils.accessibility import AccessibilityBackend
from audit.journal import ActionJournal

from core.schemas.execution_plan import ExecutionPlan
from core.verification.step_verifier import StepVerifier
from core.verification.plan_verifier import PlanVerifier
from core.execution.progress_tracker import ProgressTracker
from core.execution.failure_recovery import FailureRecoveryManager
from core.tools.autonomous_installer import AutonomousInstaller

from core.cognition.belief_state import BeliefState
from core.cognition.reasoning_engine import ReasoningEngine
from core.cognition.action_ranker import ActionRanker
from core.vision.semantic_resolver import SemanticResolver


MAX_PERCEPTION_ENTITIES = 20
MAX_PERCEPTION_JSON_BYTES = 10_000
MAX_STAGNANT_ITERS = 12


# ==================================================
# PUBLIC ENTRYPOINT
# ==================================================

def operate_main(
    *,
    terminal_prompt: str,
    execution_plan: ExecutionPlan,
    planner=None,
    observer=None,
    world_graph=None,
    max_wallclock_seconds: int = 90 * 60,
):

    if not isinstance(execution_plan, ExecutionPlan):
        raise ValueError("execution_plan must be ExecutionPlan")

    execution_plan.validate()
    PlanVerifier().verify(execution_plan)

    os_backend = OperatingSystem()

    try:
        accessibility_backend = AccessibilityBackend()
        if observer is not None:
            accessibility_backend.wire(observer=observer)
    except Exception:
        accessibility_backend = None

    journal = ActionJournal()
    input_arbitrator = InputArbitrator()
    verifier = StepVerifier()
    recovery = FailureRecoveryManager()
    progress = ProgressTracker(execution_plan)

    llm_callable = getattr(planner, "_llm_call", None)
    if not callable(llm_callable):
        raise RuntimeError("Planner LLM callable unavailable")

    installer = None
    if observer is not None:
        installer = AutonomousInstaller(
            observer=observer,
            os_backend=os_backend,
            llm_callable=llm_callable,
        )

    try:
        _execute_autonomous_loop(
            execution_plan=execution_plan,
            planner=planner,
            observer=observer,
            world_graph=world_graph,
            os_backend=os_backend,
            accessibility_backend=accessibility_backend,
            journal=journal,
            input_arbitrator=input_arbitrator,
            verifier=verifier,
            recovery=recovery,
            progress=progress,
            installer=installer,
            max_wallclock_seconds=max_wallclock_seconds,
        )
    finally:
        input_arbitrator.shutdown()


# ==================================================
# AUTONOMOUS LOOP
# ==================================================

def _execute_autonomous_loop(
    *,
    execution_plan: ExecutionPlan,
    planner,
    observer,
    world_graph,
    os_backend: OperatingSystem,
    accessibility_backend: Optional[AccessibilityBackend],
    journal: ActionJournal,
    input_arbitrator: InputArbitrator,
    verifier: StepVerifier,
    recovery: FailureRecoveryManager,
    progress: ProgressTracker,
    installer: Optional[AutonomousInstaller],
    max_wallclock_seconds: int,
):

    start_ts = time.time()
    progress.start_execution()

    journal.record({
        "event": "execution_start",
        "objective": execution_plan.objective,
    })

    belief = BeliefState()
    reasoning_engine = ReasoningEngine(planner._llm_call)
    action_ranker = ActionRanker()
    semantic_resolver = SemanticResolver(world_graph)

    iteration = 0
    stagnant_iterations = 0
    MAX_ITERATIONS = max(len(execution_plan.steps) * 5, 25)
    previous_snapshot = None

    while iteration < MAX_ITERATIONS:

        if time.time() - start_ts > max_wallclock_seconds:
            journal.record({"event": "execution_timeout"})
            raise RuntimeError("Execution wall-clock timeout exceeded")

        iteration += 1

        # ---------------- PERCEPTION ----------------

        perception_snapshot = None
        if observer:
            try:
                snap = observer.snapshot()
                perception_snapshot = snap.get("perception")
                if world_graph and isinstance(perception_snapshot, dict):
                    world_graph.update(perception_snapshot)
            except Exception:
                log_warn("Observer snapshot failed")

        world_snapshot = world_graph.snapshot() if world_graph else {}

        # ---------------- BOUND SNAPSHOT (FIX HIGH-2) ----------------

        bounded_snapshot = {}
        if isinstance(world_snapshot, dict):

            entities = world_snapshot.get("entities", [])
            if isinstance(entities, list):
                entities = entities[:MAX_PERCEPTION_ENTITIES]

            bounded_snapshot = {
                k: v
                for k, v in world_snapshot.items()
                if k != "entities"
            }
            bounded_snapshot["entities"] = entities

            # Hard byte limit
            try:
                if len(json.dumps(bounded_snapshot)) > MAX_PERCEPTION_JSON_BYTES:
                    bounded_snapshot = {}
            except Exception:
                bounded_snapshot = {}

        # ---------------- BELIEF UPDATE (FIX MEDIUM-2) ----------------

        delta = None
        if previous_snapshot and world_graph:
            delta = world_graph.compute_delta(previous_snapshot)
            belief.compute_environment_stability(delta)

        previous_snapshot = world_snapshot

        likelihoods: Dict[str, float] = {}

        if bounded_snapshot:
            focused_app = bounded_snapshot.get("focused_app")
            entity_count = len(bounded_snapshot.get("entities", []))

            if focused_app:
                likelihoods[f"app:{str(focused_app).lower()}"] = 0.9

            if entity_count > 10:
                likelihoods["ui_rich"] = 0.8
            elif entity_count > 0:
                likelihoods["ui_sparse"] = 0.7
            else:
                likelihoods["ui_empty"] = 0.5

            likelihoods["neutral"] = 0.5 if delta else 0.9

            belief.bayesian_update(likelihoods)

        # ---------------- ACTION PROPOSAL ----------------

        candidates = reasoning_engine.propose_actions(
            objective=execution_plan.objective,
            belief_summary=belief.summary(),
            perception=bounded_snapshot,
            k=4,
        )

        if not isinstance(candidates, list) or not candidates:
            raise RuntimeError("Invalid or empty action proposal")

        selected_action = action_ranker.select(candidates, belief)
        if not isinstance(selected_action, dict):
            raise RuntimeError("Invalid action selection")

        action_key = action_ranker._action_key(selected_action)

        # ---------------- AUTHORITY ----------------

        is_high_risk = selected_action.get("operation") in {"command", "install"}

        authority = input_arbitrator.evaluate(
            input_event_ts=time.monotonic(),
            high_risk=is_high_risk,
            soc_confident=belief.environment_stability > 0.7,
        )

        if authority != AuthorityDecision.CONTINUE:
            raise RuntimeError("Authority interrupted execution")

        # ---------------- EXECUTION ----------------

        try:
            input_arbitrator.soc_action_started()
            os_backend.heartbeat()

            with action_timeout(30):
                result = _execute_decision(
                    decision=selected_action,
                    execution_plan=execution_plan,
                    os_backend=os_backend,
                    accessibility_backend=accessibility_backend,
                    installer=installer,
                )

        except Exception as e:
            belief.record_action(action_key, reward=-0.5)
            stagnant_iterations += 1
            if stagnant_iterations >= MAX_STAGNANT_ITERS:
                raise RuntimeError("Execution stagnation detected")
            continue

        # ---------------- VERIFICATION ----------------

        verification = verifier.verify_step(
            step=None,
            execution_result=result,
            screenshot=perception_snapshot,
            previous_screenshot=None,
            world_graph=world_graph,
        )

        reward = float(verification.confidence) - 0.5
        belief.record_action(action_key, reward=reward)

        if not verification.success:
            stagnant_iterations += 1
            if stagnant_iterations >= MAX_STAGNANT_ITERS:
                raise RuntimeError("Execution stagnation detected")
            continue

        stagnant_iterations = 0
        belief.progress_score += verification.progress_score

        if selected_action.get("operation") == "done":
            journal.record({"event": "execution_complete"})
            return

        if belief.converged(min_iterations=3, current_iteration=iteration):
            journal.record({"event": "execution_converged"})
            return

    journal.record({"event": "execution_complete"})
