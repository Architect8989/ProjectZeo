import time
from typing import Any, Dict, Optional, List

from authority.authority_policy import AuthorityDecision
from authority.input_arbitrator import InputArbitrator

from core.safety.action_timeout import action_timeout, ActionTimeout
from core.telemetry.logger import log_warn

from operate.utils.operating_system import OperatingSystem
from utils.accessibility import AccessibilityBackend
from audit.journal import ActionJournal

from core.schemas.execution_plan import ExecutionPlan, ExecutionStep, StepType
from core.verification.step_verifier import StepVerifier
from core.verification.plan_verifier import PlanVerifier
from core.execution.progress_tracker import ProgressTracker
from core.execution.failure_recovery import FailureRecoveryManager
from core.tools.autonomous_installer import AutonomousInstaller

# --- COGNITION STACK ---
from core.cognition.belief_state import BeliefState
from core.cognition.reasoning_engine import ReasoningEngine
from core.cognition.action_ranker import ActionRanker
from core.vision.semantic_resolver import SemanticResolver


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

    installer: Optional[AutonomousInstaller] = None
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
# DECISION-THEORETIC AUTONOMOUS LOOP
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

    llm_callable = getattr(planner, "_llm_call", None)

    belief = BeliefState()
    reasoning_engine = ReasoningEngine(llm_callable)
    action_ranker = ActionRanker()
    semantic_resolver = SemanticResolver(world_graph)

    iteration = 0
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
                pass

        world_snapshot = world_graph.snapshot() if world_graph else {}

        # ---------------- BELIEF UPDATE ----------------

        likelihoods = {"neutral": 1.0}

        if previous_snapshot and world_graph:
            delta = world_graph.compute_delta(previous_snapshot)
            belief.compute_environment_stability(delta)

            if delta.get("significant_change"):
                likelihoods["neutral"] = 0.7
            else:
                likelihoods["neutral"] = 0.95

        belief.bayesian_update(likelihoods)
        previous_snapshot = world_snapshot

        # ---------------- ACTION PROPOSAL ----------------

        candidates: List[Dict[str, Any]] = reasoning_engine.propose_actions(
            objective=execution_plan.objective,
            belief_summary=belief.summary(),
            perception=world_snapshot,
            k=4,
        )

        if not candidates:
            raise RuntimeError("No actions proposed")

        # ---------------- ACTION SELECTION ----------------

        selected_action = action_ranker.select(
            candidates,
            belief,
        )

        action_key = action_ranker._action_key(selected_action)

        # ---------------- SEMANTIC GROUNDING ----------------

        if selected_action.get("operation") == "click":
            resolution = semantic_resolver.resolve(
                selected_action.get("target", "")
            )

            if resolution.get("confidence", 0) < 0.5:
                belief.record_action(action_key, reward=-0.2)
                continue

            entity = resolution.get("entity", {})
            selected_action["target"] = entity.get("coordinates")

        # ---------------- AUTHORITY CHECK ----------------

        authority = input_arbitrator.evaluate(
            input_event_ts=time.monotonic(),
            high_risk=False,
            soc_confident=True,
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

        except Exception:
            belief.record_action(action_key, reward=-0.5)
            continue

        # ---------------- VERIFICATION ----------------

        verification = verifier.verify_step(
            step=None,
            execution_result=result,
            screenshot=perception_snapshot,
            previous_screenshot=None,
            world_graph=world_graph,
        )

        reward = verification.confidence - 0.5
        belief.record_action(action_key, reward=reward)

        belief.commit(action_key, perception_snapshot or {})

        if not verification.success:
            continue

        belief.progress_score += verification.progress_score

        # ---------------- CONVERGENCE ----------------

        if belief.converged():
            journal.record({"event": "execution_converged"})
            return

        if selected_action.get("operation") == "done":
            journal.record({"event": "execution_complete"})
            return

    journal.record({"event": "execution_complete"})


# ==================================================
# DECISION EXECUTION
# ==================================================

def _execute_decision(
    *,
    decision: Dict[str, Any],
    execution_plan: ExecutionPlan,
    os_backend: OperatingSystem,
    accessibility_backend: Optional[AccessibilityBackend],
    installer: Optional[AutonomousInstaller],
) -> Dict[str, Any]:

    op = decision.get("operation")

    if op == "click":
        os_backend.click(decision.get("target"))
        return {"status": "clicked"}

    if op == "type":
        os_backend.type_text(decision.get("text", ""))
        return {"status": "typed"}

    if op == "hotkey":
        os_backend.press_keys(decision.get("keys", []))
        return {"status": "hotkey"}

    if op == "command":
        return os_backend.execute_command(decision.get("command"))

    if op == "install":
        if installer is None:
            raise RuntimeError("Installer unavailable")
        return installer.install_tool(decision.get("tool"))

    if op == "done":
        return {"status": "complete"}

    raise RuntimeError(f"Unknown operation: {op}")
