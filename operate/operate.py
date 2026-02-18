import time
import json
from typing import Any, Dict, Optional

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
from core.tools.autonomous_installer import AutonomousInstaller

from core.cognition.belief_state import BeliefState
from core.cognition.action_ranker import ActionRanker


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
            observer=observer,
            world_graph=world_graph,
            os_backend=os_backend,
            accessibility_backend=accessibility_backend,
            journal=journal,
            input_arbitrator=input_arbitrator,
            verifier=verifier,
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
    observer,
    world_graph,
    os_backend: OperatingSystem,
    accessibility_backend: Optional[AccessibilityBackend],
    journal: ActionJournal,
    input_arbitrator: InputArbitrator,
    verifier: StepVerifier,
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
    action_ranker = ActionRanker()

    iteration = 0
    stagnant_iterations = 0
    max_iterations = max(len(execution_plan.steps) * (MAX_STAGNANT_ITERS + 1), 25)

    current_step_index = 0
    previous_snapshot = None
    previous_perception = None

    while iteration < max_iterations:

        if time.time() - start_ts > max_wallclock_seconds:
            journal.record({"event": "execution_timeout"})
            raise RuntimeError("Execution wall-clock timeout exceeded")

        if current_step_index >= len(execution_plan.steps):
            journal.record({"event": "execution_complete"})
            return

        iteration += 1
        current_step = execution_plan.steps[current_step_index]

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

        # ---------------- BELIEF UPDATE ----------------

        delta = None
        if previous_snapshot and world_graph:
            delta = world_graph.compute_delta(previous_snapshot)
            belief.compute_environment_stability(delta)

        previous_snapshot = world_snapshot

        # ---------------- ACTION ----------------

        selected_action = current_step.action or {}
        if not isinstance(selected_action, dict):
            raise RuntimeError("Invalid action format")

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
                    os_backend=os_backend,
                    accessibility_backend=accessibility_backend,
                    installer=installer,
                )

        except Exception:
            belief.record_action(action_key, reward=-0.5)
            stagnant_iterations += 1
            if stagnant_iterations >= MAX_STAGNANT_ITERS:
                raise RuntimeError("Execution stagnation detected")
            continue

        # ---------------- VERIFICATION ----------------

        verification = verifier.verify_step(
            step=current_step,
            execution_result=result,
            screenshot=perception_snapshot,
            previous_screenshot=previous_perception,
            world_graph=world_graph,
        )

        reward = float(verification.confidence) - 0.5
        belief.record_action(action_key, reward=reward)

        action_rewards = belief.action_rewards.get(action_key, [])
        best_reward = max(action_rewards) if action_rewards else reward
        belief.update_regret(action_key, reward, best_reward)

        if not verification.success:
            stagnant_iterations += 1
            if stagnant_iterations >= MAX_STAGNANT_ITERS:
                raise RuntimeError("Execution stagnation detected")
            continue

        stagnant_iterations = 0
        belief.progress_score += verification.progress_score

        previous_perception = perception_snapshot
        current_step_index += 1

        if current_step.type.name == "DONE":
            journal.record({"event": "execution_complete"})
            return

        if belief.converged(
            min_iterations=3,
            current_iteration=iteration,
            plan_steps_total=len(execution_plan.steps),
            steps_completed=current_step_index,
        ):
            journal.record({"event": "execution_converged"})
            return

    journal.record({"event": "execution_complete"})


# ==================================================
# EXECUTION DISPATCH
# ==================================================

def _execute_decision(
    *,
    decision: Dict[str, Any],
    os_backend: OperatingSystem,
    accessibility_backend: Optional[AccessibilityBackend],
    installer: Optional[AutonomousInstaller],
):

    operation = decision.get("operation")

    if operation == "click":
        x = decision.get("x")
        y = decision.get("y")
        if x is None or y is None:
            raise RuntimeError("Click missing coordinates")
        os_backend.click(x, y)
        return None

    if operation == "type":
        text = decision.get("text", "")
        os_backend.type_text(text)
        return None

    if operation == "hotkey":
        keys = decision.get("keys", [])
        if not isinstance(keys, list):
            raise RuntimeError("Invalid hotkey format")
        os_backend.press_keys(keys)
        return None

    if operation == "command":
        cmd = decision.get("command")
        if not cmd:
            raise RuntimeError("Missing command")
        return os_backend.run_command(cmd)

    if operation == "install":
        if installer is None:
            raise RuntimeError("Installer unavailable")
        tool = decision.get("tool")
        if not isinstance(tool, dict):
            raise RuntimeError("Invalid tool specification")
        installer.install_tool(tool)
        return None

    raise RuntimeError(f"Unsupported operation: {operation}")
