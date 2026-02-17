import time
from typing import Any, Dict, Optional

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

    installer: Optional[AutonomousInstaller] = None
    if observer is not None:
        installer = AutonomousInstaller(
            observer=observer,
            os_backend=os_backend,
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
# AUTONOMOUS PERCEPTION–ACTION LOOP
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

    iteration = 0
    MAX_ITERATIONS = max(len(execution_plan.steps) * 3, 10)

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

        # ---------------- LLM DECISION ----------------

        if llm_callable:
            decision = llm_callable(
                messages=[{
                    "role": "system",
                    "content": f"Objective: {execution_plan.objective}"
                }],
                objective=execution_plan.objective,
                session_id="execution",
            )
        else:
            # fallback: follow static plan
            if iteration - 1 >= len(execution_plan.steps):
                break
            step = execution_plan.steps[iteration - 1]
            decision = {"operation": "execute_step", "step_id": step.id}

        if decision.get("operation") == "done":
            journal.record({"event": "execution_complete"})
            return

        # ---------------- AUTHORITY CHECK ----------------

        authority = input_arbitrator.evaluate(
            input_event_ts=time.monotonic(),
            high_risk=False,
            soc_confident=True,
        )

        if authority != AuthorityDecision.CONTINUE:
            raise RuntimeError("Authority interrupted execution")

        # ---------------- ACTION EXECUTION ----------------

        try:
            input_arbitrator.soc_action_started()
            os_backend.heartbeat()

            with action_timeout(30):
                result = _execute_decision(
                    decision=decision,
                    execution_plan=execution_plan,
                    os_backend=os_backend,
                    accessibility_backend=accessibility_backend,
                    installer=installer,
                )

        except ActionTimeout as e:
            log_warn("[EXEC] timeout during action")
            raise RuntimeError(str(e))

        # ---------------- VERIFICATION ----------------

        verification = verifier.verify_step(
            step=None,
            execution_result=result,
            screenshot=perception_snapshot,
            previous_screenshot=None,
            world_graph=world_graph,
        )

        if not verification.success:
            raise RuntimeError(verification.reason)

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

    if op == "execute_step":
        step_id = decision.get("step_id")
        step = next(
            (s for s in execution_plan.steps if s.id == step_id),
            None,
        )
        if not step:
            raise RuntimeError(f"Step {step_id} not found")
        return _execute_step(
            step=step,
            os_backend=os_backend,
            accessibility_backend=accessibility_backend,
            installer=installer,
        )

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
        return installer.install(decision.get("tool"))

    if op == "done":
        return {"status": "complete"}

    raise RuntimeError(f"Unknown operation: {op}")


# ==================================================
# LEGACY STEP EXECUTION (REUSED)
# ==================================================

def _execute_step(
    *,
    step: ExecutionStep,
    os_backend: OperatingSystem,
    accessibility_backend: Optional[AccessibilityBackend],
    installer: Optional[AutonomousInstaller],
) -> Dict[str, Any]:

    if step.type == StepType.UI_INTERACTION:
        action = step.action
        op = action.get("operation")

        if op == "click":
            os_backend.click(action.get("target"))
            return {"status": "clicked"}

        if op == "type":
            os_backend.type_text(action.get("text", ""))
            return {"status": "typed"}

        if op == "hotkey":
            os_backend.press_keys(action.get("keys", []))
            return {"status": "hotkey"}

        raise RuntimeError(f"Unknown UI operation: {op}")

    elif step.type == StepType.COMMAND_EXECUTION:
        action = step.action
        return os_backend.execute_command(
            action.get("command"),
            cwd=action.get("cwd"),
            timeout=action.get("timeout", 30),
            shell=action.get("shell", True),
        )

    elif step.type == StepType.FILE_CREATION:
        action = step.action
        os_backend.write_file(
            action.get("path"),
            action.get("content", ""),
        )
        return {"status": "file_written"}

    elif step.type == StepType.TOOL_INSTALLATION:
        if installer is None:
            raise RuntimeError("Installer not available")
        return installer.install(step.action.get("tool"))

    elif step.type == StepType.DONE:
        return {"status": "complete"}

    raise RuntimeError(f"Unsupported step type: {step.type}")
