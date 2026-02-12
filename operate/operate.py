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
    model: Optional[str],
    terminal_prompt: str,
    execution_plan: ExecutionPlan,
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
        _execute_plan(
            execution_plan=execution_plan,
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
# PLAN EXECUTION
# ==================================================

def _execute_plan(
    *,
    execution_plan: ExecutionPlan,
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
        "total_steps": len(execution_plan.steps),
    })

    for step in execution_plan.steps:

        if time.time() - start_ts > max_wallclock_seconds:
            journal.record({"event": "execution_timeout"})
            raise RuntimeError("Execution wall-clock timeout exceeded")

        for dep in step.dependencies:
            if not progress.is_completed(dep):
                raise RuntimeError("Dependency not satisfied")

        decision = input_arbitrator.evaluate(
            input_event_ts=time.monotonic(),
            high_risk=(step.type == StepType.TOOL_INSTALLATION),
            soc_confident=True,
        )

        if decision != AuthorityDecision.CONTINUE:
            _handle_authority(decision, journal)

        progress.start_step(step.id)

        journal.record({
            "event": "step_start",
            "step_id": step.id,
            "step_type": step.type.value,
            "description": step.description,
        })

        attempt_ctx = {"attempt": 0}

        while True:
            try:
                input_arbitrator.soc_action_started()
                os_backend.heartbeat()

                # Correct world graph unwrapping
                if observer and world_graph:
                    try:
                        snap = observer.snapshot()
                        wrapper = snap.get("perception")
                        if isinstance(wrapper, dict):
                            raw = wrapper.get("perception")
                            if isinstance(raw, dict) and raw.get("available"):
                                world_graph.update(raw)
                    except Exception:
                        pass

                before_screen = _extract_screen(observer)

                with action_timeout(step.estimated_duration or 30):
                    result = _execute_step(
                        step=step,
                        os_backend=os_backend,
                        accessibility_backend=accessibility_backend,
                        installer=installer,
                    )

                if observer and world_graph:
                    try:
                        snap = observer.snapshot()
                        wrapper = snap.get("perception")
                        if isinstance(wrapper, dict):
                            raw = wrapper.get("perception")
                            if isinstance(raw, dict) and raw.get("available"):
                                world_graph.update(raw)
                    except Exception:
                        pass

                after_screen = _extract_screen(observer)

                verification = verifier.verify_step(
                    step,
                    execution_result=result,
                    screenshot=after_screen,
                    previous_screenshot=before_screen,
                    world_graph=world_graph,
                )

                if not verification.success:
                    raise RuntimeError(verification.reason)

                progress.complete_step(step.id)

                journal.record({
                    "event": "step_complete",
                    "step_id": step.id,
                })

                break

            except ActionTimeout as e:
                log_warn(f"[EXEC] timeout on step {step.id}")
                action = recovery.handle_failure(step, e, attempt_ctx)

            except Exception as e:
                action = recovery.handle_failure(step, e, attempt_ctx)

            if action.action == "retry":
                attempt_ctx = action.context or attempt_ctx
                time.sleep(action.delay)
                continue

            progress.fail_step(step.id, action.reason or "fatal")
            journal.record({
                "event": "step_failed",
                "step_id": step.id,
                "reason": action.reason,
            })
            raise RuntimeError("Execution aborted")

    journal.record({"event": "execution_complete"})


# ==================================================
# INTERNAL EXECUTION HELPERS
# ==================================================

def _execute_step(
    *,
    step: ExecutionStep,
    os_backend: OperatingSystem,
    accessibility_backend: Optional[AccessibilityBackend],
    installer: Optional[AutonomousInstaller],
) -> Any:

    action = step.action or {}

    if step.type == StepType.DONE:
        return None

    if step.type == StepType.VERIFICATION:
        return None

    if step.type == StepType.COMMAND_EXECUTION:
        cmd = action.get("command")
        if not cmd:
            raise RuntimeError("Missing command")
        return os_backend.exec(cmd)

    if step.type == StepType.FILE_CREATION:
        path = action.get("path")
        content = action.get("content", "")
        if not path:
            raise RuntimeError("Missing file path")
        os_backend.write_file(path, content)
        return None

    if step.type == StepType.TOOL_INSTALLATION:
        if not installer:
            raise RuntimeError("Installer unavailable")
        tool = action.get("tool")
        if not tool:
            raise RuntimeError("Missing tool name")
        return installer.install(tool)

    if step.type == StepType.UI_INTERACTION:
        op = action.get("operation")

        if op == "click":
            os_backend.mouse(action)
            return None

        if op == "write":
            os_backend.write(action.get("content", ""))
            return None

        if op == "press":
            os_backend.press(action.get("keys", []))
            return None

        raise RuntimeError(f"Unknown UI operation: {op}")

    raise RuntimeError(f"Unhandled step type: {step.type}")


def _extract_screen(observer) -> Dict[str, Any]:
    if observer is None:
        return {"available": False, "elements": []}

    try:
        snap = observer.snapshot()
        wrapper = snap.get("perception")
        if isinstance(wrapper, dict):
            raw = wrapper.get("perception")
            if isinstance(raw, dict):
                return raw
    except Exception:
        pass

    return {"available": False, "elements": []}


def _handle_authority(decision: AuthorityDecision, journal: ActionJournal) -> None:
    if decision == AuthorityDecision.YIELD:
        journal.record({"event": "authority_yield"})
        raise RuntimeError("Authority yielded to human")

    if decision == AuthorityDecision.ABORT:
        journal.record({"event": "authority_abort"})
        raise RuntimeError("Execution aborted by authority")

    if decision == AuthorityDecision.RELEASE:
        journal.record({"event": "authority_release"})
        raise RuntimeError("Emergency release activated")

    journal.record({"event": "authority_unknown", "decision": str(decision)})
    raise RuntimeError("Unknown authority decision")
