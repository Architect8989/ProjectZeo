import time
import math
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
    max_wallclock_seconds: int = 90 * 60,
    llm_callable=None,
):
    if not isinstance(execution_plan, ExecutionPlan):
        raise ValueError("execution_plan must be ExecutionPlan")

    execution_plan.validate()
    PlanVerifier().verify(execution_plan)

    has_tool_steps = any(
        s.type == StepType.TOOL_INSTALLATION for s in execution_plan.steps
    )
    if has_tool_steps and observer is None:
        raise ValueError(
            "ExecutionPlan contains TOOL_INSTALLATION but observer missing"
        )

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

        if decision == AuthorityDecision.RELEASE:
            journal.record({"event": "authority_release"})
            raise RuntimeError("Authority released control")

        if decision == AuthorityDecision.ABORT:
            journal.record({"event": "authority_abort"})
            raise RuntimeError("Authority aborted execution")

        if decision == AuthorityDecision.YIELD:
            journal.record({"event": "authority_yield"})
            time.sleep(0.5)
            continue

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
                decision = input_arbitrator.evaluate(
                    input_event_ts=time.monotonic(),
                    high_risk=(step.type == StepType.TOOL_INSTALLATION),
                    soc_confident=True,
                )

                if decision == AuthorityDecision.RELEASE:
                    journal.record({"event": "authority_release"})
                    raise RuntimeError("Authority released control")

                if decision == AuthorityDecision.ABORT:
                    journal.record({"event": "authority_abort"})
                    raise RuntimeError("Authority aborted execution")

                input_arbitrator.soc_action_started()
                os_backend.heartbeat()

                before_screen = _extract_screen(observer)

                with action_timeout(step.estimated_duration or 30):
                    result = _execute_step(
                        step=step,
                        os_backend=os_backend,
                        accessibility_backend=accessibility_backend,
                        installer=installer,
                    )

                after_screen = _extract_screen(observer)

                verification = verifier.verify_step(
                    step,
                    execution_result=result,
                    screenshot=after_screen,
                    previous_screenshot=before_screen,
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
                time.sleep(action.delay)
                attempt_ctx = action.context or attempt_ctx
                continue

            if action.action == "alternative":
                progress.fail_step(step.id, action.reason or "alternative_failed")
                journal.record({
                    "event": "step_failed",
                    "step_id": step.id,
                    "reason": "alternative_not_implemented",
                })
                raise RuntimeError("Execution aborted (alternative not implemented)")

            progress.fail_step(step.id, action.reason or "fatal")
            journal.record({
                "event": "step_failed",
                "step_id": step.id,
            })
            raise RuntimeError("Execution aborted")

    journal.record({"event": "execution_complete"})


# ==================================================
# STEP EXECUTION
# ==================================================

def _execute_step(
    *,
    step: ExecutionStep,
    os_backend: OperatingSystem,
    accessibility_backend: Optional[AccessibilityBackend],
    installer: Optional[AutonomousInstaller],
):
    if step.type == StepType.COMMAND_EXECUTION:
        cmd = step.action.get("command")
        if not cmd:
            raise ValueError("Missing command")
        return os_backend.exec(cmd, sudo=step.action.get("sudo", False))

    elif step.type == StepType.FILE_CREATION:
        path = step.action.get("path")
        content = step.action.get("content", "")
        if not path:
            raise ValueError("Missing file path")
        os_backend.write_file(path, content)
        return None

    elif step.type == StepType.UI_INTERACTION:
        _execute_ui(step.action, os_backend)
        return None

    elif step.type == StepType.TOOL_INSTALLATION:
        if installer is None:
            raise RuntimeError("Installer unavailable")
        installer.install_tool(step.action)
        return None

    elif step.type in (StepType.VERIFICATION, StepType.DONE):
        return None

    else:
        raise ValueError(f"Unknown step type: {step.type}")


# ==================================================
# UI EXECUTION
# ==================================================

def _execute_ui(ui: Dict[str, Any], os_backend: OperatingSystem):
    op = ui.get("operation")

    if op == "click":
        x, y = ui.get("x"), ui.get("y")
        if not _valid_coord(x) or not _valid_coord(y):
            raise ValueError("Invalid coordinates")
        os_backend.mouse({"x": x, "y": y})

    elif op == "write":
        text = ui.get("content")
        if not isinstance(text, str):
            raise ValueError("Invalid text")
        os_backend.write(text)

    elif op == "press":
        keys = ui.get("keys")
        if not keys:
            raise ValueError("Missing keys")
        os_backend.press(keys)

    else:
        raise ValueError(f"Unknown UI op: {op}")


# ==================================================
# SCREEN EXTRACTION
# ==================================================

def _extract_screen(observer):
    if not observer:
        return None

    snap = observer.snapshot()
    perception = snap.get("perception")

    if not isinstance(perception, dict):
        return None

    text_parts = []
    for el in perception.get("elements", []):
        t = el.get("text")
        if isinstance(t, str) and t.strip():
            text_parts.append(t.strip())

    return {
        "available": snap.get("available", False),
        "text": " ".join(text_parts),
    }


# ==================================================
# HELPERS
# ==================================================

def _valid_coord(v: Any) -> bool:
    return (
        isinstance(v, (int, float))
        and not math.isnan(v)
        and 0.0 <= v <= 1.0
)
