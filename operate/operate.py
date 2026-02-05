import time
import math
from typing import Any, Dict, List, Optional

from authority.authority_policy import AuthorityDecision
from authority.input_arbitrator import InputArbitrator

from core.safety.action_timeout import action_timeout, ActionTimeout
from core.telemetry.logger import log_warn

from operate.utils.operating_system import OperatingSystem
from utils.accessibility import AccessibilityBackend
from audit.journal import ActionJournal

# NEW REQUIRED CONTRACTS
from core.schemas.execution_plan import ExecutionPlan, ExecutionStep, StepType
from core.verification.step_verifier import StepVerifier
from core.execution.progress_tracker import ProgressTracker
from core.execution.failure_recovery import FailureRecoveryManager


# ==================================================
# PUBLIC ENTRYPOINT
# ==================================================

def operate_main(
    *,
    model: Optional[str],               # ignored by executor (planning only)
    terminal_prompt: str,                # retained for audit/context
    execution_plan: ExecutionPlan,       # REQUIRED
    observer=None,
    screenpipe=None,
    max_wallclock_seconds: int = 90 * 60,
):
    """
    Deterministic plan executor.

    HARD GUARANTEES:
    - No planning
    - No lifecycle transitions
    - No snapshot / restore
    - Executes ONLY an ExecutionPlan
    """

    if not isinstance(execution_plan, ExecutionPlan):
        raise ValueError("execution_plan is required and must be ExecutionPlan")

    if not execution_plan.validate():
        raise ValueError("ExecutionPlan failed validation")

    os_backend = OperatingSystem()

    accessibility_backend = AccessibilityBackend()
    if observer is not None and screenpipe is not None:
        accessibility_backend.wire(observer=observer, screenpipe=screenpipe)

    journal = ActionJournal()
    input_arbitrator = InputArbitrator()

    verifier = StepVerifier(os_backend=os_backend)
    recovery = FailureRecoveryManager()
    progress = ProgressTracker(execution_plan)

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
        max_wallclock_seconds=max_wallclock_seconds,
    )


# ==================================================
# PLAN EXECUTION
# ==================================================

def _execute_plan(
    *,
    execution_plan: ExecutionPlan,
    observer,
    os_backend: OperatingSystem,
    accessibility_backend: AccessibilityBackend,
    journal: ActionJournal,
    input_arbitrator: InputArbitrator,
    verifier: StepVerifier,
    recovery: FailureRecoveryManager,
    progress: ProgressTracker,
    max_wallclock_seconds: int,
):
    start_ts = time.time()
    progress.start_execution()

    journal.record(
        event="execution_start",
        objective=execution_plan.objective,
        total_steps=len(execution_plan.steps),
    )

    for step in execution_plan.steps:
        # ---- global timeout ----
        if time.time() - start_ts > max_wallclock_seconds:
            journal.record(event="execution_timeout")
            raise RuntimeError("Execution wall-clock timeout exceeded")

        # ---- dependency enforcement ----
        for dep in step.dependencies:
            if not progress.is_completed(dep):
                journal.record(
                    event="dependency_violation",
                    step_id=step.id,
                    missing_dependency=dep,
                )
                raise RuntimeError("Dependency not satisfied")

        # ---- authority gate ----
        decision = input_arbitrator.evaluate(
            input_event_ts=time.monotonic(),
            high_risk=step.type == StepType.TOOL_INSTALLATION,
            soc_confident=True,
        )

        if decision in (AuthorityDecision.YIELD, AuthorityDecision.ABORT):
            journal.record(event=f"authority_{decision.name.lower()}")
            raise RuntimeError("Authority aborted execution")

        progress.start_step(step.id)
        journal.record(
            event="step_start",
            step_id=step.id,
            step_type=step.type.value,
            description=step.description,
        )

        attempt_ctx = {"attempt": 0}

        while True:
            try:
                os_backend.heartbeat()

                with action_timeout(step.estimated_duration or 30):
                    _execute_step(
                        step=step,
                        os_backend=os_backend,
                        accessibility_backend=accessibility_backend,
                    )

                # ---- verification ----
                verification = verifier.verify_step(step)
                if not verification.success:
                    raise RuntimeError(verification.reason)

                progress.complete_step(step.id)
                journal.record(
                    event="step_complete",
                    step_id=step.id,
                    details=verification.details,
                )
                break

            except ActionTimeout as e:
                log_warn(f"[EXEC] timeout on step {step.id}")
                action = recovery.handle_failure(step, e, attempt_ctx)

            except Exception as e:
                action = recovery.handle_failure(step, e, attempt_ctx)

            # ---- recovery decision ----
            if action.action == "retry":
                time.sleep(action.delay)
                attempt_ctx = action.context or attempt_ctx
                continue

            if action.action == "alternative":
                _execute_alternatives(
                    action.alternative_operations,
                    os_backend,
                    accessibility_backend,
                )
                continue

            # abort
            progress.fail_step(step.id, action.reason or "fatal")
            journal.record(
                event="step_failed",
                step_id=step.id,
                reason=action.reason,
            )
            raise RuntimeError("Execution aborted")

    journal.record(event="execution_complete")


# ==================================================
# STEP EXECUTION
# ==================================================

def _execute_step(
    *,
    step: ExecutionStep,
    os_backend: OperatingSystem,
    accessibility_backend: AccessibilityBackend,
):
    if step.type == StepType.COMMAND_EXECUTION:
        cmd = step.action.get("command")
        if not cmd:
            raise ValueError("Missing command")
        os_backend.exec(cmd, sudo=step.action.get("sudo", False))

    elif step.type == StepType.FILE_CREATION:
        path = step.action.get("path")
        content = step.action.get("content", "")
        if not path:
            raise ValueError("Missing file path")
        os_backend.write_file(path, content)

    elif step.type == StepType.UI_INTERACTION:
        ui = step.action
        _execute_ui(ui, os_backend)

    elif step.type == StepType.TOOL_INSTALLATION:
        # must be handled by AutonomousInstaller in future
        raise RuntimeError("AutonomousInstaller not integrated")

    elif step.type == StepType.VERIFICATION:
        return

    else:
        raise ValueError(f"Unknown step type: {step.type}")


def _execute_ui(ui: Dict[str, Any], os_backend: OperatingSystem):
    op = ui.get("op")
    if op == "click":
        x, y = ui.get("x"), ui.get("y")
        if not _valid_coord(x) or not _valid_coord(y):
            raise ValueError("Invalid coordinates")
        os_backend.mouse({"x": x, "y": y})

    elif op == "write":
        text = ui.get("text")
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


def _execute_alternatives(
    alternatives: List[Dict[str, Any]],
    os_backend: OperatingSystem,
    accessibility_backend: AccessibilityBackend,
):
    for alt in alternatives:
        # minimal deterministic handling
        if alt.get("operation") == "mkdir":
            os_backend.mkdir(alt["path"])
        elif alt.get("operation") == "tool_install":
            raise RuntimeError("AutonomousInstaller required")
        else:
            raise RuntimeError("Unknown alternative operation")


# ==================================================
# HELPERS
# ==================================================

def _valid_coord(v: Any) -> bool:
    return (
        isinstance(v, (int, float))
        and not math.isnan(v)
        and 0.0 <= v <= 1.0
   )
