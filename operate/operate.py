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
        _execute_plan(
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
# PLAN EXECUTION
# ==================================================

def _execute_plan(
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
        "total_steps": len(execution_plan.steps),
    })

    previous_snapshot: Optional[Dict[str, Any]] = None

    for step in execution_plan.steps:

        if time.time() - start_ts > max_wallclock_seconds:
            journal.record({"event": "execution_timeout"})
            raise RuntimeError("Execution wall-clock timeout exceeded")

        for dep in step.dependencies:
            if not progress.is_completed(dep):
                raise RuntimeError(f"Dependency {dep} not satisfied")

        # ---------------- PERCEPTION SYNC ----------------

        if observer and world_graph:
            try:
                snap = observer.snapshot()
                perception = snap.get("perception")
                if isinstance(perception, dict):
                    world_graph.update(perception)
            except Exception:
                pass

        # ---------------- REPLAN CHECK ----------------

        if planner and world_graph:
            current_world = world_graph.snapshot()
            planner.update_world_snapshot(current_world)

            decision = planner.should_replan(
                current_step_id=step.id,
                execution_history=progress.get_history(),
            )

            journal.record({
                "event": "replan_check",
                "step_id": step.id,
                "decision": decision,
            })

            if decision.get("replan_required"):
                raise RuntimeError("REPLAN_REQUIRED")

        # ---------------- DIVERGENCE CHECK ----------------

        if world_graph:
            current_snapshot = world_graph.snapshot()

            if previous_snapshot is not None:
                delta = world_graph.compute_delta(previous_snapshot)

                if delta.get("significant_change"):
                    journal.record({
                        "event": "divergence_detected",
                        "step_id": step.id,
                        "delta": delta,
                    })

                    if planner:
                        raise RuntimeError("REPLAN_REQUIRED")
                    else:
                        raise RuntimeError("World divergence detected")

            previous_snapshot = current_snapshot

        # ---------------- AUTHORITY CHECK ----------------

        decision = input_arbitrator.evaluate(
            input_event_ts=time.monotonic(),
            high_risk=(step.type == StepType.TOOL_INSTALLATION),
            soc_confident=True,
        )

        if decision != AuthorityDecision.CONTINUE:
            raise RuntimeError("Authority interrupted execution")

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
                        perception = snap.get("perception")
                        if isinstance(perception, dict):
                            world_graph.update(perception)
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

                if planner and world_graph:
                    planner.update_world_snapshot(world_graph.snapshot())

                journal.record({
                    "event": "step_complete",
                    "step_id": step.id,
                })

                break

            except ActionTimeout as e:
                log_warn(f"[EXEC] timeout on step {step.id}")
                if planner and world_graph:
                    action = recovery.handle_failure_with_perception(
                        step=step,
                        error=e,
                        attempt_ctx=attempt_ctx,
                        world_graph=world_graph,
                        llm_callable=planner._llm_call,
                    )
                else:
                    action = recovery.handle_failure(step, e, attempt_ctx)

            except Exception as e:
                if str(e) == "REPLAN_REQUIRED":
                    raise
                if planner and world_graph:
                    action = recovery.handle_failure_with_perception(
                        step=step,
                        error=e,
                        attempt_ctx=attempt_ctx,
                        world_graph=world_graph,
                        llm_callable=planner._llm_call,
                    )
                else:
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
# EXECUTION HELPERS (NEW – MINIMUM FIX)
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
            target = action.get("target")
            os_backend.click(target)
            return {"status": "clicked", "target": target}

        if op == "type":
            os_backend.type_text(action.get("text", ""))
            return {"status": "typed"}

        if op == "hotkey":
            os_backend.press_keys(action.get("keys", []))
            return {"status": "hotkey"}

        if op == "move":
            os_backend.move_cursor(action.get("target"))
            return {"status": "moved"}

        if op == "scroll":
            os_backend.scroll(
                action.get("direction", "down"),
                action.get("amount", 3),
            )
            return {"status": "scrolled"}

        raise RuntimeError(f"Unknown UI operation: {op}")

    elif step.type == StepType.COMMAND_EXECUTION:
        action = step.action
        command = action.get("command")
        result = os_backend.execute_command(
            command,
            cwd=action.get("cwd"),
            timeout=action.get("timeout", 30),
            shell=action.get("shell", True),
        )
        return result

    elif step.type == StepType.FILE_CREATION:
        action = step.action
        path = action.get("path")
        content = action.get("content", "")
        os_backend.write_file(path, content)
        return {"status": "file_written", "path": path}

    elif step.type == StepType.TOOL_INSTALLATION:
        if installer is None:
            raise RuntimeError("Installer not available")
        tool = step.action.get("tool")
        return installer.install(tool)

    elif step.type == StepType.DONE:
        return {"status": "complete"}

    else:
        raise RuntimeError(f"Unsupported step type: {step.type}")


def _extract_screen(observer) -> Optional[Dict[str, Any]]:
    if observer is None:
        return None
    try:
        snap = observer.snapshot()
        return snap.get("perception")
    except Exception:
        return None
