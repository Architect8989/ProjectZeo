from __future__ import annotations

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

from core.schemas.execution_plan import ExecutionPlan, StepType
from core.verification.step_verifier import StepVerifier
from core.verification.plan_verifier import PlanVerifier
from core.execution.progress_tracker import ProgressTracker
from core.tools.autonomous_installer import AutonomousInstaller

from core.cognition.belief_state import BeliefState
from core.cognition.action_ranker import ActionRanker

from config.timeouts import MAX_STAGNANT_ITERS_UI, MAX_STAGNANT_ITERS_COMMAND


MAX_PERCEPTION_ENTITIES = 20
MAX_PERCEPTION_JSON_BYTES = 10_000

# PATCH §1.11: WAIT should pause and retry, not immediately replan
WAIT_RETRY_SECONDS = 0.5
MAX_WAIT_RETRIES = 10  # 5s total wait before replanning

# Max bytes of command output stored per step in execution_log
MAX_COMMAND_OUTPUT_BYTES = 4096


class AuthorityAbortError(RuntimeError):
    pass


def operate_main(
    *,
    terminal_prompt: str,
    execution_plan: ExecutionPlan,
    planner=None,
    observer=None,
    world_graph=None,
    os_backend: Optional[OperatingSystem] = None,
    max_wallclock_seconds: int = 90 * 60,
):

    if not isinstance(execution_plan, ExecutionPlan):
        raise ValueError("execution_plan must be an ExecutionPlan instance")

    execution_plan.validate()
    PlanVerifier().verify(execution_plan)

    os_backend = os_backend or OperatingSystem()

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


def _execute_autonomous_loop(
    *,
    execution_plan: ExecutionPlan,
    observer,
    world_graph,
    os_backend: OperatingSystem,
    accessibility_backend,
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

    # PATCH §1.11: bounded command output log fed into world_graph
    execution_log: Dict[int, Dict[str, str]] = {}

    iteration = 0
    stagnant_iterations = 0

    max_iterations = max(
        len(execution_plan.steps) * (MAX_STAGNANT_ITERS_COMMAND + 1),
        25,
    )

    current_step_index = 0
    previous_snapshot = None
    previous_perception = None

    while iteration < max_iterations:

        if time.time() - start_ts > max_wallclock_seconds:
            journal.record({"event": "execution_timeout"})
            raise RuntimeError("TASK_FAILED:timeout")

        if current_step_index >= len(execution_plan.steps):
            journal.record({"event": "execution_complete"})
            return

        iteration += 1
        current_step = execution_plan.steps[current_step_index]

        # PATCH §R6: stagnation limit depends on step type
        step_type = current_step.type
        stagnant_limit = (
            MAX_STAGNANT_ITERS_COMMAND
            if step_type in (StepType.COMMAND_EXECUTION, StepType.TOOL_INSTALLATION)
            else MAX_STAGNANT_ITERS_UI
        )

        perception_snapshot = None

        if observer:
            try:
                snap = observer.snapshot()
                perception_snapshot = snap.get("perception")

                if world_graph and isinstance(perception_snapshot, dict):
                    # PATCH §1.11: merge command output into perception for LLM context
                    step_log = execution_log.get(current_step_index)
                    if step_log:
                        perception_snapshot = dict(perception_snapshot)
                        perception_snapshot["last_command_output"] = step_log

                    world_graph.update(perception_snapshot)

            except Exception:
                log_warn("Observer snapshot failed")

        world_snapshot = world_graph.snapshot() if world_graph else {}

        delta = None
        if previous_snapshot and world_graph:
            try:
                delta = world_graph.compute_delta(previous_snapshot)
                belief.compute_environment_stability(delta)
            except Exception:
                delta = None

        if isinstance(world_snapshot, dict):

            entities = world_snapshot.get("entities", [])
            if isinstance(entities, list):
                entities = entities[:MAX_PERCEPTION_ENTITIES]

            bounded = {k: v for k, v in world_snapshot.items() if k != "entities"}
            bounded["entities"] = entities

            try:
                if len(json.dumps(bounded)) <= MAX_PERCEPTION_JSON_BYTES:
                    likelihoods: Dict[str, float] = {}

                    focused_app = bounded.get("focused_app")
                    entity_count = len(bounded.get("entities", []))

                    if isinstance(focused_app, str) and focused_app.strip():
                        likelihoods[f"app:{focused_app.lower()}"] = 0.9

                    if entity_count > 10:
                        likelihoods["ui_rich"] = 0.8
                    elif entity_count > 0:
                        likelihoods["ui_sparse"] = 0.7
                    else:
                        likelihoods["ui_empty"] = 0.5

                    likelihoods["neutral"] = 0.5 if delta else 0.9
                    belief.bayesian_update(likelihoods)

            except Exception:
                pass

        previous_snapshot = world_snapshot

        raw_actions = current_step.action

        candidate_actions: List[Dict[str, Any]] = []

        if isinstance(raw_actions, dict):
            candidate_actions.append(raw_actions)
        elif isinstance(raw_actions, list):
            candidate_actions.extend(
                a for a in raw_actions if isinstance(a, dict)
            )
        else:
            raise RuntimeError("TASK_FAILED:invalid_action_format")

        if not candidate_actions:
            raise RuntimeError("TASK_FAILED:no_candidate_actions")

        selected_action = action_ranker.select(
            actions=candidate_actions,
            belief_state=belief,
        )

        action_key = action_ranker.action_key(selected_action)

        is_high_risk = selected_action.get("operation") in {
            "command", "install", "file_create"
        }

        authority = input_arbitrator.evaluate(
            input_event_ts=time.monotonic(),
            high_risk=is_high_risk,
            soc_confident=belief.environment_stability > 0.7,
        )

        if authority == AuthorityDecision.ABORT:
            raise AuthorityAbortError("Human authority abort — task terminated")

        # PATCH §1.11: WAIT and RELEASE → pause and retry
        if authority in (
            AuthorityDecision.WAIT,
            AuthorityDecision.RELEASE,
            getattr(AuthorityDecision, "YIELD", None),
        ):
            wait_retries = 0
            while wait_retries < MAX_WAIT_RETRIES:
                time.sleep(WAIT_RETRY_SECONDS)
                wait_retries += 1
                authority = input_arbitrator.evaluate(
                    input_event_ts=time.monotonic(),
                    high_risk=is_high_risk,
                    soc_confident=belief.environment_stability > 0.7,
                )
                if authority == AuthorityDecision.CONTINUE:
                    break
                if authority == AuthorityDecision.ABORT:
                    raise AuthorityAbortError(
                        "Human authority abort during wait period"
                    )
            else:
                journal.record({"event": "wait_timeout_replan"})
                raise RuntimeError("REPLAN_REQUIRED")

        if authority != AuthorityDecision.CONTINUE:
            raise RuntimeError("REPLAN_REQUIRED")

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

            # FIX-04 (RTB-03): os_backend.run_command() returns
            # subprocess.CompletedProcess, not a dict. The original guard
            # `isinstance(result, dict)` was always False for command steps,
            # so execution_log was never populated and command output was
            # never visible to the LLM in subsequent planning steps.
            if selected_action.get("operation") == "command":
                if isinstance(result, dict):
                    stdout = str(result.get("stdout", ""))[:MAX_COMMAND_OUTPUT_BYTES]
                    stderr = str(result.get("stderr", ""))[:MAX_COMMAND_OUTPUT_BYTES]
                    exit_code = result.get("returncode", result.get("exit_code"))
                elif hasattr(result, "stdout"):
                    # subprocess.CompletedProcess — decode bytes if needed
                    _stdout_raw = result.stdout
                    _stderr_raw = result.stderr
                    stdout = (
                        _stdout_raw.decode("utf-8", errors="replace")
                        if isinstance(_stdout_raw, bytes)
                        else str(_stdout_raw or "")
                    )[:MAX_COMMAND_OUTPUT_BYTES]
                    stderr = (
                        _stderr_raw.decode("utf-8", errors="replace")
                        if isinstance(_stderr_raw, bytes)
                        else str(_stderr_raw or "")
                    )[:MAX_COMMAND_OUTPUT_BYTES]
                    exit_code = getattr(result, "returncode", None)
                else:
                    stdout = stderr = ""
                    exit_code = None

                execution_log[current_step_index] = {
                    "command": selected_action.get("command", ""),
                    "stdout": stdout,
                    "stderr": stderr,
                    "exit_code": str(exit_code) if exit_code is not None else "?",
                }

                journal.record({
                    "event": "command_executed",
                    "command": selected_action.get("command"),
                    "exit_code": exit_code,
                    "stdout_bytes": len(stdout),
                    "stderr_bytes": len(stderr),
                })

        except Exception:
            belief.record_action(action_key, reward=-0.5)

            history = belief.action_rewards.get(action_key, [])
            normalized_reward = history[-1] if history else -0.5
            best_reward = max(history) if history else normalized_reward

            belief.update_regret(action_key, normalized_reward, best_reward)

            stagnant_iterations += 1
            if stagnant_iterations >= stagnant_limit:
                journal.record({"event": "stagnation_detected"})
                raise RuntimeError("REPLAN_REQUIRED")
            continue

        verification = verifier.verify_step(
            step=current_step,
            execution_result=result,
            screenshot=perception_snapshot,
            previous_screenshot=previous_perception,
            world_graph=world_graph,
        )

        raw_reward = float(verification.confidence) - 0.5
        belief.record_action(action_key, reward=raw_reward)

        history = belief.action_rewards.get(action_key, [])
        normalized_reward = history[-1] if history else raw_reward
        best_reward = max(history) if history else normalized_reward

        belief.update_regret(action_key, normalized_reward, best_reward)

        if not verification.success:
            stagnant_iterations += 1
            if stagnant_iterations >= stagnant_limit:
                journal.record({"event": "verification_stagnation"})
                raise RuntimeError("REPLAN_REQUIRED")
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

    journal.record({"event": "iteration_budget_exhausted"})
    raise RuntimeError("TASK_FAILED:iteration_budget_exhausted")


def _execute_decision(
    *,
    decision: Dict[str, Any],
    os_backend: OperatingSystem,
    accessibility_backend,
    installer: Optional[AutonomousInstaller],
):
    """
    Dispatch a single action decision to the OS backend.

    FIX-5: click operations now use os_backend.mouse({"x": x, "y": y})
    which correctly handles LLM-supplied NORMALIZED coordinates (0.0–1.0).
    The prior os_backend.click(float(x), float(y)) expected PIXEL coordinates,
    causing all clicks to land near screen origin (0, 0).

    Operation mapping:
      click       → os_backend.mouse()       — normalized 0.0–1.0 coords
      type/write  → os_backend.type_text()
      hotkey/press→ os_backend.press_keys()
      command     → os_backend.run_command() — returns CompletedProcess
      file_create → os_backend.write_file()  — §R1
      verify      → os_backend.run_command() or visual no-op — §R1
      install     → AutonomousInstaller.install_tool()
      done        → no-op
    """

    if not isinstance(decision, dict):
        raise RuntimeError("TASK_FAILED:invalid_decision_payload")

    operation = decision.get("operation")
    if not isinstance(operation, str):
        raise RuntimeError("TASK_FAILED:missing_operation")

    operation = operation.lower().strip()

    # ----------------------------------------------------------
    # CLICK — FIX-5: use mouse() which handles normalized coords
    # ----------------------------------------------------------
    if operation == "click":
        x = decision.get("x")
        y = decision.get("y")
        if x is None or y is None:
            raise RuntimeError("TASK_FAILED:click_missing_coordinates")

        try:
            x_f = float(x)
            y_f = float(y)
        except (TypeError, ValueError):
            raise RuntimeError(
                f"TASK_FAILED:click_invalid_coordinates x={x!r} y={y!r}"
            )

        # mouse() calls _click_at_percentage() — expects normalized 0.0–1.0
        os_backend.mouse({"x": x_f, "y": y_f})
        return None

    # ----------------------------------------------------------
    # TYPE / WRITE
    # ----------------------------------------------------------
    if operation in ("type", "write"):
        text = decision.get("text") or decision.get("content")
        if not isinstance(text, str):
            raise RuntimeError("TASK_FAILED:invalid_text_payload")
        os_backend.type_text(text)
        return None

    # ----------------------------------------------------------
    # HOTKEY / PRESS
    # ----------------------------------------------------------
    if operation in ("hotkey", "press"):
        keys = decision.get("keys")
        if not isinstance(keys, list) or not keys:
            raise RuntimeError("TASK_FAILED:invalid_hotkey_format")
        os_backend.press_keys(keys)
        return None

    # ----------------------------------------------------------
    # COMMAND EXECUTION
    # ----------------------------------------------------------
    if operation == "command":
        command = decision.get("command")
        if not isinstance(command, str) or not command.strip():
            raise RuntimeError("TASK_FAILED:invalid_command")
        return os_backend.run_command(command)

    # ----------------------------------------------------------
    # FILE CREATION — §R1
    # ----------------------------------------------------------
    if operation == "file_create":
        path = decision.get("path")
        content = decision.get("content", "")
        if not isinstance(path, str) or not path.strip():
            raise RuntimeError("TASK_FAILED:file_create_missing_path")
        if not isinstance(content, str):
            raise RuntimeError("TASK_FAILED:file_create_invalid_content")
        os_backend.write_file(path.strip(), content)
        return {"operation": "file_create", "path": path, "success": True}

    # ----------------------------------------------------------
    # VERIFICATION — §R1
    # ----------------------------------------------------------
    if operation == "verify":
        command = decision.get("command")
        if isinstance(command, str) and command.strip():
            result = os_backend.run_command(command.strip())
            return result
        # No command → visual verification pass-through
        return {"operation": "verify", "result": "visual_check", "success": True}

    # ----------------------------------------------------------
    # TOOL INSTALLATION
    # ----------------------------------------------------------
    if operation == "install":
        if installer is None:
            raise RuntimeError("TASK_FAILED:installer_unavailable")
        tool = decision.get("tool")
        if not isinstance(tool, dict):
            raise RuntimeError("TASK_FAILED:invalid_tool_specification")
        installer.install_tool(tool)
        return None

    # ----------------------------------------------------------
    # DONE
    # ----------------------------------------------------------
    if operation == "done":
        return None

    raise RuntimeError(f"TASK_FAILED:unsupported_operation:{operation}")
