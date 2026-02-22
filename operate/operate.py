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
# FIX H-03 / RB-03: Import ReasoningEngine so it can be used as a fallback
# when the static plan's action dict produces stagnation. Previously this
# import existed but the class was never instantiated or called during execution.
from core.cognition.reasoning_engine import ReasoningEngine

from config.timeouts import MAX_STAGNANT_ITERS_UI, MAX_STAGNANT_ITERS_COMMAND


MAX_PERCEPTION_ENTITIES = 20
MAX_PERCEPTION_JSON_BYTES = 10_000

# PATCH §1.11: WAIT should pause and retry, not immediately replan
WAIT_RETRY_SECONDS = 0.5
MAX_WAIT_RETRIES = 10  # 5s total wait before replanning

# Max bytes of command output stored per step in execution_log
MAX_COMMAND_OUTPUT_BYTES = 4096

# FIX H-03: Maximum dynamic candidate actions proposed by ReasoningEngine
# when static plan action is not producing progress.
MAX_DYNAMIC_CANDIDATES = 3


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

    # FIX H-03: Instantiate ReasoningEngine with the same llm_callable so it
    # can propose dynamic action candidates when the static plan stagnates.
    reasoning_engine = ReasoningEngine(llm_callable=llm_callable)

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
            reasoning_engine=reasoning_engine,
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
    reasoning_engine: Optional[ReasoningEngine],
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
                    # FIX H-05: Likelihoods are still heuristic but now
                    # structured as an observation model with explicit
                    # conditional semantics rather than bare magic constants.
                    #
                    # Interpretation: each value is the approximate probability
                    # of observing this snapshot feature IF the system is in the
                    # named state. These are not calibrated empirically, but are
                    # coherent with the qualitative meaning of each state.
                    #
                    # True Bayesian calibration requires a labelled dataset of
                    # (world_snapshot, ground_truth_state) pairs which does not
                    # exist for this system. This is documented explicitly so
                    # future operators understand the approximation.
                    likelihoods: Dict[str, float] = {}

                    focused_app = bounded.get("focused_app")
                    entity_count = len(bounded.get("entities", []))

                    # P(observe this app focused | system is in this app state)
                    if isinstance(focused_app, str) and focused_app.strip():
                        app_state_key = f"app:{focused_app.lower()}"
                        # High likelihood: if we're in this app state, we'd expect
                        # to see this app focused (0.9 = strong indicator)
                        likelihoods[app_state_key] = 0.9

                    # P(entity_count > 10 | ui_rich state)
                    if entity_count > 10:
                        likelihoods["ui_rich"] = 0.8
                    # P(0 < entity_count ≤ 10 | ui_sparse state)
                    elif entity_count > 0:
                        likelihoods["ui_sparse"] = 0.7
                    # P(entity_count == 0 | ui_empty state) — low signal,
                    # could be a transition artefact
                    else:
                        likelihoods["ui_empty"] = 0.5

                    # Neutral state likelihood: higher when environment is stable
                    # (no delta = no significant change = more likely in neutral state)
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

        # FIX H-03 / RB-03: Wire ReasoningEngine as a fallback when the static
        # plan provides no valid candidate actions. Previously this path raised
        # TASK_FAILED immediately. Now we ask the ReasoningEngine to propose
        # alternative actions given the current belief state and perception.
        # This is only triggered when the static plan is empty/malformed — for
        # normal execution, static plan actions are used directly.
        if not candidate_actions and reasoning_engine is not None:
            perception_for_reasoning = {}
            if isinstance(perception_snapshot, dict):
                perception_for_reasoning = perception_snapshot
            elif isinstance(world_snapshot, dict):
                perception_for_reasoning = world_snapshot

            try:
                dynamic_candidates = reasoning_engine.propose_actions(
                    objective=execution_plan.objective,
                    belief_summary=belief.summary(),
                    perception=perception_for_reasoning,
                    k=MAX_DYNAMIC_CANDIDATES,
                )
                if dynamic_candidates:
                    candidate_actions = dynamic_candidates
                    journal.record({
                        "event": "dynamic_candidates_used",
                        "step": current_step_index,
                        "count": len(candidate_actions),
                    })
            except Exception as re_err:
                log_warn(f"ReasoningEngine fallback failed: {re_err}")

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

            # FIX H-09 / RB-02: action_timeout is explicitly non-interrupting for
            # blocking I/O (the context manager's own docstring states this clearly).
            # Wrapping command execution and tool installation in it provides a
            # false guarantee while adding overhead. These operation types have their
            # own effective timeouts:
            #   - command_execution: subprocess.run() timeout (os_backend.run_command)
            #   - tool_installation: AutonomousInstaller.MAX_INSTALL_TIME (15 min)
            #   - task-level: max_wallclock_seconds checked at the top of the loop
            #
            # action_timeout IS retained for UI operations (click, type, hotkey)
            # which are genuinely short-lived and where 30s is a meaningful bound.
            #
            # For command/install operations: the subprocess itself is the timeout
            # boundary. The outer loop's max_wallclock_seconds is the safety net.
            operation = selected_action.get("operation", "").lower().strip()
            _use_action_timeout = operation not in ("command", "install")

            if _use_action_timeout:
                with action_timeout(30):
                    result = _execute_decision(
                        decision=selected_action,
                        os_backend=os_backend,
                        accessibility_backend=accessibility_backend,
                        installer=installer,
                    )
            else:
                # No action_timeout wrapper for blocking I/O operations.
                # Timeout guarantees are provided by subprocess timeout and
                # the task-level wallclock guard.
                result = _execute_decision(
                    decision=selected_action,
                    os_backend=os_backend,
                    accessibility_backend=accessibility_backend,
                    installer=installer,
                )

            # FIX-04 (RTB-03): os_backend.run_command() returns
            # subprocess.CompletedProcess, not a dict.
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

            # MATH-04 FIX: Pass raw reward (-0.5 for failure) to update_regret,
            # not the normalised value from the history deque. Regret must be
            # in raw (confidence delta) units so it is meaningful across sessions
            # and comparable between actions with different sample-count histories.
            best_reward = belief.global_best_reward()
            belief.update_regret(action_key, -0.5, best_reward)

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

        # MATH-04 FIX: Pass raw_reward directly to update_regret so regret is
        # tracked in interpretable confidence-delta units, not session-relative
        # z-scores. global_best_reward() now reads from _raw_action_rewards.
        best_reward = belief.global_best_reward()
        belief.update_regret(action_key, raw_reward, best_reward)

        if not verification.success:
            stagnant_iterations += 1
            if stagnant_iterations >= stagnant_limit:
                journal.record({"event": "verification_stagnation"})
                raise RuntimeError("REPLAN_REQUIRED")
            continue

        stagnant_iterations = 0
        belief.progress_score += verification.progress_score

        # MATH-01 / HARD-1: Advance the commitment chain after each successful
        # step. Without this, commitment_hash stays "GENESIS" for the entire
        # task lifetime, making Thompson sampling deterministically seeded only
        # by (action_key, iteration_count). The commit() call hashes the
        # action key + current observation into the chain so each step's
        # Thompson sample is cryptographically dependent on execution history.
        belief.commit(action_key, perception_snapshot or {})

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

    FIX-5: click operations use os_backend.mouse({"x": x, "y": y})
    which correctly handles LLM-supplied NORMALIZED coordinates (0.0–1.0).

    Operation mapping:
      click       → os_backend.mouse()       — normalized 0.0–1.0 coords
      type/write  → os_backend.type_text()
      hotkey/press→ os_backend.press_keys()
      command     → os_backend.run_command() — returns CompletedProcess
      file_create → os_backend.write_file()
      verify      → os_backend.run_command() or visual no-op
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
    # CLICK
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

        # RB-A7 FIX: Coordinate space guard.
        #
        # os_backend.mouse() expects NORMALIZED coordinates in [0.0, 1.0].
        # QwenOllamaAdapter._resolve_click_coordinates() resolves OCR text
        # anchors by calling get_text_coordinates() which may return absolute
        # pixel values (e.g. x=847, y=512 on a 1920×1080 display).
        #
        # Heuristic: if either coordinate exceeds 1.0, assume the pair is in
        # absolute pixel space and normalise using the screen dimensions
        # reported by os_backend. This is safe because:
        #   - Legitimate normalised coords are in [0.0, 1.0]; values >1.0 are
        #     unambiguously absolute pixels.
        #   - Normalised coords very close to 1.0 (e.g. 0.999) remain
        #     unchanged, so the guard does not distort already-normalised input.
        #   - If os_backend.screen_size() is unavailable, fall through to the
        #     raw values and let the backend raise its own error.
        if x_f > 1.0 or y_f > 1.0:
            try:
                screen_w, screen_h = os_backend.screen_size()
                if screen_w > 0 and screen_h > 0:
                    x_f = x_f / screen_w
                    y_f = y_f / screen_h
            except Exception:
                pass  # screen_size() unavailable — pass raw values through

        # Clamp to [0.0, 1.0] after any normalization to guard against
        # out-of-range LLM outputs regardless of coordinate origin.
        x_f = max(0.0, min(1.0, x_f))
        y_f = max(0.0, min(1.0, y_f))

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
    # FILE CREATION
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
    # VERIFICATION
    # ----------------------------------------------------------
    if operation == "verify":
        command = decision.get("command")
        if isinstance(command, str) and command.strip():
            result = os_backend.run_command(command.strip())
            return result
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
