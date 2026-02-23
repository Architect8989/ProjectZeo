from __future__ import annotations

import time
import json
from typing import Any, Dict, Optional, List

from authority.authority_policy import AuthorityDecision
from authority.input_arbitrator import InputArbitrator

from core.safety.action_timeout import action_timeout, run_with_timeout
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
from core.cognition.reasoning_engine import ReasoningEngine

from policy.engine import PolicyEngine, PolicyViolationError

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
    watchdog=None,
    prior_belief_state: Optional[dict] = None,
    belief_state_out: Optional[list] = None,
):
    """
    Main execution entry point.

    prior_belief_state (MATH-NEW-03 FIX):
        Optional dict from BeliefState.to_dict() from a previous replan attempt.
        When provided, the BeliefState is reconstructed from it via
        BeliefState.from_dict(), preserving action counts, regret history, Welford
        statistics, and Thompson counter state across replans. This prevents the
        bandit from forgetting that certain actions were ineffective and re-exploring
        them from scratch after a replan.

    belief_state_out (MATH-NEW-03 FIX):
        Optional single-element list. When provided, the final serialized
        BeliefState is placed in it (belief_state_out[0]) after the loop exits
        so the caller (main.py) can forward it to the next replan via
        prior_belief_state. Cleared and repopulated on every call.
    """
    if not isinstance(execution_plan, ExecutionPlan):
        raise ValueError("execution_plan must be an ExecutionPlan instance")

    execution_plan.validate()
    PlanVerifier().verify(execution_plan)

    os_backend = os_backend or OperatingSystem()

    policy_engine = PolicyEngine()
    try:
        import os as _os_mod, yaml as _yaml  # type: ignore[import]
        _policy_path = _os_mod.path.join(
            _os_mod.path.dirname(__file__), "..", "policy.yaml"
        )
        if _os_mod.path.exists(_policy_path):
            with open(_policy_path, "r") as _pf:
                _pcfg = _yaml.safe_load(_pf) or {}
            _allowed = _pcfg.get("allowed_apps")
            if isinstance(_allowed, list):
                policy_engine = PolicyEngine(allowed_apps=set(_allowed))
    except ImportError:
        pass
    except Exception:
        pass

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

    # SI-03 FIX: Use public get_llm_callable() instead of private attribute.
    if hasattr(planner, "get_llm_callable"):
        llm_callable = planner.get_llm_callable()
    else:
        llm_callable = getattr(planner, "_llm_call", None)
    if not callable(llm_callable):
        raise RuntimeError("Planner LLM callable unavailable")

    installer = None
    if observer is not None:
        _shared_client = getattr(planner, "_ollama_client", None)
        installer = AutonomousInstaller(
            observer=observer,
            os_backend=os_backend,
            llm_callable=llm_callable,
            shared_ollama_client=_shared_client,
        )

    reasoning_engine = ReasoningEngine(llm_callable=llm_callable)

    try:
        _execute_autonomous_loop(
            terminal_prompt=terminal_prompt,
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
            policy_engine=policy_engine,
            max_wallclock_seconds=max_wallclock_seconds,
            watchdog=watchdog,
            prior_belief_state=prior_belief_state,
            belief_state_out=belief_state_out,
        )
    finally:
        input_arbitrator.shutdown()


def _execute_autonomous_loop(
    *,
    terminal_prompt: str,
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
    policy_engine: PolicyEngine,
    max_wallclock_seconds: int,
    watchdog=None,
    prior_belief_state: Optional[dict] = None,
    belief_state_out: Optional[list] = None,
):

    start_ts = time.time()
    progress.start_execution()

    journal.record({
        "event": "execution_start",
        "objective": execution_plan.objective,
    })

    # MATH-NEW-03 FIX: Restore BeliefState from a prior replan when available.
    #
    # Root cause: each replan calls operate_main() fresh, constructing a new
    # BeliefState from scratch. The bandit forgets all action counts, regret
    # history, Welford statistics, and Thompson counter state from prior
    # attempts. On a second replan for the same intent, low-reward actions
    # that were clearly identified as ineffective in the first run are
    # re-explored from the uninformative uniform prior.
    #
    # Fix: when prior_belief_state is provided (a dict from BeliefState.to_dict()
    # persisted by the caller after the previous operate_main() call), reconstruct
    # via BeliefState.from_dict(). The reconstructed instance preserves action
    # counts, reward history, Welford stats, commitment_hash, and Thompson
    # counters, so the second replan can exploit what was already learned.
    #
    # A fresh BeliefState is always used when no prior state is provided (first
    # run) or when restoration fails (safe fallback — task continues with
    # degraded cross-replan learning but no crash).
    belief: BeliefState
    if prior_belief_state is not None:
        try:
            belief = BeliefState.from_dict(prior_belief_state, intent_hash=terminal_prompt)
            journal.record({"event": "belief_state_restored", "from_prior": True})
        except Exception as _bs_err:
            belief = BeliefState(intent_hash=terminal_prompt)
            journal.record({
                "event": "belief_state_restore_failed",
                "error": str(_bs_err),
                "fallback": "fresh_state",
            })
    else:
        belief = BeliefState(intent_hash=terminal_prompt)

    # MATH-NEW-01 FIX: set_plan_horizon() is called fresh on every
    # operate_main() invocation, so each replan re-tunes regret decay to the
    # new plan's actual step count. There is no frozen-initial-plan-length
    # issue because the loop does not survive across replans.
    _plan_real_steps = max(len(execution_plan.steps) - 1, 1)
    belief.set_plan_horizon(_plan_real_steps)

    # MATH-NEW-02 FIX: Tune EXPLOIT_SATURATION_N to the plan horizon so that
    # short plans don't remain in exploration-heavy weighting for their entire
    # execution. ActionRanker.set_plan_horizon() sets saturation to
    # max(10, plan_steps * 2): short plans saturate at ~20 visits, long plans
    # at proportionally higher values.
    action_ranker = ActionRanker()
    action_ranker.set_plan_horizon(_plan_real_steps)

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

    try:
        while iteration < max_iterations:

            if time.time() - start_ts > max_wallclock_seconds:
                journal.record({"event": "execution_timeout"})
                raise RuntimeError("TASK_FAILED:timeout")

            if watchdog is not None:
                watchdog.check()

            if current_step_index >= len(execution_plan.steps):
                journal.record({"event": "execution_complete"})
                return

            input_arbitrator.soc_action_started()
            input_arbitrator.clear_emergency_reclaim()

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
                            app_state_key = f"app:{focused_app.lower()}"
                            likelihoods[app_state_key] = 0.9

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

            _soc_initial = belief.environment_stability > 0.7

            authority = input_arbitrator.evaluate(
                input_event_ts=time.monotonic(),
                high_risk=is_high_risk,
                soc_confident=_soc_initial,
            )

            if authority == AuthorityDecision.ABORT:
                raise AuthorityAbortError("Human authority abort — task terminated")

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

                _policy_node = None
                if accessibility_backend is not None:
                    try:
                        _policy_node = accessibility_backend.get_focused_node()
                    except Exception:
                        _policy_node = None

                if _policy_node is None:
                    _focused_app = ""
                    if isinstance(world_snapshot, dict):
                        _focused_app = str(world_snapshot.get("focused_app", ""))

                    _sentinel_app = _focused_app if _focused_app.strip() else "__unknown_app__"

                    class _LightweightNode:
                        """Minimal AT-SPI-compatible node built from perception."""
                        def getRoleName(self):
                            return "unknown"
                        @property
                        def name(self):
                            return ""
                        def getApplication(self):
                            class _App:
                                name = _sentinel_app
                            return _App()

                    _policy_node = _LightweightNode()

                _policy_verdict, _policy_reason = policy_engine.validate(
                    _policy_node,
                    selected_action.get("operation", ""),
                )

                if _policy_verdict == PolicyEngine.DENY:
                    journal.record({
                        "event": "policy_violation",
                        "action": selected_action.get("operation"),
                        "reason": _policy_reason,
                        "step": current_step_index,
                    })
                    raise PolicyViolationError(
                        f"PolicyEngine DENY: {_policy_reason}"
                    )

                if _policy_verdict == PolicyEngine.REQUIRE_HUMAN_CONFIRMATION:
                    journal.record({
                        "event": "policy_human_confirmation_required",
                        "action": selected_action.get("operation"),
                        "reason": _policy_reason,
                        "step": current_step_index,
                    })
                    raise RuntimeError("REPLAN_REQUIRED")

                if is_high_risk:
                    soc_confident = belief.consecutive_high_stability_count >= 3
                else:
                    soc_confident = belief.environment_stability > 0.7

                operation = selected_action.get("operation", "").lower().strip()
                _use_action_timeout = operation not in ("command", "install")

                if _use_action_timeout:
                    result = run_with_timeout(
                        lambda: _execute_decision(
                            decision=selected_action,
                            os_backend=os_backend,
                            accessibility_backend=accessibility_backend,
                            installer=installer,
                        ),
                        seconds=30,
                        operation_hint=operation,
                    )
                else:
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

            except PolicyViolationError as pve:
                # FIX RB-NEW-01 / SI-NEW-01: Absorb policy violations as stagnation
                # events rather than terminating the task immediately.
                #
                # Root cause: the previous code raised RuntimeError("TASK_FAILED:policy_blocked"),
                # which propagated past the replan logic in main.py and triggered
                # _force_safe_shutdown(). A single policy denial permanently ended the task.
                #
                # Fix: record the block, score the action as a failure, and increment
                # the stagnation counter. When the stagnation limit is reached the
                # REPLAN_REQUIRED path fires, giving the planner the opportunity to
                # route around the blocked application. This matches the design intent
                # in policy/engine.py and audit finding RB-NEW-01.
                #
                # Operators who need immediate termination on any policy denial should
                # subclass PolicyEngine and raise a non-REPLAN RuntimeError directly
                # from validate().
                journal.record({
                    "event": "policy_blocked",
                    "reason": str(pve),
                    "step": current_step_index,
                    "action": selected_action.get("operation"),
                })
                belief.record_action(action_key, reward=-0.5)
                best_reward = belief.global_best_reward()
                if best_reward is not None:
                    belief.update_regret(action_key, -0.5, best_reward)
                belief.commit(
                    action_key,
                    {"outcome": "policy_denied", "step": current_step_index},
                )
                stagnant_iterations += 1
                if stagnant_iterations >= stagnant_limit:
                    journal.record({"event": "policy_stagnation_replan"})
                    raise RuntimeError("REPLAN_REQUIRED")
                continue

            except Exception:
                belief.record_action(action_key, reward=-0.5)

                best_reward = belief.global_best_reward()
                if best_reward is not None:
                    belief.update_regret(action_key, -0.5, best_reward)

                belief.commit(
                    action_key,
                    {"outcome": "failure", "step": current_step_index},
                )

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

            # MATH-04 FIX: Pass raw_reward to update_regret (interpretable units).
            # HAR-1 (MATH-1): Skip regret update when global_best_reward() is None.
            best_reward = belief.global_best_reward()
            if best_reward is not None:
                belief.update_regret(action_key, raw_reward, best_reward)

            if not verification.success:
                stagnant_iterations += 1
                if stagnant_iterations >= stagnant_limit:
                    journal.record({"event": "verification_stagnation"})
                    raise RuntimeError("REPLAN_REQUIRED")
                continue

            stagnant_iterations = 0
            belief.progress_score += verification.progress_score

            # MATH-01 / HARD-1: Advance the commitment chain after each successful step.
            belief.commit(action_key, perception_snapshot or {})

            previous_perception = perception_snapshot
            current_step_index += 1

            if current_step.type.name == "DONE":
                journal.record({"event": "execution_complete"})
                return

            _real_steps = max(len(execution_plan.steps) - 1, 1)
            if belief.converged(
                min_iterations=3,
                current_iteration=iteration,
                plan_steps_total=_real_steps,
                steps_completed=current_step_index,
            ):
                journal.record({"event": "execution_converged"})
                return

        journal.record({"event": "iteration_budget_exhausted"})
        raise RuntimeError("TASK_FAILED:iteration_budget_exhausted")

    finally:
        # MATH-NEW-03 FIX: Capture the final BeliefState for the caller so it
        # can be forwarded to the next replan via prior_belief_state. This
        # fires on all exit paths (success, REPLAN_REQUIRED, TASK_FAILED, etc.)
        # so the caller always gets the most recent bandit state.
        if belief_state_out is not None:
            try:
                belief_state_out.clear()
                belief_state_out.append(belief.to_dict())
            except Exception:
                pass  # Serialization failure must never mask the original exception.


def _execute_decision(
    *,
    decision: Dict[str, Any],
    os_backend: OperatingSystem,
    accessibility_backend,
    installer: Optional[AutonomousInstaller],
):

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

        if x_f > 1.0 or y_f > 1.0:
            try:
                screen_w, screen_h = os_backend.screen_size()
                if screen_w > 0 and screen_h > 0:
                    x_f = x_f / screen_w
                    y_f = y_f / screen_h
                else:
                    raise RuntimeError(
                        "TASK_FAILED:click_screen_size_zero "
                        f"screen=({screen_w},{screen_h}) raw=({x_f},{y_f})"
                    )
            except RuntimeError:
                raise
            except Exception as _sse:
                raise RuntimeError(
                    f"TASK_FAILED:click_screen_size_unavailable "
                    f"raw=({x_f},{y_f}) reason={_sse}"
                ) from _sse

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
        # FIX-8: Return success=None (inconclusive) — not success=True.
        return {"operation": "verify", "result": "visual_check", "success": None}

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
