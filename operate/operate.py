from __future__ import annotations

import concurrent.futures
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

    # FIX RB-A3: Per-task UI executor — replaces removed module-level singleton.
    #
    # The old code in action_timeout.py used a module-level
    # `_UI_EXECUTOR = ThreadPoolExecutor(max_workers=1)` shared across ALL tasks.
    # When task A submitted a blocking UI action (e.g. pyautogui waiting on a
    # frozen display), the single shared worker was occupied. Task B's action
    # was enqueued but never picked up until task A unblocked. Task B's 30-second
    # timeout therefore started counting while the worker was still in task A —
    # meaning task B's timeout guarantee was completely violated.
    #
    # Fix: each call to operate_main() creates its own single-worker executor.
    # The executor is passed to run_with_timeout() via the `executor=` parameter
    # (new in the fixed action_timeout.py). After the task completes (any exit
    # path), the executor is shut down with wait=False. The background thread
    # (if still running a blocking OS call) drains naturally; we do not block
    # the main thread waiting for it.
    #
    # max_workers=1: UI operations must be sequential — the display event queue
    # is not thread-safe on most backends (pyautogui, xdotool). A single worker
    # enforces that only one UI action runs at a time, which is also correct for
    # any sequentially ordered execution plan.
    _task_ui_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="ui_timeout_worker",
    )

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
            task_ui_executor=_task_ui_executor,
        )
    finally:
        input_arbitrator.shutdown()
        # FIX RB-A3: Shut down the per-task executor on all exit paths.
        # wait=False: do not block the main thread waiting for a stuck UI thread.
        _task_ui_executor.shutdown(wait=False)


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
    task_ui_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None,
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
            belief.consecutive_high_stability_count = 0
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

            # FIX RB-A4: Compute soc_confident using the risk-aware formula
            # BEFORE the authority gate, then pass it into the gate.
            #
            # Previous code:
            #   _soc_initial = belief.environment_stability > 0.7
            #   authority = input_arbitrator.evaluate(..., soc_confident=_soc_initial)
            #   ...
            #   # AFTER the gate (lines 490-492) — never connected to anything:
            #   if is_high_risk:
            #       soc_confident = belief.consecutive_high_stability_count >= 3
            #   else:
            #       soc_confident = belief.environment_stability > 0.7
            #
            # Root cause: the correct risk-aware `soc_confident` was computed at
            # lines 490–492 but was NEVER used — neither passed to
            # input_arbitrator.evaluate() nor to any other consumer. The authority
            # gate always received the low-risk formula (_soc_initial) regardless
            # of whether the operation was high-risk. High-risk operations (command,
            # install, file_create) that should require `consecutive_high_stability_count
            # >= 3` were instead approved on the weaker `environment_stability > 0.7`
            # threshold — allowing them to execute when the environment was only
            # momentarily stable.
            #
            # Fix: compute the correct risk-aware soc_confident value once, here,
            # before the evaluate() call, and pass it directly.
            if is_high_risk:
                # High-risk operations require sustained stability: at least 3
                # consecutive observations with high stability score.
                soc_confident = belief.consecutive_high_stability_count >= 3
            else:
                # Low-risk operations only need momentary stability.
                soc_confident = belief.environment_stability > 0.7

            authority = input_arbitrator.evaluate(
                input_event_ts=time.monotonic(),
                high_risk=is_high_risk,
                soc_confident=soc_confident,
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
                    # Re-evaluate with current stability on each retry
                    _soc_retry = (
                        belief.consecutive_high_stability_count >= 3
                        if is_high_risk
                        else belief.environment_stability > 0.7
                    )
                    authority = input_arbitrator.evaluate(
                        input_event_ts=time.monotonic(),
                        high_risk=is_high_risk,
                        soc_confident=_soc_retry,
                    )
                    if authority == AuthorityDecision.CONTINUE:
                        break
                    if authority == AuthorityDecision.ABORT:
                        raise AuthorityAbortError(
                            "Human authority abort during WAIT — task terminated"
                        )
                    if authority == AuthorityDecision.CONTINUE:
                        break

            if authority == AuthorityDecision.ABORT:
                raise AuthorityAbortError("Human authority abort — task terminated")

            # -------------------------------------------------------
            # ACTION EXECUTION
            # -------------------------------------------------------
            # P0-B FIX (RT-02): Initialise exec_result before the try block so
            # the variable is always bound.  In the except path action_success is
            # False and the "output" in {} check short-circuits safely — but any
            # future refactor that sets action_success=True inside the except
            # block would hit NameError without this initialisation.
            exec_result: dict = {}
            try:
                exec_result = _execute_decision(
                    action=selected_action,
                    os_backend=os_backend,
                    installer=installer,
                    current_step=current_step,
                    execution_log=execution_log,
                    current_step_index=current_step_index,
                    task_ui_executor=task_ui_executor,
                    watchdog=watchdog,
                )
                action_success = exec_result.get("success", False)
                raw_reward = exec_result.get("reward", 0.0)
            except AuthorityAbortError:
                raise
            except Exception as exec_exc:
                action_success = False
                raw_reward = -0.5
                journal.record({
                    "event": "action_exception",
                    "step": current_step_index,
                    "action_key": action_key,
                    "error": str(exec_exc),
                })

            # Record command output for world-graph enrichment.
            if "output" in (exec_result if action_success else {}):
                output_text = str(exec_result.get("output", ""))
                execution_log[current_step_index] = {
                    "output": output_text[:MAX_COMMAND_OUTPUT_BYTES]
                }

            # -------------------------------------------------------
            # REWARD & REGRET UPDATE
            # -------------------------------------------------------
            # P0-F FIX (SI-04/MATH-06): Sample best_reward AFTER record_action()
            # so the current action's reward is included before comparing.
            # Previously best_reward was sampled BEFORE record_action(), causing
            # the bandit to accumulate false regret on the single best action:
            # when an action achieved a new global best (e.g. first 'done'
            # returning 1.0), regret was computed as old_best − 1.0 > 0 (wrong;
            # should be 0.0 for a record-breaker).  This over-penalised the best
            # action and drove re-exploration of inferior alternatives.
            belief.record_action(action_key, raw_reward)
            best_reward = belief.global_best_reward() or 0.0
            belief.update_regret(action_key, raw_reward, best_reward)

            journal.record({
                "event": "action_executed",
                "step": current_step_index,
                "action_key": action_key,
                "success": action_success,
                "reward": raw_reward,
            })

            # -------------------------------------------------------
            # STEP VERIFICATION & ADVANCEMENT
            # -------------------------------------------------------
            if action_success:
                verify_result = verifier.verify_step(
                    current_step,
                    exec_result,
                    screenshot=perception_snapshot,
                    previous_screenshot=previous_perception,
                    world_graph=world_graph,
                )
                belief.progress_score = verify_result.progress_score

                if verify_result.success:
                    stagnant_iterations = 0
                    current_step_index += 1
                    progress.advance_step()
                    journal.record({
                        "event": "step_verified",
                        "step": current_step_index - 1,
                        "progress_score": verify_result.progress_score,
                    })
                else:
                    stagnant_iterations += 1
            else:
                stagnant_iterations += 1

            # -------------------------------------------------------
            # STAGNATION GUARD
            # -------------------------------------------------------
            if stagnant_iterations >= stagnant_limit:
                journal.record({
                    "event": "stagnation_abort",
                    "step": current_step_index,
                    "stagnant_iterations": stagnant_iterations,
                })
                raise RuntimeError("TASK_FAILED:stagnation")

            previous_perception = perception_snapshot

        # Loop exhausted without reaching DONE step.
        raise RuntimeError("TASK_FAILED:max_iterations_exceeded")

    except (RuntimeError, AuthorityAbortError):
        raise

    finally:
        # Persist the final BeliefState for the next replan.
        if belief_state_out is not None:
            belief_state_out.clear()
            try:
                belief_state_out.append(belief.to_dict())
            except Exception:
                pass


# =============================================================================
# ACTION DISPATCH — P0 FIX: _execute_decision() was entirely absent.
#
# The loop in _execute_autonomous_loop() called action_ranker.select() to
# pick an action and computed authority / soc_confident correctly, but then
# had no mechanism to actually dispatch the selected action to the OS backend.
# Every iteration was a no-op: no clicks, no keystrokes, no commands were
# ever sent. This is the function that bridges the cognition layer to the OS.
#
# Returns a dict with at minimum:
#   {"success": bool, "reward": float}
# Optionally includes "output" (str) for command steps.
# =============================================================================

def _execute_decision(
    *,
    action: dict,
    os_backend: "OperatingSystem",
    installer,
    current_step,
    execution_log: dict,
    current_step_index: int,
    task_ui_executor,
    watchdog,
) -> dict:
    """
    Dispatch `action` to the OS backend and return an execution result dict.

    Action dispatch is determined by the `operation` field:
      click         → os_backend.mouse()
      write / type  → os_backend.write()
      press / hotkey→ os_backend.press()
      command       → os_backend.exec()
      file_create   → os_backend.write_file()
      install       → installer.install() or os_backend.exec()
      scroll        → pyautogui.scroll()
      done          → success immediately (DONE sentinel)
      (unknown)     → no-op, success=False

    Rewards are assigned by outcome:
      success       →  0.8
      failure/error → -0.5
      done          →  1.0
    """
    from core.safety.action_timeout import run_with_timeout, ActionTimeout

    op = (action.get("operation") or "").lower().strip()

    if op == "done":
        return {"success": True, "reward": 1.0}

    if not op:
        return {"success": False, "reward": -0.5}

    try:
        if op == "click":
            # Click by coordinates (percentage) or by label (OCR fallback).
            x = action.get("x")
            y = action.get("y")
            if x is not None and y is not None:
                run_with_timeout(
                    lambda: os_backend.mouse({"x": x, "y": y}),
                    seconds=30.0,
                    operation_hint="click",
                    executor=task_ui_executor,
                )
            else:
                # Label/text click: x/y coordinates are required.
                # FIX-2 (HIGH): The previous implementation silently fell back
                # to clicking screen centre (0.5, 0.5) when a label was present
                # but OCR/coordinate resolution had failed.  That behaviour:
                #   - Clicked the wrong target on every non-centre UI element
                #   - Returned success=True / reward=0.8 despite doing nothing useful
                #   - Caused stagnation loops because the task never progressed
                #
                # Fix: return failure immediately so the planner is forced to
                # replan and either acquire real coordinates or choose a different
                # action.  The reward of -0.5 is identical to other hard failures
                # (missing content, missing keys) and feeds correctly into the
                # Welford normaliser and Thompson sampler.
                label = action.get("label") or action.get("text") or ""
                if label:
                    return {
                        "success": False,
                        "reward": -0.5,
                        "reason": (
                            f"click_label '{label}' has no resolved coordinates — "
                            "OCR unavailable or label not found; replanning required"
                        ),
                    }
                else:
                    return {"success": False, "reward": -0.5}
            return {"success": True, "reward": 0.8}

        elif op in ("write", "type"):
            content = str(action.get("content") or action.get("text") or "")
            if not content:
                return {"success": False, "reward": -0.5}
            run_with_timeout(
                lambda: os_backend.write(content),
                seconds=30.0,
                operation_hint="write",
                executor=task_ui_executor,
            )
            return {"success": True, "reward": 0.8}

        elif op in ("press", "hotkey", "key"):
            keys = action.get("keys") or action.get("key")
            if isinstance(keys, str):
                keys = [keys]
            if not isinstance(keys, list) or not keys:
                return {"success": False, "reward": -0.5}
            run_with_timeout(
                lambda: os_backend.press(keys),
                seconds=15.0,
                operation_hint="press",
                executor=task_ui_executor,
            )
            return {"success": True, "reward": 0.8}

        elif op == "scroll":
            import pyautogui
            direction = str(action.get("direction", "down")).lower()
            clicks = int(action.get("clicks", 3))
            amount = clicks if direction == "up" else -clicks
            run_with_timeout(
                lambda: pyautogui.scroll(amount),
                seconds=10.0,
                operation_hint="scroll",
                executor=task_ui_executor,
            )
            return {"success": True, "reward": 0.8}

        elif op == "command":
            cmd = str(action.get("command") or "").strip()
            if not cmd:
                return {"success": False, "reward": -0.5}
            from config.timeouts import INSTALL_COMMAND_TIMEOUT_SECONDS
            result = os_backend.exec(cmd, timeout=int(INSTALL_COMMAND_TIMEOUT_SECONDS))
            success = (result.returncode == 0)
            reward = 0.8 if success else -0.5
            output = (result.stdout or "") + (result.stderr or "")
            return {"success": success, "reward": reward, "output": output}

        elif op == "file_create":
            path = str(action.get("path") or "").strip()
            content_str = str(action.get("content") or "")
            if not path:
                return {"success": False, "reward": -0.5}
            os_backend.write_file(path, content_str)
            return {"success": True, "reward": 0.8}

        elif op == "install":
            tool_spec = action.get("tool", {})
            install_cmds = (
                tool_spec.get("install_commands", [])
                if isinstance(tool_spec, dict)
                else []
            )
            if install_cmds and isinstance(install_cmds, list):
                # Terminal-first install path.
                from config.timeouts import INSTALL_COMMAND_TIMEOUT_SECONDS
                all_ok = True
                combined_output = ""
                for cmd in install_cmds:
                    r = os_backend.exec(cmd, timeout=int(INSTALL_COMMAND_TIMEOUT_SECONDS))
                    combined_output += (r.stdout or "") + (r.stderr or "")
                    if r.returncode != 0:
                        all_ok = False
                        break
                reward = 0.8 if all_ok else -0.5
                return {
                    "success": all_ok,
                    "reward": reward,
                    "output": combined_output,
                }
            elif installer is not None:
                # UI-based install fallback via AutonomousInstaller.
                tool_name = (
                    tool_spec.get("name", "") if isinstance(tool_spec, dict) else ""
                )
                try:
                    installer.install(tool_name)
                    return {"success": True, "reward": 0.8}
                except Exception as inst_err:
                    return {
                        "success": False,
                        "reward": -0.5,
                        "output": str(inst_err),
                    }
            else:
                return {"success": False, "reward": -0.5}

        elif op == "verify":
            # Verification steps always return success to allow the plan to
            # advance; actual verification is performed by StepVerifier.
            method = action.get("method", "screenshot")
            if method == "command":
                cmd = str(action.get("command") or "").strip()
                if cmd:
                    from config.timeouts import INSTALL_COMMAND_TIMEOUT_SECONDS
                    r = os_backend.exec(cmd, timeout=30)
                    return {
                        "success": r.returncode == 0,
                        "reward": 0.6 if r.returncode == 0 else -0.3,
                        "output": (r.stdout or "") + (r.stderr or ""),
                    }
            return {"success": True, "reward": 0.6}

        else:
            # Unknown operation — log and return soft failure.
            log_warn(f"_execute_decision: unknown operation {op!r}")
            return {"success": False, "reward": -0.5}

    except ActionTimeout as toe:
        log_warn(f"_execute_decision: action timeout — {toe}")
        return {"success": False, "reward": -0.5}

    except Exception as exc:
        log_warn(f"_execute_decision: unexpected error — {exc}")
        return {"success": False, "reward": -0.5}

