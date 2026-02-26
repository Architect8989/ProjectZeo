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


try:
    import pyautogui as _pyautogui
    _PYAUTOGUI_AVAILABLE: bool = True
except ImportError:
    _pyautogui = None  # type: ignore[assignment]
    _PYAUTOGUI_AVAILABLE: bool = False

MAX_PERCEPTION_ENTITIES = 20
MAX_PERCEPTION_JSON_BYTES = 10_000


REPLAN_SIGNAL: str = "REPLAN_REQUIRED"

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

            
            _focused_app_for_policy = (
                world_snapshot.get("focused_app", "__unknown_app__")
                if isinstance(world_snapshot, dict)
                else "__unknown_app__"
            )
            _policy_decision, _policy_reason = policy_engine.validate_action_dict(
                selected_action,
                focused_app=_focused_app_for_policy,
            )

            if _policy_decision == PolicyEngine.DENY:
                belief.record_action(action_key, -0.5)
                best_reward = belief.global_best_reward() or 0.0
                belief.update_regret(action_key, -0.5, best_reward)
                journal.record({
                    "event": "policy_deny",
                    "step": current_step_index,
                    "action_key": action_key,
                    "reason": _policy_reason,
                })
                stagnant_iterations += 1
                if stagnant_iterations >= stagnant_limit:
                    raise RuntimeError(REPLAN_SIGNAL)
                previous_perception = perception_snapshot
                continue

            if _policy_decision == PolicyEngine.REQUIRE_HUMAN_CONFIRMATION:
                # Pause and wait for human to approve (up to MAX_WAIT_RETRIES
                # slots) before re-evaluating.  If authority clears → proceed.
                journal.record({
                    "event": "policy_human_confirmation_required",
                    "step": current_step_index,
                    "action_key": action_key,
                    "reason": _policy_reason,
                })
                _phc_wait = 0
                _phc_approved = False
                while _phc_wait < MAX_WAIT_RETRIES:
                    time.sleep(WAIT_RETRY_SECONDS)
                    _phc_wait += 1
                    _re_policy, _ = policy_engine.validate_action_dict(
                        selected_action,
                        focused_app=_focused_app_for_policy,
                    )
                    if _re_policy == PolicyEngine.ALLOW:
                        _phc_approved = True
                        break
                if not _phc_approved:
                    # Still blocked — treat as stagnation.
                    belief.record_action(action_key, -0.3)
                    best_reward = belief.global_best_reward() or 0.0
                    belief.update_regret(action_key, -0.3, best_reward)
                    stagnant_iterations += 1
                    if stagnant_iterations >= stagnant_limit:
                        raise RuntimeError(REPLAN_SIGNAL)
                    previous_perception = perception_snapshot
                    continue

            is_high_risk = selected_action.get("operation") in {
                "command", "install", "file_create"
            }

            
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
                
                _entropy = belief.entropy() if hasattr(belief, "entropy") else 0.0
                journal.record({
                    "event": "replan_trigger",
                    "replan_signal": REPLAN_SIGNAL,
                    "step": current_step_index,
                    "stagnant_iterations": stagnant_iterations,
                    "stagnant_limit": stagnant_limit,
                    "iteration": iteration,
                    "belief_entropy": round(_entropy, 4),
                })
                raise RuntimeError(REPLAN_SIGNAL)

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
            # RT-05 FIX: Use the module-level _pyautogui alias (imported at the
            # top of this file) and the _PYAUTOGUI_AVAILABLE flag instead of
            # deferring `import pyautogui` here.
            #
            # If pyautogui is unavailable, return a structured failure with a
            # clear reason so operators can diagnose the dependency gap.  The
            # failure reward (-0.5) is identical to other hard failures and
            # feeds correctly into the Welford normaliser and Thompson sampler.
            if not _PYAUTOGUI_AVAILABLE:
                return {
                    "success": False,
                    "reward": -0.5,
                    "reason": (
                        "pyautogui_unavailable: scroll operation requires pyautogui. "
                        "Install with: pip install pyautogui"
                    ),
                }
            direction = str(action.get("direction", "down")).lower()
            clicks = int(action.get("clicks", 3))
            amount = clicks if direction == "up" else -clicks
            run_with_timeout(
                lambda: _pyautogui.scroll(amount),
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
            # RT-B1 FIX: Include "returncode" so StepVerifier._verify_command()
            # can extract it from the dict.  Previously the dict lacked this key,
            # causing hasattr(result, "returncode") to be False, making every
            # COMMAND_EXECUTION step fail verification unconditionally.
            return {
                "success": success,
                "reward": reward,
                "output": output,
                "returncode": result.returncode,
            }

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

