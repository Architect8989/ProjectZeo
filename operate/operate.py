from __future__ import annotations

import concurrent.futures
import hashlib
import os
import sys
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


# pyautogui is optional — import once at module level (FIX RT-05)
try:
    import pyautogui as _pyautogui
    _PYAUTOGUI_AVAILABLE: bool = True
except ImportError:
    _pyautogui = None  # type: ignore[assignment]
    _PYAUTOGUI_AVAILABLE: bool = False


# ------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------

MAX_PERCEPTION_ENTITIES = 20
MAX_PERCEPTION_JSON_BYTES = 10_000

REPLAN_SIGNAL: str = "REPLAN_REQUIRED"

# WAIT should pause and retry, not immediately replan (PATCH §1.11)
WAIT_RETRY_SECONDS = 0.5
MAX_WAIT_RETRIES = 10  # 5 seconds total before giving up and replanning

# Max bytes of command output stored per step in execution_log (bounded, not unlimited)
MAX_COMMAND_OUTPUT_BYTES = 4096

# Maximum dynamic candidates from ReasoningEngine on stagnant steps (H-03 fix)
MAX_DYNAMIC_CANDIDATES = 3



_SIGNAL_DIR: str = "/tmp"
_SIGNAL_PREFIX: str = "projectzeo_approve_"


def _approval_signal_path(action_key: str) -> str:
    """Return the path of the approval signal file for this action key."""
    return os.path.join(_SIGNAL_DIR, f"{_SIGNAL_PREFIX}{action_key}.signal")


def _write_approval_signal(action_key: str, action: dict, reason: str) -> str:
    """Write the pending-approval signal file and return its path."""
    path = _approval_signal_path(action_key)
    try:
        content = json.dumps(
            {
                "action_key": action_key,
                "action": action,
                "reason": reason,
                "instruction": (
                    f"Delete this file to approve the action: {path}\n"
                    "The file will be cleaned up automatically after "
                    f"{MAX_WAIT_RETRIES * WAIT_RETRY_SECONDS:.0f}s."
                ),
            },
            indent=2,
        )
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
    except OSError as e:
        # /tmp write failure is non-fatal; log and continue to timed-out denial
        print(
            f"[OPERATE] Warning: could not write approval signal file {path!r}: {e}",
            file=sys.stderr,
        )
    return path


def _remove_approval_signal(path: str) -> None:
    """Remove the signal file; ignore errors (already deleted by user, race, etc.)."""
    try:
        os.remove(path)
    except OSError:
        pass


class AuthorityAbortError(RuntimeError):
    """Raised when a human-authority decision requires immediate task termination."""
    pass


# =========================================================================
# PUBLIC ENTRY POINT
# =========================================================================

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
) -> None:
    
    if not isinstance(execution_plan, ExecutionPlan):
        raise ValueError("execution_plan must be an ExecutionPlan instance")

    execution_plan.validate()
    PlanVerifier().verify(execution_plan)

    os_backend = os_backend or OperatingSystem()

    # ------------------------------------------------------------------
    # PolicyEngine — load allowed apps from policy.yaml if available
    # ------------------------------------------------------------------
    policy_engine = PolicyEngine()
    try:
        import os as _os_mod
        import yaml as _yaml  # type: ignore[import]
        _policy_path = _os_mod.path.join(
            _os_mod.path.dirname(__file__), "..", "policy.yaml"
        )
        if _os_mod.path.exists(_policy_path):
            with open(_policy_path, "r", encoding="utf-8") as _pf:
                _pcfg = _yaml.safe_load(_pf) or {}
            _allowed = _pcfg.get("allowed_apps")
            if isinstance(_allowed, list):
                policy_engine = PolicyEngine(allowed_apps=set(_allowed))
    except ImportError:
        # FIX RB-3: Log the missing dependency explicitly rather than silently
        # ignoring the failure.  Operators must know the policy file was not
        # loaded so they can install pyyaml or accept the default allowlist.
        print(
            "[operate_main] WARNING: pyyaml is not installed — policy.yaml was not loaded. "
            "PolicyEngine is running with the built-in default application allowlist. "
            "To enable custom policy configuration, install pyyaml: "
            "  pip install pyyaml",
            file=sys.stderr,
        )
    except Exception as _policy_err:
        # Non-fatal: policy.yaml may be malformed; log and continue with defaults
        print(
            f"[operate_main] WARNING: Failed to load policy.yaml: {_policy_err}. "
            "PolicyEngine is running with the built-in default application allowlist.",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # Accessibility backend (optional AT-SPI path)
    # ------------------------------------------------------------------
    try:
        accessibility_backend = AccessibilityBackend()
        if observer is not None:
            accessibility_backend.wire(observer=observer)
    except Exception:
        accessibility_backend = None

    # ------------------------------------------------------------------
    # Core services
    # ------------------------------------------------------------------
    journal = ActionJournal()
    input_arbitrator = InputArbitrator()
    verifier = StepVerifier()
    progress = ProgressTracker(execution_plan)

    # SI-03 FIX: Use public get_llm_callable() — never reach into private _llm_call
    if hasattr(planner, "get_llm_callable"):
        llm_callable = planner.get_llm_callable()
    else:
        llm_callable = getattr(planner, "_llm_call", None)
    if not callable(llm_callable):
        raise RuntimeError(
            "Planner LLM callable unavailable — planner must expose get_llm_callable() "
            "or have a callable _llm_call attribute."
        )

    # AutonomousInstaller — wired when observer is available
    installer: Optional[AutonomousInstaller] = None
    if observer is not None:
        _shared_client = getattr(planner, "_ollama_client", None)
        installer = AutonomousInstaller(
            observer=observer,
            os_backend=os_backend,
            llm_callable=llm_callable,
            shared_ollama_client=_shared_client,
        )

    reasoning_engine = ReasoningEngine(llm_callable=llm_callable)

    # FIX RB-A3: Per-task executor — isolates timeout guarantees between tasks.
    # A module-level singleton executor shares ONE worker thread; a blocking call
    # in one task starves the next.  Creating per-task gives each task its own
    # isolated thread pool.
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
        # wait=False: never block the main thread waiting for a stuck UI thread.
        _task_ui_executor.shutdown(wait=False)


# =========================================================================
# AUTONOMOUS EXECUTION LOOP
# =========================================================================

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
) -> None:

    start_ts = time.time()
    progress.start_execution()

    journal.record({
        "event": "execution_start",
        "objective": execution_plan.objective,
    })

    # ------------------------------------------------------------------
    # BeliefState — reconstruct from prior replan if available
    # ------------------------------------------------------------------
    belief: BeliefState
    if prior_belief_state is not None:
        try:
            belief = BeliefState.from_dict(
                prior_belief_state, intent_hash=terminal_prompt
            )
            belief.consecutive_high_stability_count = 0  # reset stability on replan
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

    # MATH-NEW-01: set_plan_horizon() called fresh per operate_main() invocation
    # so each replan re-tunes regret decay to the new plan's step count
    _plan_real_steps = max(len(execution_plan.steps) - 1, 1)
    belief.set_plan_horizon(_plan_real_steps)

    # MATH-NEW-02: Tune ActionRanker saturation to the plan horizon so that
    # short plans don't remain exploration-heavy for their entire execution
    action_ranker = ActionRanker()
    action_ranker.set_plan_horizon(_plan_real_steps)

    # Bounded command output log fed into world_graph for context enrichment
    execution_log: Dict[int, Dict[str, str]] = {}

    
    _visited_action_keys: dict = {}  # ordered set: {action_key: True}
    _VISITED_ACTION_MAX = 200        # cap at 200 unique action keys per task run

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

            # Wall-clock timeout
            if time.time() - start_ts > max_wallclock_seconds:
                journal.record({"event": "execution_timeout"})
                raise RuntimeError("TASK_FAILED:timeout")

            if watchdog is not None:
                watchdog.check()

            if current_step_index >= len(execution_plan.steps):
                journal.record({"event": "execution_complete"})
                return

            # Heartbeat: keep InputArbitrator watchdog alive
            input_arbitrator.soc_action_started()
            input_arbitrator.clear_emergency_reclaim()

            iteration += 1
            current_step = execution_plan.steps[current_step_index]

            # Stagnation limit varies by step type (PATCH §R6)
            step_type = current_step.type
            stagnant_limit = (
                MAX_STAGNANT_ITERS_COMMAND
                if step_type in (StepType.COMMAND_EXECUTION, StepType.TOOL_INSTALLATION)
                else MAX_STAGNANT_ITERS_UI
            )

            # ------------------------------------------------------------------
            # Perception snapshot
            # ------------------------------------------------------------------
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

            # ------------------------------------------------------------------
            # World-graph delta & belief update
            # ------------------------------------------------------------------
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

            # ------------------------------------------------------------------
            # Candidate action selection
            # ------------------------------------------------------------------
            raw_actions = current_step.action
            candidate_actions: List[Dict[str, Any]] = []

            if isinstance(raw_actions, dict):
                candidate_actions.append(raw_actions)
            elif isinstance(raw_actions, list):
                candidate_actions.extend(
                    a for a in raw_actions if isinstance(a, dict)
                )

            # Fallback: ask ReasoningEngine for dynamic candidates on stagnant steps
            if not candidate_actions and reasoning_engine is not None:
                perception_for_reasoning: Dict[str, Any] = {}
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
                        # IH-4: Filter out candidates already tried to prevent
                        # infinite UCB exploration of perpetually-novel candidates.
                        fresh_candidates = [
                            c for c in dynamic_candidates
                            if action_ranker.action_key(c) not in _visited_action_keys
                        ]
                        # Use fresh candidates if available; fall back to all dynamic
                        # candidates only if every one has been tried (exploration exhausted).
                        candidate_actions = fresh_candidates or dynamic_candidates
                        journal.record({
                            "event": "dynamic_candidates_used",
                            "step": current_step_index,
                            "count": len(candidate_actions),
                            "fresh_count": len(fresh_candidates),
                        })
                except Exception as re_err:
                    log_warn(f"ReasoningEngine fallback failed: {re_err}")

            if not candidate_actions:
                raise RuntimeError("TASK_FAILED:no_candidate_actions")

            # Thompson-UCB-EU composite selection via ActionRanker
            selected_action = action_ranker.select(
                actions=candidate_actions,
                belief_state=belief,
            )
            action_key = action_ranker.action_key(selected_action)

            # IH-4: Record action key as visited. Evict oldest when full.
            if len(_visited_action_keys) >= _VISITED_ACTION_MAX:
                _oldest = next(iter(_visited_action_keys))
                del _visited_action_keys[_oldest]
            _visited_action_keys[action_key] = True

            # ------------------------------------------------------------------
            # Policy gate
            # ------------------------------------------------------------------
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
                # RT-05 / SI-04 FIX: Cap best_reward at 0.9 to prevent DONE
                # sentinel (reward=1.0) from permanently inflating the regret
                # reference for all subsequent actions.
                best_reward = min(belief.global_best_reward() or 0.0, 0.9)
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
                journal.record({
                    "event": "policy_human_confirmation_required",
                    "step": current_step_index,
                    "action_key": action_key,
                    "reason": _policy_reason,
                })

                # RB-LOW-1 FIX: Signal-file human-approval mechanism.
                #
                # Previous implementation re-called policy_engine.validate_action_dict()
                # with identical arguments inside the loop.  The function is
                # deterministic — given the same action and focused_app it always
                # returns the same REQUIRE_HUMAN_CONFIRMATION result.  The loop
                # therefore NEVER reached the ALLOW branch; human approval was
                # structurally unreachable.
                #
                # Fix: write a signal file to /tmp and poll for its ABSENCE.
                # The user deletes the file to approve.  The policy engine is
                # NOT re-called inside the loop — only the file-system state is
                # checked.  This makes the approval path reachable.
                _signal_path = _write_approval_signal(
                    action_key, selected_action, _policy_reason or ""
                )
                print(
                    f"[OPERATE] HUMAN CONFIRMATION REQUIRED\n"
                    f"  Action : {selected_action.get('operation')!r}\n"
                    f"  Reason : {_policy_reason}\n"
                    f"  Approve: delete the file → {_signal_path}",
                    file=sys.stderr,
                )

                _phc_wait = 0
                _phc_approved = False
                try:
                    while _phc_wait < MAX_WAIT_RETRIES:
                        time.sleep(WAIT_RETRY_SECONDS)
                        _phc_wait += 1
                        # Approved when the signal file no longer exists
                        if not os.path.exists(_signal_path):
                            _phc_approved = True
                            break
                finally:
                    # Always clean up — whether approved, timed-out, or interrupted
                    _remove_approval_signal(_signal_path)

                if not _phc_approved:
                    journal.record({
                        "event": "policy_human_confirmation_denied",
                        "step": current_step_index,
                        "action_key": action_key,
                        "waited_seconds": _phc_wait * WAIT_RETRY_SECONDS,
                    })
                    belief.record_action(action_key, -0.3)
                    # RT-05 / SI-04 FIX: Cap best_reward at 0.9 to prevent DONE
                    # sentinel (reward=1.0) from permanently inflating the regret
                    # reference for all subsequent actions.
                    best_reward = min(belief.global_best_reward() or 0.0, 0.9)
                    belief.update_regret(action_key, -0.3, best_reward)
                    stagnant_iterations += 1
                    if stagnant_iterations >= stagnant_limit:
                        raise RuntimeError(REPLAN_SIGNAL)
                    previous_perception = perception_snapshot
                    continue
                else:
                    journal.record({
                        "event": "policy_human_confirmation_approved",
                        "step": current_step_index,
                        "action_key": action_key,
                    })

            # ------------------------------------------------------------------
            # Authority evaluation
            # ------------------------------------------------------------------
            is_high_risk = selected_action.get("operation") in {
                "command", "install", "file_create"
            }
            if is_high_risk:
                # High-risk requires 3 consecutive stable observations
                soc_confident = belief.consecutive_high_stability_count >= 3
            else:
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

            if authority == AuthorityDecision.ABORT:
                raise AuthorityAbortError("Human authority abort — task terminated")

            # ------------------------------------------------------------------
            # Execute action
            # ------------------------------------------------------------------
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

            # FIX MF-3: Accumulate command output (success AND failure) per step.
            # Original: execution_log[current_step_index] was overwritten on each
            # successful action, so the LLM saw only the LAST successful output when
            # replanning. On stagnant iterations (all failures), execution_log was
            # never updated at all — the LLM had no failure context to diagnose the
            # stagnation. Fix: append to a list per step index, capped at 5 entries.
            # Both success and failure outputs are recorded so the LLM receives full
            # stagnation history rather than only the last successful output.
            if "output" in exec_result and exec_result.get("output"):
                output_text = str(exec_result.get("output", ""))[:MAX_COMMAND_OUTPUT_BYTES]
                _step_entry = execution_log.setdefault(current_step_index, {"outputs": []})
                _step_outputs = _step_entry.get("outputs", [])
                if len(_step_outputs) < 5:  # cap at 5 entries to bound context size
                    _step_outputs.append({
                        "success": action_success,
                        "output": output_text,
                        "iteration": iteration,
                    })
                    _step_entry["outputs"] = _step_outputs
                    _step_entry["last_output"] = output_text  # convenience alias
                    execution_log[current_step_index] = _step_entry

            # Bandit update
            belief.record_action(action_key, raw_reward)
            # RT-05 / SI-04 FIX: Cap best_reward at 0.9 to prevent DONE
            # sentinel (reward=1.0) from permanently inflating the regret
            # reference for all subsequent actions.
            best_reward = min(belief.global_best_reward() or 0.0, 0.9)
            belief.update_regret(action_key, raw_reward, best_reward)

            journal.record({
                "event": "action_executed",
                "step": current_step_index,
                "action_key": action_key,
                "success": action_success,
                "reward": raw_reward,
            })

            # ------------------------------------------------------------------
            # Step verification & advancement
            # ------------------------------------------------------------------
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

            # ------------------------------------------------------------------
            # Stagnation guard
            # ------------------------------------------------------------------
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

        # Loop exhausted without reaching DONE step
        raise RuntimeError("TASK_FAILED:max_iterations_exceeded")

    except (RuntimeError, AuthorityAbortError):
        raise

    finally:
        # Persist the final BeliefState on ALL exit paths (success, failure, exception)
        if belief_state_out is not None:
            belief_state_out.clear()
            try:
                belief_state_out.append(belief.to_dict())
            except Exception:
                pass  # Serialisation failure must never propagate on shutdown path


# =========================================================================
# ACTION DISPATCHER
# =========================================================================

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

    Returns
    -------
    dict with keys:
        success   : bool
        reward    : float  (0.8 success, -0.5 failure, 1.0 done)
        output    : str    (command stdout+stderr, if applicable)
        returncode: int    (command exit code — RT-B1 fix, required by StepVerifier)
        reason    : str    (optional; describes why an action could not execute)

    Dispatch table (by `operation` field):
        click        → os_backend.mouse()
        write/type   → os_backend.write()
        press/hotkey → os_backend.press()
        command      → os_backend.exec()
        file_create  → os_backend.write_file()
        install      → installer.install_tool() or os_backend.exec()
        scroll       → pyautogui.scroll()
        verify       → os_backend.exec() (command method) or no-op (screenshot)
        done         → immediate success (DONE sentinel)
        (unknown)    → structured failure, reward -0.5
    """
    from core.safety.action_timeout import run_with_timeout, ActionTimeout

    op = (action.get("operation") or "").lower().strip()

    # DONE sentinel — always succeeds immediately with maximum reward
    if op == "done":
        return {"success": True, "reward": 1.0}

    if not op:
        return {"success": False, "reward": -0.5, "reason": "empty operation field"}

    try:
        # ------------------------------------------------------------------
        # CLICK
        # ------------------------------------------------------------------
        if op == "click":
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
                return {"success": False, "reward": -0.5, "reason": "click: no x/y or label"}
            return {"success": True, "reward": 0.8}

        # ------------------------------------------------------------------
        # WRITE / TYPE
        # ------------------------------------------------------------------
        elif op in ("write", "type"):
            content = str(action.get("content") or action.get("text") or "")
            if not content:
                return {"success": False, "reward": -0.5, "reason": "write: empty content"}
            run_with_timeout(
                lambda: os_backend.write(content),
                seconds=30.0,
                operation_hint="write",
                executor=task_ui_executor,
            )
            return {"success": True, "reward": 0.8}

        # ------------------------------------------------------------------
        # PRESS / HOTKEY / KEY
        # ------------------------------------------------------------------
        elif op in ("press", "hotkey", "key"):
            keys = action.get("keys") or action.get("key")
            if isinstance(keys, str):
                keys = [keys]
            if not isinstance(keys, list) or not keys:
                return {"success": False, "reward": -0.5, "reason": "press: no keys specified"}
            run_with_timeout(
                lambda: os_backend.press(keys),
                seconds=15.0,
                operation_hint="press",
                executor=task_ui_executor,
            )
            return {"success": True, "reward": 0.8}

        # ------------------------------------------------------------------
        # SCROLL (FIX RT-05: uses module-level _pyautogui alias)
        # ------------------------------------------------------------------
        elif op == "scroll":
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

        # ------------------------------------------------------------------
        # COMMAND
        # ------------------------------------------------------------------
        elif op == "command":
            cmd = str(action.get("command") or "").strip()
            if not cmd:
                return {"success": False, "reward": -0.5, "reason": "command: empty command"}
            from config.timeouts import INSTALL_COMMAND_TIMEOUT_SECONDS
            result = os_backend.exec(cmd, timeout=int(INSTALL_COMMAND_TIMEOUT_SECONDS))
            success = (result.returncode == 0)
            reward = 0.8 if success else -0.5
            output = (result.stdout or "") + (result.stderr or "")
            # FIX RT-B1: Include returncode so StepVerifier._verify_command()
            # can extract it.  Previously the key was absent, making every
            # COMMAND_EXECUTION step fail verification unconditionally.
            return {
                "success": success,
                "reward": reward,
                "output": output,
                "returncode": result.returncode,
            }

        # ------------------------------------------------------------------
        # FILE CREATE
        # ------------------------------------------------------------------
        elif op == "file_create":
            path = str(action.get("path") or "").strip()
            content_str = str(action.get("content") or "")
            if not path:
                return {"success": False, "reward": -0.5, "reason": "file_create: no path"}
            os_backend.write_file(path, content_str)
            return {"success": True, "reward": 0.8}

        # ------------------------------------------------------------------
        # INSTALL
        # ------------------------------------------------------------------
        elif op == "install":
            tool_spec = action.get("tool", {})
            install_cmds = (
                tool_spec.get("install_commands", [])
                if isinstance(tool_spec, dict)
                else []
            )
            if install_cmds and isinstance(install_cmds, list):
                # Terminal-first install path
                from config.timeouts import INSTALL_COMMAND_TIMEOUT_SECONDS
                all_ok = True
                combined_output = ""
                for cmd in install_cmds:
                    r = os_backend.exec(
                        cmd, timeout=int(INSTALL_COMMAND_TIMEOUT_SECONDS)
                    )
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
                #
                # RB-CRIT-2 FIX: The previous code called installer.install(tool_name)
                # where tool_name is a str.  AutonomousInstaller exposes only
                # install_tool(tool: Dict) — there is no install(str) method.
                # Every UI-based install path therefore raised AttributeError.
                #
                # Fix: pass the full tool_spec dict (which the caller already has)
                # to install_tool(), which is the method the class actually defines.
                # If tool_spec is not a dict (e.g. None or a stray string), we
                # construct a minimal dict with a 'name' key so install_tool()
                # always receives a well-formed argument.
                if not isinstance(tool_spec, dict):
                    tool_spec = {"name": str(tool_spec) if tool_spec else ""}
                try:
                    installer.install_tool(tool_spec)
                    return {"success": True, "reward": 0.8}
                except Exception as inst_err:
                    return {
                        "success": False,
                        "reward": -0.5,
                        "output": str(inst_err),
                    }
            else:
                return {
                    "success": False,
                    "reward": -0.5,
                    "reason": (
                        "install: no install_commands in tool spec and installer unavailable. "
                        "Ensure install_commands is populated in the plan step."
                    ),
                }

        # ------------------------------------------------------------------
        # VERIFY
        # ------------------------------------------------------------------
        elif op == "verify":
            method = action.get("method", "screenshot")
            if method == "command":
                cmd = str(action.get("command") or "").strip()
                if cmd:
                    r = os_backend.exec(cmd, timeout=30)
                    return {
                        "success": r.returncode == 0,
                        "reward": 0.6 if r.returncode == 0 else -0.3,
                        "output": (r.stdout or "") + (r.stderr or ""),
                        "returncode": r.returncode,
                    }
            # Screenshot verify — StepVerifier handles the actual comparison
            return {"success": True, "reward": 0.6}

        # ------------------------------------------------------------------
        # UNKNOWN OPERATION
        # ------------------------------------------------------------------
        else:
            log_warn(f"_execute_decision: unknown operation {op!r}")
            return {
                "success": False,
                "reward": -0.5,
                "reason": (
                    f"unknown operation {op!r}. Valid operations: "
                    "click, write, type, press, hotkey, key, scroll, "
                    "command, file_create, install, verify, done."
                ),
            }

    except ActionTimeout as toe:
        log_warn(f"_execute_decision: action timeout [{op}] — {toe}")
        return {"success": False, "reward": -0.5, "reason": f"action_timeout: {toe}"}

    except Exception as exc:
        log_warn(f"_execute_decision: unexpected error [{op}] — {exc}")
        return {
            "success": False,
            "reward": -0.5,
            "reason": f"unexpected_error: {exc}",
        }
