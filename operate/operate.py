from __future__ import annotations

import concurrent.futures
import hashlib
import os
import sys
import tempfile
import time
import json
from typing import Any, Dict, Optional, List

from authority.authority_policy import AuthorityDecision
from authority.input_arbitrator import InputArbitrator

from core.safety.action_timeout import run_with_timeout
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




MAX_PERCEPTION_ENTITIES = 20
MAX_PERCEPTION_JSON_BYTES = 10_000

REPLAN_SIGNAL: str = "REPLAN_REQUIRED"

# WAIT should pause and retry, not immediately replan (PATCH §1.11)
WAIT_RETRY_SECONDS = 0.5

def _resolve_confirm_timeout() -> int:
    """Read PROJECTZEO_CONFIRM_TIMEOUT_SECONDS env var; default 60s."""
    raw = os.environ.get("PROJECTZEO_CONFIRM_TIMEOUT_SECONDS", "")
    try:
        val = int(raw.strip())
        if val > 0:
            return val
    except (ValueError, AttributeError):
        pass
    return 60  # default: 60 seconds


_CONFIRM_TIMEOUT_SECONDS: int = _resolve_confirm_timeout()
MAX_WAIT_RETRIES: int = max(int(_CONFIRM_TIMEOUT_SECONDS / WAIT_RETRY_SECONDS), 1)


_CONFIRM_TIMEOUT_LOGGED: list = [False]  # mutable container — reset per task call

# Max bytes of command output stored per step in execution_log (bounded, not unlimited)
MAX_COMMAND_OUTPUT_BYTES = 4096

# Maximum dynamic candidates from ReasoningEngine on stagnant steps (H-03 fix)
MAX_DYNAMIC_CANDIDATES = 3


import os as _os_sig
import secrets as _secrets_mod


_SESSION_TOKEN: str = _secrets_mod.token_hex(16)
_SIGNAL_DIR_BASE: str = tempfile.gettempdir()
_SIGNAL_DIR: str = _os_sig.path.join(
    _SIGNAL_DIR_BASE,
    f"projectzeo_{_SESSION_TOKEN}",
)
try:
    _os_sig.makedirs(_SIGNAL_DIR, mode=0o700, exist_ok=True)
    # Enforce mode even if directory already existed.
    _os_sig.chmod(_SIGNAL_DIR, 0o700)
except OSError:
    # Fallback to /tmp with session-secret prefix — still better than no secret.
    _SIGNAL_DIR = _SIGNAL_DIR_BASE

_SIGNAL_PREFIX: str = "approve_"


def _approval_signal_path(action_key: str) -> str:
    return _os_sig.path.join(
        _SIGNAL_DIR,
        f"{_SIGNAL_PREFIX}{action_key}.signal",
    )


def _write_approval_signal(action_key: str, action: dict, reason: str) -> str:
    
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
                    f"{_CONFIRM_TIMEOUT_SECONDS}s."
                ),
            },
            indent=2,
        )
        
        fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
    except FileExistsError:
        # Another process/thread already created the approval file; this
        # is a benign race — the gate is still in place.
        pass
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
    
    prior_step_index: Optional[int] = None,
    prior_execution_log: Optional[dict] = None,  # MAJ-1 FIX: restore from checkpoint
) -> None:
    
   
    
    if not _CONFIRM_TIMEOUT_LOGGED[0]:
        _CONFIRM_TIMEOUT_LOGGED[0] = True
        print(
            f"[OPERATE] Human confirmation timeout: {_CONFIRM_TIMEOUT_SECONDS}s "
            f"({MAX_WAIT_RETRIES} retries × {WAIT_RETRY_SECONDS}s). "
            "Override with PROJECTZEO_CONFIRM_TIMEOUT_SECONDS env var.",
            file=sys.stderr,
        )

    if not isinstance(execution_plan, ExecutionPlan):
        raise ValueError("execution_plan must be an ExecutionPlan instance")

    
    if not execution_plan.validate():
        raise ValueError(
            "ExecutionPlan failed validation on entry to operate_main(). "
            "The plan may have been mutated after creation or serialised/deserialised "
            "incorrectly. Re-run planner.create_plan() to obtain a fresh valid plan."
        )
    PlanVerifier().verify(execution_plan)

    os_backend = os_backend or OperatingSystem()

    
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
            # M1+M2 FIX: Use from_policy_yaml() to load ALL policy sections
            # (denied_apps, high_risk_apps, filesystem.allowed_write_paths, etc.)
            policy_engine = PolicyEngine.from_policy_yaml(_pcfg)
    except ImportError:
        
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

    
    _auto_discovered_names: list = []
    try:
        import psutil as _psutil_ad  # noqa: PLC0415
        _seen: set = set()
        for _proc in _psutil_ad.process_iter(["name"]):
            try:
                _pname = (_proc.info.get("name") or "").strip()
                if _pname and _pname not in _seen:
                    _seen.add(_pname)
                    _auto_discovered_names.append(_pname)
            except (_psutil_ad.NoSuchProcess, _psutil_ad.AccessDenied):
                continue
        if _auto_discovered_names:
            print(
                f"[operate_main] Process fingerprint: {len(_auto_discovered_names)} running "
                f"processes observed at task start (NOT added to allowlist). "
                f"Sample: {sorted(_auto_discovered_names)[:8]}. "
                "Add apps explicitly via policy.yaml allowed_apps.",
                file=sys.stderr,
            )
    except ImportError:
        print(
            "[operate_main] WARNING: psutil not installed — cannot fingerprint "
            "running apps. Fix: pip install psutil",
            file=sys.stderr,
        )
    except Exception as _disc_err:
        print(
            f"[operate_main] WARNING: process fingerprint failed: {_disc_err}.",
            file=sys.stderr,
        )

    # Also add binaries from the environment fingerprint (tools found on PATH)
    try:
        _fp_tools = (
            (world_graph.snapshot().get("environment", {}) if world_graph else {})
            or {}
        )
        # environment_fingerprint "tools" is a dict {name: bool}
        _fp_tools_dict = _fp_tools.get("tools", {})
        if isinstance(_fp_tools_dict, dict):
            for _tool_name, _present in _fp_tools_dict.items():
                if _present and isinstance(_tool_name, str):
                    policy_engine.allow_app(_tool_name)
    except Exception:
        pass  # env fingerprint is best-effort

    
    _LIKELIHOOD_DEFAULTS: dict = {
        "app_match_with_delta":  0.95,
        "app_match_no_delta":    0.75,
        "ui_rich":               1.50,
        "ui_sparse":             1.20,
        "ui_empty":              0.80,
        "neutral_with_delta":    1.00,
        "neutral_no_delta":      0.80,
        "ENTITY_RICH_THRESHOLD": 10,
    }
    _likelihood_cfg: dict = dict(_LIKELIHOOD_DEFAULTS)
    try:
        import os as _os_lh
        import json as _json_lh
        _lh_path = _os_lh.path.join(
            _os_lh.path.dirname(__file__), "..", "likelihoods.json"
        )
        if _os_lh.path.exists(_lh_path):
            with open(_lh_path, "r", encoding="utf-8") as _lhf:
                _lh_raw = _json_lh.load(_lhf)
            if isinstance(_lh_raw, dict):
                # Validate: all expected keys must be present and numeric
                _missing = [k for k in _LIKELIHOOD_DEFAULTS if k not in _lh_raw]
                _bad_type = [
                    k for k, v in _lh_raw.items()
                    if k in _LIKELIHOOD_DEFAULTS and not isinstance(v, (int, float))
                ]
                
                _bad_range = [
                    k for k, v in _lh_raw.items()
                    if k in _LIKELIHOOD_DEFAULTS
                    and isinstance(v, (int, float))
                    and v <= 0
                ]
                if _missing or _bad_type or _bad_range:
                    raise ValueError(
                        f"likelihoods.json validation failed — "
                        f"missing keys: {_missing}, bad types: {_bad_type}, "
                        f"non-positive ratios: {_bad_range} "
                        "(all likelihood ratios must be > 0)"
                    )
                _likelihood_cfg.update(_lh_raw)
                print(
                    f"[operate_main] Loaded Bayesian likelihood ratios from "
                    f"likelihoods.json: {_likelihood_cfg}",
                    file=sys.stderr,
                )
            else:
                raise ValueError("likelihoods.json root must be a JSON object")
        else:
            # File absent is normal on first deploy — silently use defaults.
            print(
                "[operate_main] likelihoods.json not found — using hardcoded "
                "default Bayesian likelihood ratios. To calibrate, create "
                "likelihoods.json alongside policy.yaml. See H2 fix comment "
                "for the schema and calibration guidance.",
                file=sys.stderr,
            )
    except Exception as _lh_err:
        print(
            f"[operate_main] WARNING H2: Failed to load likelihoods.json: "
            f"{_lh_err}. Using hardcoded defaults.",
            file=sys.stderr,
        )
        _likelihood_cfg = dict(_LIKELIHOOD_DEFAULTS)

    
    _likelihood_cfg["ENTITY_RICH_THRESHOLD"] = int(
        _likelihood_cfg.get("ENTITY_RICH_THRESHOLD", 10)
    )

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

    
    from core.memory.playbook_store import load_playbook as _load_playbook, save_playbook as _save_playbook
    _prior_playbook_actions: list = []
    try:
        _pb = _load_playbook(terminal_prompt)
        if isinstance(_pb, list) and _pb:
            _prior_playbook_actions = _pb
            print(
                f"[operate_main] H2: Loaded playbook for intent "
                f"({len(_prior_playbook_actions)} prior actions). "
                "These will be offered as first candidates on stagnant steps.",
                file=sys.stderr,
            )
    except Exception as _pb_load_err:
        print(
            f"[operate_main] H2 WARNING: playbook load failed: {_pb_load_err}. "
            "Proceeding without playbook warm-start.",
            file=sys.stderr,
        )

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

    
    installer: Optional[AutonomousInstaller] = None
    if observer is not None:
        _shared_client = getattr(planner, "_ollama_client", None)
        installer = AutonomousInstaller(
            observer=observer,
            os_backend=os_backend,
            llm_callable=llm_callable,
            shared_ollama_client=_shared_client,
            policy_engine=policy_engine,
        )

    
    reasoning_engine = ReasoningEngine(
        llm_callable=llm_callable,
        ollama_client=getattr(planner, "_ollama_client", None),
    )

    
    _task_ui_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="ui_timeout_worker",
    )

    try:
        
        _created_files_ledger: List[str] = []

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
            prior_playbook_actions=_prior_playbook_actions,
            prior_step_index=prior_step_index,
            prior_execution_log=prior_execution_log,  # MAJ-1 FIX
            created_files_ledger=_created_files_ledger,
        )
        
        try:
            _journal_actions = journal.get_all() if hasattr(journal, "get_all") else []
            if _journal_actions:
                _save_playbook(terminal_prompt, _journal_actions)
                print(
                    f"[operate_main] H2: Saved playbook for intent "
                    f"({len(_journal_actions)} actions). Future identical tasks "
                    "will warm-start from this run's successful action sequence.",
                    file=sys.stderr,
                )
        except Exception as _pb_save_err:
            print(
                f"[operate_main] H2 WARNING: playbook save failed: {_pb_save_err}.",
                file=sys.stderr,
            )

        try:
            from core.safety.checkpoint_store import clear_checkpoint as _clear_cp
            _clear_cp()
        except Exception:
            pass  # non-fatal
    finally:
        input_arbitrator.shutdown()
        
        import threading as _threading
        _ui_threads = [t for t in _threading.enumerate()
                       if t.name.startswith("ui_timeout_worker")]
        for _t in _ui_threads:
            _t.join(timeout=2.0)
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
    prior_playbook_actions: Optional[List[dict]] = None,
    
    prior_step_index: Optional[int] = None,
    
    prior_execution_log: Optional[dict] = None,
    
    created_files_ledger: Optional[List[str]] = None,
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

    
    _plan_real_steps = max(len(execution_plan.steps) - 1, 1)
    belief.set_plan_horizon(_plan_real_steps)

    
    action_ranker = ActionRanker()
    action_ranker.set_plan_horizon(_plan_real_steps)

    
    
    if prior_execution_log and isinstance(prior_execution_log, dict):
        try:
            execution_log: Dict[int, Dict[str, str]] = {
                int(k): v for k, v in prior_execution_log.items()
                if isinstance(v, dict)
            }
            print(
                f"[OPERATE] MAJ-1: Restored execution_log from checkpoint with "
                f"{len(execution_log)} step(s).",
                file=sys.stderr,
            )
        except Exception as _el_err:
            print(
                f"[OPERATE] MAJ-1: execution_log restore failed ({_el_err}); "                "starting with empty log.",
                file=sys.stderr,
            )
            execution_log = {}
    else:
        execution_log: Dict[int, Dict[str, str]] = {}

    _visited_action_keys: dict = {}  # ordered set: {action_key: True}
    if prior_belief_state is not None:
        try:
            _persisted_visited = prior_belief_state.get("_visited_action_keys", [])
            if isinstance(_persisted_visited, list):
                _visited_action_keys = {k: True for k in _persisted_visited if isinstance(k, str)}
        except Exception:
            _visited_action_keys = {}
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

    
    if prior_step_index is not None and prior_step_index > 0:
        _max_valid = max(len(execution_plan.steps) - 1, 0)
        _safe_index = min(int(prior_step_index), _max_valid)
        if _safe_index > 0:
            current_step_index = _safe_index
            journal.record({
                "event": "crash_recovery_step_fastforward",
                "prior_step_index": prior_step_index,
                "fast_forwarded_to": current_step_index,
                "plan_step_count": len(execution_plan.steps),
            })
            print(
                f"[operate] BUG-5: Crash recovery fast-forward: skipping steps "
                f"0–{current_step_index - 1} (already completed before crash). "
                f"Resuming at step {current_step_index}.",
                file=sys.stderr,
            )
            # Advance the progress tracker to match the recovered position
            for _ in range(current_step_index):
                try:
                    progress.advance_step()
                except Exception:
                    pass

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
            
            _LONG_RUNNING_KEYWORDS = frozenset({
                "render", "compile", "build", "download", "install", "export",
                "encode", "transcode", "generate", "train", "convert",
            })
            if step_type in (StepType.COMMAND_EXECUTION, StepType.TOOL_INSTALLATION):
                _desc_lower = current_step.description.lower()
                if any(_kw in _desc_lower for _kw in _LONG_RUNNING_KEYWORDS):
                    stagnant_limit = stagnant_limit * 10

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
                            
                            perception_snapshot["last_command_output"] = step_log.get(
                                "last_output", ""
                            )
                        world_graph.update(perception_snapshot)
                except Exception:
                    log_warn("Observer snapshot failed")

            
            try:
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

                            

                            has_delta = bool(delta)

                        
                            _entity_rich_thresh = _likelihood_cfg["ENTITY_RICH_THRESHOLD"]

                            if isinstance(focused_app, str) and focused_app.strip():
                                app_state_key = f"app:{focused_app.lower()}"
                                likelihoods[app_state_key] = (
                                    _likelihood_cfg["app_match_with_delta"] if has_delta
                                    else _likelihood_cfg["app_match_no_delta"]
                                )

                            if entity_count > _entity_rich_thresh:
                                likelihoods["ui_rich"] = _likelihood_cfg["ui_rich"]
                            elif entity_count > 0:
                                likelihoods["ui_sparse"] = _likelihood_cfg["ui_sparse"]
                            else:
                                likelihoods["ui_empty"] = _likelihood_cfg["ui_empty"]

                            # Neutral baseline
                            likelihoods["neutral"] = (
                                _likelihood_cfg["neutral_with_delta"] if has_delta
                                else _likelihood_cfg["neutral_no_delta"]
                            )
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

               
                if candidate_actions:
                    try:
                        from core.security.injection_markers import contains_injection_marker as _cim_plan
                        _safe_plan = []
                        for _pa in candidate_actions:
                            _pa_text = " ".join(
                                str(_pa.get(f, ""))
                                for f in ("command", "content", "path")
                            )
                            if _cim_plan(_pa_text):
                                log_warn(
                                    f"[H-03] Plan-step action BLOCKED — injection marker "
                                    f"detected in command/content/path: {_pa_text[:80]!r}. "
                                    "Step dropped."
                                )
                            else:
                                _safe_plan.append(_pa)
                        candidate_actions = _safe_plan
                    except ImportError:
                        pass

                # Fallback: ask ReasoningEngine for dynamic candidates on stagnant steps
                if not candidate_actions and reasoning_engine is not None:
                    perception_for_reasoning: Dict[str, Any] = {}
                    if isinstance(perception_snapshot, dict):
                        perception_for_reasoning = perception_snapshot
                    elif isinstance(world_snapshot, dict):
                        perception_for_reasoning = world_snapshot

                    
                    _playbook_candidates: List[Dict[str, Any]] = []
                    if prior_playbook_actions:
                        _playbook_candidates = [
                            a for a in prior_playbook_actions
                            if isinstance(a, dict)
                            and action_ranker.action_key(a) not in _visited_action_keys
                        ]
                        if _playbook_candidates:
                            journal.record({
                                "event": "playbook_candidates_injected",
                                "step": current_step_index,
                                "count": len(_playbook_candidates),
                            })

                    try:
                        dynamic_candidates = reasoning_engine.propose_actions(
                            objective=execution_plan.objective,
                            belief_summary=belief.summary(),
                            perception=perception_for_reasoning,
                            k=MAX_DYNAMIC_CANDIDATES,
                        )
                        if dynamic_candidates:
                            
                            try:
                                from core.security.injection_markers import contains_injection_marker as _cim
                                _safe_dynamic: list = []
                                for _dc in dynamic_candidates:
                                    _dc_cmd = str(_dc.get("command", "")) + " " + str(_dc.get("content", ""))
                                    if _cim(_dc_cmd):
                                        log_warn(
                                            f"[CRIT-6] Dynamic candidate BLOCKED — injection marker "                                            f"detected in command/content: {_dc_cmd[:80]!r}. "                                            "Candidate dropped.")
                                    else:
                                        _safe_dynamic.append(_dc)
                                dynamic_candidates = _safe_dynamic
                            except ImportError:
                                pass  # security module unavailable — proceed without this check

                            fresh_candidates = [
                                c for c in dynamic_candidates
                                if action_ranker.action_key(c) not in _visited_action_keys
                            ]
                            
                            candidate_actions = fresh_candidates or dynamic_candidates
                            if candidate_actions:
                                journal.record({
                                    "event": "dynamic_candidates_used",
                                    "step": current_step_index,
                                    "count": len(candidate_actions),
                                    "fresh_count": len(fresh_candidates),
                                })
                    except Exception as re_err:
                        log_warn(f"ReasoningEngine fallback failed: {re_err}")

                    
                    if _playbook_candidates:
                        candidate_actions = _playbook_candidates + candidate_actions

                if not candidate_actions:
                    raise RuntimeError("TASK_FAILED:no_candidate_actions")

                # Thompson-UCB-EU composite selection via ActionRanker
                selected_action = action_ranker.select(
                    actions=candidate_actions,
                    belief_state=belief,
                )
                action_key = action_ranker.action_key(selected_action)

                
                _policy_decision: str = PolicyEngine.DENY
                _policy_reason: str = "not_evaluated"

                # IH-4: Record action key as visited. Evict oldest when full.
                if len(_visited_action_keys) >= _VISITED_ACTION_MAX:
                    _oldest = next(iter(_visited_action_keys))
                    del _visited_action_keys[_oldest]
                _visited_action_keys[action_key] = True

                # NEW: Semantic loop detection (A→B→A cycle)
                if hasattr(belief, "record_action_key_for_loop_detection"):
                    belief.record_action_key_for_loop_detection(action_key)
                if (
                    hasattr(belief, "detect_semantic_loop")
                    and belief.detect_semantic_loop()
                ):
                    journal.record({
                        "event": "semantic_loop_detected",
                        "step": current_step_index,
                        "action_key": action_key,
                    })
                    log_warn(
                        f"[OPERATE] Semantic loop detected (A→B→A cycle) at "
                        f"step {current_step_index}. Triggering replan."
                    )
                    raise RuntimeError(REPLAN_SIGNAL)

                
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

            except (AuthorityAbortError, RuntimeError):
                # AuthorityAbortError and REPLAN_SIGNAL must propagate unchanged.
                raise
            except Exception as _iter_exc:
                # P1 RT-C: Transient failure (LLM parse error, corrupt screenshot,
                # Ollama network timeout) -> stagnation increment, not process crash.
                log_warn(
                    f"[operate] Transient per-iteration failure (step "
                    f"{current_step_index}, iter {iteration}): "
                    f"{type(_iter_exc).__name__}: {_iter_exc}. "
                    "Converting to stagnation increment."
                )
                journal.record({
                    "event": "per_iteration_transient_failure",
                    "step": current_step_index,
                    "iteration": iteration,
                    "error_type": type(_iter_exc).__name__,
                    "error": str(_iter_exc),
                })
                stagnant_iterations += 1
                if stagnant_iterations >= stagnant_limit:
                    raise RuntimeError(REPLAN_SIGNAL)
                previous_perception = perception_snapshot
                continue  # RTB-01 FIX: removed duplicate unreachable continue

            if _policy_decision == PolicyEngine.REQUIRE_HUMAN_CONFIRMATION:
                journal.record({
                    "event": "policy_human_confirmation_required",
                    "step": current_step_index,
                    "action_key": action_key,
                    "reason": _policy_reason,
                })

                
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
                
                try:
                    import subprocess as _sp
                    _notif_body = (
                        f"Action: {selected_action.get('operation')}\n"
                        f"Reason: {_policy_reason}\n"
                        f"Delete to approve: {_signal_path}"
                    )
                    if _sp.run(["which", "notify-send"], capture_output=True).returncode == 0:
                        _sp.run(
                            [
                                "notify-send",
                                "--urgency=critical",
                                "--expire-time=30000",
                                "ProjectZeo: Human Approval Required",
                                _notif_body,
                            ],
                            timeout=5,
                            capture_output=True,
                        )
                    elif _sp.run(["which", "osascript"], capture_output=True).returncode == 0:
                        # macOS
                        _sp.run(
                            ["osascript", "-e",
                             f'display notification "{_notif_body}" with title "ProjectZeo Approval"'],
                            timeout=5,
                            capture_output=True,
                        )
                except Exception:
                    pass  # Notification failure is never fatal

                _phc_wait = 0
                _phc_approved = False
                try:
                    while _phc_wait < MAX_WAIT_RETRIES:
                        time.sleep(WAIT_RETRY_SECONDS)
                        _phc_wait += 1
                        try:
                            _file_present = os.path.exists(_signal_path)
                        except OSError:
                            # C-05 FIX: Filesystem error → treat as still pending,
                            # never auto-approve on transient stat failures.
                            continue
                        if not _file_present:
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

            
            _dispatch_action = selected_action
            if (
                created_files_ledger is not None
                and str(selected_action.get("operation", "")).lower() == "file_create"
            ):
                _dispatch_action = dict(selected_action)
                _dispatch_action["_created_files_ledger"] = created_files_ledger

            exec_result: dict = {}
            try:
                exec_result = _execute_decision(
                    action=_dispatch_action,
                    os_backend=os_backend,
                    installer=installer,
                    current_step=current_step,
                    execution_log=execution_log,
                    current_step_index=current_step_index,
                    task_ui_executor=task_ui_executor,
                    watchdog=watchdog,
                    # GAP-1 FIX: Pass browser context for Playwright routing
                    focused_app=(
                        world_snapshot.get("focused_app", "")
                        if isinstance(world_snapshot, dict)
                        else ""
                    ),
                    prefer_playwright=True,  # controlled by policy.yaml in future
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

            
            if "output" in exec_result and exec_result.get("output"):
                _raw_output = str(exec_result.get("output", ""))[:MAX_COMMAND_OUTPUT_BYTES]
                # CRIT-NEW: Scrub potential credentials from command output
                # before storing in execution_log/checkpoint (cat .env attack).
                import re as _re_cred
                _CRED_RE = _re_cred.compile(
                    r"(?:password|passwd|secret|token|api[_\-]?key|auth[_\-]?token"
                    r"|bearer|private[_\-]?key|aws[_\-]?secret|access[_\-]?key"
                    r")\s*[:=]\s*\S+",
                    _re_cred.IGNORECASE,
                )
                output_text = _CRED_RE.sub(
                    lambda m: m.group(0).split(":")[0].split("=")[0] + "=<REDACTED>",
                    _raw_output,
                )
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
                    
                    try:
                        from core.safety.checkpoint_store import save_checkpoint as _save_cp
                        _save_cp({
                            "intent": terminal_prompt,
                            "step_index": current_step_index,
                            "belief_state": belief.to_dict(),
                            "execution_log": {
                                str(k): v for k, v in execution_log.items()
                            },
                        })
                    except Exception as _cp_err:
                        # Non-fatal: checkpoint failure must never block execution.
                        log_warn(f"[CHECKPOINT] save_checkpoint failed: {_cp_err}")
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
                _bs_dict = belief.to_dict()
                
                _bs_dict["_visited_action_keys"] = list(_visited_action_keys.keys())
                
                if created_files_ledger:
                    _bs_dict["_created_files_ledger"] = list(created_files_ledger)
                belief_state_out.append(_bs_dict)
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
    focused_app: str = "",          # GAP-1 FIX: passed from world_snapshot.focused_app
    prefer_playwright: bool = True,  # GAP-1 FIX: from policy.yaml browser.prefer_playwright
) -> dict:
    
    from core.safety.action_timeout import run_with_timeout, ActionTimeout

    op = (action.get("operation") or "").lower().strip()

    
    try:
        from pyautogui import FailSafeException as _FailSafeException
    except ImportError:
        _FailSafeException = None  # type: ignore[assignment,misc]

    # DONE sentinel — always succeeds immediately with maximum reward
    if op == "done":
        return {"success": True, "reward": 1.0}

    if not op:
        return {"success": False, "reward": -0.5, "reason": "empty operation field"}

    
    if op in ("command", "install", "file_create", "verify"):
        _dangerous_text = ""
        if op == "command":
            _dangerous_text = str(action.get("command", ""))
        elif op == "file_create":
            _dangerous_text = str(action.get("path", "")) + " " + str(action.get("content", ""))
        elif op == "install":
            _tool = action.get("tool", {})
            if isinstance(_tool, dict):
                _dangerous_text = " ".join(
                    str(c) for c in _tool.get("install_commands", [])
                )
        elif op == "verify":
            _dangerous_text = str(action.get("command", ""))

        if _dangerous_text.strip():
            try:
                from core.planner.execution_planner import ExecutionPlanner as _EP  # noqa: PLC0415
                from core.security.injection_markers import normalize_for_injection_check as _norm_dp  # noqa: PLC0415
                _compiled = getattr(_EP, "_dispatch_compiled_patterns", None)
                if _compiled is None:
                    import re as _re_dp
                    _compiled = [
                        _re_dp.compile(p, _re_dp.IGNORECASE) for p in _EP.DANGEROUS_PATTERNS
                    ]
                    _EP._dispatch_compiled_patterns = _compiled
                _normalized_dangerous = _norm_dp(_dangerous_text)
                for _pat in _compiled:
                    if _pat.search(_normalized_dangerous):
                        log_warn(
                            f"[BUG-11] DANGEROUS_PATTERNS match at DISPATCH time "
                            f"for op={op!r}: pattern={_pat.pattern!r}. "
                            "Blocking execution."
                        )
                        return {
                            "success": False,
                            "reward": -1.0,
                            "reason": (
                                f"dangerous_pattern_blocked_at_dispatch: "
                                f"pattern={_pat.pattern!r} matched in {op!r} operation. "
                                "Execution blocked for safety."
                            ),
                        }
            except Exception as _dp_err:
                log_warn(f"[BUG-11] dispatch-time DANGEROUS_PATTERNS check failed: {_dp_err}")

    try:
        # ------------------------------------------------------------------
        
        _BROWSER_OPS = {"click", "write", "type", "fill", "scroll", "navigate", "goto"}
        if (
            op in _BROWSER_OPS
            and prefer_playwright
            and focused_app
        ):
            try:
                from operate.utils.browser_backend import (  # noqa: PLC0415
                    get_browser_backend,
                    is_browser_app,
                )
                if is_browser_app(focused_app):
                    _bb = get_browser_backend()
                    if _bb is not None:
                        _br_result = _bb.execute_action(action)
                        if _br_result.get("success"):
                            return _br_result
                        # If Playwright failed (element not found), fall through
                        # to pyautogui coordinate path as graceful degradation.
                        log_warn(
                            f"[GAP-1] Playwright failed for op={op!r} on "
                            f"focused_app={focused_app!r}: "
                            f"{_br_result.get('reason')}. "
                            "Falling back to pyautogui coordinate path."
                        )
            except ImportError:
                pass  # playwright not installed — proceed with pyautogui silently
            except Exception as _br_err:
                log_warn(
                    f"[GAP-1] BrowserBackend routing error for op={op!r}: {_br_err}. "
                    "Falling back to pyautogui."
                )

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
            
            _ledger = action.get("_created_files_ledger")
            if isinstance(_ledger, list):
                if path not in _ledger:
                    _ledger.append(path)
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
                
                _install_preview = "; ".join(str(c) for c in install_cmds[:3])
                import secrets as _secrets_install
                _ak = _secrets_install.token_hex(16)  # M4 FIX: secure key
                _sig_path = _write_approval_signal(
                    _ak,
                    action,
                    reason=f"Install/sudo requires confirmation: {_install_preview[:120]}",
                )
                print(
                    f"[OPERATE] Install requires human approval. "
                    f"Commands: {_install_preview!r}. "
                    f"APPROVE: delete {_sig_path}  |  "
                    f"Timeout: {_CONFIRM_TIMEOUT_SECONDS}s → auto-denied.",
                    file=sys.stderr,
                )
                _waited = 0.0
                _approved = False
                while _waited < _CONFIRM_TIMEOUT_SECONDS:
                    time.sleep(WAIT_RETRY_SECONDS)
                    _waited += WAIT_RETRY_SECONDS
                    try:
                        _file_present = os.path.exists(_sig_path)
                    except OSError:
                        # C-05 FIX: fail-closed on filesystem error
                        continue
                    if not _file_present:
                        _approved = True
                        break
                if not _approved:
                    _remove_approval_signal(_sig_path)
                    return {
                        "success": False,
                        "reward": -1.0,
                        "reason": (
                            f"Install confirmation timed out after "
                            f"{_CONFIRM_TIMEOUT_SECONDS}s. Commands were: {_install_preview!r}."
                        ),
                    }

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
                
                if not isinstance(tool_spec, dict):
                    tool_spec = {"name": str(tool_spec) if tool_spec else ""}
                try:
                    install_result = installer.install_tool(tool_spec)
                    
                    install_output = ""
                    if isinstance(install_result, dict):
                        install_output = install_result.get("output", "") or ""
                    elif isinstance(install_result, str):
                        install_output = install_result
                    install_ok = (
                        install_result is True
                        or (isinstance(install_result, dict) and install_result.get("success", False))
                    )
                    return {
                        "success": install_ok,
                        "reward": 0.8 if install_ok else -0.5,
                        "output": install_output,
                        "returncode": 0 if install_ok else 1,
                    }
                except Exception as inst_err:
                    return {
                        "success": False,
                        "reward": -0.5,
                        "output": str(inst_err),
                        "returncode": 1,
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
                    
                    _VERIFY_SAFE_PREFIXES: frozenset = frozenset({
                        "which ",
                        "command -v ",
                        "test -f ",
                        "test -d ",
                        "test -e ",
                        "test -x ",
                        "stat ",
                        "ls ",
                        "echo ",
                        "cat /proc/version",
                        "python --version",
                        "python3 --version",
                        "python -c \"import ",
                        "python3 -c \"import ",
                        "node --version",
                        "node -v",
                        "npm --version",
                        "npm -v",
                        "git --version",
                        "git -v",
                        "java -version",
                        "java --version",
                        "go version",
                        "rustc --version",
                        "cargo --version",
                        "docker --version",
                        "pip --version",
                        "pip3 --version",
                        "pip show ",
                        "dpkg -l ",
                        "rpm -q ",
                        "brew list ",
                        "type ",
                    })
                    cmd_lower = cmd.lower().lstrip()
                    _verify_allowed = any(
                        cmd_lower.startswith(prefix)
                        for prefix in _VERIFY_SAFE_PREFIXES
                    )
                    if not _verify_allowed:
                        log_warn(
                            f"[H-01] verify command BLOCKED — not in safe read-only "
                            f"allowlist: {cmd[:120]!r}. Use method='screenshot' for "
                            "non-probe verification."
                        )
                        return {
                            "success": False,
                            "reward": -1.0,
                            "reason": (
                                f"verify command blocked: {cmd[:80]!r} is not in the "
                                "safe verify-command allowlist. Verify commands must be "
                                "read-only probes (which, test, stat, --version, etc.)."
                            ),
                        }
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
        
        _exc_type = type(exc).__name__
        if _exc_type == "FailSafeException" or (
            _FailSafeException is not None and isinstance(exc, _FailSafeException)
        ):
            log_warn(
                f"_execute_decision: pyautogui FailSafeException [{op}] — "
                "cursor reached a screen corner. Bandit penalised with reward=-1.0. "
                "LLM must avoid coordinates at screen edges (within ~5px of any border)."
            )
            return {
                "success": False,
                "reward": -1.0,  # severe: strong bandit de-prioritization
                "reason": (
                    "pyautogui_failsafe: cursor reached a screen corner — pyautogui "
                    "FAILSAFE triggered. Action blocked for safety. "
                    "Use coordinates away from screen edges (avoid 0,0 / W,0 / 0,H / W,H)."
                ),
            }

        log_warn(f"_execute_decision: unexpected error [{op}] — {exc}")
        return {
            "success": False,
            "reward": -0.5,
            "reason": f"unexpected_error: {exc}",
        }

