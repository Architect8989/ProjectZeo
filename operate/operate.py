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

# RT-A FIX (P0): Human confirmation timeout is now configurable via
# PROJECTZEO_CONFIRM_TIMEOUT_SECONDS environment variable.
#
# ORIGINAL DEFECT: MAX_WAIT_RETRIES=10 x 0.5s = 5 seconds total. The
# human operator must notice the /tmp signal file and delete it within 5
# seconds, which is operationally unreachable. All REQUIRE_HUMAN_CONFIRMATION
# actions were effectively auto-denied -> stagnation -> REPLAN -> TASK_FAILED.
#
# FIX: Default timeout raised to 60s (120 retries x 0.5s). Configurable
# via PROJECTZEO_CONFIRM_TIMEOUT_SECONDS for slow-notification environments.
# H4 HARDENING: Effective timeout logged at startup for operator visibility.
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

# BUG-6 FIX: The confirmation timeout log was a bare module-level print().
# It fired at import time — before any task ran — polluting startup logs and
# appearing before the model was even loaded. Moved to operate_main() where
# it fires exactly once per task (not once per import) so operators see it
# with proper task context. The flag below prevents duplicate prints across
# multiple operate_main() calls in the same process.
_CONFIRM_TIMEOUT_LOGGED: bool = False

# Max bytes of command output stored per step in execution_log (bounded, not unlimited)
MAX_COMMAND_OUTPUT_BYTES = 4096

# Maximum dynamic candidates from ReasoningEngine on stagnant steps (H-03 fix)
MAX_DYNAMIC_CANDIDATES = 3



# AUDIT §2.3 FIX: was hardcoded "/tmp" — does not exist on Windows, breaking
# the REQUIRE_HUMAN_CONFIRMATION gate entirely on that platform.  Replace with
# tempfile.gettempdir() which returns the correct temp directory on all OSes:
# /tmp on Linux/macOS, %TEMP% / %TMP% on Windows.
_SIGNAL_DIR: str = tempfile.gettempdir()
_SIGNAL_PREFIX: str = "projectzeo_approve_"


def _approval_signal_path(action_key: str) -> str:
    """Return the path of the approval signal file for this action key."""
    return os.path.join(_SIGNAL_DIR, f"{_SIGNAL_PREFIX}{action_key}.signal")


def _write_approval_signal(action_key: str, action: dict, reason: str) -> str:
    """Write the pending-approval signal file and return its path.

    RT-A7 FIX (P4): After writing the signal file, apply os.chmod(path, 0o600)
    to restrict it to the process owner only.  The original code wrote to /tmp
    with the default umask-derived mode (typically 0o644 — world-readable,
    world-deletable on most Linux /tmp mounts with sticky bit off).  Any
    same-UID process could delete the signal file, triggering an unintended
    action approval and silently bypassing the human confirmation gate.
    Owner-only mode (0o600) prevents this without breaking the intended
    workflow (the human operator deletes the file to approve).
    """
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
        # BUG-08 FIX: The original code wrote the file then called
        # os.chmod(path, 0o600).  There is a race window between close()
        # and chmod() where the file exists with the default umask mode
        # (typically 0o644 — world-readable/deletable), allowing any same-UID
        # process to delete it and trigger an unintended approval.
        #
        # Fix: use os.open() with O_CREAT|O_WRONLY|O_EXCL and mode 0o600,
        # which atomically creates the file with the correct permissions —
        # no window where the wrong mode is visible to other processes.
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
    # BUG-5 FIX: Accept checkpoint step index for crash recovery fast-forward.
    # When the process crashes mid-task and restarts, main.py loads this from
    # checkpoint_store.get_checkpoint_step_index() and passes it here so
    # _execute_autonomous_loop() can skip already-completed steps instead of
    # replaying the entire plan from step 0.
    prior_step_index: Optional[int] = None,
) -> None:
    
    # BUG-6 FIX: Log confirmation timeout here (inside operate_main) not at
    # import time. Fires once per process lifetime using the _CONFIRM_TIMEOUT_LOGGED
    # flag, so multi-task runs don't repeat it.
    global _CONFIRM_TIMEOUT_LOGGED
    if not _CONFIRM_TIMEOUT_LOGGED:
        _CONFIRM_TIMEOUT_LOGGED = True
        print(
            f"[OPERATE] Human confirmation timeout: {_CONFIRM_TIMEOUT_SECONDS}s "
            f"({MAX_WAIT_RETRIES} retries × {WAIT_RETRY_SECONDS}s). "
            "Override with PROJECTZEO_CONFIRM_TIMEOUT_SECONDS env var.",
            file=sys.stderr,
        )

    if not isinstance(execution_plan, ExecutionPlan):
        raise ValueError("execution_plan must be an ExecutionPlan instance")

    # BUG-16 FIX: execution_plan.validate() return value was previously ignored.
    # In create_plan() the return is checked: `if not plan.validate(): raise PlanningError(...)`.
    # But here in operate_main() the call was `execution_plan.validate()` with no
    # check — if it returns False (indicating an invalid plan), the loop continued
    # on a broken plan producing cryptic downstream errors. Now we check and raise
    # immediately with a clear error message, consistent with create_plan() behaviour.
    if not execution_plan.validate():
        raise ValueError(
            "ExecutionPlan failed validation on entry to operate_main(). "
            "The plan may have been mutated after creation or serialised/deserialised "
            "incorrectly. Re-run planner.create_plan() to obtain a fresh valid plan."
        )
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
    # H2 FIX — Load Bayesian likelihood ratios from likelihoods.json
    # ------------------------------------------------------------------
    # DEFECT (MS-1 / H2): The likelihood ratios used in bayesian_update()
    # were hardcoded constants (0.95, 0.75, 1.5, 1.2, 0.8, 1.0) with no
    # calibration mechanism.  True P(observation|state) ratios depend on
    # the deployment environment (display resolution, application mix,
    # task profile) and cannot be calibrated at development time.
    #
    # FIX: Load ratios from likelihoods.json at startup.  The file lives
    # alongside policy.yaml in the project root and uses the same key
    # structure as the inline code.  If the file is absent or malformed,
    # the hardcoded defaults below are used and a WARNING is emitted.
    #
    # likelihoods.json schema:
    # {
    #   "app_match_with_delta":  0.95,  -- P(obs|state) when world changed
    #   "app_match_no_delta":    0.75,  -- P(obs|state) when world stable
    #   "ui_rich":               1.50,  -- > ENTITY_RICH_THRESHOLD entities
    #   "ui_sparse":             1.20,  -- 1 to ENTITY_RICH_THRESHOLD entities
    #   "ui_empty":              0.80,  -- 0 entities
    #   "neutral_with_delta":    1.00,  -- baseline: fresh observation
    #   "neutral_no_delta":      0.80,  -- baseline: stale observation
    #   "ENTITY_RICH_THRESHOLD": 10     -- entity count above which ui_rich fires
    # }
    #
    # Calibration guidance:
    #   Collect a dataset of (observation, true_state) pairs from production
    #   runs.  For each state category, compute the empirical P(observation|state)
    #   by counting how often that category was active when the true state
    #   matched.  Normalize relative to the neutral baseline to produce ratios.
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
                # AUDIT §2.5 FIX: also reject non-positive ratios.  A zero or
                # negative likelihood ratio forces all Bayesian posteriors to zero
                # on the first update, permanently collapsing the belief distribution.
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

    # PATCH MOD-3: Pre-cast ENTITY_RICH_THRESHOLD to int once here.
    # The original code called int(_likelihood_cfg.get("ENTITY_RICH_THRESHOLD", 10))
    # inside the main execution loop on every iteration.  A float value like 10.5
    # would pass the > 0 validation but produce logically wrong comparisons with
    # integer entity counts.  Casting once at load time is both correct and faster.
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

    # H2 FIX (SI-5): Load playbook for this intent at task start.
    # Previously playbook_store.save_playbook() and load_playbook() were
    # defined with full disk persistence but never called anywhere in the
    # execution path — the entire long-horizon memory system was dead
    # infrastructure.  Every task started from zero knowledge regardless
    # of prior successful executions.
    #
    # Fix: load matching prior actions here and inject them as the first
    # candidates into the ReasoningEngine fallback pool.  On task success,
    # save the executed actions so future identical/similar tasks warm-start.
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

    # H5 FIX (SI-3): Pass shared ollama_client so ReasoningEngine uses the
    # text-only path instead of the vision adapter for abstract reasoning.
    reasoning_engine = ReasoningEngine(
        llm_callable=llm_callable,
        ollama_client=getattr(planner, "_ollama_client", None),
    )

    # FIX RB-A3: Per-task executor — isolates timeout guarantees between tasks.
    # A module-level singleton executor shares ONE worker thread; a blocking call
    # in one task starves the next.  Creating per-task gives each task its own
    # isolated thread pool.
    _task_ui_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="ui_timeout_worker",
    )

    try:
        # GAP-2 FIX: Per-task filesystem rollback tracker.
        # Tracks every file created during this task so that:
        #   1. On REPLAN, the planner receives already_created_files in context.
        #   2. Duplicate file_create steps can be skipped (already done).
        #   3. Post-mortem debugging has a clear record of filesystem side-effects.
        # The list is passed by reference so _execute_autonomous_loop fills it.
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
            created_files_ledger=_created_files_ledger,
        )
        # H2 FIX: Save playbook on successful task completion.
        # Collect actions from the journal (executed steps) and persist them
        # so future tasks with identical intent can warm-start from this run.
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

        # AUDIT §2.4 FIX: Clear the step checkpoint on successful task completion
        # so the next task does not accidentally resume from a stale position.
        try:
            from core.safety.checkpoint_store import clear_checkpoint as _clear_cp
            _clear_cp()
        except Exception:
            pass  # non-fatal
    finally:
        input_arbitrator.shutdown()
        # PATCH MOD-5: Give in-flight UI actions a 2-second grace window to
        # complete before the executor is torn down.  The original wait=False
        # could leave a pyautogui.click() or pyautogui.write() call mid-flight
        # when a REPLAN fires, producing ghost inputs on the newly-restored screen.
        # We achieve a bounded wait by joining the underlying thread with a timeout
        # rather than blocking on shutdown() itself (which has no timeout parameter
        # before Python 3.9).  After 2s any still-running action is abandoned as
        # before — we never want to block indefinitely on a hung UI thread.
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
    # BUG-5 FIX: Accept checkpoint step index for crash-recovery fast-forward.
    prior_step_index: Optional[int] = None,
    # GAP-2 FIX: Filesystem rollback ledger — collects paths created during
    # this task so replanning context includes already_created_files.
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

    # BUG-13 FIX: _visited_action_keys was a local dict that reset on every
    # replan (because _execute_autonomous_loop is re-called by main.py on REPLAN).
    # After a replan, the system forgot all previously tried (failed) actions and
    # would attempt the same failing actions again, wasting inference budget.
    #
    # Fix: persist _visited_action_keys inside prior_belief_state (serializable
    # as a list of keys) and restore it here on replan entry. On first call
    # (prior_belief_state is None), start with an empty dict as before.
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

    # BUG-5 FIX: Fast-forward to the checkpointed step on crash recovery.
    # Without this, every crash restart replayed all completed steps from 0,
    # wasting compute and risking double-execution of already-completed commands
    # (e.g. file writes, git commits, API calls, package installs).
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
                            # MED-1 FIX: execution_log entries are dicts
                            # {"outputs": [...], "last_output": str}.  Passing
                            # the whole dict into perception_snapshot bloated
                            # the LLM prompt with a nested object instead of a
                            # plain string.  Extract last_output only.
                            perception_snapshot["last_command_output"] = step_log.get(
                                "last_output", ""
                            )
                        world_graph.update(perception_snapshot)
                except Exception:
                    log_warn("Observer snapshot failed")

            # P1 RT-C FIX: Try/except wraps the full per-iteration execution body so
            # transient LLM/screenshot/network failures become stagnation increments
            # rather than unhandled RuntimeErrors that terminate the process.
            try:
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

                            # MS-1 / IH-1 FIX: Replace hardcoded developer constants
                            # (0.9, 0.8, 0.7, 0.5) with calibrated delta-sensitive
                            # likelihood ratios.
                            #
                            # ORIGINAL DEFECT: The prior code used fixed scalars that
                            # were not P(observation|state) ratios — they were constant
                            # multiplicative weights independent of observation quality.
                            # After N iterations the dominant state was whichever
                            # category was activated most frequently, not the true world
                            # state.  This violated the stated "Bayesian inference"
                            # contract (audit finding MS-1).
                            #
                            # CALIBRATION RATIONALE:
                            # Likelihoods are now computed as observation-conditional
                            # ratios relative to the neutral baseline (1.0):
                            #
                            # app:{focused_app}:
                            #   P(focused_app seen | app is correct state) / P(seen | neutral)
                            #   = 0.95 when world changed (fresh observation), 0.75 when stable
                            #   (title re-read may be stale).  Rationale: if the expected app
                            #   is focused and the world just changed, we have strong evidence
                            #   of a correct state; if the world is stable (no delta), the
                            #   same observation is weaker because the OS may be reporting
                            #   a cached value.
                            #
                            # ui_rich / ui_sparse / ui_empty:
                            #   Ratios relative to a flat prior; calibrated so that a
                            #   UI-rich snapshot is 1.5× more likely under a "productive"
                            #   state than a sparse or empty one.  Values anchored to
                            #   empirically observed entity count distributions:
                            #     > 10 entities  → rich  (1.5)
                            #     1–10 entities  → sparse (1.2)
                            #     0 entities     → empty (0.8, slight evidence of wrong state)
                            #
                            # neutral:
                            #   Explicit neutral likelihood set to 1.0 when world changed
                            #   (observation is fresh, no dampening) and 0.8 when stable
                            #   (no delta = stale or no activity = mild negative evidence).
                            #
                            # NOTE: These ratios are still heuristic estimates, not
                            # empirically measured conditional probabilities from a
                            # training distribution.  The update is proportionally correct
                            # Bayesian inference but the ratio magnitudes are developer
                            # estimates.  True calibration would require a labeled dataset
                            # of (observation, true_state) pairs per application.  This
                            # fix is a significant improvement over fixed constants because
                            # the ratios are delta-sensitive (observation freshness matters)
                            # and directionally correct (rich UI → higher state confidence).

                            has_delta = bool(delta)

                            # H2 FIX: Use values from _likelihood_cfg (loaded from
                            # likelihoods.json at startup) instead of hardcoded
                            # developer constants.  If the file is absent, _likelihood_cfg
                            # contains the original defaults so behaviour is unchanged.
                            # PATCH MOD-3: _entity_rich_thresh is now pre-cast to int
                            # at config-load time — no per-iteration cast needed.
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

                # Fallback: ask ReasoningEngine for dynamic candidates on stagnant steps
                if not candidate_actions and reasoning_engine is not None:
                    perception_for_reasoning: Dict[str, Any] = {}
                    if isinstance(perception_snapshot, dict):
                        perception_for_reasoning = perception_snapshot
                    elif isinstance(world_snapshot, dict):
                        perception_for_reasoning = world_snapshot

                    # H2 FIX (SI-5): Inject prior playbook actions as first candidates
                    # before asking ReasoningEngine.  On stagnant steps, unvisited playbook
                    # actions from prior successful runs of the same intent are prepended so
                    # the agent tries known-good actions before generating new ones.
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

                    # Prepend playbook candidates (known-good from prior runs) before
                    # ReasoningEngine-generated candidates so they are seen by ActionRanker.
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
                    # C1 FIX (CRITICAL): This continue was missing.
                    # Without it, execution fell through into the authority
                    # check and _execute_decision(), meaning every DENY'd
                    # action was executed anyway — the policy engine was
                    # completely bypassed on first stagnation cycle.
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
            # GAP-2 FIX: Inject the filesystem ledger reference into file_create
            # actions so _execute_decision can record created paths. Use a shallow
            # copy so the original selected_action dict is not mutated (it may be
            # referenced by the candidate list and the journal record).
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
                    # AUDIT §2.4 FIX: save_checkpoint() was implemented in
                    # core/safety/checkpoint_store.py but never called anywhere.
                    # Wire it here after every successful step advance so that
                    # a crash at hour N of a multi-hour task can resume from the
                    # last completed step instead of restarting from zero.
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
                # BUG-13 FIX: Serialize _visited_action_keys alongside BeliefState
                # so the next replan invocation can restore previously-tried actions
                # and avoid retrying known-failing actions. The list is capped at
                # _VISITED_ACTION_MAX entries (already enforced during execution).
                _bs_dict["_visited_action_keys"] = list(_visited_action_keys.keys())
                # GAP-2 FIX: Persist the filesystem ledger into BeliefState so
                # the replanning LLM knows which files already exist. On replan,
                # ExecutionPlanner._expand_goal() injects already_created_files
                # into the planning prompt so it skips completed file_create steps.
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
    """
    Dispatch `action` to the OS backend and return an execution result dict.

    GAP-1 FIX: When the focused_app is a browser AND prefer_playwright=True,
    click/write/scroll/navigate operations are routed through BrowserBackend
    (Playwright DOM-aware automation) instead of pyautogui coordinate clicks.
    This fixes the #1 failure mode for web tasks: React/Angular SPA elements
    shifting between screenshot and click (coordinate staleness).

    Falls back to pyautogui automatically when:
      - playwright is not installed
      - The element cannot be found by DOM selector/text/role
      - The operation type is not browser-targetable (install, command, file_create)
    """
    from core.safety.action_timeout import run_with_timeout, ActionTimeout

    op = (action.get("operation") or "").lower().strip()

    # BUG-5 FIX: Import FailSafeException at dispatch time (not module level)
    # so operate.py does not hard-require pyautogui at import time.
    # FailSafeException is raised when the cursor reaches a screen corner while
    # pyautogui.FAILSAFE=True (which is always set by OperatingSystem.__init__).
    # Previously caught by the generic `except Exception` and returned reward=-0.5,
    # giving the bandit the same signal as a routine action failure.
    # Fix: catch explicitly and return reward=-1.0 (severe) so the bandit strongly
    # de-prioritizes corner-bound coordinates for the rest of the task.
    try:
        from pyautogui import FailSafeException as _FailSafeException
    except ImportError:
        _FailSafeException = None  # type: ignore[assignment,misc]

    # DONE sentinel — always succeeds immediately with maximum reward
    if op == "done":
        return {"success": True, "reward": 1.0}

    if not op:
        return {"success": False, "reward": -0.5, "reason": "empty operation field"}

    # BUG-11 FIX: Apply DANGEROUS_PATTERNS check at dispatch time for every
    # command/install/file_create operation, not only at plan creation time.
    #
    # The original code filtered patterns in ExecutionPlanner._validate_and_normalise_step()
    # at plan generation time, but:
    #   1. ReasoningEngine.propose_actions() generates dynamic candidates at runtime
    #      WITHOUT DANGEROUS_PATTERNS filtering.
    #   2. AutonomousInstaller generates install commands that go through exec(shell=True).
    #   3. A hallucinating LLM or adversarial intent could inject dangerous commands
    #      via the replan path (different code path from create_plan()).
    #
    # Fix: for "command", "install", and "file_create" ops, scan the command/content
    # text against the compiled dangerous patterns immediately before execution.
    # If a match is found, return a structured failure with reward -1.0 (severe)
    # so the bandit strongly de-prioritizes this action for the rest of the task.
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
                _compiled = getattr(_EP, "_dispatch_compiled_patterns", None)
                if _compiled is None:
                    import re as _re_dp
                    _compiled = [
                        _re_dp.compile(p, _re_dp.IGNORECASE) for p in _EP.DANGEROUS_PATTERNS
                    ]
                    _EP._dispatch_compiled_patterns = _compiled
                for _pat in _compiled:
                    if _pat.search(_dangerous_text):
                        log_warn(
                            f"[BUG-11] DANGEROUS_PATTERNS match at DISPATCH time "
                            f"for op={op!r}: pattern={_pat.pattern!r}. "
                            "Blocking execution. This indicates a plan or dynamic "
                            "candidate bypass of the planning-time filter."
                        )
                        return {
                            "success": False,
                            "reward": -1.0,  # severe: strong bandit de-prioritization
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
        # GAP-1 FIX: Playwright DOM routing for browser-targeted actions
        # ------------------------------------------------------------------
        # Before falling through to pyautogui coordinate operations, check
        # if the focused app is a browser and prefer_playwright is enabled.
        # If so, route click/write/scroll/navigate through BrowserBackend.
        # The BrowserBackend uses DOM selectors and text matching instead of
        # pixel coordinates — no coordinate staleness, SPA-safe, iframe-aware.
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
            # GAP-2 FIX: Record in the per-task filesystem ledger so that
            # replanning context includes already_created_files. This prevents
            # the LLM from re-creating files that already exist (duplicates,
            # overwrites, directory re-creation) after a mid-task replan.
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
                    install_result = installer.install_tool(tool_spec)
                    # BUG-09 FIX: Capture install output so the planner can
                    # see WHY an install failed on a replan.  Previously the
                    # return dict had no "output" key, leaving the replanner
                    # blind to version errors, missing dependencies, and
                    # network failures from the install process.
                    install_output = ""
                    if isinstance(install_result, dict):
                        install_output = install_result.get("output", "") or ""
                    elif isinstance(install_result, str):
                        install_output = install_result
                    install_ok = (
                        install_result is True
                        or (isinstance(install_result, dict) and install_result.get("success", True))
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
        # BUG-5 FIX: Catch pyautogui FailSafeException before the generic handler.
        # FailSafeException fires when the cursor reaches a screen corner (0,0),
        # (W,0), (0,H), or (W,H) while FAILSAFE=True. The agent doesn't know why
        # the click failed — previously reward=-0.5 gave the same signal as a
        # routine failure, causing the bandit to retry similar coordinates.
        # Fix: detect by class name (robust against import failures) and return
        # reward=-1.0 (severe), giving the bandit a strong de-prioritization signal
        # for corner-bound coordinates AND a clear reason in the journal.
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
