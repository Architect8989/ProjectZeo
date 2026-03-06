from __future__ import annotations

# =============================================================================
# ARCHITECTURE: GII REASONING LOOP
# =============================================================================
# AUDIT-CRIT-4 FIX: Replaced the scripted ExecutionPlan step-iteration model
# with a pure goal-directed reasoning loop:
#
#   while not goal_complete:
#       world  = observe()
#       action = reason(world, goal)
#       evaluate_safety(action)
#       execute(action)
#
# The ExecutionPlan is now used ONLY as a scaffold (high-level phase guidance)
# passed to PerStepReasoner.  It is NOT iterated.  There is NO current_step_index.
# Goal completion is signalled by the reasoner emitting {"operation": "done"}.
#
# Files changed: operate/operate.py (this file), main.py, core/planner/execution_planner.py
# =============================================================================

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


# pyautogui is optional — import once at module level
try:
    import pyautogui as _pyautogui
    _PYAUTOGUI_AVAILABLE: bool = True
except ImportError:
    _pyautogui = None  # type: ignore[assignment]
    _PYAUTOGUI_AVAILABLE: bool = False


MAX_PERCEPTION_ENTITIES = 20
MAX_PERCEPTION_JSON_BYTES = 10_000

REPLAN_SIGNAL: str = "REPLAN_REQUIRED"

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
    return 60


_CONFIRM_TIMEOUT_SECONDS: int = _resolve_confirm_timeout()
MAX_WAIT_RETRIES: int = max(int(_CONFIRM_TIMEOUT_SECONDS / WAIT_RETRY_SECONDS), 1)

_CONFIRM_TIMEOUT_LOGGED: list = [False]

MAX_COMMAND_OUTPUT_BYTES = 4096

# AUDIT-HIGH-6 FIX: Module-level credential scrubbing regex (compiled once at import).
import re as _re_module
_CRED_SCRUB_RE = _re_module.compile(
    r"(?:password|passwd|secret|token|api[_\-]?key|auth[_\-]?token"
    r"|bearer|private[_\-]?key|aws[_\-]?secret|access[_\-]?key"
    r"|database[_\-]?url|db[_\-]?password|connection[_\-]?string"
    r"|encryption[_\-]?key|signing[_\-]?key|client[_\-]?secret"
    r"|x[_\-]?api[_\-]?key|authorization"
    r")\s*[:=]\s*\S+",
    _re_module.IGNORECASE,
)

# SEC-4 FIX: Regex to detect password-like values typed into fields.
# Short alphanumeric strings with special chars are redacted from write/type journal entries.
_TYPED_CREDENTIAL_RE = _re_module.compile(
    r"(?:password|passwd|secret|token|api.?key|bearer|private.?key"
    r"|aws.?secret|access.?key|auth.?token)\s*[:=]?\s*\S+",
    _re_module.IGNORECASE,
)


def _scrub_credentials(text: str) -> str:
    """Replace credential values with <REDACTED> in command output."""
    if not isinstance(text, str) or not text:
        return text
    return _CRED_SCRUB_RE.sub(
        lambda m: m.group(0).split(":")[0].split("=")[0] + "=<REDACTED>",
        text,
    )


def _scrub_write_type_content(action: dict) -> dict:
    """
    SEC-4 FIX: Scrub sensitive content from write/type action dicts before journaling.
    The content field of write/type operations may contain typed passwords.
    We redact credential patterns and also redact if the action has a password role.
    Returns a copy with content scrubbed if necessary.
    """
    op = str(action.get("operation", "")).lower()
    if op not in ("write", "type"):
        return action
    content = str(action.get("content") or action.get("text") or "")
    if not content:
        return action
    # Scrub if content matches credential patterns
    scrubbed = _scrub_credentials(content)
    # Also scrub if action context indicates password role
    role = str(action.get("role") or action.get("field_type") or "").lower()
    if "password" in role or "secret" in role or "token" in role:
        scrubbed = "<REDACTED:password_field>"
    if scrubbed != content:
        action_copy = dict(action)
        if "content" in action_copy:
            action_copy["content"] = scrubbed
        if "text" in action_copy:
            action_copy["text"] = scrubbed
        return action_copy
    return action


MAX_DYNAMIC_CANDIDATES = 3

import os as _os_sig
import secrets as _secrets_mod

# H-08 FIX: Per-session secure signal directory.
_SESSION_TOKEN: str = _secrets_mod.token_hex(16)
_SIGNAL_DIR_BASE: str = tempfile.gettempdir()
_SIGNAL_DIR: str = _os_sig.path.join(
    _SIGNAL_DIR_BASE,
    f"projectzeo_{_SESSION_TOKEN}",
)
try:
    _os_sig.makedirs(_SIGNAL_DIR, mode=0o700, exist_ok=True)
    _os_sig.chmod(_SIGNAL_DIR, 0o700)
except OSError:
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
        # SEC-4 FIX: Scrub write/type content before writing to approval signal
        safe_action = _scrub_write_type_content(action)
        content = json.dumps(
            {
                "action_key": action_key,
                "action": safe_action,
                "reason": reason,
                "instruction": (
                    f"CREATE file {path}.APPROVE to approve the action.\n"
                    "(Denial is automatic after "
                    f"{_CONFIRM_TIMEOUT_SECONDS}s."
                ),
            },
            indent=2,
        )
        fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
    except FileExistsError:
        pass
    except OSError as e:
        print(
            f"[OPERATE] Warning: could not write approval signal file {path!r}: {e}",
            file=sys.stderr,
        )
    return path


def _remove_approval_signal(path: str) -> None:
    """Remove the signal file; ignore errors."""
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
    prior_step_index: Optional[int] = None,   # DEPRECATED — kept for API compat; ignored
    prior_execution_log: Optional[dict] = None,
    gii_controller=None,
) -> None:
    """
    Main entry point for autonomous task execution.

    ARCHITECTURE NOTE (AUDIT-CRIT-4):
        The prior_step_index parameter is DEPRECATED and IGNORED.
        The execution loop no longer iterates ExecutionPlan.steps.
        Instead it runs a pure `while not goal_complete` reasoning loop
        where PerStepReasoner/GIIController decides each action from world state.
    """
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
            "Re-run planner.create_plan() to obtain a fresh valid plan."
        )
    PlanVerifier().verify(execution_plan)

    os_backend = os_backend or OperatingSystem()

    # AUDIT-BLOCKER-5 FIX: Missing policy.yaml must be a FATAL startup error.
    _allow_default_policy = os.environ.get("PROJECTZEO_ALLOW_DEFAULT_POLICY", "0").strip() == "1"
    policy_engine = PolicyEngine()
    try:
        import os as _os_mod
        import yaml as _yaml  # type: ignore[import]
        _policy_path = _os_mod.path.join(
            _os_mod.path.dirname(__file__), "..", "policy.yaml"
        )
        if not _os_mod.path.exists(_policy_path):
            if _allow_default_policy:
                print(
                    "[operate_main] WARNING: policy.yaml not found. "
                    "Running with built-in defaults (PROJECTZEO_ALLOW_DEFAULT_POLICY=1 set). "
                    "Create policy.yaml for production deployments.",
                    file=sys.stderr,
                )
            else:
                raise RuntimeError(
                    "FATAL: policy.yaml not found at expected path: "
                    f"{_os_mod.path.abspath(_policy_path)}. "
                    "A policy.yaml file is required for production deployments. "
                    "Create one based on the template in the repository. "
                    "To bypass this check (development only), set: "
                    "PROJECTZEO_ALLOW_DEFAULT_POLICY=1"
                )
        else:
            with open(_policy_path, "r", encoding="utf-8") as _pf:
                _pcfg = _yaml.safe_load(_pf) or {}
            policy_engine = PolicyEngine.from_policy_yaml(_pcfg)
            print(
                f"[operate_main] policy.yaml loaded from {_os_mod.path.abspath(_policy_path)}",
                file=sys.stderr,
            )
    except RuntimeError:
        raise
    except ImportError:
        if _allow_default_policy:
            print(
                "[operate_main] WARNING: pyyaml not installed — policy.yaml not loaded. "
                "Install pyyaml for full policy support: pip install pyyaml",
                file=sys.stderr,
            )
        else:
            raise RuntimeError(
                "FATAL: pyyaml is not installed. Cannot load policy.yaml. "
                "Install it: pip install pyyaml"
            )
    except Exception as _policy_err:
        print(
            f"[operate_main] FATAL: Failed to load policy.yaml: {_policy_err}. ",
            file=sys.stderr,
        )
        if not _allow_default_policy:
            raise RuntimeError(f"FATAL: policy.yaml load failed: {_policy_err}") from _policy_err

    # Process fingerprint
    _auto_discovered_names: list = []
    try:
        import psutil as _psutil_ad
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
                f"processes observed at task start. "
                f"Sample: {sorted(_auto_discovered_names)[:8]}. "
                "Add apps explicitly via policy.yaml allowed_apps.",
                file=sys.stderr,
            )
    except ImportError:
        print(
            "[operate_main] WARNING: psutil not installed — cannot fingerprint running apps.",
            file=sys.stderr,
        )
    except Exception as _disc_err:
        print(f"[operate_main] WARNING: process fingerprint failed: {_disc_err}.", file=sys.stderr)

    # Environment fingerprint tools
    try:
        _fp_tools = (
            (world_graph.snapshot().get("environment", {}) if world_graph else {})
            or {}
        )
        _fp_tools_dict = _fp_tools.get("tools", {})
        if isinstance(_fp_tools_dict, dict):
            for _tool_name, _present in _fp_tools_dict.items():
                if _present and isinstance(_tool_name, str):
                    policy_engine.allow_app(_tool_name)
    except Exception:
        pass

    # Bayesian likelihood ratios
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
                        f"non-positive ratios: {_bad_range}"
                    )
                _likelihood_cfg.update(_lh_raw)
                print(
                    f"[operate_main] Loaded Bayesian likelihood ratios from likelihoods.json.",
                    file=sys.stderr,
                )
            else:
                raise ValueError("likelihoods.json root must be a JSON object")
        else:
            print(
                "[operate_main] likelihoods.json not found — using hardcoded defaults.",
                file=sys.stderr,
            )
    except Exception as _lh_err:
        print(
            f"[operate_main] WARNING: Failed to load likelihoods.json: {_lh_err}. Using defaults.",
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

    # AUDIT-BLOCKER-3 FIX: Pre-task restoration scope disclosure.
    _task_involves_browser = any(
        kw in terminal_prompt.lower()
        for kw in ("browser", "firefox", "chrome", "chromium", "web", "url", "http", "download")
    )
    _task_involves_documents = any(
        kw in terminal_prompt.lower()
        for kw in ("document", "file", "edit", "write", "save", "libreoffice", "word", "spreadsheet")
    )
    print(
        "\n[RESTORATION DISCLOSURE] This task will be operated by ProjectZeo GII.\n"
        "Restoration scope is LIMITED:\n"
        "  ✓ Cursor position will be restored\n"
        "  ✓ Window focus will be restored (best-effort, title matching)\n"
        "  ✗ Browser tabs, URLs, and scroll position will NOT be restored\n"
        "  ✗ Clipboard contents will NOT be restored\n"
        "  ✗ Unsaved document state will NOT be restored\n"
        "  ✗ Terminal session state (cwd, env vars) will NOT be restored\n"
        + (
            "  ⚠ BROWSER TASK DETECTED: Restoration will be incomplete if task fails.\n"
            if _task_involves_browser else ""
        )
        + (
            "  ⚠ DOCUMENT TASK DETECTED: Save your work before proceeding.\n"
            if _task_involves_documents else ""
        )
        + "Proceeding with task execution...\n",
        file=sys.stderr,
    )

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
                f"[operate_main] Loaded playbook for intent "
                f"({len(_prior_playbook_actions)} prior actions).",
                file=sys.stderr,
            )
    except Exception as _pb_load_err:
        print(
            f"[operate_main] WARNING: playbook load failed: {_pb_load_err}.",
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
            prior_execution_log=prior_execution_log,
            created_files_ledger=_created_files_ledger,
            gii_controller=gii_controller,
            likelihood_cfg=_likelihood_cfg,
        )

        try:
            _journal_actions = journal.get_all() if hasattr(journal, "get_all") else []
            if _journal_actions:
                _save_playbook(terminal_prompt, _journal_actions)
                print(
                    f"[operate_main] Saved playbook for intent "
                    f"({len(_journal_actions)} actions).",
                    file=sys.stderr,
                )
        except Exception as _pb_save_err:
            print(f"[operate_main] WARNING: playbook save failed: {_pb_save_err}.", file=sys.stderr)

        try:
            from core.safety.checkpoint_store import clear_checkpoint as _clear_cp
            _clear_cp()
        except Exception:
            pass
    finally:
        input_arbitrator.shutdown()
        import threading as _threading
        _ui_threads = [t for t in _threading.enumerate()
                       if t.name.startswith("ui_timeout_worker")]
        for _t in _ui_threads:
            _t.join(timeout=2.0)
        _task_ui_executor.shutdown(wait=False)


# =========================================================================
# GII AUTONOMOUS EXECUTION LOOP
# =========================================================================
# AUDIT-CRIT-4 FIX: Replaced the scripted plan-step iteration model with a
# pure goal-directed reasoning loop.  There is NO current_step_index.
# The ExecutionPlan is used only as a scaffold (high-level phase descriptions)
# passed to PerStepReasoner for guidance — it is NOT iterated.
# =========================================================================

def _execute_autonomous_loop(
    *,
    terminal_prompt: str,
    execution_plan: ExecutionPlan,           # Used as scaffold only — NOT iterated
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
    prior_execution_log: Optional[dict] = None,
    created_files_ledger: Optional[List[str]] = None,
    gii_controller=None,
    likelihood_cfg: Optional[dict] = None,
) -> None:

    start_ts = time.time()
    progress.start_execution()

    _likelihood_cfg = likelihood_cfg or {
        "app_match_with_delta": 0.95, "app_match_no_delta": 0.75,
        "ui_rich": 1.50, "ui_sparse": 1.20, "ui_empty": 0.80,
        "neutral_with_delta": 1.00, "neutral_no_delta": 0.80,
        "ENTITY_RICH_THRESHOLD": 10,
    }

    journal.record({
        "event": "execution_start",
        "objective": execution_plan.objective,
        "execution_model": "GII_REASONING_LOOP",
        "note": "current_step_index removed — pure goal-directed loop",
    })

    # ------------------------------------------------------------------
    # BeliefState — reconstruct from prior replan if available
    # ------------------------------------------------------------------
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

    # Scaffold phases from the execution plan (guidance, not iteration targets)
    scaffold_phases: List[Dict[str, Any]] = []
    for _step in execution_plan.steps:
        _stype = str(getattr(getattr(_step, "type", None), "value", getattr(_step, "type", "")))
        if _stype.lower() not in ("done", ""):
            scaffold_phases.append({
                "description": str(getattr(_step, "description", "")),
                "type": _stype,
            })

    _plan_real_steps = max(len(scaffold_phases), 1)
    belief.set_plan_horizon(_plan_real_steps)

    action_ranker = ActionRanker()
    action_ranker.set_plan_horizon(_plan_real_steps)

    # Restore execution log from checkpoint
    execution_log: Dict[int, Dict[str, Any]] = {}
    if prior_execution_log and isinstance(prior_execution_log, dict):
        try:
            execution_log = {
                int(k): v for k, v in prior_execution_log.items()
                if isinstance(v, dict)
            }
            print(
                f"[OPERATE] Restored execution_log from checkpoint with "
                f"{len(execution_log)} iteration(s).",
                file=sys.stderr,
            )
        except Exception as _el_err:
            print(f"[OPERATE] execution_log restore failed ({_el_err}); starting fresh.", file=sys.stderr)
            execution_log = {}

    # Visited action keys — persisted across replans
    _visited_action_keys: dict = {}
    if prior_belief_state is not None:
        try:
            _persisted_visited = prior_belief_state.get("_visited_action_keys", [])
            if isinstance(_persisted_visited, list):
                _visited_action_keys = {k: True for k in _persisted_visited if isinstance(k, str)}
        except Exception:
            _visited_action_keys = {}
    # AUDIT-MEDIUM FIX: Cap raised from 200 → 1000
    _VISITED_ACTION_MAX = 1000
    _PERMANENT_DENY_ACTION_KEYS: set = set()

    # ------------------------------------------------------------------
    # Initialize PerStepReasoner for the GII reasoning loop.
    # GIIController has its own internal PSR; fall back to standalone.
    # ------------------------------------------------------------------
    _per_step_reasoner = None
    if gii_controller is not None and gii_controller.enabled:
        _per_step_reasoner = getattr(gii_controller, "_per_step_reasoner", None)

    if _per_step_reasoner is None:
        try:
            from core.cognition.per_step_reasoner import PerStepReasoner as _PSR
            _llm_for_psr = None
            if gii_controller is not None:
                _llm_for_psr = getattr(gii_controller, "_llm", None)
            if _llm_for_psr is None and reasoning_engine is not None:
                _llm_for_psr = getattr(reasoning_engine, "_llm_callable", None)
            if callable(_llm_for_psr):
                _cr_for_psr = (
                    gii_controller.consequence_reasoner
                    if gii_controller is not None and hasattr(gii_controller, "consequence_reasoner")
                    else None
                )
                _per_step_reasoner = _PSR(
                    llm_callable=_llm_for_psr,
                    objective=terminal_prompt,
                    scaffold_steps=scaffold_phases,
                    consequence_reasoner=_cr_for_psr,
                )
                print(
                    "[OPERATE] GII loop: PerStepReasoner initialised as primary action source.",
                    file=sys.stderr,
                )
        except Exception as _psr_err:
            log_warn(f"[LOOP] PerStepReasoner init failed: {_psr_err}. Will use GII/fallback.")

    # ------------------------------------------------------------------
    # Loop state
    # ------------------------------------------------------------------
    iteration: int = 0
    stagnant_iterations: int = 0
    goal_complete: bool = False

    # Max iterations based on scaffold complexity
    max_iterations: int = max(_plan_real_steps * (MAX_STAGNANT_ITERS_COMMAND + 1), 25)
    # In the GII loop, stagnation limit is fixed (no per-step-type variation)
    stagnant_limit: int = MAX_STAGNANT_ITERS_COMMAND

    previous_snapshot: Optional[dict] = None
    previous_perception = None

    try:
        while not goal_complete and iteration < max_iterations:

            # ----------------------------------------------------------------
            # Wall-clock timeout
            # ----------------------------------------------------------------
            if time.time() - start_ts > max_wallclock_seconds:
                journal.record({"event": "execution_timeout"})
                raise RuntimeError("TASK_FAILED:timeout")

            if watchdog is not None:
                watchdog.check()

            # Heartbeat
            input_arbitrator.soc_action_started()
            input_arbitrator.clear_emergency_reclaim()

            iteration += 1

            # ================================================================
            # STEP 1: OBSERVE — Capture current world state
            # ================================================================
            perception_snapshot = None
            if observer:
                try:
                    snap = observer.snapshot()
                    perception_snapshot = snap.get("perception")
                    if world_graph and isinstance(perception_snapshot, dict):
                        _iter_log = execution_log.get(iteration)
                        if _iter_log:
                            perception_snapshot = dict(perception_snapshot)
                            perception_snapshot["last_command_output"] = _iter_log.get(
                                "last_output", ""
                            )
                        world_graph.update(perception_snapshot)
                except Exception:
                    log_warn("Observer snapshot failed")

            try:
                world_snapshot = world_graph.snapshot() if world_graph else {}

                # Bayesian belief update from world delta
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
                            focused_app_b = bounded.get("focused_app")
                            entity_count_b = len(bounded.get("entities", []))
                            has_delta_b = bool(delta)
                            _entity_rich_thresh = _likelihood_cfg["ENTITY_RICH_THRESHOLD"]

                            if isinstance(focused_app_b, str) and focused_app_b.strip():
                                app_state_key = f"app:{focused_app_b.lower()}"
                                likelihoods[app_state_key] = (
                                    _likelihood_cfg["app_match_with_delta"] if has_delta_b
                                    else _likelihood_cfg["app_match_no_delta"]
                                )

                            if entity_count_b > _entity_rich_thresh:
                                likelihoods["ui_rich"] = _likelihood_cfg["ui_rich"]
                            elif entity_count_b > 0:
                                likelihoods["ui_sparse"] = _likelihood_cfg["ui_sparse"]
                            else:
                                likelihoods["ui_empty"] = _likelihood_cfg["ui_empty"]

                            likelihoods["neutral"] = (
                                _likelihood_cfg["neutral_with_delta"] if has_delta_b
                                else _likelihood_cfg["neutral_no_delta"]
                            )
                            belief.bayesian_update(likelihoods)
                    except Exception:
                        pass

                previous_snapshot = world_snapshot

                # ============================================================
                # STEP 2: REASON — Get next action from GII reasoning chain
                # ============================================================
                selected_action: Optional[Dict[str, Any]] = None
                action_source: str = "unknown"

                # Primary: GIIController.decide_next_action()
                if gii_controller is not None and gii_controller.enabled:
                    _ws_for_gii = world_snapshot if isinstance(world_snapshot, dict) else {}
                    _gii_action, _gii_reason = gii_controller.decide_next_action(
                        _ws_for_gii,
                        perception=perception_snapshot if isinstance(perception_snapshot, dict) else None,
                    )
                    if _gii_action is not None:
                        # D-12 FIX: Scan GII reasoning output for injection markers
                        _gii_text_to_scan = " ".join(
                            str(_gii_action.get(f, ""))
                            for f in ("thought", "command", "content", "path", "summary")
                        )
                        try:
                            from core.security.injection_markers import contains_injection_marker as _cim_gii
                            if _cim_gii(_gii_text_to_scan):
                                log_warn(
                                    "[D-12] GII action BLOCKED — injection marker in "
                                    f"VL model reasoning output: {_gii_text_to_scan[:120]!r}. "
                                    "Continuing to PerStepReasoner."
                                )
                                _gii_action = None
                        except ImportError:
                            _lower_gii = _gii_text_to_scan.lower()
                            if "ignore previous instructions" in _lower_gii or \
                               "ignore all previous" in _lower_gii:
                                log_warn("[D-12] GII action BLOCKED — injection marker (inline check).")
                                _gii_action = None
                    if _gii_action is not None:
                        selected_action = _gii_action
                        action_source = "gii_controller"

                # Secondary: PerStepReasoner (when GII disabled or returned nothing)
                if selected_action is None and _per_step_reasoner is not None:
                    _ws_for_psr = world_snapshot if isinstance(world_snapshot, dict) else {}
                    _psr_action, _psr_reason = _per_step_reasoner.next_action(
                        _ws_for_psr,
                        perception=perception_snapshot if isinstance(perception_snapshot, dict) else None,
                    )
                    if _psr_action is not None:
                        selected_action = _psr_action
                        action_source = "per_step_reasoner"

                # Tertiary: Playbook warm-start + ReasoningEngine dynamic candidates
                if selected_action is None:
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
                                "iteration": iteration,
                                "count": len(_playbook_candidates),
                            })

                    _dynamic_candidates: List[Dict[str, Any]] = []
                    if reasoning_engine is not None:
                        try:
                            _dc = reasoning_engine.propose_actions(
                                objective=execution_plan.objective,
                                belief_summary=belief.summary(),
                                perception=perception_for_reasoning,
                                k=MAX_DYNAMIC_CANDIDATES,
                            )
                            if _dc:
                                # Injection marker scan on dynamic candidates
                                try:
                                    from core.security.injection_markers import contains_injection_marker as _cim
                                    _safe_dc: list = []
                                    for _dc_item in _dc:
                                        _dc_cmd = str(_dc_item.get("command", "")) + " " + str(_dc_item.get("content", ""))
                                        if _cim(_dc_cmd):
                                            log_warn(
                                                f"[CRIT-6] Dynamic candidate BLOCKED — injection marker "
                                                f"detected: {_dc_cmd[:80]!r}. Candidate dropped."
                                            )
                                        else:
                                            _safe_dc.append(_dc_item)
                                    _dc = _safe_dc
                                except ImportError:
                                    pass
                                fresh_dc = [
                                    c for c in _dc
                                    if action_ranker.action_key(c) not in _visited_action_keys
                                ]
                                _dynamic_candidates = fresh_dc or _dc
                                if _dynamic_candidates:
                                    journal.record({
                                        "event": "dynamic_candidates_used",
                                        "iteration": iteration,
                                        "count": len(_dynamic_candidates),
                                    })
                        except Exception as re_err:
                            log_warn(f"ReasoningEngine fallback failed: {re_err}")

                    # Merge: playbook first (warm-start), then dynamic
                    candidate_actions = _playbook_candidates + _dynamic_candidates

                    # H-03 FIX: Injection marker scan on ALL candidate actions
                    if candidate_actions:
                        try:
                            from core.security.injection_markers import contains_injection_marker as _cim_plan
                            _safe_plan: list = []
                            for _pa in candidate_actions:
                                _pa_text = " ".join(
                                    str(_pa.get(f, ""))
                                    for f in ("command", "content", "path")
                                )
                                if _cim_plan(_pa_text):
                                    log_warn(
                                        f"[H-03] Candidate action BLOCKED — injection marker "
                                        f"detected: {_pa_text[:80]!r}. Step dropped."
                                    )
                                else:
                                    _safe_plan.append(_pa)
                            candidate_actions = _safe_plan
                        except ImportError:
                            pass

                    if candidate_actions:
                        selected_action = action_ranker.select(
                            actions=candidate_actions,
                            belief_state=belief,
                        )
                        action_source = "reasoning_fallback"

                if selected_action is None:
                    raise RuntimeError("TASK_FAILED:no_candidate_actions")

                action_key = action_ranker.action_key(selected_action)

                # Record as visited (evict oldest when at cap)
                if len(_visited_action_keys) >= _VISITED_ACTION_MAX:
                    _oldest = next(iter(_visited_action_keys))
                    del _visited_action_keys[_oldest]
                _visited_action_keys[action_key] = True

                # Semantic loop detection (A→B→A cycle)
                if hasattr(belief, "record_action_key_for_loop_detection"):
                    belief.record_action_key_for_loop_detection(action_key)
                if (
                    hasattr(belief, "detect_semantic_loop")
                    and belief.detect_semantic_loop()
                ):
                    journal.record({
                        "event": "semantic_loop_detected",
                        "iteration": iteration,
                        "action_key": action_key,
                    })
                    log_warn(
                        f"[OPERATE] Semantic loop detected (A→B→A cycle) at "
                        f"iteration {iteration}. Triggering replan."
                    )
                    raise RuntimeError(REPLAN_SIGNAL)

                # ============================================================
                # STEP 3: POLICY — Validate against PolicyEngine
                # ============================================================
                _policy_decision: str = PolicyEngine.DENY
                _policy_reason: str = "not_evaluated"

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
                    if gii_controller is not None:
                        gii_controller.record_denial(action_key)
                    journal.record({
                        "event": "policy_deny",
                        "iteration": iteration,
                        "action_key": action_key,
                        "reason": _policy_reason,
                    })
                    stagnant_iterations += 1
                    if stagnant_iterations >= stagnant_limit:
                        raise RuntimeError(REPLAN_SIGNAL)
                    previous_perception = perception_snapshot
                    continue

                # ============================================================
                # STEP 4: SAFETY — ConsequenceReasoner for ALL executable ops
                # AUDIT-HIGH-2 FIX: Runs for command/file_create/install
                # regardless of GII mode. Safety and GII mode are independent.
                # ============================================================
                _op_for_cr = str(selected_action.get("operation", "")).lower()
                _consequence_reasoner_instance = None
                if gii_controller is not None and hasattr(gii_controller, "consequence_reasoner"):
                    _consequence_reasoner_instance = gii_controller.consequence_reasoner

                if (
                    _consequence_reasoner_instance is not None
                    and _policy_decision != PolicyEngine.DENY
                    and _op_for_cr in ("command", "file_create", "install")
                ):
                    try:
                        _cr_result = _consequence_reasoner_instance.evaluate(
                            action=selected_action,
                            objective=terminal_prompt,
                            step_description=str(selected_action.get("thought", "")),
                        )
                        from core.safety.consequence_reasoner import SafetyDecision as _SafetyDecision
                        if _cr_result.decision == _SafetyDecision.DENY:
                            belief.record_action(action_key, -0.8)
                            if gii_controller is not None:
                                gii_controller.record_denial(action_key)
                            journal.record({
                                "event": "consequence_reasoner_deny",
                                "iteration": iteration,
                                "action_key": action_key,
                                "reason": _cr_result.reason,
                                "tier_reached": _cr_result.tier_reached,
                            })
                            stagnant_iterations += 1
                            if stagnant_iterations >= stagnant_limit:
                                raise RuntimeError(REPLAN_SIGNAL)
                            previous_perception = perception_snapshot
                            continue
                        elif _cr_result.decision == _SafetyDecision.REQUIRE_HUMAN_CONFIRMATION:
                            if _policy_decision != PolicyEngine.REQUIRE_HUMAN_CONFIRMATION:
                                _policy_decision = PolicyEngine.REQUIRE_HUMAN_CONFIRMATION
                                _policy_reason = (
                                    f"ConsequenceReasoner (Tier{_cr_result.tier_reached}): "
                                    f"{_cr_result.reason}"
                                )
                                journal.record({
                                    "event": "consequence_reasoner_require_confirmation",
                                    "iteration": iteration,
                                    "action_key": action_key,
                                    "reason": _policy_reason,
                                })
                    except Exception as _cr_exc:
                        import logging as _cr_log
                        _cr_log.getLogger(__name__).warning(
                            "[operate] ConsequenceReasoner error (fail-closed for IRREVERSIBLE): %s",
                            _cr_exc,
                        )

            except (AuthorityAbortError, RuntimeError):
                raise
            except Exception as _iter_exc:
                log_warn(
                    f"[operate] Transient per-iteration failure (iter {iteration}): "
                    f"{type(_iter_exc).__name__}: {_iter_exc}. "
                    "Converting to stagnation increment."
                )
                journal.record({
                    "event": "per_iteration_transient_failure",
                    "iteration": iteration,
                    "error_type": type(_iter_exc).__name__,
                    "error": str(_iter_exc),
                })
                stagnant_iterations += 1
                if stagnant_iterations >= stagnant_limit:
                    raise RuntimeError(REPLAN_SIGNAL)
                previous_perception = perception_snapshot
                continue

            # ================================================================
            # Human confirmation gate (AUDIT-CRIT-3: create-to-approve)
            # ================================================================
            if _policy_decision == PolicyEngine.REQUIRE_HUMAN_CONFIRMATION:
                journal.record({
                    "event": "policy_human_confirmation_required",
                    "iteration": iteration,
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
                    f"  Approve: CREATE file → {_signal_path}.APPROVE",
                    file=sys.stderr,
                )

                try:
                    import subprocess as _sp
                    _notif_body = (
                        f"Action: {selected_action.get('operation')}\n"
                        f"Reason: {_policy_reason}\n"
                        f"Create to approve: {_signal_path}.APPROVE"
                    )
                    if _sp.run(["which", "notify-send"], capture_output=True).returncode == 0:
                        _sp.run(
                            ["notify-send", "--urgency=critical", "--expire-time=30000",
                             "ProjectZeo: Human Approval Required", _notif_body],
                            timeout=5, capture_output=True,
                        )
                    elif _sp.run(["which", "osascript"], capture_output=True).returncode == 0:
                        _sp.run(
                            ["osascript", "-e",
                             f'display notification "{_notif_body}" with title "ProjectZeo Approval"'],
                            timeout=5, capture_output=True,
                        )
                except Exception:
                    pass

                # AUDIT-CRITICAL-4 FIX: Create-to-approve (fail-closed)
                _approve_path = _signal_path + ".APPROVE"
                _phc_wait = 0
                _phc_approved = False
                try:
                    while _phc_wait < MAX_WAIT_RETRIES:
                        time.sleep(WAIT_RETRY_SECONDS)
                        _phc_wait += 1
                        try:
                            _approve_present = os.path.exists(_approve_path)
                        except OSError:
                            continue
                        if _approve_present:
                            try:
                                os.remove(_approve_path)
                            except OSError:
                                pass
                            _phc_approved = True
                            break
                finally:
                    _remove_approval_signal(_signal_path)
                    try:
                        os.remove(_approve_path)
                    except OSError:
                        pass

                if not _phc_approved:
                    journal.record({
                        "event": "policy_human_confirmation_denied",
                        "iteration": iteration,
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
                        "iteration": iteration,
                        "action_key": action_key,
                    })

            # ================================================================
            # Authority evaluation
            # ================================================================
            is_high_risk = selected_action.get("operation") in {"command", "install", "file_create"}
            if is_high_risk:
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

            # ================================================================
            # STEP 5: EXECUTE — Dispatch action
            # ================================================================
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
                    execution_log=execution_log,
                    iteration=iteration,
                    task_ui_executor=task_ui_executor,
                    watchdog=watchdog,
                    focused_app=(
                        world_snapshot.get("focused_app", "")
                        if isinstance(world_snapshot, dict) else ""
                    ),
                    prefer_playwright=True,
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
                    "iteration": iteration,
                    "action_key": action_key,
                    "error": str(exec_exc),
                })

            # Store scrubbed command output
            if "output" in exec_result and exec_result.get("output"):
                _raw_output = str(exec_result.get("output", ""))[:MAX_COMMAND_OUTPUT_BYTES]
                output_text = _scrub_credentials(_raw_output)
                _step_entry = execution_log.setdefault(iteration, {"outputs": []})
                _step_outputs = _step_entry.get("outputs", [])
                if len(_step_outputs) < 5:
                    _step_outputs.append({
                        "success": action_success,
                        "output": output_text,
                        "iteration": iteration,
                    })
                    _step_entry["outputs"] = _step_outputs
                    _step_entry["last_output"] = output_text
                    execution_log[iteration] = _step_entry

            # Record outcome to GIIController and PerStepReasoner for history
            if gii_controller is not None and gii_controller.enabled:
                try:
                    gii_controller.record_outcome(
                        selected_action,
                        success=action_success,
                        output=str(exec_result.get("output", ""))[:500],
                    )
                except Exception:
                    pass

            if _per_step_reasoner is not None:
                try:
                    _per_step_reasoner.record_outcome(
                        selected_action,
                        success=action_success,
                        output=str(exec_result.get("output", ""))[:500],
                    )
                except Exception:
                    pass

            # Bandit update
            belief.record_action(action_key, raw_reward)
            best_reward = min(belief.global_best_reward() or 0.0, 0.9)
            belief.update_regret(action_key, raw_reward, best_reward)

            # SEC-4 FIX: Scrub write/type content before journaling action
            _journal_op = str(selected_action.get("operation", "")).lower()

            journal.record({
                "event": "action_executed",
                "iteration": iteration,
                "action_key": action_key,
                "operation": _journal_op,
                "source": action_source,
                "success": action_success,
                "reward": raw_reward,
            })

            # ================================================================
            # STEP 6: Goal completion detection
            # ================================================================
            if _journal_op == "done":
                journal.record({"event": "execution_complete", "iteration": iteration})
                goal_complete = True
                break

            # ================================================================
            # Step verification and stagnation tracking
            # ================================================================
            if action_success:
                stagnant_iterations = 0
                progress.advance_step()

                # Save checkpoint
                try:
                    from core.safety.checkpoint_store import save_checkpoint as _save_cp
                    _save_cp({
                        "intent": terminal_prompt,
                        "iteration": iteration,
                        "belief_state": belief.to_dict(),
                        "execution_log": {str(k): v for k, v in execution_log.items()},
                    })
                except Exception as _cp_err:
                    log_warn(f"[CHECKPOINT] save_checkpoint failed: {_cp_err}")
            else:
                stagnant_iterations += 1

            # Stagnation guard
            if stagnant_iterations >= stagnant_limit:
                _entropy = belief.entropy() if hasattr(belief, "entropy") else 0.0
                journal.record({
                    "event": "replan_trigger",
                    "replan_signal": REPLAN_SIGNAL,
                    "iteration": iteration,
                    "stagnant_iterations": stagnant_iterations,
                    "stagnant_limit": stagnant_limit,
                    "belief_entropy": round(_entropy, 4),
                })
                raise RuntimeError(REPLAN_SIGNAL)

            previous_perception = perception_snapshot

        # Loop exhausted without "done" action
        if not goal_complete:
            raise RuntimeError("TASK_FAILED:max_iterations_exceeded")

    except (RuntimeError, AuthorityAbortError):
        raise

    finally:
        # Persist final BeliefState on ALL exit paths
        if belief_state_out is not None:
            belief_state_out.clear()
            try:
                _bs_dict = belief.to_dict()
                _bs_dict["_visited_action_keys"] = list(_visited_action_keys.keys())
                if created_files_ledger:
                    _bs_dict["_created_files_ledger"] = list(created_files_ledger)
                belief_state_out.append(_bs_dict)
            except Exception:
                pass


# =========================================================================
# ACTION DISPATCHER
# =========================================================================

def _execute_decision(
    *,
    action: dict,
    os_backend: "OperatingSystem",
    installer,
    execution_log: dict,
    iteration: int,
    task_ui_executor,
    watchdog,
    focused_app: str = "",
    prefer_playwright: bool = True,
) -> dict:
    """
    Dispatch a single action to the appropriate OS backend.

    AUDIT-CRIT-4: The 'current_step' parameter has been removed.
    The dispatcher no longer needs step context — safety gates in the
    reasoning loop have already validated the action before dispatch.
    """
    from core.safety.action_timeout import run_with_timeout, ActionTimeout

    op = (action.get("operation") or "").lower().strip()

    try:
        from pyautogui import FailSafeException as _FailSafeException
    except ImportError:
        _FailSafeException = None  # type: ignore[assignment,misc]

    # DONE sentinel
    if op == "done":
        return {"success": True, "reward": 1.0}

    if not op:
        return {"success": False, "reward": -0.5, "reason": "empty operation field"}

    # Dispatch-time DANGEROUS_PATTERNS check (belt-and-suspenders)
    if op in ("command", "install", "file_create", "verify"):
        _dangerous_text = ""
        if op == "command":
            _dangerous_text = str(action.get("command", ""))
        elif op == "file_create":
            _dangerous_text = str(action.get("path", "")) + " " + str(action.get("content", ""))
        elif op == "install":
            _tool = action.get("tool", {})
            if isinstance(_tool, dict):
                _dangerous_text = " ".join(str(c) for c in _tool.get("install_commands", []))
        elif op == "verify":
            _dangerous_text = str(action.get("command", ""))

        if _dangerous_text.strip():
            try:
                from core.planner.execution_planner import ExecutionPlanner as _EP
                from core.security.injection_markers import normalize_for_injection_check as _norm_dp
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
                            f"for op={op!r}: pattern={_pat.pattern!r}. Blocking execution."
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
        # Browser routing via Playwright (when available and applicable)
        _BROWSER_OPS = {"click", "write", "type", "fill", "scroll", "navigate", "goto"}
        if op in _BROWSER_OPS and prefer_playwright and focused_app:
            try:
                from operate.utils.browser_backend import (
                    get_browser_backend,
                    is_browser_app,
                )
                if is_browser_app(focused_app):
                    _bb = get_browser_backend()
                    if _bb is not None:
                        _br_result = _bb.execute_action(action)
                        if _br_result.get("success"):
                            return _br_result
                        log_warn(
                            f"[GAP-1] Playwright failed for op={op!r} on "
                            f"focused_app={focused_app!r}: "
                            f"{_br_result.get('reason')}. "
                            "Falling back to pyautogui."
                        )
            except ImportError:
                pass
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
        # SEC-4 FIX: content is never stored in journal (action_key is a hash).
        # _scrub_write_type_content() provides defence-in-depth for any path
        # that would log the raw action dict.
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
        # SCROLL
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
                _ak = _secrets_install.token_hex(16)
                _sig_path = _write_approval_signal(
                    _ak, action,
                    reason=f"Install/sudo requires confirmation: {_install_preview[:120]}",
                )
                _approve_path_install = _sig_path + ".APPROVE"
                print(
                    f"[OPERATE] Install requires human approval. "
                    f"Commands: {_install_preview!r}. "
                    f"APPROVE: create file {_approve_path_install}  |  "
                    f"Timeout: {_CONFIRM_TIMEOUT_SECONDS}s → auto-denied.",
                    file=sys.stderr,
                )
                _waited = 0.0
                _approved = False
                while _waited < _CONFIRM_TIMEOUT_SECONDS:
                    time.sleep(WAIT_RETRY_SECONDS)
                    _waited += WAIT_RETRY_SECONDS
                    try:
                        _approve_present = os.path.exists(_approve_path_install)
                    except OSError:
                        continue
                    if _approve_present:
                        try:
                            os.remove(_approve_path_install)
                        except OSError:
                            pass
                        _approved = True
                        break
                if not _approved:
                    _remove_approval_signal(_sig_path)
                    try:
                        os.remove(_approve_path_install)
                    except OSError:
                        pass
                    return {
                        "success": False,
                        "reward": -1.0,
                        "reason": (
                            f"Install confirmation timed out after "
                            f"{_CONFIRM_TIMEOUT_SECONDS}s. Commands were: {_install_preview!r}."
                        ),
                    }

                from config.timeouts import INSTALL_COMMAND_TIMEOUT_SECONDS
                all_ok = True
                combined_output = ""
                for cmd in install_cmds:
                    r = os_backend.exec(cmd, timeout=int(INSTALL_COMMAND_TIMEOUT_SECONDS))
                    combined_output += (r.stdout or "") + (r.stderr or "")
                    if r.returncode != 0:
                        all_ok = False
                        break
                return {
                    "success": all_ok,
                    "reward": 0.8 if all_ok else -0.5,
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
                        "install: no install_commands in tool spec and installer unavailable."
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
                        "which ", "command -v ", "test -f ", "test -d ", "test -e ", "test -x ",
                        "stat ", "ls ", "echo ", "cat /proc/version",
                        "python --version", "python3 --version",
                        "python -c \"import ", "python3 -c \"import ",
                        "node --version", "node -v", "npm --version", "npm -v",
                        "git --version", "git -v", "java -version", "java --version",
                        "go version", "rustc --version", "cargo --version",
                        "docker --version", "pip --version", "pip3 --version",
                        "pip show ", "dpkg -l ", "rpm -q ", "brew list ", "type ",
                    })
                    cmd_lower = cmd.lower().lstrip()
                    _verify_allowed = any(cmd_lower.startswith(prefix) for prefix in _VERIFY_SAFE_PREFIXES)
                    if not _verify_allowed:
                        log_warn(
                            f"[H-01] verify command BLOCKED — not in safe read-only "
                            f"allowlist: {cmd[:120]!r}."
                        )
                        return {
                            "success": False,
                            "reward": -1.0,
                            "reason": (
                                f"verify command blocked: {cmd[:80]!r} is not in the "
                                "safe verify-command allowlist."
                            ),
                        }
                    r = os_backend.exec(cmd, timeout=30)
                    return {
                        "success": r.returncode == 0,
                        "reward": 0.6 if r.returncode == 0 else -0.3,
                        "output": (r.stdout or "") + (r.stderr or ""),
                        "returncode": r.returncode,
                    }
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
                "cursor reached a screen corner."
            )
            return {
                "success": False,
                "reward": -1.0,
                "reason": (
                    "pyautogui_failsafe: cursor reached a screen corner — pyautogui "
                    "FAILSAFE triggered. Use coordinates away from screen edges."
                ),
            }

        log_warn(f"_execute_decision: unexpected error [{op}] — {exc}")
        return {
            "success": False,
            "reward": -0.5,
            "reason": f"unexpected_error: {exc}",
        }
