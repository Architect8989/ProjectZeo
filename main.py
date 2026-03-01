from __future__ import annotations

import time
import os
import signal
import atexit
import sys
import threading
from typing import Callable, Optional

from core.mode_controller import (
    ModeController,
    SystemMode,
    ArmedTimeoutError,
    ModeTransitionError,
    PlanningTimeoutError,
)
from core.mode_controller import VisionUnavailableError, ObserverUnavailableError
from core.intent_listener import IntentListener
from core.environment_fingerprint import collect_environment_fingerprint

from observer.observer_core import ObserverCore, ObserverBlindnessError
from observer.observer_loop import ObserverLoop

from core.vision.vision_runtime import VisionRuntime
from core.vision.world_graph import WorldGraph

from state.serializer import AuthorityStateSerializer
from operate.utils.operating_system import OperatingSystem
from operate.operate import operate_main

from restoration.snapshot_provider import SnapshotProvider, SnapshotProviderError
from restoration.restore_provider import RestoreProvider
from restoration.restore_verifier import RestoreVerifier, RestorationVerificationError

from core.planner.execution_planner import ExecutionPlanner
from core.safety.runtime_watchdog import RuntimeWatchdog, WatchdogViolation

from adapters.apis_safety_layer import uninstall_patches
from core.cognition.belief_state import BeliefState


# ------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------

# PATCH §1.5: reduced from 2.0s to 0.25s for lower intent-to-task latency
HEARTBEAT_INTERVAL = 0.25

MAX_TASK_SECONDS = int(
    os.environ.get("PROJECTZEO_MAX_TASK_SECONDS", str(90 * 60))
)   # Default: 90 minutes.  Set to 0 for unlimited (use with caution).
    # BUG-13 FIX: Hard-coded 90-minute wall killed multi-hour tasks silently.
    # Override: PROJECTZEO_MAX_TASK_SECONDS=<seconds>  (0 = unlimited)

MAX_REPLANS = int(
    os.environ.get("PROJECTZEO_MAX_REPLANS", "3")
)   # Default: 3 replan attempts before TASK_FAILED.
    # BUG-14 FIX: Hard-coded 3 was too low for iterative/debugging tasks.
    # Override: PROJECTZEO_MAX_REPLANS=<count>  (0 = unlimited, not recommended)

# AUDIT §2.6 FIX: Absolute replan ceiling that fires regardless of MAX_REPLANS.
# When PROJECTZEO_MAX_REPLANS=0 (unlimited), there is no guard in the outer
# main() loop — a task whose inner loop always hits max_iterations and raises
# REPLAN_REQUIRED will replan forever, consuming unbounded LLM inference budget.
# This ceiling is intentionally NOT configurable via env var; it is a hard safety
# backstop that should never be needed in correct operation.
ABSOLUTE_REPLAN_CEILING: int = 50

# BUG-2 FIX: WARMUP_TIMEOUT_SECONDS raised from 150s to 300s (env-configurable).
# Root cause: CPU-only inference of Qwen2.5-VL 7B takes 60-90s per call.
# 3 consecutive frames = 3×90s = 270s > 150s old limit. EVERY CPU deployment
# hit timeout and proceeded blind (empty world model at task start).
# Fix: default 300s covers worst-case CPU warmup. GPU users can lower it.
# Override: PROJECTZEO_WARMUP_TIMEOUT_SECONDS=<seconds>
WARMUP_TIMEOUT_SECONDS: float = float(
    os.environ.get("PROJECTZEO_WARMUP_TIMEOUT_SECONDS", "300")
)
WARMUP_STABLE_FRAMES: int = int(
    os.environ.get("PROJECTZEO_WARMUP_STABLE_FRAMES", "3")
)

# Hard timeout for the restoration phase (thread-enforced)
RESTORE_TIMEOUT_SECONDS = 60.0

# Max retries for transient vision/observer health failures at task start
HEALTH_RETRY_MAX = 5
HEALTH_RETRY_INTERVAL = 1.0

# Vision restart grace period after ObserverBlindnessError
VISION_RESTART_GRACE_SECONDS = 5.0


# ------------------------------------------------------------------
# MODULE-LEVEL STATE
# ------------------------------------------------------------------

_TASK_START: Optional[float] = None
_TASK_LOCK = threading.Lock()
_SHUTDOWN_EVENT = threading.Event()


# ------------------------------------------------------------------
# THREAD-SAFE TASK TIMING
# ------------------------------------------------------------------

def _set_task_start(ts: float) -> None:
    global _TASK_START
    with _TASK_LOCK:
        _TASK_START = ts


def _clear_task_start() -> None:
    global _TASK_START
    with _TASK_LOCK:
        _TASK_START = None


def _get_task_start() -> Optional[float]:
    with _TASK_LOCK:
        return _TASK_START


# ------------------------------------------------------------------
# BELIEF STATE HELPER (FIX RB-6)
# ------------------------------------------------------------------

def _safe_belief_snapshot(belief_state_out: list) -> dict:
    """
    FIX RB-6: Safe accessor for belief_state_out[0].

    Returns the serialised BeliefState dict when available, or an empty dict
    when the list is empty or the access raises.  This prevents IndexError from
    masking the original exception in the restoration finally block.

    Root cause: if operate_main() raises during init (before the finally block
    in _execute_autonomous_loop appends to belief_state_out), then
    belief_state_out is empty.  Accessing [0] directly raises IndexError and
    replaces the original exception in the traceback.
    """
    try:
        return belief_state_out[0] if belief_state_out else {}
    except Exception:
        return {}


# ------------------------------------------------------------------
# SAFE SHUTDOWN
# ------------------------------------------------------------------

def _force_safe_shutdown(os_backend, auth_state, reason: str) -> None:
    """
    Emergency failsafe: release all automated input and mark auth state safe.
    Best-effort — never raises.
    """
    try:
        os_backend.force_release_all(reason=reason)
    except Exception:
        pass
    try:
        auth_state.force_safe_state()
    except Exception:
        pass
    print(f"[SAFE-SHUTDOWN] {reason}", file=sys.stderr)


def _signal_handler(signum, frame) -> None:
    _SHUTDOWN_EVENT.set()


def _install_signal_handlers() -> None:
    """
    Register SIGINT, SIGTERM, and SIGQUIT (Unix) to set the shutdown event.

    Previously only SIGINT was handled; SIGTERM from systemd/Kubernetes would
    kill the process immediately without triggering the atexit sequence or
    running _force_safe_shutdown().
    """
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    if hasattr(signal, "SIGQUIT"):
        signal.signal(signal.SIGQUIT, _signal_handler)


# PATCH CRIT-1: _INTERACTIVE_MODE was a module-level boolean frozen at import
# time.  run.py sets PROJECTZEO_INTERACTIVE AFTER importing main, so the flag
# was always False regardless of the --interactive CLI flag.  Fix: read the env
# var live inside _interactive_print() on every call — zero overhead, correct.
def _interactive_print(msg: str) -> None:
    """Print msg to stdout when PROJECTZEO_INTERACTIVE=1; no-op otherwise."""
    if os.environ.get("PROJECTZEO_INTERACTIVE", "").strip() == "1":
        print(msg, flush=True)


# GAP-5 FIX: Write task result to temp/task_result.json after each task so:
#   1. The operator can see the outcome without reading JSONL logs.
#   2. --status flag in run.py can surface the last result.
#   3. External monitoring scripts can poll this file.
def _write_task_result(
    *,
    intent: str,
    success: bool,
    error: Optional[str] = None,
    steps_completed: Optional[int] = None,
) -> None:
    """GAP-5 FIX: Persist task outcome to temp/task_result.json."""
    import json as _json
    import datetime as _dt

    _root = os.path.dirname(os.path.abspath(__file__))
    _temp = os.path.join(_root, "temp")
    _path = os.path.join(_temp, "task_result.json")
    _tmp = _path + ".tmp"

    payload = {
        "success": success,
        "intent": intent,
        "completed_at": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "error": error,
        "steps_completed": steps_completed,
    }

    try:
        os.makedirs(_temp, exist_ok=True)
        with open(_tmp, "w", encoding="utf-8") as f:
            _json.dump(payload, f, indent=2)
        os.replace(_tmp, _path)
    except Exception as _e:
        print(f"[MAIN] GAP-5: Could not write task_result.json: {_e}", file=sys.stderr)


# ------------------------------------------------------------------
# STARTUP VALIDATION
# ------------------------------------------------------------------

# PATCH CRIT-2: _validate_runtime_dependencies() removed from main.py.
# run.py runs a consolidated, authoritative dependency check (with pyautogui
# display access, pyyaml, playwright, EasyOCR, Wayland tooling) before calling
# main().  A second check here produced duplicate stderr output and used
# different severity levels for the same conditions, causing confusing
# discrepancies on Wayland sessions.


# ------------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------------

def _ingest_latest_perception(observer, world_graph) -> bool:
    """Pull the latest observer snapshot into the world graph. Returns True on success."""
    snap = observer.snapshot()
    if not isinstance(snap, dict):
        return False
    if not snap.get("perception_available"):
        return False
    perception = snap.get("perception")
    if not isinstance(perception, dict):
        return False
    world_graph.ingest(perception)
    return True


def _enforce_task_timeout() -> None:
    """Raise RuntimeError if the current task has exceeded MAX_TASK_SECONDS."""
    start = _get_task_start()
    if start is None:
        return
    # BUG-13 FIX: MAX_TASK_SECONDS == 0 means unlimited (no timeout enforced).
    if MAX_TASK_SECONDS > 0 and (time.time() - start) > MAX_TASK_SECONDS:
        raise RuntimeError("TASK_FAILED:timeout")


def _safe_begin_restoration(mode: ModeController) -> bool:
    """
    Attempt to transition mode to RESTORING.  Returns True if restoration should
    proceed, False if it should be skipped (e.g. task never reached EXECUTING).

    PATCH §1.5: begin_restoration() only accepts EXECUTING mode.
    """
    current = mode.mode

    if current is SystemMode.EXECUTING:
        try:
            mode.begin_restoration()
            return True
        except Exception:
            return False

    if current is SystemMode.RESTORING:
        return True  # Already in RESTORING — proceed

    if current in (SystemMode.PLANNING, SystemMode.ARMED, SystemMode.OBSERVER):
        try:
            mode.force_observer()
        except Exception:
            pass
        return False  # Task never started executing — skip restoration

    return False


def _shutdown_executor(executor, wait: bool = False) -> None:
    """
    Python 3.8-compatible ThreadPoolExecutor shutdown.

    Python ≥3.9 supports cancel_futures=True; earlier versions do not.
    """
    if sys.version_info >= (3, 9):
        executor.shutdown(wait=wait, cancel_futures=True)
    else:
        executor.shutdown(wait=wait)


# ------------------------------------------------------------------
# MAIN ENTRY
# ------------------------------------------------------------------

def main(llm_callable: Callable, model_name: str) -> None:
    """
    Main entry point for the ProjectZeo autonomous execution kernel.

    Parameters
    ----------
    llm_callable : callable
        Vision LLM adapter callable (get_next_action).
    model_name : str
        Model identifier string (e.g. "qwen2.5-vl:7b-instruct").

    Raises
    ------
    RuntimeError
        If llm_callable is not callable, model_name is invalid, or a FATAL
        startup dependency is missing.
    """
    if not callable(llm_callable):
        raise RuntimeError("llm_callable must be a callable")

    if not isinstance(model_name, str) or not model_name.strip():
        raise RuntimeError("model_name must be a non-empty string")

    # PATCH CRIT-2: dependency validation is done once in run.py before main()
    # is ever called.  Removed duplicate check here — see run.py's
    # _validate_runtime_dependencies() for the authoritative pre-flight check.

    # ------------------------------------------------------------------
    # Core infrastructure
    # ------------------------------------------------------------------
    os_backend = OperatingSystem()
    state_path = os.path.join(os.getcwd(), ".authority_state.json")
    auth_state = AuthorityStateSerializer(state_path)

    observer = ObserverCore()
    vision_runtime = VisionRuntime(model_name=model_name)
    world_graph = WorldGraph()

    observer_loop = ObserverLoop(
        observer=observer,
        vision_runtime=vision_runtime,
        world_graph=world_graph,
    )

    mode = ModeController()
    mode.inject_llm_callable(llm_callable)

    _install_signal_handlers()
    atexit.register(lambda: _force_safe_shutdown(os_backend, auth_state, "atexit"))

    watchdog = RuntimeWatchdog()
    env_fingerprint = collect_environment_fingerprint()
    persisted = auth_state.load()

    # Crash recovery: restore BeliefState from persisted state if available
    _crash_recovery_belief_state: "dict | None" = persisted.get("belief_state_full")
    # BUG-5 FIX: Also load checkpoint step_index for crash recovery.
    # The original code restored BeliefState but always restarted current_step_index=0,
    # causing all completed steps to be replayed from scratch on crash recovery.
    # This wasted compute and could corrupt already-modified files / already-run commands.
    # Fix: load the step_index from checkpoint_store and pass it into operate_main()
    # so _execute_autonomous_loop can fast-forward to the last completed position.
    _crash_recovery_step_index: "int | None" = None
    try:
        from core.safety.checkpoint_store import get_checkpoint_step_index as _get_cp_idx
        _crash_recovery_step_index = _get_cp_idx()
        if _crash_recovery_step_index is not None:
            print(
                f"[MAIN] BUG-5: Crash recovery: restoring from checkpoint step_index="
                f"{_crash_recovery_step_index}. Steps 0–{_crash_recovery_step_index - 1} "
                "will be skipped (already completed before crash).",
                file=sys.stderr,
            )
    except Exception as _cp_load_err:
        print(
            f"[MAIN] BUG-5: Checkpoint step_index load failed: {_cp_load_err}. "
            "Starting from step 0 (full replay on crash recovery).",
            file=sys.stderr,
        )

    if persisted.get("dirty") or persisted.get("restore_required"):
        try:
            os_backend.force_release_all(reason="crash_recovery")
        except Exception:
            pass
        auth_state.force_safe_state()
        mode.force_observer()

    vision_runtime.start()
    observer_loop.start()

    # BUG-8 FIX: Register the live VisionRuntime with QwenOllamaAdapter so that
    # action-decision calls reuse the already-captured frame instead of taking a
    # second independent screenshot (which creates a race-condition stale-state).
    try:
        from adapters.qwen_ollama_adapter import set_shared_vision_runtime as _svr
        _svr(vision_runtime)
    except Exception as _svr_err:
        print(
            f"[MAIN] BUG-8: Could not register VisionRuntime with adapter: {_svr_err}. "
            "Falling back to independent capture (no frame sharing).",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # Warmup — extended to 150s for CPU inference compatibility
    # P4 FIX (RT-1): Two-phase warmup.
    # Phase 1 (REQUIRED): vision AND observer healthy — 3 consecutive healthy
    #   frames.  Does NOT require entity_count > 0.  A bare desktop (zero open
    #   windows) is a valid operating environment; blocking on entity_count > 0
    #   caused a 150-second penalty on every cold-start in minimal Xvfb sessions.
    # Phase 2 (OPTIONAL): entity-warm — at least one visible UI entity.
    #   Tracked separately; planning proceeds without it but the world model is
    #   noted as entity-empty for diagnostics.
    # ------------------------------------------------------------------
    stable_frames = 0
    entity_warm = False
    warmup_deadline = time.time() + WARMUP_TIMEOUT_SECONDS
    _warmup_achieved = False

    while time.time() < warmup_deadline:
        if observer.is_healthy() and vision_runtime.is_healthy():
            if _ingest_latest_perception(observer, world_graph):
                stable_frames += 1
                if world_graph.entity_count() > 0:
                    entity_warm = True
                if stable_frames >= WARMUP_STABLE_FRAMES:
                    _warmup_achieved = True
                    break
        else:
            stable_frames = 0
        time.sleep(0.1)

    if not _warmup_achieved:
        # Include observer/vision health state in the warning for diagnostics
        _obs_healthy = observer.is_healthy()
        _vis_healthy = vision_runtime.is_healthy()
        print(
            f"[WARMUP] WARNING: warmup did not reach {WARMUP_STABLE_FRAMES} stable frames "
            f"within {WARMUP_TIMEOUT_SECONDS:.0f}s. "
            f"observer_healthy={_obs_healthy}, vision_healthy={_vis_healthy}, "
            f"entity_count={world_graph.entity_count()}, stable_frames={stable_frames}. "
            "Proceeding with degraded world model. "
            "If running on a bare desktop, ensure at least one application window is "
            "visible before arming, or increase WARMUP_TIMEOUT_SECONDS.",
            file=sys.stderr,
        )
        try:
            auth_state.record_event({
                "event": "warmup_degraded",
                "stable_frames_achieved": stable_frames,
                "required": WARMUP_STABLE_FRAMES,
                "entity_count": world_graph.entity_count(),
                "entity_warm": entity_warm,
                "observer_healthy": _obs_healthy,
                "vision_healthy": _vis_healthy,
                "timeout_seconds": WARMUP_TIMEOUT_SECONDS,
            })
        except Exception:
            pass
    elif not entity_warm:
        # Vision/observer healthy but no UI entities seen — bare desktop.
        # Task will proceed but planner starts with an empty world model.
        print(
            "[WARMUP] INFO: warmup succeeded (observer+vision healthy) but no UI "
            "entities were detected (bare desktop). Planning proceeds with empty world "
            "model. Open application windows before submitting tasks for best results.",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # Restoration infrastructure
    # ------------------------------------------------------------------
    snapshot_provider = SnapshotProvider(
        observer=observer,
        os_backend=os_backend,
        mode_controller=mode,
    )
    restore_provider = RestoreProvider(
        os_backend=os_backend,
        mode_controller=mode,
        snapshot_provider=snapshot_provider,
        authority_state=auth_state,  # HIGH-7 FIX: enables duplicate-restore warning flag
    )
    # P1 FIX (RT-3 / SI-2 / HAR-07): Wire authority_state into RestoreVerifier.
    # Previously constructed without authority_state=auth_state, so
    # self._authority_state was always None inside RestoreVerifier, making the
    # _emit_verification_warning() → authority_state.verification_warning = True
    # path permanently dead code.  Restoration verification failures were never
    # recorded in the authority audit file, preventing post-mortem analysis.
    # Fix: pass auth_state so that HAR-07 structured audit events actually fire.
    restore_verifier = RestoreVerifier(
        os_backend=os_backend,
        mode_controller=mode,
        cursor_tolerance_px=5,
        authority_state=auth_state,
    )

    intent_listener = IntentListener(mode, snapshot_provider)
    intent_listener.start()

    _health_failures = 0
    _current_planner = None

    try:

        while not _SHUTDOWN_EVENT.is_set():

            try:
                _enforce_task_timeout()

                # Watchdog check
                try:
                    watchdog.check()
                except WatchdogViolation as wv:
                    print(f"[MAIN] WatchdogViolation: {wv}", file=sys.stderr)
                    _force_safe_shutdown(os_backend, auth_state, f"watchdog:{wv}")
                    break

                mode.update_observer_health(observer.is_healthy())
                mode.update_vision_status(vision_runtime.is_healthy())

                if mode.mode == SystemMode.PLANNING:
                    mode.check_planning_timeout()
                if mode.mode == SystemMode.ARMED:
                    mode.check_armed_timeout()

                if not mode.is_armed():
                    time.sleep(HEARTBEAT_INTERVAL)
                    continue

                # --------------------------------------------------------
                # Task start
                # --------------------------------------------------------
                _set_task_start(time.time())
                # Reset watchdog per-task so the 1-hour wall-clock limit applies
                # per task, not per process lifetime
                watchdog.start_time = time.time()
                replan_count = 0

                snapshot_id = mode.consume_snapshot()
                if not snapshot_id:
                    raise RuntimeError("Missing snapshot")

                _ingest_latest_perception(observer, world_graph)

                intent = mode.get_intent()
                if not intent or not intent.strip():
                    raise RuntimeError("Invalid intent")

                # Shut down previous planner's executor to avoid thread leaks
                if _current_planner is not None:
                    try:
                        _shutdown_executor(_current_planner._executor, wait=False)
                    except Exception:
                        pass

                planner = ExecutionPlanner(
                    llm_call=mode.get_llm_callable(),
                    environment_fingerprint=env_fingerprint,
                    world_graph=world_graph,
                )
                _current_planner = planner

                # Health gate before planning
                # BUG-3 FIX: Pause observer_loop BEFORE begin_planning() so the
                # background vision model (VisionRuntime) is not running concurrently
                # with the planning LLM calls. Previously the loop was only paused
                # during EXECUTING mode. During PLANNING mode on a VRAM-limited GPU:
                #   - observer_loop → VisionRuntime._call_model() every 0.5s
                #   - ExecutionPlanner._expand_goal() → client.chat() on same model
                # Both called the same Qwen2.5-VL model simultaneously → OOM or
                # mutual timeout cascades. Now we pause before any planning LLM call.
                try:
                    observer_loop.pause()
                except Exception:
                    pass

                for _health_attempt in range(HEALTH_RETRY_MAX):
                    try:
                        mode.begin_planning()
                        _interactive_print(f"[MODE] PLANNING — {intent[:72]}")
                        _health_failures = 0
                        break
                    except (VisionUnavailableError, ObserverUnavailableError) as health_err:
                        _health_failures += 1
                        if _health_failures >= HEALTH_RETRY_MAX:
                            raise
                        print(
                            f"[MAIN] Health check failed "
                            f"({_health_attempt + 1}/{HEALTH_RETRY_MAX}): "
                            f"{health_err} — retrying in {HEALTH_RETRY_INTERVAL}s",
                            file=sys.stderr,
                        )
                        time.sleep(HEALTH_RETRY_INTERVAL)
                        mode.update_observer_health(observer.is_healthy())
                        mode.update_vision_status(vision_runtime.is_healthy())

                # Planning (CPU-monitored by watchdog)
                # RT-C FIX: Wrap planner.create_plan() in a retry loop so that
                # transient LLM failures (OOM subprocess kill, network timeout,
                # malformed JSON after all internal retries) do not immediately
                # terminate the process. The planner's internal retry
                # (_call_llm_text, max_retries=2) handles per-call transient
                # failures; this outer loop handles persistent failures across
                # multiple planning attempts (up to _PLAN_MAX_RETRIES=3).
                # After exhausting retries, the exception propagates normally
                # and is caught by the outer except Exception → _force_safe_shutdown.
                _PLAN_MAX_RETRIES = 3
                _plan_attempt = 0
                while True:
                    _plan_attempt += 1
                    try:
                        watchdog.pause_cpu()
                        execution_plan = planner.create_plan(
                            objective=intent,
                            requirements={
                                "environment": env_fingerprint,
                                "tools": env_fingerprint.get("tools", []),
                            },
                            high_level_steps=[{"goal": intent}],
                        )
                        break  # success — exit retry loop
                    except Exception as _plan_err:
                        watchdog.resume_cpu()
                        if _plan_attempt >= _PLAN_MAX_RETRIES:
                            print(
                                f"[MAIN] Planning failed after {_plan_attempt} "
                                f"attempt(s): {_plan_err}. Propagating error.",
                                file=sys.stderr,
                            )
                            raise  # exhaust retries — let outer handler deal with it
                        _plan_delay = min(2.0 ** _plan_attempt, 30.0)  # exp backoff, cap 30s
                        print(
                            f"[MAIN] Planning attempt {_plan_attempt}/{_PLAN_MAX_RETRIES} "
                            f"failed: {type(_plan_err).__name__}: {_plan_err}. "
                            f"Retrying in {_plan_delay:.1f}s...",
                            file=sys.stderr,
                        )
                        time.sleep(_plan_delay)
                    finally:
                        try:
                            watchdog.resume_cpu()
                        except Exception:
                            pass

                mode.attach_execution_plan(f"plan_{int(time.time())}")
                mode.mark_planning_complete()

                # BUG-3 FIX: Resume observer_loop briefly after planning so the
                # world graph gets at least one fresh frame before execution starts.
                # We immediately re-pause it when entering EXECUTING mode below.
                try:
                    observer_loop.resume()
                    time.sleep(0.5)  # allow one world graph update
                    observer_loop.pause()  # pause again for GPU during EXECUTING
                except Exception:
                    pass

                auth_state.persist(
                    execution_mode="EXECUTING",
                    automation_active=True,
                    restore_required=True,
                    last_snapshot_id=snapshot_id,
                    dirty=True,
                )

                mode.execute()
                _interactive_print(f"[MODE] EXECUTING — intent: {intent[:72]}")
                # EXECUTING so QwenOllamaAdapter gets the full GPU.
                # Observer uses cached world graph state during execution.
                try:
                    observer_loop.pause()
                except Exception:
                    pass

                _task_succeeded = False
                _belief_state_out: list = []
                _prior_belief_state: Optional[dict] = _crash_recovery_belief_state
                _crash_recovery_belief_state = None  # consume once
                # BUG-5 FIX: Use checkpoint step_index for crash recovery fast-forward.
                _prior_step_index: Optional[int] = _crash_recovery_step_index
                _crash_recovery_step_index = None  # consume once

                try:
                    while not _SHUTDOWN_EVENT.is_set():
                        _enforce_task_timeout()

                        try:
                            # BUG-1 FIX: Wire max_wallclock_seconds from MAX_TASK_SECONDS.
                            # Previously operate_main() always used its hardcoded default
                            # (90 minutes) regardless of PROJECTZEO_MAX_TASK_SECONDS.
                            # Two competing timeouts: the outer _enforce_task_timeout() read
                            # MAX_TASK_SECONDS but the inner wallclock used 90*60 always.
                            # Fix: pass MAX_TASK_SECONDS explicitly. When MAX_TASK_SECONDS==0
                            # (unlimited outer), use 90*60 as inner limit — still bounded
                            # to prevent runaway tasks consuming infinite LLM budget.
                            _inner_wallclock = (
                                MAX_TASK_SECONDS
                                if MAX_TASK_SECONDS > 0
                                else (90 * 60)
                            )
                            operate_main(
                                terminal_prompt=intent,
                                execution_plan=execution_plan,
                                planner=planner,
                                observer=observer,
                                world_graph=world_graph,
                                os_backend=os_backend,
                                max_wallclock_seconds=_inner_wallclock,
                                watchdog=watchdog,
                                prior_belief_state=_prior_belief_state,
                                belief_state_out=_belief_state_out,
                                prior_step_index=_prior_step_index,
                            )
                            _task_succeeded = True
                            _interactive_print(f"[MODE] TASK COMPLETED — {intent[:72]}")
                            _write_task_result(intent=intent, success=True)
                            break

                        except RuntimeError as e:
                            if str(e) != "REPLAN_REQUIRED":
                                raise

                            # FIX RB-6: Use _safe_belief_snapshot() — never access [0] directly
                            _prior_belief_state = _safe_belief_snapshot(_belief_state_out) or None

                            replan_count += 1
                            # BUG-14 FIX: MAX_REPLANS == 0 means unlimited replans.
                            if MAX_REPLANS > 0 and replan_count > MAX_REPLANS:
                                raise RuntimeError("TASK_FAILED:max_replans_exceeded")
                            # AUDIT §2.6 FIX: Hard safety backstop — fires even
                            # when MAX_REPLANS=0 (unlimited) to prevent infinite
                            # LLM consumption on permanently-stagnant tasks.
                            if replan_count > ABSOLUTE_REPLAN_CEILING:
                                raise RuntimeError(
                                    "TASK_FAILED:absolute_replan_ceiling_exceeded "
                                    f"({ABSOLUTE_REPLAN_CEILING} replans). "
                                    "Set PROJECTZEO_MAX_REPLANS to a lower value "
                                    "or investigate why the task cannot make progress."
                                )
                            mode.begin_replan_sequence()

                            try:
                                mode.force_observer()

                                try:
                                    new_snapshot_id = snapshot_provider.take_snapshot()
                                except SnapshotProviderError as snap_err:
                                    print(
                                        f"[MAIN] Replan snapshot failed (using prior): {snap_err}",
                                        file=sys.stderr,
                                    )
                                    new_snapshot_id = snapshot_id

                                # RB-MED-2 FIX: Detect and warn when the replan snapshot
                                # fallback produces a no-op restoration.
                                #
                                # Root cause: if take_snapshot() fails and new_snapshot_id
                                # falls back to the existing snapshot_id, then the later call
                                # to restore_provider.restore_snapshot(snapshot_id) in the
                                # finally block will find "already completed" in the ledger
                                # and return silently without restoring anything.  No
                                # exception is raised, no log is emitted — the no-op is
                                # invisible.
                                #
                                # Fix: detect the equality here, immediately after the
                                # fallback assignment, and emit a structured warning so
                                # operators know restoration will be skipped.
                                if new_snapshot_id == snapshot_id:
                                    print(
                                        "[MAIN] WARNING RESTORATION_SKIPPED: replan_snapshot_fallback — "
                                        f"new_snapshot_id == snapshot_id == {snapshot_id!r}. "
                                        "The replan snapshot failed; the prior snapshot ID will be "
                                        "used for restoration.  The ledger may treat this as "
                                        "'already completed' and perform no restoration.  "
                                        "Verify environment state manually after task completion.",
                                        file=sys.stderr,
                                    )

                                mode.attach_snapshot(new_snapshot_id)
                                mode.arm_for_replan(intent)

                                _ingest_latest_perception(observer, world_graph)
                                planner.update_world_snapshot(world_graph.snapshot())

                                mode.begin_planning()
                                env_fingerprint = collect_environment_fingerprint()
                                planner.refresh_environment(env_fingerprint)

                                try:
                                    watchdog.pause_cpu()
                                    execution_plan = planner.create_plan(
                                        objective=intent,
                                        requirements={
                                            "environment": env_fingerprint,
                                            "tools": env_fingerprint.get("tools", []),
                                        },
                                        high_level_steps=[{"goal": intent}],
                                    )
                                finally:
                                    watchdog.resume_cpu()

                                mode.attach_execution_plan(
                                    f"plan_replan_{replan_count}_{int(time.time())}"
                                )
                                mode.mark_planning_complete()
                                # BUG-9 FIX: Explicitly update vision status before execute()
                                # in the replan sequence. The outer heartbeat loop updates
                                # _vision_ok every HEARTBEAT_INTERVAL, but the replan sequence
                                # runs synchronously inside the task loop — the heartbeat may
                                # not have ticked recently. If _vision_ok is stale-False,
                                # mode.execute() raises VisionUnavailableError and kills the
                                # replan even though vision is actually healthy.
                                mode.update_vision_status(vision_runtime.is_healthy())
                                mode.update_observer_health(observer.is_healthy())
                                mode.execute()
                                snapshot_id = new_snapshot_id

                            finally:
                                mode.end_replan_sequence()

                finally:
                    should_restore = _safe_begin_restoration(mode)

                    if should_restore:
                        try:
                            _restore_exc: list = []

                            def _do_restore() -> None:
                                try:
                                    restore_provider.restore_snapshot(snapshot_id)
                                except Exception as _e:
                                    _restore_exc.append(_e)

                            _restore_thread = threading.Thread(
                                target=_do_restore, daemon=True
                            )
                            _restore_thread.start()
                            _restore_thread.join(timeout=RESTORE_TIMEOUT_SECONDS)

                            if _restore_thread.is_alive():
                                raise RuntimeError(
                                    f"restore_snapshot() timed out after "
                                    f"{RESTORE_TIMEOUT_SECONDS}s — "
                                    "window manager may be unresponsive"
                                )

                            if _restore_exc:
                                raise _restore_exc[0]

                            # RD-03 FIX: Call complete_execution() BEFORE verify().
                            #
                            # Root cause of original sequencing defect:
                            #   1. restore_provider.restore_snapshot(snapshot_id) runs
                            #      in RESTORING mode and calls RestoreProvider._verify()
                            #      internally (mode-check: expects RESTORING) ✓
                            #   2. mode.complete_execution() was called here ← FIXED ORDER
                            #   3. restore_verifier.verify() was called here, which
                            #      internally calls _verify_execution_mode() expecting
                            #      OBSERVER mode ← this is CORRECT
                            #
                            # The original code had complete_execution() AFTER verify(),
                            # meaning verify() ran while mode was still RESTORING. This
                            # caused _verify_execution_mode() to raise:
                            #   RestorationVerificationError(expected OBSERVER, got RESTORING)
                            # which looked like a restoration failure but was a sequencing
                            # failure. Any unexpected exception between restore and
                            # complete_execution() would also leave the mode in RESTORING
                            # indefinitely.
                            #
                            # Fix: call complete_execution() immediately after restore
                            # succeeds and before any secondary verification runs.
                            # This ensures the mode is OBSERVER before verify() checks it.
                            mode.complete_execution()

                            snap_obj = snapshot_provider.get_snapshot(snapshot_id)
                            if snap_obj is not None:
                                try:
                                    restore_verifier.verify(snap_obj)
                                except RestorationVerificationError as rve:
                                    print(
                                        f"[MAIN] RestoreVerifier HARD FAILURE: {rve}",
                                        file=sys.stderr,
                                    )
                                    raise RestorationVerificationError(
                                        f"Post-restore verification failed: {rve}"
                                    )
                            else:
                                # RT-D FIX (P2): Snapshot TTL has expired between
                                # capture and retrieval (3-hour TTL, configurable).
                                # Previously this branch was a silent no-op: verify()
                                # was skipped and auth_state was written with
                                # restore_required=False, falsely signaling a clean
                                # verified exit when verification was never performed.
                                #
                                # FIX: Log at WARNING level so the unverified
                                # restoration is visible in all log aggregators.
                                # Also set auth_state.verification_warning=True so
                                # that monitoring can detect unverified restorations
                                # in the authority audit record (aligns with HAR-07
                                # RestoreVerifier structured audit event pattern).
                                print(
                                    f"[MAIN] WARNING RT-D: Snapshot {snapshot_id!r} "
                                    "has expired (TTL elapsed between capture and "
                                    "retrieval). Post-restore verification was SKIPPED. "
                                    "The system reports clean restoration but the "
                                    "restoration was NOT independently verified. "
                                    "Consider reducing task duration or increasing "
                                    "PROJECTZEO_SNAPSHOT_TTL_SECONDS.",
                                    file=sys.stderr,
                                )
                                try:
                                    auth_state.verification_warning = True
                                except Exception:
                                    pass  # auth_state may not support this attribute

                            # IH-05 FIX: Track whether verification was actually
                            # performed.  If the snapshot expired (snap_obj is None),
                            # we must not write restore_required=False — that would be
                            # a false safety record.  Instead write restore_required=True
                            # so that monitoring and the next run know verification was
                            # skipped.  Only write restore_required=False when
                            # verify() actually completed without raising.
                            _verification_performed = snap_obj is not None

                            # IH-05 FIX: Flush regret on successful task completion.
                            # Stale regret from a successfully-completed task must not
                            # carry over into crash-recovery execution of a new task.
                            # flush_regret_on_success() zeroes all action regret before
                            # the BeliefState is serialised into auth_state.
                            if _task_succeeded and _belief_state_out:
                                try:
                                    _bs_obj = BeliefState.from_dict(
                                        _belief_state_out[0], intent_hash=""
                                    )
                                    _bs_obj.flush_regret_on_success()
                                    _belief_state_out[0] = _bs_obj.to_dict()
                                except Exception:
                                    pass  # Non-fatal: flush failure must not block auth persist

                            # FIX RB-6: Use _safe_belief_snapshot() — never index directly
                            _bs = _safe_belief_snapshot(_belief_state_out)
                            auth_state.persist(
                                execution_mode="OBSERVER",
                                automation_active=False,
                                # IH-05 FIX: Only claim clean restoration when
                                # verification actually ran.  Expired TTL → skip
                                # → restore_required=True so next run re-verifies.
                                restore_required=not _verification_performed,
                                last_snapshot_id=None,
                                dirty=False,
                                thompson_state=(
                                    {
                                        "_iteration_counter": _bs.get("_iteration_counter", 0),
                                        "_sample_counter": _bs.get("_sample_counter", 0),
                                        "commitment_chain_hash": _bs.get(
                                            "commitment_chain_hash",
                                            _bs.get("commitment_hash", ""),
                                        ),
                                    }
                                    if _bs else None
                                ),
                                belief_state_full=_bs if _bs else None,
                            )

                        except Exception as cleanup_err:
                            _force_safe_shutdown(
                                os_backend,
                                auth_state,
                                f"restoration_failure:{cleanup_err}",
                            )
                            raise
                    else:
                        try:
                            # FIX RB-6: Use _safe_belief_snapshot() here too
                            _bs = _safe_belief_snapshot(_belief_state_out)
                            auth_state.persist(
                                execution_mode="OBSERVER",
                                automation_active=False,
                                restore_required=False,
                                last_snapshot_id=None,
                                dirty=False,
                                thompson_state=(
                                    {
                                        "_iteration_counter": _bs.get("_iteration_counter", 0),
                                        "_sample_counter": _bs.get("_sample_counter", 0),
                                        "commitment_chain_hash": _bs.get(
                                            "commitment_chain_hash",
                                            _bs.get("commitment_hash", ""),
                                        ),
                                    }
                                    if _bs else None
                                ),
                                belief_state_full=_bs if _bs else None,
                            )
                        except Exception:
                            pass

                    # Environment re-fingerprint after successful task
                    if _task_succeeded:
                        try:
                            env_fingerprint = collect_environment_fingerprint()
                        except Exception:
                            pass

                    observer.reset_for_new_task()
                    world_graph.reset()
                    _clear_task_start()

                    # GPU CONTENTION FIX: resume observer after task complete
                    try:
                        observer_loop.resume()
                    except Exception:
                        pass

            except ArmedTimeoutError as ate:
                print(f"[MAIN] {ate}", file=sys.stderr)
                mode.force_observer()
                _clear_task_start()
                time.sleep(HEARTBEAT_INTERVAL)
                continue

            except PlanningTimeoutError as pte:
                # C2 FIX: PlanningTimeoutError was not imported or handled.
                # It was raised by mode.check_planning_timeout() and caught by
                # the generic `except Exception` handler, which called
                # _force_safe_shutdown() and broke the main loop — terminating
                # the entire process for what is a recoverable per-task failure.
                #
                # Fix: catch it explicitly, force OBSERVER mode, clear the task
                # timer, and continue the heartbeat loop.  The IntentListener
                # will re-arm on the next intent file write.  Observer and
                # VisionRuntime keep running; no restart needed.
                print(
                    f"[MAIN] PlanningTimeoutError: {pte} — aborting current task, "
                    "returning to OBSERVER. No process restart required.",
                    file=sys.stderr,
                )
                _write_task_result(
                    intent=locals().get("intent", "(planning_timeout)"),
                    success=False,
                    error=f"PlanningTimeoutError: {pte}",
                )
                try:
                    mode.force_observer()
                except Exception:
                    pass
                try:
                    observer_loop.resume()
                except Exception:
                    pass
                _clear_task_start()
                time.sleep(HEARTBEAT_INTERVAL)
                continue

            except ObserverBlindnessError as obe:
                print(
                    f"[MAIN] Observer blind: {obe} — attempting vision restart",
                    file=sys.stderr,
                )
                try:
                    # FIX-RESTART: Stop both vision_runtime AND observer_loop,
                    # then restart both. The original code only restarted
                    # vision_runtime but not observer_loop — the loop remained
                    # stopped (or its _stop_event was set), so no new frames
                    # were ever delivered to ObserverCore after blindness recovery.
                    observer_loop.stop()
                    vision_runtime.stop()
                    time.sleep(2.0)
                    vision_runtime.start()
                    observer_loop.start()
                    time.sleep(VISION_RESTART_GRACE_SECONDS)
                    observer.reset_for_new_task()
                    world_graph.reset()
                    _clear_task_start()
                    continue
                except Exception as restart_err:
                    print(
                        f"[MAIN] Vision restart failed: {restart_err} — shutting down",
                        file=sys.stderr,
                    )
                    break

            except Exception as e:
                _err_str = str(e)
                _write_task_result(
                    intent=locals().get("intent", "(unknown)"),
                    success=False,
                    error=_err_str[:200],
                )
                _interactive_print(f"[MODE] TASK FAILED — {_err_str[:80]}")
                _force_safe_shutdown(os_backend, auth_state, f"main_loop_failure:{e}")
                break

            time.sleep(HEARTBEAT_INTERVAL)

    finally:
        # Shut down last active planner executor to avoid thread leaks
        if _current_planner is not None:
            try:
                _shutdown_executor(_current_planner._executor, wait=False)
            except Exception:
                pass

        for _cleanup_fn, _cleanup_name in [
            (intent_listener.stop, "intent_listener"),
            (observer_loop.stop, "observer_loop"),
            (vision_runtime.stop, "vision_runtime"),
        ]:
            try:
                _cleanup_fn()
            except Exception as _e:
                print(
                    f"[MAIN] Warning: {_cleanup_name}.stop() raised: {_e}",
                    file=sys.stderr,
                )

        # Restore builtins.open — best-effort, non-fatal on shutdown path
        try:
            uninstall_patches()
        except Exception:
            pass

        _force_safe_shutdown(os_backend, auth_state, "shutdown")
