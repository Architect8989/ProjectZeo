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


def _shutdown_executor(executor, wait: bool = False) -> None:
    """
    RB-08 FIX: Python 3.8 compatibility for ThreadPoolExecutor.shutdown().
    """
    if sys.version_info >= (3, 9):
        executor.shutdown(wait=wait, cancel_futures=True)
    else:
        executor.shutdown(wait=wait)


# PATCH §1.5: reduced from 2.0s to 0.25s for lower intent-to-task latency
HEARTBEAT_INTERVAL = 0.25

MAX_TASK_SECONDS = 90 * 60
MAX_REPLANS = 3

# FIX-4: Extended from 8s to 150s
WARMUP_TIMEOUT_SECONDS = 150.0
WARMUP_STABLE_FRAMES = 3

# DEF-4: Hard timeout for the restoration phase.
RESTORE_TIMEOUT_SECONDS = 60.0

# PATCH §main-7: max retries for transient vision/observer health failures
HEALTH_RETRY_MAX = 5
HEALTH_RETRY_INTERVAL = 1.0

# EVO-4: vision restart on ObserverBlindnessError before giving up
VISION_RESTART_GRACE_SECONDS = 5.0


_TASK_START: Optional[float] = None
_TASK_LOCK = threading.Lock()
_SHUTDOWN_EVENT = threading.Event()


# ============================================================
# THREAD-SAFE TASK TIMING
# ============================================================

def _set_task_start(ts: float):
    global _TASK_START
    with _TASK_LOCK:
        _TASK_START = ts


def _clear_task_start():
    global _TASK_START
    with _TASK_LOCK:
        _TASK_START = None


def _get_task_start():
    with _TASK_LOCK:
        return _TASK_START


# ============================================================
# SAFE SHUTDOWN
# ============================================================

def _force_safe_shutdown(os_backend, auth_state, reason: str):
    try:
        os_backend.force_release_all(reason=reason)
    except Exception:
        pass
    try:
        auth_state.force_safe_state()
    except Exception:
        pass
    print(f"[SAFE-SHUTDOWN] {reason}", file=sys.stderr)


def _signal_handler(signum, frame):
    _SHUTDOWN_EVENT.set()


def _install_signal_handlers():
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    if hasattr(signal, "SIGQUIT"):
        signal.signal(signal.SIGQUIT, _signal_handler)


# ============================================================
# UTILITIES
# ============================================================

def _ingest_latest_perception(observer, world_graph) -> bool:
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


def _enforce_task_timeout():
    start = _get_task_start()
    if start is None:
        return
    if (time.time() - start) > MAX_TASK_SECONDS:
        raise RuntimeError("TASK_FAILED:timeout")


def _safe_begin_restoration(mode: ModeController) -> bool:
    """
    PATCH §1.5: begin_restoration() only accepts EXECUTING mode.
    Returns True if restoration should proceed, False if it should be skipped.
    """
    current = mode.mode

    if current is SystemMode.EXECUTING:
        try:
            mode.begin_restoration()
            return True
        except Exception:
            return False

    if current is SystemMode.RESTORING:
        return True

    if current in (SystemMode.PLANNING, SystemMode.ARMED, SystemMode.OBSERVER):
        try:
            mode.force_observer()
        except Exception:
            pass
        return False

    return False


# ============================================================
# MAIN ENTRY
# ============================================================

def main(llm_callable: Callable, model_name: str):

    if not callable(llm_callable):
        raise RuntimeError("llm_callable must be a callable")

    if not isinstance(model_name, str) or not model_name.strip():
        raise RuntimeError("model_name must be a non-empty string")

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

    # FIX SI-4: On crash recovery, try to restore the full BeliefState so that
    # the first post-crash task continues learning from where it left off rather
    # than re-exploring all actions from a virgin uniform prior.
    _crash_recovery_belief_state: "dict | None" = persisted.get("belief_state_full")

    if persisted.get("dirty") or persisted.get("restore_required"):
        try:
            os_backend.force_release_all(reason="crash_recovery")
        except Exception:
            pass
        auth_state.force_safe_state()
        mode.force_observer()

    vision_runtime.start()
    observer_loop.start()

    # --------------------------------------------------------
    # Warmup — extended to 150s for CPU inference compat.
    # --------------------------------------------------------
    stable_frames = 0
    warmup_deadline = time.time() + WARMUP_TIMEOUT_SECONDS
    _warmup_achieved = False

    while time.time() < warmup_deadline:
        if observer.is_healthy() and vision_runtime.is_healthy():
            if _ingest_latest_perception(observer, world_graph):
                if world_graph.entity_count() > 0:
                    stable_frames += 1
                    if stable_frames >= WARMUP_STABLE_FRAMES:
                        _warmup_achieved = True
                        break
                else:
                    stable_frames = 0
        time.sleep(0.1)

    if not _warmup_achieved:
        print(
            f"[WARMUP] WARNING: warmup did not reach {WARMUP_STABLE_FRAMES} stable frames "
            f"within {WARMUP_TIMEOUT_SECONDS:.0f}s. "
            f"entity_count={world_graph.entity_count()}, stable_frames={stable_frames}. "
            f"Proceeding with degraded world model. "
            f"If running on a bare desktop, ensure at least one application window is "
            f"visible before arming, or increase WARMUP_TIMEOUT_SECONDS.",
            file=sys.stderr,
        )
        try:
            auth_state.record_event({
                "event": "warmup_degraded",
                "stable_frames_achieved": stable_frames,
                "required": WARMUP_STABLE_FRAMES,
                "entity_count": world_graph.entity_count(),
                "timeout_seconds": WARMUP_TIMEOUT_SECONDS,
            })
        except Exception:
            pass

    snapshot_provider = SnapshotProvider(
        observer=observer,
        os_backend=os_backend,
        mode_controller=mode,
    )

    restore_provider = RestoreProvider(
        os_backend=os_backend,
        mode_controller=mode,
        snapshot_provider=snapshot_provider,
    )

    restore_verifier = RestoreVerifier(
        os_backend=os_backend,
        mode_controller=mode,
        cursor_tolerance_px=5,
    )

    intent_listener = IntentListener(mode, snapshot_provider)
    intent_listener.start()

    _health_failures = 0
    _current_planner = None

    try:

        while not _SHUTDOWN_EVENT.is_set():

            try:
                _enforce_task_timeout()

                # FIX-C3 (RB-4): Call watchdog.check() on every heartbeat so
                # runtime safety limits are enforced.  WatchdogViolation is a
                # subclass of RuntimeError and is caught by the outer
                # `except Exception` block, which calls _force_safe_shutdown().
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

                _set_task_start(time.time())
                # FIX-C3: Reset watchdog start time when a new task begins so the
                # 1-hour wall-clock limit applies per-task, not per-process.
                watchdog.start_time = time.time()
                replan_count = 0

                snapshot_id = mode.consume_snapshot()
                if not snapshot_id:
                    raise RuntimeError("Missing snapshot")

                _ingest_latest_perception(observer, world_graph)

                intent = mode.get_intent()
                if not intent or not intent.strip():
                    raise RuntimeError("Invalid intent")

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

                for _health_attempt in range(HEALTH_RETRY_MAX):
                    try:
                        mode.begin_planning()
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

                mode.attach_execution_plan(f"plan_{int(time.time())}")
                mode.mark_planning_complete()

                auth_state.persist(
                    execution_mode="EXECUTING",
                    automation_active=True,
                    restore_required=True,
                    last_snapshot_id=snapshot_id,
                    dirty=True,
                )

                mode.execute()

                _task_succeeded = False

                
                _belief_state_out: list = []
                # FIX SI-4: Seed first task with crash-recovery BeliefState
                # when available; clear after first use so subsequent tasks
                # start fresh (normal behavior for non-crash runs).
                _prior_belief_state: Optional[dict] = _crash_recovery_belief_state
                _crash_recovery_belief_state = None  # consume once

                try:
                    while not _SHUTDOWN_EVENT.is_set():
                        _enforce_task_timeout()

                        try:
                            operate_main(
                                terminal_prompt=intent,
                                execution_plan=execution_plan,
                                planner=planner,
                                observer=observer,
                                world_graph=world_graph,
                                os_backend=os_backend,
                                watchdog=watchdog,
                                prior_belief_state=_prior_belief_state,   # MATH-NEW-03
                                belief_state_out=_belief_state_out,       # MATH-NEW-03
                            )
                            _task_succeeded = True
                            break

                        except RuntimeError as e:
                            if str(e) != "REPLAN_REQUIRED":
                                raise

                            # Capture belief state for the next replan attempt.
                            _prior_belief_state = (
                                _belief_state_out[0] if _belief_state_out else None
                            )

                            replan_count += 1
                            if replan_count > MAX_REPLANS:
                                raise RuntimeError("TASK_FAILED:max_replans_exceeded")
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

                                mode.attach_snapshot(new_snapshot_id)

                                # FIX-C5 (RB-6): Use mode.arm_for_replan(intent) instead
                                # of mode.arm(intent) in the replan sequence.
                                #
                                # arm_for_replan() was added specifically as the
                                # self-documenting, safe-by-construction path for
                                # replanning.  Using arm() here is structurally
                                # incorrect: if the arm() guard is ever reinstated
                                # (e.g. to prevent concurrent arming), all replan
                                # sequences will raise ModeTransitionError silently
                                # and arm_for_replan() will remain unreachable dead code.
                                #
                                # Using the purpose-built method makes the contract
                                # explicit and future-proof against arm() guard changes.
                                mode.arm_for_replan(intent)

                                _ingest_latest_perception(observer, world_graph)
                                planner.update_world_snapshot(world_graph.snapshot())

                                mode.begin_planning()

                                env_fingerprint = collect_environment_fingerprint()
                                planner.refresh_environment(env_fingerprint)

                                # RB-1 FIX: Pause CPU monitoring during replan LLM call.
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
                                mode.execute()

                                snapshot_id = new_snapshot_id

                            finally:
                                mode.end_replan_sequence()

                finally:
                    should_restore = _safe_begin_restoration(mode)

                    if should_restore:
                        try:
                            _restore_exc: list = []

                            def _do_restore():
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

                            auth_state.persist(
                                execution_mode="OBSERVER",
                                automation_active=False,
                                restore_required=False,
                                last_snapshot_id=None,
                                dirty=False,
                                thompson_state=(
                                    {
                                        # RT-02 FIX: BeliefState.to_dict() serializes these
                                        # counters with underscore-prefix keys ("_iteration_counter",
                                        # "_sample_counter").  The previous code used the no-underscore
                                        # form, so .get() always returned the default 0, causing the
                                        # persisted thompson_state stub to be all-zeros on every task
                                        # completion.  After a crash, the Thompson stub was always
                                        # zero — losing per-session uniqueness.
                                        #
                                        # Fix: use the canonical underscore-prefixed key names that
                                        # match BeliefState.to_dict().  AuthorityStateSerializer.persist()
                                        # accepts both forms and normalizes them internally.
                                        "_iteration_counter": _belief_state_out[0].get("_iteration_counter", 0),
                                        "_sample_counter":    _belief_state_out[0].get("_sample_counter", 0),
                                        "commitment_chain_hash": _belief_state_out[0].get(
                                            "commitment_chain_hash",
                                            _belief_state_out[0].get("commitment_hash", ""),
                                        ),
                                    }
                                    if _belief_state_out else None
                                ),
                                # FIX SI-4: Persist full BeliefState for
                                # crash-recovery bandit continuity.
                                belief_state_full=(
                                    _belief_state_out[0] if _belief_state_out else None
                                ),
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
                            auth_state.persist(
                                execution_mode="OBSERVER",
                                automation_active=False,
                                restore_required=False,
                                last_snapshot_id=None,
                                dirty=False,
                                thompson_state=(
                                    {
                                        # RT-02 FIX (second site): same correction as above.
                                        "_iteration_counter": _belief_state_out[0].get("_iteration_counter", 0),
                                        "_sample_counter":    _belief_state_out[0].get("_sample_counter", 0),
                                        "commitment_chain_hash": _belief_state_out[0].get(
                                            "commitment_chain_hash",
                                            _belief_state_out[0].get("commitment_hash", ""),
                                        ),
                                    }
                                    if _belief_state_out else None
                                ),
                                # FIX SI-4: Persist full BeliefState.
                                belief_state_full=(
                                    _belief_state_out[0] if _belief_state_out else None
                                ),
                            )
                        except Exception:
                            pass

                    if _task_succeeded:
                        try:
                            env_fingerprint = collect_environment_fingerprint()
                        except Exception:
                            pass

                    observer.reset_for_new_task()
                    world_graph.reset()
                    _clear_task_start()

            except ArmedTimeoutError as ate:
                print(f"[MAIN] {ate}", file=sys.stderr)
                mode.force_observer()
                _clear_task_start()
                time.sleep(HEARTBEAT_INTERVAL)
                continue

            except ObserverBlindnessError as obe:
                print(
                    f"[MAIN] Observer blind: {obe} — attempting vision restart",
                    file=sys.stderr,
                )
                try:
                    vision_runtime.stop()
                    time.sleep(2.0)
                    vision_runtime.start()
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
                _force_safe_shutdown(os_backend, auth_state, f"main_loop_failure:{e}")
                break

            time.sleep(HEARTBEAT_INTERVAL)

    finally:
        # GAP-3: shutdown last active planner executor on exit
        if _current_planner is not None:
            try:
                _shutdown_executor(_current_planner._executor, wait=False)
            except Exception:
                pass
        try:
            intent_listener.stop()
        except Exception:
            pass
        try:
            observer_loop.stop()
        except Exception:
            pass
        try:
            vision_runtime.stop()
        except Exception:
            pass

        
        try:
            uninstall_patches()
        except Exception:
            pass  # Non-fatal — shutdown path must not raise

        _force_safe_shutdown(os_backend, auth_state, "shutdown")
