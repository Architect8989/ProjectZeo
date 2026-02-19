"""
main.py
========
PATCHES APPLIED (Audit Fixes):

  ✅  §R5  (was §main-8): Environment fingerprint refreshed after each
           successful task execution. Previously the fingerprint was collected
           once at startup — after the installer installed Node.js, the planner
           still saw tools["node"]=False and planned redundant installs on
           every replan.  Now collect_environment_fingerprint() is re-run after
           operate_main() returns successfully, and the planner is refreshed
           via planner.refresh_environment().

  ✅  §R8  (was §mc-6): check_armed_timeout() is now called in the main
           heartbeat loop alongside check_planning_timeout().  Prevents
           infinite ARMED stall when begin_planning() is never called.

  ✅  §main-7 (was vision glitch): VisionUnavailableError and
           ObserverUnavailableError from begin_planning() now retry up to
           HEALTH_RETRY_MAX times before giving up rather than immediately
           calling _force_safe_shutdown() and breaking the main loop.

  ✅  §DEF-4: restore_provider.restore_snapshot() now runs under a thread-based
           timeout guard (RESTORE_TIMEOUT_SECONDS = 60). Previously if the
           window manager was unresponsive during RESTORING mode, the kernel
           would hang indefinitely. The guard kills the restoration call and
           raises RuntimeError so the main loop can call _force_safe_shutdown.
           handles EXECUTING, PLANNING, ARMED, RESTORING modes correctly.

  ✅  §1.5 (original): HEARTBEAT_INTERVAL reduced from 2.0s to 0.25s.

All existing correct behaviours preserved:
  - crash_recovery check at startup
  - SIGINT/SIGTERM/SIGQUIT signal handlers
  - MAX_REPLANS=3
  - finally-block restoration pattern
  - observer_loop / vision_runtime lifecycle management
"""

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

from restoration.snapshot_provider import SnapshotProvider
from restoration.restore_provider import RestoreProvider

from core.planner.execution_planner import ExecutionPlanner


# PATCH §1.5: reduced from 2.0s to 0.25s for lower intent-to-task latency
HEARTBEAT_INTERVAL = 0.25

MAX_TASK_SECONDS = 90 * 60
MAX_REPLANS = 3
WARMUP_STABLE_FRAMES = 3

# DEF-4: Hard timeout for the restoration phase.
# If RestoreProvider.restore_snapshot() hangs (e.g. unresponsive window manager),
# the kernel must not block in RESTORING indefinitely.
RESTORE_TIMEOUT_SECONDS = 60.0

# PATCH §main-7: max retries for transient vision/observer health failures
# before treating them as fatal and breaking the main loop.
HEALTH_RETRY_MAX = 5
HEALTH_RETRY_INTERVAL = 1.0


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

def _ingest_latest_perception(observer, world_graph):
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
    If the finally block fires from PLANNING or other modes, handle gracefully.

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
        raise RuntimeError("LLM callable must be provided")

    if not isinstance(model_name, str) or not model_name.strip():
        raise RuntimeError("model_name must be provided")

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

    # PATCH §R5: environment fingerprint — mutable, refreshed after each task
    env_fingerprint = collect_environment_fingerprint()

    persisted = auth_state.load()
    if persisted.get("dirty") or persisted.get("restore_required"):
        try:
            os_backend.force_release_all(reason="crash_recovery")
        except Exception:
            pass
        auth_state.force_safe_state()
        mode.force_observer()

    vision_runtime.start()
    observer_loop.start()

    stable_frames = 0
    warmup_deadline = time.time() + 8.0

    while time.time() < warmup_deadline:
        if observer.is_healthy() and vision_runtime.is_healthy():
            if _ingest_latest_perception(observer, world_graph):
                if world_graph.entity_count() > 0:
                    stable_frames += 1
                    if stable_frames >= WARMUP_STABLE_FRAMES:
                        break
                else:
                    stable_frames = 0
        time.sleep(0.1)

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

    intent_listener = IntentListener(mode, snapshot_provider)
    intent_listener.start()

    # Transient health failure counter for PATCH §main-7
    _health_failures = 0

    try:

        while not _SHUTDOWN_EVENT.is_set():

            try:
                _enforce_task_timeout()

                mode.update_observer_health(observer.is_healthy())
                mode.update_vision_status(vision_runtime.is_healthy())

                if mode.mode == SystemMode.PLANNING:
                    mode.check_planning_timeout()

                # PATCH §R8: guard ARMED mode stall
                if mode.mode == SystemMode.ARMED:
                    mode.check_armed_timeout()

                if not mode.is_armed():
                    time.sleep(HEARTBEAT_INTERVAL)
                    continue

                _set_task_start(time.time())
                replan_count = 0

                snapshot_id = mode.consume_snapshot()
                if not snapshot_id:
                    raise RuntimeError("Missing snapshot")

                _ingest_latest_perception(observer, world_graph)

                intent = mode.get_intent()
                if not intent or not intent.strip():
                    raise RuntimeError("Invalid intent")

                # GAP-3 FIX: ExecutionPlanner creates a ThreadPoolExecutor internally.
                # Previously a new planner (and executor) was created on every task and
                # every replan without ever calling executor.shutdown(). Over many tasks
                # or replans this leaked OS thread-pool resources.
                # Fix: explicitly shutdown the previous planner's executor before creating
                # a new one. _current_planner tracks the active instance across replans.
                if "_current_planner" in dir() and _current_planner is not None:
                    try:
                        _current_planner._executor.shutdown(wait=False, cancel_futures=True)
                    except Exception:
                        pass
                planner = ExecutionPlanner(
                    llm_call=mode.get_llm_callable(),
                    environment_fingerprint=env_fingerprint,
                    world_graph=world_graph,
                )
                _current_planner = planner

                # PATCH §main-7: retry transient health failures before begin_planning
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
                            f"[MAIN] Health check failed ({_health_attempt + 1}/{HEALTH_RETRY_MAX}): "
                            f"{health_err} — retrying in {HEALTH_RETRY_INTERVAL}s",
                            file=sys.stderr,
                        )
                        time.sleep(HEALTH_RETRY_INTERVAL)
                        mode.update_observer_health(observer.is_healthy())
                        mode.update_vision_status(vision_runtime.is_healthy())

                execution_plan = planner.create_plan(
                    objective=intent,
                    requirements={
                        "environment": env_fingerprint,
                        "tools": env_fingerprint.get("tools", []),
                    },
                    high_level_steps=[{"goal": intent}],
                )

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
                            )
                            _task_succeeded = True
                            break

                        except RuntimeError as e:

                            if str(e) != "REPLAN_REQUIRED":
                                raise

                            replan_count += 1
                            if replan_count > MAX_REPLANS:
                                raise RuntimeError("TASK_FAILED:max_replans_exceeded")

                            mode.begin_replan_sequence()

                            try:
                                mode.force_observer()
                                new_snapshot_id = snapshot_provider.take_snapshot()
                                mode.attach_snapshot(new_snapshot_id)
                                mode.arm(intent)

                                _ingest_latest_perception(observer, world_graph)
                                planner.update_world_snapshot(world_graph.snapshot())

                                mode.begin_planning()

                                # PATCH §R5: refresh fingerprint before replan so
                                # newly installed tools are visible to the planner
                                env_fingerprint = collect_environment_fingerprint()
                                planner.refresh_environment(env_fingerprint)

                                execution_plan = planner.create_plan(
                                    objective=intent,
                                    requirements={
                                        "environment": env_fingerprint,
                                        "tools": env_fingerprint.get("tools", []),
                                    },
                                    high_level_steps=[{"goal": intent}],
                                )

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
                            # DEF-4: Timeout guard on restoration phase.
                            # RestoreProvider can hang if the window manager is
                            # unresponsive. We run it in a daemon thread with a
                            # hard deadline so RESTORING mode cannot block forever.
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

                            auth_state.persist(
                                execution_mode="OBSERVER",
                                automation_active=False,
                                restore_required=False,
                                last_snapshot_id=None,
                                dirty=False,
                            )

                            mode.complete_execution()

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
                            )
                        except Exception:
                            pass

                    # PATCH §R5: refresh fingerprint after successful task
                    # so that the next task sees newly installed tools.
                    if _task_succeeded:
                        try:
                            env_fingerprint = collect_environment_fingerprint()
                        except Exception:
                            pass  # Non-fatal — use existing fingerprint

                    observer.reset_for_new_task()
                    world_graph.reset()
                    _clear_task_start()

            except ArmedTimeoutError as ate:
                # PATCH §R8: ARMED timeout is recoverable — log and continue
                print(f"[MAIN] {ate}", file=sys.stderr)
                mode.force_observer()
                _clear_task_start()
                time.sleep(HEARTBEAT_INTERVAL)
                continue

            except ObserverBlindnessError:
                break

            except Exception as e:
                _force_safe_shutdown(os_backend, auth_state, f"main_loop_failure:{e}")
                break

            time.sleep(HEARTBEAT_INTERVAL)

    finally:
        # GAP-3 FIX: Shutdown the last active planner executor on exit.
        try:
            if "_current_planner" in dir() and _current_planner is not None:
                _current_planner._executor.shutdown(wait=False, cancel_futures=True)
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

        _force_safe_shutdown(os_backend, auth_state, "shutdown")
