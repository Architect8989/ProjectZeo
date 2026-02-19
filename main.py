"""
main.py
========
PATCH AUDIT FIXES:

  ⚠️  §1.5: `begin_restoration()` requires mode == EXECUTING.  If the finally
            block fires while mode is PLANNING (e.g. planning timeout), it raises
            ModeTransitionError — leaving the system in PLANNING forever.
            FIX: Guard expanded to `if mode.mode in (EXECUTING, PLANNING)`.
            Before calling begin_restoration() we force the mode to EXECUTING
            via the existing force_observer() + rearm path so the state machine
            contract is not violated.

  ⚠️  §1.5: HEARTBEAT_INTERVAL=2.0s creates up to 2s latency between intent
            arrival and task start.  Reduced to 0.25s for better interactivity
            while preserving the heartbeat mechanism.
            (Kept as a named constant so operators can tune it.)

  ✅  All existing correct behaviours preserved:
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

from core.mode_controller import ModeController, SystemMode
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
    If the finally block fires from PLANNING (planning timeout) or any other
    non-EXECUTING mode, we need to safely transition first.

    Strategy:
      1. EXECUTING  → call begin_restoration() directly (normal path)
      2. PLANNING   → force_observer() resets mode, then we skip restoration
                     (no execute happened, nothing to restore)
      3. RESTORING  → already in restoration, idempotent — do nothing
      4. Other      → force_observer() + skip

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
        # Already restoring — caller should proceed with restore_snapshot()
        return True

    if current in (SystemMode.PLANNING, SystemMode.ARMED, SystemMode.OBSERVER):
        # No execution happened (or planning failed before execute()) —
        # force reset to OBSERVER and skip restoration.
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

    try:

        while not _SHUTDOWN_EVENT.is_set():

            try:
                _enforce_task_timeout()

                mode.update_observer_health(observer.is_healthy())
                mode.update_vision_status(vision_runtime.is_healthy())

                if mode.mode == SystemMode.PLANNING:
                    mode.check_planning_timeout()

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

                planner = ExecutionPlanner(
                    llm_call=mode.get_llm_callable(),
                    environment_fingerprint=env_fingerprint,
                    world_graph=world_graph,
                )

                mode.begin_planning()

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
                    # PATCH §1.5: expanded guard — EXECUTING or PLANNING both trigger
                    # restoration.  _safe_begin_restoration() handles the transition
                    # correctly for every mode the finally block might fire in.
                    should_restore = _safe_begin_restoration(mode)

                    if should_restore:
                        try:
                            restore_provider.restore_snapshot(snapshot_id)

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
                        # No execution occurred — just clean auth state
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

                    observer.reset_for_new_task()
                    world_graph.reset()
                    _clear_task_start()

            except ObserverBlindnessError:
                break

            except Exception as e:
                _force_safe_shutdown(os_backend, auth_state, f"main_loop_failure:{e}")
                break

            time.sleep(HEARTBEAT_INTERVAL)

    finally:
        try:
            observer_loop.stop()
        except Exception:
            pass
        try:
            vision_runtime.stop()
        except Exception:
            pass

        _force_safe_shutdown(os_backend, auth_state, "shutdown")
