import time
import os
import signal
import atexit
import sys
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


HEARTBEAT_INTERVAL = 2.0
MAX_TASK_SECONDS = 90 * 60
MAX_REPLANS = 3

TASK_START = None


# ==================================================
# SAFE SHUTDOWN
# ==================================================

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


def _install_signal_handlers(os_backend, auth_state):
    def _signal_handler(signum, frame):
        _force_safe_shutdown(os_backend, auth_state, f"signal-{signum}")
        os._exit(1)

    atexit.register(_force_safe_shutdown, os_backend, auth_state, "atexit")
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    if hasattr(signal, "SIGQUIT"):
        signal.signal(signal.SIGQUIT, _signal_handler)


# ==================================================
# PERCEPTION INGESTION
# ==================================================

def _ingest_latest_perception(observer, world_graph):
    snap = observer.snapshot()
    if not isinstance(snap, dict):
        return

    if not snap.get("perception_available"):
        return

    perception = snap.get("perception")
    if not isinstance(perception, dict):
        return

    world_graph.ingest(perception)


# ==================================================
# MAIN LOOP
# ==================================================

def main(llm_callable: Callable[[str], str]):
    global TASK_START

    if not callable(llm_callable):
        raise RuntimeError("LLM callable must be provided")

    # Runtime-owned dependencies (moved from global scope)
    os_backend = OperatingSystem()
    state_path = os.path.join(os.getcwd(), ".authority_state.json")
    auth_state = AuthorityStateSerializer(state_path)

    observer = ObserverCore()
    vision_runtime = VisionRuntime()
    world_graph = WorldGraph()

    observer_loop = ObserverLoop(
        observer=observer,
        vision_runtime=vision_runtime,
        world_graph=world_graph,
    )

    mode = ModeController()
    mode.inject_llm_callable(llm_callable)

    _install_signal_handlers(os_backend, auth_state)

    print("[BOOT] ProjectZeo kernel starting")

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

    # Warmup perception
    warmup_deadline = time.time() + 5.0
    while time.time() < warmup_deadline:
        if observer.is_healthy() and vision_runtime.is_healthy():
            _ingest_latest_perception(observer, world_graph)
            if world_graph.entity_count() > 0:
                break
        time.sleep(0.1)

    snapshot_provider = SnapshotProvider(
        observer=observer,
        os_backend=os_backend,
        mode_controller=mode,
    )

    restore_provider = RestoreProvider(
        os_backend=os_backend,
        mode_controller=mode,
    )

    intent_listener = IntentListener(mode, snapshot_provider)
    intent_listener.start()

    while True:
        try:
            mode.update_observer_health(observer.is_healthy())
            mode.update_vision_status(vision_runtime.is_healthy())

            if mode.mode == SystemMode.PLANNING:
                mode.check_planning_timeout()

            if mode.is_armed():

                TASK_START = time.time()
                replan_count = 0

                snapshot_id: Optional[str] = mode.consume_snapshot()
                if not snapshot_id:
                    raise RuntimeError("Missing snapshot")

                _ingest_latest_perception(observer, world_graph)
                mode.begin_planning()

                intent = mode.get_intent()
                if not intent or not intent.strip():
                    raise RuntimeError("Invalid intent")

                planner = ExecutionPlanner(
                    llm_call=mode.get_llm_callable(),
                    environment_fingerprint=env_fingerprint,
                    world_graph=world_graph,
                )

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
                consumed_intent = mode.consume_intent()

                try:
                    while True:
                        try:
                            operate_main(
                                terminal_prompt=consumed_intent,
                                execution_plan=execution_plan,
                                planner=planner,
                                observer=observer,
                                world_graph=world_graph,
                            )
                            break

                        except RuntimeError as e:
                            if str(e) != "REPLAN_REQUIRED":
                                raise

                            replan_count += 1
                            if replan_count > MAX_REPLANS:
                                raise RuntimeError("Max replans exceeded")

                            _ingest_latest_perception(observer, world_graph)
                            planner.update_world_snapshot(world_graph.snapshot())

                            execution_plan = planner.create_plan(
                                objective=consumed_intent,
                                requirements={
                                    "environment": env_fingerprint,
                                    "tools": env_fingerprint.get("tools", []),
                                },
                                high_level_steps=[{"goal": consumed_intent}],
                            )

                            mode.begin_planning()
                            mode.attach_execution_plan(
                                f"plan_replan_{replan_count}_{int(time.time())}"
                            )
                            mode.mark_planning_complete()
                            mode.execute()

                finally:
                    mode.begin_restoration()

                    restore_provider.restore_snapshot(snapshot_id)

                    auth_state.persist(
                        execution_mode="OBSERVER",
                        automation_active=False,
                        restore_required=False,
                        last_snapshot_id=None,
                        dirty=False,
                    )

                    mode.complete_execution()

                    observer.reset_for_new_task()
                    world_graph.reset()
                    TASK_START = None

            if TASK_START and (time.time() - TASK_START) > MAX_TASK_SECONDS:
                raise RuntimeError("task_timeout")

        except ObserverBlindnessError:
            _force_safe_shutdown(os_backend, auth_state, "observer_blindness")
            os._exit(1)

        except Exception as e:
            _force_safe_shutdown(os_backend, auth_state, f"main_loop_failure:{e}")
            os._exit(1)

        time.sleep(HEARTBEAT_INTERVAL)
