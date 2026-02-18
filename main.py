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

TASK_START: Optional[float] = None


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


def _enforce_task_timeout(os_backend, auth_state):
    global TASK_START
    if TASK_START is None:
        return
    if (time.time() - TASK_START) > MAX_TASK_SECONDS:
        _force_safe_shutdown(os_backend, auth_state, "task_timeout")
        os._exit(1)


def main(llm_callable: Callable, model_name: str):

    global TASK_START

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

    _install_signal_handlers(os_backend, auth_state)

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
        snapshot_provider=snapshot_provider,
    )

    intent_listener = IntentListener(mode, snapshot_provider)
    intent_listener.start()

    while True:

        try:

            _enforce_task_timeout(os_backend, auth_state)

            mode.update_observer_health(observer.is_healthy())
            mode.update_vision_status(vision_runtime.is_healthy())

            if mode.mode == SystemMode.PLANNING:
                mode.check_planning_timeout()

            if not mode.is_armed():
                time.sleep(HEARTBEAT_INTERVAL)
                continue

            TASK_START = time.time()
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

            while True:

                _enforce_task_timeout(os_backend, auth_state)

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

                    msg = str(e)

                    if msg == "REPLAN_REQUIRED":

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

                        continue

                    if msg.startswith("TASK_FAILED"):
                        raise

                    raise

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

        except ObserverBlindnessError:
            _force_safe_shutdown(os_backend, auth_state, "observer_blindness")
            os._exit(1)

        except Exception as e:
            _force_safe_shutdown(os_backend, auth_state, f"main_loop_failure:{e}")
            os._exit(1)

        time.sleep(HEARTBEAT_INTERVAL)
