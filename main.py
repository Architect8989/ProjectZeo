import time
import os
import signal
import atexit
import sys
from typing import Callable, Optional

from core.mode_controller import ModeController
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
TASK_START = None
MAX_TASK_SECONDS = 90 * 60


OS_BACKEND = OperatingSystem()

STATE_PATH = os.path.join(os.getcwd(), ".authority_state.json")
AUTH_STATE = AuthorityStateSerializer(STATE_PATH)

observer = ObserverCore()
vision_runtime = VisionRuntime()
world_graph = WorldGraph()

observer_loop = ObserverLoop(
    observer=observer,
    vision_runtime=vision_runtime,
    world_graph=world_graph,
)

mode = ModeController()


# ==================================================
# LLM INJECTION (PLANNING ONLY)
# ==================================================

def create_llm_callable() -> Callable[[str], str]:
    from anthropic import Anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    client = Anthropic(api_key=api_key)

    def llm_call(prompt: str) -> str:
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    return llm_call


mode.inject_llm_callable(create_llm_callable())


SNAPSHOT_PROVIDER = SnapshotProvider(
    observer=observer,
    os_backend=OS_BACKEND,
    mode_controller=mode,
)

RESTORE_PROVIDER = RestoreProvider(
    os_backend=OS_BACKEND,
    mode_controller=mode,
)


# ==================================================
# SAFE SHUTDOWN
# ==================================================

def _force_safe_shutdown(reason: str):
    try:
        OS_BACKEND.force_release_all(reason=reason)
    except Exception:
        pass

    try:
        AUTH_STATE.force_safe_state()
    except Exception:
        pass

    print(f"[SAFE-SHUTDOWN] {reason}", file=sys.stderr)


def _signal_handler(signum, frame):
    _force_safe_shutdown(f"signal-{signum}")
    os._exit(1)


atexit.register(_force_safe_shutdown, "atexit")
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)
if hasattr(signal, "SIGQUIT"):
    signal.signal(signal.SIGQUIT, _signal_handler)


# ==================================================
# ROOT MAIN
# ==================================================

def _ingest_latest_perception():
    """
    Strict perception unwrap.
    ObserverCore now stores flat perception payload.
    """
    snap = observer.snapshot()
    if not isinstance(snap, dict):
        return

    perception = snap.get("perception")
    if not isinstance(perception, dict):
        return

    if not perception.get("available"):
        return

    world_graph.ingest(perception)


def main():
    global TASK_START

    print("[BOOT] ProjectZeo kernel starting")

    env_fingerprint = collect_environment_fingerprint()

    persisted = AUTH_STATE.load()
    if persisted.get("dirty") or persisted.get("restore_required"):
        try:
            OS_BACKEND.force_release_all(reason="crash_recovery")
        except Exception:
            pass

        AUTH_STATE.force_safe_state()
        mode.force_observer()

    vision_runtime.start()
    observer_loop.start()

    print("[BOOT] Waiting for first valid perception frame...")
    warmup_deadline = time.time() + 5.0

    while time.time() < warmup_deadline:
        if observer.is_healthy() and vision_runtime.is_healthy():
            try:
                _ingest_latest_perception()
                if world_graph.entity_count() > 0:
                    break
            except Exception:
                pass
        time.sleep(0.1)

    intent_listener = IntentListener(mode, SNAPSHOT_PROVIDER)
    intent_listener.start()

    while True:
        try:
            mode.update_observer_health(observer.is_healthy())
            mode.update_vision_status(vision_runtime.is_healthy())

            if mode.is_armed():

                TASK_START = time.time()

                snapshot_id: Optional[str] = mode.consume_snapshot()
                if not snapshot_id:
                    raise RuntimeError("Missing snapshot at execution boundary")

                # Correct ingestion before planning
                _ingest_latest_perception()

                mode.begin_planning()

                intent = mode.get_intent()
                if not intent or not intent.strip():
                    raise RuntimeError("Armed without valid intent")

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

                plan_id = f"plan_{int(time.time())}_{abs(hash(intent))}"
                mode.attach_execution_plan(plan_id)
                mode.mark_planning_complete()

                AUTH_STATE.persist(
                    execution_mode="EXECUTING",
                    automation_active=True,
                    restore_required=True,
                    last_snapshot_id=snapshot_id,
                    dirty=True,
                )

                try:
                    mode.execute()
                    consumed_intent = mode.consume_intent()

                    operate_main(
                        model=None,
                        terminal_prompt=consumed_intent,
                        execution_plan=execution_plan,
                        observer=observer,
                        world_graph=world_graph,
                    )

                finally:
                    mode.begin_restoration()

                    try:
                        RESTORE_PROVIDER.restore_snapshot(snapshot_id)
                    except Exception as e:
                        _force_safe_shutdown(f"restoration_failed:{e}")
                        os._exit(1)

                    AUTH_STATE.persist(
                        execution_mode="OBSERVER",
                        automation_active=False,
                        restore_required=False,
                        last_snapshot_id=None,
                        dirty=False,
                    )

                    mode.complete_execution()

                    try:
                        observer.reset_for_new_task()
                    except Exception:
                        pass

                    try:
                        world_graph.reset()
                    except Exception as e:
                        print(f"[WARN] world_graph reset failed: {e}", file=sys.stderr)

                    TASK_START = None

            if TASK_START and (time.time() - TASK_START) > MAX_TASK_SECONDS:
                _force_safe_shutdown("task_timeout")
                os._exit(1)

        except ObserverBlindnessError:
            _force_safe_shutdown("observer_blindness")
            os._exit(1)

        except Exception as e:
            _force_safe_shutdown(f"main_loop_failure:{e}")
            os._exit(1)

        time.sleep(HEARTBEAT_INTERVAL)


if __name__ == "__main__":
    main()
