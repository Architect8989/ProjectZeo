import time
import os
import signal
import atexit
import sys
from typing import Callable

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

# ==================================================
# EXECUTION TIME BOUND (HARD SAFETY)
# ==================================================

TASK_START = None
MAX_TASK_SECONDS = 90 * 60  # 90 minutes hard cap


# ==================================================
# PROCESS SINGLETONS
# ==================================================

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
# LLM INJECTION (AUTHORITATIVE FIX)
# ==================================================

def create_llm_callable() -> Callable[[str], str]:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)

    def llm_call(prompt: str) -> str:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    return llm_call


# 🔒 Inject ONCE, before any planning can occur
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
# SAFE SHUTDOWN (FAIL-CLOSED)
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
signal.signal(signal.SIGQUIT, _signal_handler)


# ==================================================
# ROOT MAIN
# ==================================================

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

    intent_listener = IntentListener(mode)
    intent_listener.start()

    while True:
        try:
            mode.update_observer_health(observer.is_healthy())
            mode.update_vision_status(vision_runtime.is_healthy())

            if mode.is_armed():
                TASK_START = time.time()

                snapshot_id = SNAPSHOT_PROVIDER.take_snapshot()
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
                        llm_callable=mode.get_llm_callable(),
                    )

                finally:
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

                    mode.force_observer()
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
