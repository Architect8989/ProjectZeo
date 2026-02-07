import time
import os
import signal
import atexit
import sys

from core.mode_controller import ModeController
from core.intent_listener import IntentListener
from core.environment_fingerprint import collect_environment_fingerprint

from observer.observer_core import ObserverCore, ObserverBlindnessError
from observer.screenpipe_adapter import ScreenpipeAdapter, ScreenpipeBlindnessError
from observer.perception_engine import PerceptionEngine

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
perception = PerceptionEngine()
mode = ModeController()

SNAPSHOT_PROVIDER = SnapshotProvider(
    observer=observer,
    screenpipe=None,
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

    print("[BOOT] System starting")

    # --------------------------------------------------
    # ENVIRONMENT FINGERPRINT
    # --------------------------------------------------
    env_fingerprint = collect_environment_fingerprint()

    # --------------------------------------------------
    # CRASH RECOVERY GATE
    # --------------------------------------------------
    persisted = AUTH_STATE.load()
    if persisted.get("dirty") or persisted.get("restore_required"):
        OS_BACKEND.force_release_all(reason="crash_recovery")
        AUTH_STATE.force_safe_state()
        mode.force_observer()

    # --------------------------------------------------
    # SCREENPIPE (REQUIRED)
    # --------------------------------------------------
    screenpipe = ScreenpipeAdapter()

    if not screenpipe.self_test():
        raise RuntimeError("Screenpipe self-test failed")

    SNAPSHOT_PROVIDER.screenpipe = screenpipe
    print("[SCREENPIPE] Connected and healthy")

    # --------------------------------------------------
    # INTENT LISTENER
    # --------------------------------------------------
    intent_listener = IntentListener(mode)
    intent_listener.start()

    print("[OBSERVER] Active")

    # ==================================================
    # MAIN LOOP
    # ==================================================

    while True:
        try:
            # ----------------------------------------------
            # PERCEPTION
            # ----------------------------------------------
            screen_state = screenpipe.read()
            ui_snapshot = perception.process(screen_state)

            observer.attach_screen_state(screen_state)
            observer.attach_ui_snapshot(ui_snapshot)
            observer.tick()

            mode.update_vision_status(screen_state.get("available") is True)
            mode.update_observer_health(observer.is_healthy())

            # ----------------------------------------------
            # EXECUTION TRIGGER
            # ----------------------------------------------
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
                        screenpipe=screenpipe,
                        llm_callable=mode.get_llm_callable(),
                    )

                finally:
                    # ---- FAIL-CLOSED RESTORATION ----
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

            # ----------------------------------------------
            # HARD WALL-CLOCK ENFORCEMENT
            # ----------------------------------------------
            if TASK_START and (time.time() - TASK_START) > MAX_TASK_SECONDS:
                _force_safe_shutdown("task_timeout")
                os._exit(1)

        except (ObserverBlindnessError, ScreenpipeBlindnessError):
            _force_safe_shutdown("perception_blindness")
            os._exit(1)

        except Exception as e:
            _force_safe_shutdown(f"main_loop_failure:{e}")
            os._exit(1)

        time.sleep(HEARTBEAT_INTERVAL)


if __name__ == "__main__":
    main()
