import time
import os
import signal
import atexit
import sys

from core.mode_controller import ModeController
from core.intent_listener import IntentListener
from core.environment_fingerprint import collect_environment_fingerprint

from observer.observer_core import ObserverCore, ObserverBlindnessError
from observer.screenpipe_adapter import ScreenpipeAdapter
from observer.perception_engine import PerceptionEngine

from state.serializer import AuthorityStateSerializer
from operate.utils.operating_system import OperatingSystem
from operate.operate import operate_main

from restoration.snapshot_provider import SnapshotProvider
from restoration.restore_provider import RestoreProvider

from core.planner.execution_planner import ExecutionPlanner

HEARTBEAT_INTERVAL = 2.0

# ==================================================
# PROCESS AUTHORITY (SINGLETONS)
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
        OS_BACKEND.force_release_all()
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
# ROOT MAIN (LIFECYCLE AUTHORITY)
# ==================================================

def main():
    print("[BOOT] System starting")

    # ---- environment probe (non-fatal) ----
    collect_environment_fingerprint()

    # ---- crash recovery gate ----
    persisted = AUTH_STATE.load()
    if persisted.get("dirty") or persisted.get("restore_required"):
        OS_BACKEND.force_release_all()
        AUTH_STATE.force_safe_state()
        mode.force_observer()

    # ---- REQUIRED DEPENDENCY: Screenpipe must already be running ----
    screenpipe = ScreenpipeAdapter()
    SNAPSHOT_PROVIDER.screenpipe = screenpipe

    # ---- intent listener ----
    intent_listener = IntentListener(mode)
    intent_listener.start()

    print("[OBSERVER] Active")

    while True:
        try:
            # ==================================================
            # OBSERVER LOOP
            # ==================================================
            screen_state = screenpipe.read()
            ui_snapshot = perception.process(screen_state)

            observer.attach_screen_state(screen_state)
            observer.attach_ui_snapshot(ui_snapshot)
            observer.tick()

            mode.update_vision_status(screen_state.get("available", False))
            mode.update_observer_health(observer.is_healthy())

            # ==================================================
            # EXECUTION TRIGGER
            # ==================================================
            if mode.is_armed():
                # ---- SNAPSHOT (OBSERVER ONLY) ----
                snapshot_id = SNAPSHOT_PROVIDER.take_snapshot()

                # ---- PLANNING ----
                mode.begin_planning()

                # READ intent WITHOUT consuming
                intent = mode.get_intent()
                if not intent:
                    raise RuntimeError("Armed with no intent")

                planner = ExecutionPlanner(
                    llm_call=lambda prompt: prompt  # placeholder
                )

                execution_plan = planner.create_plan(
                    objective=intent,
                    requirements={},
                    high_level_steps=[
                        {"goal": intent}
                    ],
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

                    # NOW consume intent (EXECUTING only)
                    consumed_intent = mode.consume_intent()

                    operate_main(
                        model=None,
                        terminal_prompt=consumed_intent,
                        execution_plan=execution_plan,
                        observer=observer,
                        screenpipe=screenpipe,
                    )

                finally:
                    # ---- RESTORE (FAIL-CLOSED) ----
                    try:
                        RESTORE_PROVIDER.restore_snapshot(snapshot_id)
                    except Exception:
                        pass

                    AUTH_STATE.persist(
                        execution_mode="OBSERVER",
                        automation_active=False,
                        restore_required=False,
                        last_snapshot_id=None,
                        dirty=False,
                    )

                    mode.force_observer()

        except ObserverBlindnessError:
            _force_safe_shutdown("observer-blindness")
            os._exit(1)

        except Exception as e:
            _force_safe_shutdown(f"main-loop-failure: {e}")
            os._exit(1)

        time.sleep(HEARTBEAT_INTERVAL)


if __name__ == "__main__":
    main()
