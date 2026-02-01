import time
import os
import signal
import atexit

from core.mode_controller import ModeController, ModeTransitionError
from core.intent_listener import IntentListener
from core.environment_fingerprint import collect_environment_fingerprint

from observer.observer_core import ObserverCore, ObserverBlindnessError
from observer.screenpipe_adapter import ScreenpipeAdapter
from observer.perception_engine import PerceptionEngine

# Crash-proof authority state
from state.serializer import AuthorityStateSerializer

# OS safety backend
from operate.utils.operating_system import OperatingSystem

# SOC entrypoint
from operate.main import main_entry as soc_execute_main

# RESTORATION
from restoration.snapshot_provider import SnapshotProvider
from restoration.restore_provider import RestoreProvider


HEARTBEAT_INTERVAL = 2.0

# --------------------------------------------------
# GLOBAL SINGLETONS
# --------------------------------------------------

OS_BACKEND = OperatingSystem()
STATE_PATH = os.path.join(os.getcwd(), ".authority_state.json")
AUTH_STATE = AuthorityStateSerializer(STATE_PATH)

SNAPSHOT_PROVIDER = SnapshotProvider(
    observer=None,
    screenpipe=None,
    os_backend=OS_BACKEND,
)
RESTORE_PROVIDER = RestoreProvider(os_backend=OS_BACKEND)


# --------------------------------------------------
# PROCESS SAFETY
# --------------------------------------------------

def _force_safe_shutdown(reason: str, *args, **kwargs):
    try:
        OS_BACKEND.force_release_all()
    except Exception:
        pass

    try:
        AUTH_STATE.force_safe_state()
    except Exception:
        pass

    print(f"[SAFE-SHUTDOWN] {reason}")


def _signal_handler(signum, frame):
    _force_safe_shutdown(f"signal:{signum}")
    os._exit(1)


atexit.register(_force_safe_shutdown, "atexit")
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGQUIT, _signal_handler)


# --------------------------------------------------
# ROOT MAIN
# --------------------------------------------------

def main():
    print("[BOOT] System starting")

    # ---- ENVIRONMENT FINGERPRINT ----
    env_fingerprint = collect_environment_fingerprint()
    print("[ENV] Fingerprint collected")
    for k, v in env_fingerprint.items():
        print(f"[ENV] {k}: {v}")

    # ---- LOAD AUTH STATE ----
    persisted = AUTH_STATE.load()

    if persisted.get("dirty") or persisted.get("restore_required"):
        print("[RECOVERY] Unsafe prior shutdown detected")
        OS_BACKEND.force_release_all()
        AUTH_STATE.force_safe_state()

    # ---- MODE CONTROLLER ----
    mode = ModeController()

    if persisted.get("dirty"):
        try:
            mode.force_observer()
        except Exception:
            pass

    # ---- INTENT LISTENER ----
    intent_listener = IntentListener(mode)
    intent_listener.start()

    # ---- OBSERVER STACK ----
    observer = ObserverCore()
    screenpipe = ScreenpipeAdapter()
    perception = PerceptionEngine()

    SNAPSHOT_PROVIDER._observer = observer
    SNAPSHOT_PROVIDER._screenpipe = screenpipe

    print(f"[STATE] Mode = {mode.mode.value}")
    print("[OBSERVER] Watching screen (read-only)")
    print("[INTENT] Type intent and press Enter")

    # --------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------

    while True:
        try:
            # ---- SCREEN + PERCEPTION ----
            screen_state = screenpipe.read()
            ui_snapshot = perception.process(screen_state)

            observer.attach_screen_state(screen_state)
            observer.attach_ui_snapshot(ui_snapshot)
            observer_state = observer.tick()

            # 🔧 FIX (NEW-002): update authority health + vision
            mode.update_vision_status(screen_state.get("available", False))
            mode.update_observer_health(observer.is_healthy())

            heartbeat = {
                "mode": mode.mode.value,
                "uptime": observer_state["uptime_seconds"],
                "ticks": observer_state["tick_count"],
                "screen_available": screen_state.get("available"),
                "screen_stale": screen_state.get("stale"),
                "ui_stable": ui_snapshot.stable,
                "ui_elements": len(ui_snapshot.elements),
                "ui_dialogs": len(ui_snapshot.dialogs),
            }

            print(f"[HEARTBEAT] {heartbeat}")

            # --------------------------------------------------
            # EXECUTION TRANSACTION (AUTHORITATIVE)
            # --------------------------------------------------

            if mode.is_armed():
                try:
                    # 🔒 AUTHORITATIVE MODE TRANSITION
                    mode.execute("root-main-execution")

                    # 🔧 FIX (NEW-003): consume intent immediately
                    current_intent = mode.consume_intent()

                except ModeTransitionError as e:
                    print(f"[EXECUTION] Transition blocked: {e}")
                    mode.force_observer()
                    continue

                print("[EXECUTION] Intent authorized — snapshotting")

                snapshot_id = SNAPSHOT_PROVIDER.take_snapshot()

                AUTH_STATE.persist(
                    execution_mode="EXECUTING",
                    automation_active=True,
                    restore_required=True,
                    last_snapshot_id=snapshot_id,
                    dirty=True,
                )

                try:
                    print("[EXECUTION] Launching SOC")

                    soc_execute_main(
                        model=None,
                        terminal_prompt=current_intent,
                        voice_mode=False,
                        verbose_mode=False,
                    )

                    print("[EXECUTION] SOC finished — restoring")

                    RESTORE_PROVIDER.restore_snapshot(snapshot_id)

                    AUTH_STATE.persist(
                        execution_mode="OBSERVER",
                        automation_active=False,
                        restore_required=False,
                        last_snapshot_id=None,
                        dirty=False,
                    )

                except Exception as e:
                    print(f"[EXECUTION] Fatal error: {e}")
                    AUTH_STATE.force_safe_state()
                    OS_BACKEND.force_release_all()

                    try:
                        RESTORE_PROVIDER.restore_snapshot(snapshot_id)
                    except Exception:
                        pass

                finally:
                    try:
                        mode.force_observer()
                    except Exception:
                        pass

        except ObserverBlindnessError:
            print("[FATAL] Observer blindness detected — shutting down")
            _force_safe_shutdown("observer-blindness")
            os._exit(1)

        except Exception as fatal:
            print(f"[FATAL] Main loop error: {fatal}")
            _force_safe_shutdown("main-loop-failure")
            os._exit(1)

        time.sleep(HEARTBEAT_INTERVAL)


if __name__ == "__main__":
    main()
