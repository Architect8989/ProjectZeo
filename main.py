import time
import os
import signal
import atexit

from core.mode_controller import ModeController
from core.intent_listener import IntentListener
from core.environment_fingerprint import collect_environment_fingerprint

from observer.observer_core import ObserverCore
from observer.screenpipe_adapter import ScreenpipeAdapter
from observer.perception_engine import PerceptionEngine

# Crash-proof authority state
from state.serializer import AuthorityStateSerializer

# OS safety backend
from operate.utils.operating_system import OperatingSystem

# SOC entrypoint
from operate.main import main as soc_execute_main

# 🔥 RESTORATION (NEW WIRES)
from restoration.snapshot_provider import take_snapshot
from restoration.restore_provider import restore_snapshot


HEARTBEAT_INTERVAL = 2.0

# --------------------------------------------------
# GLOBAL SINGLETONS
# --------------------------------------------------

OS_BACKEND = OperatingSystem()
STATE_PATH = os.path.join(os.getcwd(), ".authority_state.json")
AUTH_STATE = AuthorityStateSerializer(STATE_PATH)


# --------------------------------------------------
# PROCESS SAFETY
# --------------------------------------------------

def _force_safe_shutdown(reason: str):
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

    print(f"[STATE] Mode = {mode.mode.value}")
    print("[OBSERVER] Watching screen (read-only)")
    print("[INTENT] Type intent and press Enter")

    # --------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------

    while True:

        # 1. Observer heartbeat
        observer_state = observer.tick()

        # 2. Screen feed
        screen_state = screenpipe.read()

        # 3. Perception
        ui_snapshot = perception.process(screen_state)

        # 4. Attach to observer
        observer.attach_screen_state(screen_state)
        observer.attach_ui_snapshot(ui_snapshot)

        # 5. Heartbeat log
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
        # EXECUTION TRANSACTION
        # --------------------------------------------------

        if mode.is_armed():
            print("[EXECUTION] Intent armed — snapshotting")

            # ---- SNAPSHOT BEFORE HIJACK ----
            snapshot_id = take_snapshot()

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
                    terminal_prompt=mode.consume_intent(),
                    voice_mode=False,
                    verbose_mode=False,
                )

                print("[EXECUTION] SOC finished — restoring")

                restore_snapshot(snapshot_id)

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

                # Best effort restore
                try:
                    restore_snapshot(snapshot_id)
                except Exception:
                    pass

            finally:
                try:
                    mode.force_observer()
                except Exception:
                    pass

        time.sleep(HEARTBEAT_INTERVAL)


if __name__ == "__main__":
    main()
