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

# NEW: crash-proof authority state
from state.serializer import AuthorityStateSerializer

# NEW: OS safety backend
from operate.utils.operating_system import OperatingSystem

# NEW: SOC entrypoint (transactional execution)
from operate.main import main as soc_execute_main


HEARTBEAT_INTERVAL = 2.0

# --------------------------------------------------
# GLOBAL SINGLETONS (ROOT OWNS THESE)
# --------------------------------------------------

OS_BACKEND = OperatingSystem()
STATE_PATH = os.path.join(os.getcwd(), ".authority_state.json")
AUTH_STATE = AuthorityStateSerializer(STATE_PATH)


# --------------------------------------------------
# PROCESS-LEVEL SAFETY
# --------------------------------------------------

def _force_safe_shutdown(reason: str):
    """
    Absolute last-resort safety.
    Must be safe to call multiple times.
    """
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


# Register safety hooks
atexit.register(_force_safe_shutdown, "atexit")
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGQUIT, _signal_handler)


# --------------------------------------------------
# ROOT MAIN
# --------------------------------------------------

def main():
    print("[BOOT] System starting")

    # --- Phase 0: Environment fingerprint (one-time, read-only) ---
    env_fingerprint = collect_environment_fingerprint()
    print("[ENV] Fingerprint collected")
    for k, v in env_fingerprint.items():
        print(f"[ENV] {k}: {v}")

    # --- Load crash-proof authority state ---
    persisted = AUTH_STATE.load()

    if persisted.get("dirty") or persisted.get("restore_required"):
        print("[RECOVERY] Unsafe prior shutdown detected")
        # Force pessimistic state immediately
        OS_BACKEND.force_release_all()
        AUTH_STATE.force_safe_state()

    # --- Authority core ---
    mode = ModeController()

    # Force OBSERVER if coming from recovery
    if persisted.get("dirty"):
        try:
            mode.force_observer()
        except Exception:
            pass

    # --- Intent input (CLI) ---
    intent_listener = IntentListener(mode)
    intent_listener.start()

    # --- Observer + perception stack ---
    observer = ObserverCore()
    screenpipe = ScreenpipeAdapter()
    perception = PerceptionEngine()

    print(f"[STATE] Mode = {mode.mode.value}")
    print("[OBSERVER] Live observation + understanding (no execution)")
    print("[INTENT] Type intent and press Enter to arm system")

    # --------------------------------------------------
    # MAIN OBSERVER LOOP (DAEMON)
    # --------------------------------------------------

    while True:
        # 1. Witness time
        observer_state = observer.tick()

        # 2. Read raw screen feed (read-only)
        screen_state = screenpipe.read()

        # 3. Derive semantic understanding (read-only)
        ui_snapshot = perception.process(screen_state)

        # 4. Attach perception to observer truth
        observer.attach_screen_state(screen_state)
        observer.attach_ui_snapshot(ui_snapshot)

        # 5. Structured heartbeat (truth only)
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
        # EXECUTION HANDOFF (ROOT → SOC)
        # --------------------------------------------------

        if mode.is_armed():
            print("[EXECUTION] Intent armed — invoking SOC")

            # Persist authority transition BEFORE execution
            AUTH_STATE.persist(
                execution_mode="EXECUTING",
                automation_active=True,
                restore_required=True,
                last_snapshot_id=None,
                dirty=True,
            )

            try:
                soc_execute_main(
                    model=None,
                    terminal_prompt=mode.consume_intent(),
                    voice_mode=False,
                    verbose_mode=False,
                )

                # Successful completion → safe state
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

            finally:
                # Always return to OBSERVER
                try:
                    mode.force_observer()
                except Exception:
                    pass

        time.sleep(HEARTBEAT_INTERVAL)


if __name__ == "__main__":
    main()
