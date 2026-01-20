import time

from core.mode_controller import ModeController
from core.intent_listener import IntentListener

from observer.observer_core import ObserverCore
from observer.screenpipe_adapter import ScreenpipeAdapter
from observer.perception_engine import PerceptionEngine


HEARTBEAT_INTERVAL = 2.0


def main():
    print("[BOOT] System starting")

    # Authority core
    mode = ModeController()

    # Intent input (CLI)
    intent_listener = IntentListener(mode)
    intent_listener.start()

    # Observer + perception stack
    observer = ObserverCore()
    screenpipe = ScreenpipeAdapter()
    perception = PerceptionEngine()

    print(f"[STATE] Mode = {mode.mode.value}")
    print("[OBSERVER] Live observation + understanding (no execution)")
    print("[INTENT] Type intent and press Enter to arm system")

    while True:
        # 1. Witness time
        observer_state = observer.tick()

        # 2. Read raw screen feed
        screen_state = screenpipe.read()

        # 3. Phase 2B: derive semantic understanding (read-only)
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
        time.sleep(HEARTBEAT_INTERVAL)


if __name__ == "__main__":
    main()
