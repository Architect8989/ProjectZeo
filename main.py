import time
import os

from core.mode_controller import ModeController
from observer.observer_core import ObserverCore
from observer.screenpipe_adapter import ScreenpipeAdapter
from observer.perception_engine import PerceptionEngine


HEARTBEAT_INTERVAL = 2.0
INTENT_FILE = "/tmp/arm_system.intent"


def check_for_intent(mode: ModeController):
    if not os.path.exists(INTENT_FILE):
        return

    try:
        with open(INTENT_FILE, "r") as f:
            reason = f.read().strip() or "explicit user intent"

        if mode.mode.value == "OBSERVER":
            mode.arm(reason=reason)
            print("[INTENT] System armed")

    except Exception as e:
        print(f"[INTENT] Error reading intent: {e}")


def main():
    print("[BOOT] System starting")

    mode = ModeController()
    observer = ObserverCore()
    screenpipe = ScreenpipeAdapter()
    perception = PerceptionEngine()

    print(f"[STATE] Mode = {mode.mode.value}")
    print("[OBSERVER] Live observation + understanding (no execution)")

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

        # 5. Check explicit human intent (arming only)
        check_for_intent(mode)

        # 6. Structured heartbeat
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
