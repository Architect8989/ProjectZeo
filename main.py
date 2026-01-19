import time
import os

from core.mode_controller import ModeController
from observer.observer_core import ObserverCore
from observer.screenpipe_adapter import ScreenpipeAdapter


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

    print(f"[STATE] Mode = {mode.mode.value}")
    print("[OBSERVER] Live observation running (no execution)")

    while True:
        observer_state = observer.tick()
        screen_state = screenpipe.read()

        observer.attach_screen_state(screen_state)

        check_for_intent(mode)

        heartbeat = {
            "mode": mode.mode.value,
            "uptime": observer_state["uptime_seconds"],
            "ticks": observer_state["tick_count"],
            "screen_available": screen_state["available"],
            "screen_stale": screen_state.get("stale"),
        }

        print(f"[HEARTBEAT] {heartbeat}")
        time.sleep(HEARTBEAT_INTERVAL)


if __name__ == "__main__":
    main()
