import time

from core.mode_controller import ModeController
from observer.observer_core import ObserverCore
from observer.screenpipe_adapter import ScreenpipeAdapter


HEARTBEAT_INTERVAL = 2.0


def main():
    print("[BOOT] System starting")

    mode = ModeController()
    observer = ObserverCore()
    screenpipe = ScreenpipeAdapter()

    print(f"[STATE] Mode = {mode.mode.value}")
    print("[OBSERVER] Live observation enabled (no execution)")

    while True:
        observer_state = observer.tick()
        screen_state = screenpipe.read()

        observer.attach_screen_state(screen_state)

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
