import time

from core.mode_controller import ModeController
from observer.observer_core import ObserverCore
from observer.screenpipe_adapter import ScreenpipeAdapter


def main():
    print("[BOOT] System starting")

    # Authority + subsystems
    mode = ModeController()
    observer = ObserverCore()
    screenpipe = ScreenpipeAdapter()

    print(f"[STATE] Mode = {mode.mode}")
    print("[OBSERVER] Idle observer running (read-only)")

    while True:
        # Tick observer (witness)
        observer_state = observer.tick()

        # Poll screenpipe adapter (safe, may be unavailable)
        screen_state = screenpipe.read()

        # Structured heartbeat (single source of truth)
        heartbeat = {
            "mode": mode.mode,
            "uptime_seconds": observer_state.get("uptime_seconds"),
            "tick_count": observer_state.get("tick_count"),
            "screenpipe_available": screen_state.get("available"),
        }

        print(f"[HEARTBEAT] {heartbeat}")

        time.sleep(2)


if __name__ == "__main__":
    main()
