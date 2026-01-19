import time
from core.mode_controller import ModeController
from observer.observer_core import ObserverCore

def main():
    print("[BOOT] System starting")
    
    mode = ModeController()
    observer = ObserverCore()

    print(f"[STATE] Mode = {mode.mode}")
    print("[OBSERVER] Idle observer running")

    while True:
        observer.tick()
        print("[HEARTBEAT] Observer alive")
        time.sleep(2)

if __name__ == "__main__":
    main()
