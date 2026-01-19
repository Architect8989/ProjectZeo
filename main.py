from core.mode_controller import ModeController
from observer.observer_core import ObserverCore

def main():
    mode = ModeController()
    observer = ObserverCore()
    while True:
        observer.tick()

if __name__ == "__main__":
    main()
