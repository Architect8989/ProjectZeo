import threading
import time
import traceback
from typing import Optional, Dict, Any

from observer.observer_core import ObserverCore, ObserverBlindnessError
from core.vision.vision_runtime import VisionRuntime


class ObserverLoop:
    """
    Continuous observer daemon.

    ROLE:
    - Continuously pulls perception from VisionRuntime
    - Feeds ObserverCore with perception state only
    - NEVER mutates world state
    - NEVER plans
    - NEVER acts
    - NEVER changes mode
    """

    DEFAULT_TICK_INTERVAL = 0.20  # 5 Hz baseline

    def __init__(
        self,
        *,
        observer: ObserverCore,
        vision_runtime: VisionRuntime,
        tick_interval: float = DEFAULT_TICK_INTERVAL,
    ):
        self._observer = observer
        self._vision = vision_runtime
        self._tick_interval = max(0.05, float(tick_interval))

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self._lock = threading.Lock()

    # ==================================================
    # LIFECYCLE
    # ==================================================

    def start(self) -> None:
        with self._lock:
            if self._running:
                return

            self._stop_event.clear()
            self._running = True

            self._thread = threading.Thread(
                target=self._run,
                name="observer-loop",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return

            self._stop_event.set()
            thread = self._thread
            self._thread = None
            self._running = False

        if thread:
            thread.join(timeout=2.0)

    # ==================================================
    # MAIN LOOP
    # ==================================================

    def _run(self) -> None:
        while not self._stop_event.is_set():
            start_ts = time.monotonic()

            try:
                perception: Optional[Dict[str, Any]] = self._vision.get_latest()

                if (
                    isinstance(perception, dict)
                    and perception.get("available") is True
                ):
                    # Observer receives RAW perception only.
                    self._observer.attach_perception_state(
                        {
                            "available": True,
                            "frame_ts": perception.get("frame_ts"),
                            "perception": perception,
                        }
                    )
                else:
                    self._observer.attach_perception_state(
                        {
                            "available": False,
                            "frame_ts": None,
                            "perception": None,
                        }
                    )

                # Authoritative tick
                self._observer.tick()

            except ObserverBlindnessError:
                # Blindness is authoritative inside ObserverCore.
                pass

            except Exception:
                traceback.print_exc()

                try:
                    self._observer.attach_perception_state(
                        {
                            "available": False,
                            "frame_ts": None,
                            "perception": None,
                        }
                    )
                except Exception:
                    pass

            elapsed = time.monotonic() - start_ts
            sleep_for = self._tick_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    # ==================================================
    # INTROSPECTION
    # ==================================================

    def is_running(self) -> bool:
        with self._lock:
            return self._running
