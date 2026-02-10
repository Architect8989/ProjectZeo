# observer/observer_loop.py

import threading
import time
import traceback
from typing import Optional

from observer.observer_core import ObserverCore, ObserverBlindnessError
from core.vision.vision_runtime import VisionRuntime
from core.vision.world_graph import WorldGraph


class ObserverLoop:
    """
    Continuous observer daemon.

    ROLE:
    - Keeps the system ALIVE
    - Continuously perceives the real OS screen
    - Builds world understanding passively
    - Feeds ObserverCore with ground truth
    - Never reasons
    - Never acts
    - Never plans

    THIS IS THE HEARTBEAT OF THE SYSTEM.
    """

    DEFAULT_TICK_INTERVAL = 0.20  # seconds (5 Hz baseline)

    def __init__(
        self,
        *,
        observer: ObserverCore,
        vision_runtime: VisionRuntime,
        world_graph: WorldGraph,
        tick_interval: float = DEFAULT_TICK_INTERVAL,
    ):
        self._observer = observer
        self._vision = vision_runtime
        self._world = world_graph
        self._tick_interval = max(0.05, float(tick_interval))

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._running = False

    # --------------------------------------------------
    # LIFECYCLE
    # --------------------------------------------------

    def start(self) -> None:
        if self._running:
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="observer-loop",
            daemon=True,
        )
        self._thread.start()
        self._running = True

    def stop(self) -> None:
        if not self._running:
            return

        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

        self._running = False

    # --------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------

    def _run(self) -> None:
        """
        Infinite passive perception loop.

        Failure semantics:
        - Vision failures mark screen unavailable
        - Persistent failures cause observer blindness
        - Loop NEVER exits unless stop() is called
        """
        while not self._stop_event.is_set():
            start_ts = time.monotonic()

            try:
                # 1. Capture + perceive (NO DECISIONS)
                perception = self._vision.capture_and_perceive()

                # perception schema (guaranteed by VisionRuntime):
                # {
                #   "available": bool,
                #   "frame_ts": float,
                #   "ui_snapshot": UISnapshot,
                #   "raw_frame": optional
                # }

                # 2. Update world model (pure accumulation)
                if perception.get("available"):
                    self._world.update(perception)

                # 3. Attach to observer core
                self._observer.attach_screen_state(
                    {
                        "available": perception.get("available", False),
                        "frame_ts": perception.get("frame_ts"),
                    }
                )

                self._observer.attach_ui_snapshot(
                    perception.get("ui_snapshot")
                )

                # 4. Advance observer clock
                self._observer.tick()

            except ObserverBlindnessError:
                # Observer blindness is authoritative.
                # Do not suppress, do not crash.
                pass

            except Exception:
                # Vision/runtime failures are treated as transient blindness
                try:
                    self._observer.attach_screen_state(
                        {"available": False, "frame_ts": None}
                    )
                except Exception:
                    pass

                # Never crash the loop
                traceback.print_exc()

            # 5. Tick pacing
            elapsed = time.monotonic() - start_ts
            sleep_for = self._tick_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    # --------------------------------------------------
    # INTROSPECTION
    # --------------------------------------------------

    def is_running(self) -> bool:
        return self._running
