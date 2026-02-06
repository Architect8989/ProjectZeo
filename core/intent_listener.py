import threading
import time
import sys
import select
import os
from typing import Optional


class IntentListener:
    """
    Deterministic intent ingestion.

    Guarantees:
    - NEVER blocks main thread
    - Accepts intent ONLY in OBSERVER mode
    - No intent overwrite
    - Works in interactive + non-interactive environments
    - Clean shutdown
    """

    POLL_INTERVAL = 0.1  # seconds
    INTENT_FILE = "/tmp/projectzeo.intent"  # optional external trigger

    def __init__(self, mode_controller):
        self.mode = mode_controller
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ==================================================
    # Lifecycle
    # ==================================================

    def start(self):
        if self._thread is not None:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._listen_loop,
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._running = False

    # ==================================================
    # Core loop
    # ==================================================

    def _listen_loop(self):
        while self._running:
            try:
                # Only accept intent in OBSERVER mode
                if self.mode.mode.name != "OBSERVER":
                    time.sleep(self.POLL_INTERVAL)
                    continue

                intent = self._read_intent()

                if not intent:
                    time.sleep(self.POLL_INTERVAL)
                    continue

                # Atomic authority transition
                self.mode.arm(intent=intent)
                print(f"[INTENT] Armed: {intent}")

            except Exception as e:
                # Illegal transition, mode violation, etc.
                print(f"[INTENT] Rejected: {e}")
                time.sleep(self.POLL_INTERVAL)

    # ==================================================
    # Input sources
    # ==================================================

    def _read_intent(self) -> Optional[str]:
        """
        Tries multiple non-blocking intent sources:
        1. Interactive stdin (TTY)
        2. Intent file (non-interactive / automation)

        Returns intent string or None.
        """

        # ---- interactive stdin ----
        if sys.stdin and sys.stdin.isatty():
            ready, _, _ = select.select(
                [sys.stdin],
                [],
                [],
                0.0,
            )

            if ready:
                line = sys.stdin.readline()
                if not line:
                    return None

                line = line.strip()
                return line if line else None

        # ---- non-interactive intent file ----
        if os.path.exists(self.INTENT_FILE):
            try:
                with open(self.INTENT_FILE, "r", encoding="utf-8") as f:
                    data = f.read().strip()

                os.remove(self.INTENT_FILE)

                return data if data else None

            except Exception:
                return None

        return None
