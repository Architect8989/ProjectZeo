import threading
import time


class IntentListener:
    """
    CLI intent ingestion.

    Rules:
    - Only accepts intent in OBSERVER mode
    - No intent overwrite
    - No spam during execution
    - Deterministic shutdown
    """

    POLL_INTERVAL = 0.1  # seconds

    def __init__(self, mode_controller):
        self.mode = mode_controller
        self._running = True
        self._thread = None

    def start(self):
        if self._thread is not None:
            return

        self._thread = threading.Thread(
            target=self._listen,
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._running = False

    def _listen(self):
        while self._running:
            try:
                # Only accept intent in OBSERVER mode
                if self.mode.mode.name != "OBSERVER":
                    time.sleep(self.POLL_INTERVAL)
                    continue

                raw = input()

                if raw is None:
                    self._running = False
                    return

                raw = raw.strip()
                if not raw:
                    continue

                # Single atomic authority call
                self.mode.arm(reason=raw)
                print(f"[INTENT] Armed: {raw}")

            except EOFError:
                self._running = False
                return

            except Exception as e:
                # Illegal transition, vision unavailable, etc.
                print(f"[INTENT] Rejected: {e}")
                time.sleep(self.POLL_INTERVAL)
