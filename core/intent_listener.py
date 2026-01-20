import threading

class IntentListener:
    def __init__(self, mode_controller):
        self.mode = mode_controller
        self._running = True

    def start(self):
        thread = threading.Thread(target=self._listen, daemon=True)
        thread.start()

    def _listen(self):
        while self._running:
            try:
                raw = input().strip()
                if not raw:
                    continue

                if self.mode.mode.value == "OBSERVER":
                    self.mode.arm(reason=raw)
                    print(f"[INTENT] Armed via CLI: {raw}")
                else:
                    print("[INTENT] Ignored (already armed)")

            except Exception as e:
                print(f"[INTENT] CLI error: {e}")
