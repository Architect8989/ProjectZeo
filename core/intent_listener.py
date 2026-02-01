import threading


class IntentListener:
    def __init__(self, mode_controller):
        self.mode = mode_controller
        self._running = True

    def start(self):
        thread = threading.Thread(
            target=self._listen,
            daemon=True,
        )
        thread.start()

    def stop(self):
        """
        Explicit shutdown hook.
        """
        self._running = False

    def _listen(self):
        while self._running:
            try:
                raw = input()

                # stdin may return None or empty on some environments
                if raw is None:
                    print("[INTENT] stdin closed — stopping listener")
                    self._running = False
                    return

                raw = raw.strip()
                if not raw:
                    continue

                # 🔴 SINGLE AUTHORITY CALL — NO TOCTOU
                self.mode.arm(reason=raw)
                print(f"[INTENT] Armed via CLI: {raw}")

            except EOFError:
                # Deterministic shutdown on stdin close
                print("[INTENT] EOF received — stopping listener")
                self._running = False
                return

            except Exception as e:
                # Includes illegal transitions, vision errors, etc.
                print(f"[INTENT] Rejected: {e}")
