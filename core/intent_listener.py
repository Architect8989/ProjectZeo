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
    - Snapshot taken BEFORE arming
    - No intent overwrite
    - Works in interactive + non-interactive environments
    - Clean shutdown
    """

    POLL_INTERVAL = 0.1  # seconds
    INTENT_FILE = "/tmp/projectzeo.intent"

    def __init__(self, mode_controller, snapshot_provider):
        self.mode = mode_controller
        self.snapshot_provider = snapshot_provider
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
                # Accept intent ONLY in OBSERVER mode
                if self.mode.mode.name != "OBSERVER":
                    time.sleep(self.POLL_INTERVAL)
                    continue

                intent = self._read_intent()

                if not intent:
                    time.sleep(self.POLL_INTERVAL)
                    continue

                # ---- SNAPSHOT FIRST (HARD GUARANTEE) ----
                try:
                    snapshot_id = self.snapshot_provider.take_snapshot()
                except Exception as e:
                    print(f"[INTENT] Snapshot failed, rejecting: {e}")
                    time.sleep(self.POLL_INTERVAL)
                    continue

                if not snapshot_id:
                    print("[INTENT] Snapshot returned invalid ID, rejecting")
                    time.sleep(self.POLL_INTERVAL)
                    continue

                # ---- AUTHORITATIVE SNAPSHOT ATTACH ----
                try:
                    self.mode.attach_snapshot(snapshot_id)
                except Exception as e:
                    print(f"[INTENT] Snapshot attach failed: {e}")
                    time.sleep(self.POLL_INTERVAL)
                    continue

                # ---- THEN ARM ----
                try:
                    self.mode.arm(intent=intent)
                except Exception as e:
                    print(f"[INTENT] Arm rejected: {e}")
                    time.sleep(self.POLL_INTERVAL)
                    continue

                print(f"[INTENT] Armed: {intent}")

            except Exception as e:
                print(f"[INTENT] Rejected: {e}")
                time.sleep(self.POLL_INTERVAL)

    # ==================================================
    # Input sources
    # ==================================================

    def _read_intent(self) -> Optional[str]:

        # ---- interactive stdin ----
        if sys.stdin and sys.stdin.isatty():
            try:
                ready, _, _ = select.select(
                    [sys.stdin],
                    [],
                    [],
                    0.0,
                )
            except Exception:
                ready = []

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
