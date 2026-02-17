import threading
import time
import sys
import select
import os
import stat
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
    - Secure file-based ingestion
    """

    POLL_INTERVAL = 0.1  # seconds
    INTENT_FILE = "/tmp/projectzeo.intent"
    INTENT_MAX_BYTES = 4096

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
                if self.mode.mode.name != "OBSERVER":
                    time.sleep(self.POLL_INTERVAL)
                    continue

                intent = self._read_intent()

                if not intent:
                    time.sleep(self.POLL_INTERVAL)
                    continue

                # ---- SNAPSHOT FIRST ----
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

                # ---- ATTACH SNAPSHOT ----
                try:
                    self.mode.attach_snapshot(snapshot_id)
                except Exception as e:
                    print(f"[INTENT] Snapshot attach failed: {e}")
                    time.sleep(self.POLL_INTERVAL)
                    continue

                # ---- ARM ----
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
                ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            except Exception:
                ready = []

            if ready:
                line = sys.stdin.readline()
                if not line:
                    return None

                line = line.strip()
                return line if line else None

        # ---- secure file ingestion ----
        path = self.INTENT_FILE

        if not os.path.exists(path):
            return None

        try:
            st = os.stat(path, follow_symlinks=False)

            # Must be regular file
            if not stat.S_ISREG(st.st_mode):
                os.remove(path)
                return None

            # Must be owned by current user
            if st.st_uid != os.getuid():
                os.remove(path)
                return None

            # Size bounds
            if st.st_size <= 0 or st.st_size > self.INTENT_MAX_BYTES:
                os.remove(path)
                return None

            with open(path, "r", encoding="utf-8") as f:
                data = f.read(self.INTENT_MAX_BYTES)

        except Exception:
            try:
                os.remove(path)
            except Exception:
                pass
            return None

        # Remove after successful read
        try:
            os.remove(path)
        except Exception:
            pass

        intent = data.strip()
        return intent if intent else None
