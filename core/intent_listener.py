from __future__ import annotations

import threading
import time
import sys
import select
import os
import stat
import tempfile
from typing import Optional


class IntentListener:

    POLL_INTERVAL = 0.1
    INTENT_MAX_BYTES = 4096

    # PATCH §1.7: replace hardcoded /tmp with platform-aware temp dir
    INTENT_FILE: str = os.path.join(tempfile.gettempdir(), "projectzeo.intent")

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
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

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

                try:
                    snapshot_id = self.snapshot_provider.take_snapshot()
                except Exception:
                    time.sleep(self.POLL_INTERVAL)
                    continue

                if not snapshot_id:
                    time.sleep(self.POLL_INTERVAL)
                    continue

                try:
                    self.mode.attach_snapshot(snapshot_id)
                except Exception:
                    time.sleep(self.POLL_INTERVAL)
                    continue

                try:
                    self.mode.arm(intent=intent)
                except Exception:
                    time.sleep(self.POLL_INTERVAL)
                    continue

            except Exception:
                time.sleep(self.POLL_INTERVAL)

    # ==================================================
    # Input sources
    # ==================================================

    def _read_intent(self) -> Optional[str]:

        # ---- Non-blocking interactive stdin ----
        if sys.stdin and sys.stdin.isatty():
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            except Exception:
                ready = []

            if ready:
                try:
                    data = sys.stdin.readline()
                except Exception:
                    return None

                if not data:
                    return None

                intent = data.strip()
                return intent if intent else None

        # ---- Secure file ingestion ----
        # PATCH §1.7: use self.INTENT_FILE (platform-aware) instead of /tmp hardcode
        path = self.INTENT_FILE

        if not os.path.exists(path):
            return None

        fd = None
        data = None
        should_delete = False

        try:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(path, flags)

            st = os.fstat(fd)

            if not stat.S_ISREG(st.st_mode):
                return None

            # PATCH §1.7: uid check is skipped on Windows where getuid() doesn't exist
            current_uid = getattr(os, "getuid", lambda: None)()
            if current_uid is not None and st.st_uid != current_uid:
                return None

            # FIX-03 (RTB-07): The original check `(st.st_mode & 0o077) != 0`
            # rejected any file with group or world read permissions. On Linux,
            # the default umask is 0o022, producing mode 0o644 (group-readable).
            # This caused intent files written with the default umask to be
            # silently consumed and discarded — the user's intent was lost with
            # no diagnostic. The README documented no chmod 600 requirement.
            #
            # Revised policy: only reject files that are world-writable (0o002)
            # or group-writable (0o020), which are genuine privilege-escalation
            # risks. Group-readable (0o040) and world-readable (0o004) files
            # are accepted, consistent with the default umask on most systems.
            if (st.st_mode & 0o022) != 0:
                return None

            if st.st_size <= 0 or st.st_size > self.INTENT_MAX_BYTES:
                return None

            raw = os.read(fd, self.INTENT_MAX_BYTES)
            data = raw.decode("utf-8", errors="strict")
            should_delete = True

        except Exception:
            return None

        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except Exception:
                    pass

        if should_delete:
            try:
                os.remove(path)
            except Exception:
                pass

        if not data:
            return None

        intent = data.strip()
        return intent if intent else None

