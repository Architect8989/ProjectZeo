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

            # FIX F-01 / RB-06: The prior check `(st.st_mode & 0o022) != 0` rejected
            # files with the GROUP-WRITE bit (0o020) set. On Linux with umask 002
            # (standard in many multi-user environments), files are created with mode
            # 0o664 — group-readable AND group-writable — causing every intent file to
            # be silently discarded with no diagnostic. The system would never arm.
            #
            # Corrected policy: reject ONLY genuinely dangerous permissions:
            #   - World-writable (0o002): any unprivileged user can modify the file,
            #     enabling privilege escalation via intent injection.
            #
            # Group-writable (0o020) is NOT rejected because:
            #   - It requires group membership to exploit (not arbitrary users).
            #   - It is the default permission produced by umask 002, which is
            #     standard on many developer and CI systems.
            #   - The README documents no requirement to chmod 600 intent files.
            #
            # Operators in high-security deployments should enforce umask 027 or
            # chmod 600 manually; this is not enforced at the code level to avoid
            # breaking the most common deployment configurations.
            if (st.st_mode & 0o002) != 0:
                print(
                    f"[IntentListener] Rejected intent file — world-writable: {path}",
                    file=sys.stderr,
                )
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

        # RB-A5 FIX: If data was successfully read but is empty or whitespace-only,
        # emit a diagnostic warning to stderr and do NOT delete the file.
        # Previously the file was silently consumed and discarded, leaving the
        # operator with no indication of why the system failed to arm. The operator
        # must re-write a valid intent to arm; discarding their (possibly accidental)
        # whitespace-only file without warning creates a silent failure mode.
        if should_delete and data is not None:
            intent = data.strip()
            if not intent:
                print(
                    f"[IntentListener] Discarded intent file — content is empty or "
                    f"whitespace-only: {path!r}. Re-write the file with a valid intent "
                    f"string to arm the system. File has NOT been deleted.",
                    file=sys.stderr,
                )
                # Do not delete: leave the file in place so the operator can inspect it.
                return None
            # Non-empty content — safe to delete and return.
            try:
                os.remove(path)
            except Exception:
                pass
            return intent

        if not data:
            return None

        intent = data.strip()
        return intent if intent else None
