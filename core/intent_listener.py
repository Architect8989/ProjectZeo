from __future__ import annotations

import threading
import time
import sys
import select
import os
import stat
import pathlib
from typing import Optional


def atomic_write_intent(intent_text: str, intent_file: str = None) -> None:
    """
    Write an intent string to the intent file atomically.

    RB-06 FIX: The IntentListener polls at 100ms. A non-atomic write (open,
    write, close) creates a race window where the listener can read a partial
    intent if it polls between the open and the final close.

    Fix: write to a .tmp sibling file, then os.replace() to the target path.
    os.replace() is atomic on POSIX (rename syscall) and atomic on Windows
    (MoveFileExW with MOVEFILE_REPLACE_EXISTING). The listener only ever
    reads complete files.
    """
    if intent_file is None:
        intent_file = IntentListener.INTENT_FILE

    encoded = intent_text.encode("utf-8")
    if len(encoded) > IntentListener.INTENT_MAX_BYTES:
        raise ValueError(
            f"Intent text exceeds {IntentListener.INTENT_MAX_BYTES} byte limit "
            f"({len(encoded)} bytes encoded)"
        )
    if not encoded:
        raise ValueError("Intent text must be non-empty")

    tmp_path = intent_file + ".tmp"
    try:
        with open(tmp_path, "wb") as f:
            f.write(encoded)
        os.replace(tmp_path, intent_file)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class IntentListener:

    POLL_INTERVAL = 0.1
    INTENT_MAX_BYTES = 4096

    # FIX RB-3: Replace tempfile.gettempdir() path with a project-relative path.
    #
    # Bug: INTENT_FILE = os.path.join(tempfile.gettempdir(), "projectzeo.intent")
    # resolves to /tmp/projectzeo.intent on Linux.  The bundled intent file
    # lives at temp/arm_system.intent (relative to project root).  The listener
    # polled /tmp/projectzeo.intent while the file to be read was elsewhere —
    # the system never entered ARMED state in default deployment.
    #
    # Fix: resolve the path relative to this file's location at import time.
    # This is robust against:
    #   - Different working directories at process startup
    #   - Containerised deployments that mount the project at a non-CWD path
    #   - Operator scripts that write to temp/arm_system.intent expecting it to
    #     be picked up automatically
    #
    # Deployment note: operators can override INTENT_FILE after import by
    # assigning IntentListener.INTENT_FILE = "/custom/path" before calling
    # IntentListener.__init__(). This class attribute is mutable.
    _PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
    INTENT_FILE: str = str(_PROJECT_ROOT / "temp" / "arm_system.intent")

    # ARM_PREFIX: content starting with this prefix has the prefix stripped
    # before use as the task objective.  The bundled intent file ships with
    # "ARM: investigate deployment issue"; the "ARM: " is a human-readable
    # signal, not part of the actual task description that the LLM receives.
    _ARM_PREFIX = "ARM:"

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
                    raw = sys.stdin.readline()
                    if len(raw.encode("utf-8", errors="replace")) > self.INTENT_MAX_BYTES:
                        print(
                            f"[IntentListener] Discarded stdin intent — exceeds "
                            f"{self.INTENT_MAX_BYTES} byte limit.",
                            file=sys.stderr,
                        )
                        return None
                    data = raw
                except Exception:
                    return None

                if not data:
                    return None

                intent = data.strip()
                # FIX RB-3: Strip ARM: prefix from stdin input too
                intent = self._strip_arm_prefix(intent)
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

            current_uid = getattr(os, "getuid", lambda: None)()
            if current_uid is not None and st.st_uid != current_uid:
                return None

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

        if should_delete and data is not None:
            intent = data.strip()
            if not intent:
                print(
                    f"[IntentListener] Discarded intent file — content is empty or "
                    f"whitespace-only: {path!r}. Re-write the file with a valid intent "
                    f"string to arm the system. File has NOT been deleted.",
                    file=sys.stderr,
                )
                return None

            # FIX RB-3: Strip "ARM: " prefix before using as task objective.
            # The bundled intent file ships with "ARM: investigate deployment issue".
            # Without stripping, the literal string "ARM: " becomes the first word
            # of the LLM prompt, degrading plan quality.
            intent = self._strip_arm_prefix(intent)
            if not intent:
                print(
                    f"[IntentListener] Discarded intent file — content was only the "
                    f"ARM: prefix with no task text: {path!r}.",
                    file=sys.stderr,
                )
                return None

            try:
                os.remove(path)
            except Exception:
                pass
            return intent

        if not data:
            return None

        intent = data.strip()
        intent = self._strip_arm_prefix(intent)
        return intent if intent else None

    # ==================================================
    # Helpers
    # ==================================================

    def _strip_arm_prefix(self, text: str) -> str:
        """
        FIX RB-3: Strip the 'ARM:' prefix if present (case-insensitive).

        The bundled temp/arm_system.intent ships with:
            ARM: investigate deployment issue

        Without stripping, "ARM: " becomes the first word of the task objective
        sent to the LLM, degrading plan quality and causing confusing logs.
        """
        stripped = text.strip()
        if stripped.upper().startswith(self._ARM_PREFIX):
            stripped = stripped[len(self._ARM_PREFIX):].strip()
        return stripped
