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
    Write intent text atomically to the intent file.

    RB-4 / H5 FIX: Explicitly set the temp file mode to 0o644 before rename.
    Under umask 0000 (Docker --privileged default), files created without an
    explicit chmod inherit mode 0666 (world-writable). _peek_intent() rejects
    world-writable (bit 0o002) and group-writable (bit 0o020) files. Without
    this chmod, every intent file written inside a privileged container was
    permanently rejected, leaving the system in OBSERVER indefinitely.

    os.chmod() is called BEFORE os.replace() so the file is never visible on
    the filesystem with the wrong mode — os.replace() is atomic on POSIX.
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
        # RB-4 FIX: Force mode 0o644 before atomic rename so the intent file
        # is never world-writable (0o002) or group-writable (0o020) at rest.
        try:
            os.chmod(tmp_path, 0o644)
        except OSError:
            pass  # Best-effort — chmod may not be supported on some filesystems
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

    # RB-CRIT-3 FIX: INTENT_FILE was hardcoded to
    #   <project_root>/temp/arm_system.intent
    # but the shipped intent file lives at:
    #   <project_root>/arm_system.intent  (project root, no subdirectory)
    #
    # Runtime verification:
    #   $ ls ProjectZeo-main/arm_system.intent      → exists (29 bytes)
    #   $ ls ProjectZeo-main/temp/arm_system.intent → no such file
    #
    # _listen_loop() called os.path.exists(self.INTENT_FILE) on every poll
    # tick. With the wrong path this always returned False, leaving the system
    # in OBSERVER mode indefinitely. Arming via the file mechanism was
    # permanently blocked regardless of the file content or permissions.
    #
    # Fix: point INTENT_FILE at the project root (no temp/ subdirectory).
    # The atomic_write_intent() helper and sidecar writers use os.path.dirname()
    # on INTENT_FILE to locate the sidecar directory — they are automatically
    # correct after this change (they now write sidecars to the project root,
    # which is acceptable; the temp/ subdirectory serves no operational purpose).
    _PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
    INTENT_FILE: str = str(_PROJECT_ROOT / "arm_system.intent")

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

                
                intent, intent_file_path = self._peek_intent()

                if not intent:
                    time.sleep(self.POLL_INTERVAL)
                    continue

                # ---- Arm sequence ----
                arm_failure_reason: Optional[str] = None
                snapshot_id: Optional[str] = None

                try:
                    snapshot_id = self.snapshot_provider.take_snapshot()
                except Exception as _snap_err:
                    arm_failure_reason = f"snapshot_failed: {_snap_err}"

                if arm_failure_reason is None and not snapshot_id:
                    arm_failure_reason = "snapshot_returned_none"

                if arm_failure_reason is None:
                    try:
                        self.mode.attach_snapshot(snapshot_id)
                    except Exception as _attach_err:
                        arm_failure_reason = f"attach_snapshot_failed: {_attach_err}"

                if arm_failure_reason is None:
                    try:
                        self.mode.arm(intent=intent)
                    except Exception as _arm_err:
                        arm_failure_reason = f"arm_failed: {_arm_err}"

                if arm_failure_reason is not None:
                    # Arming failed — leave intent file in place so operator
                    # does not have to re-write it. Write a sidecar so the
                    # failure reason is visible without watching stderr.
                    self._write_arm_failure_sidecar(arm_failure_reason, intent)
                    print(
                        f"[IntentListener] Arm sequence failed ({arm_failure_reason}). "
                        "Intent file preserved — system will retry on next poll.",
                        file=sys.stderr,
                    )
                    time.sleep(self.POLL_INTERVAL)
                    continue

                # ---- Arm succeeded — now safe to consume (delete) the intent file ----
                self._consume_intent(intent_file_path)
                self._write_arm_success_sidecar()

            except Exception:
                time.sleep(self.POLL_INTERVAL)

    # ==================================================
    # Input sources
    # ==================================================

    def _peek_intent(self) -> tuple:
        """
        RB-6 FIX: Read the intent without deleting the file.

        Returns (intent_text, file_path_or_None).
          - intent_text: the stripped, prefix-cleaned intent string, or None
          - file_path_or_None: the path of the intent file that was read
            (None for stdin input or when no intent was found)

        The caller is responsible for calling _consume_intent(file_path)
        after successfully arming, and for calling _write_arm_failure_sidecar()
        on failure.
        """
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
                        return None, None
                    data = raw
                except Exception:
                    return None, None

                if not data:
                    return None, None

                intent = self._strip_arm_prefix(data.strip())
                return (intent if intent else None), None

        # ---- Secure file peek (no deletion) ----
        path = self.INTENT_FILE

        if not os.path.exists(path):
            return None, None

        fd = None
        data = None

        try:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(path, flags)

            st = os.fstat(fd)

            if not stat.S_ISREG(st.st_mode):
                return None, None

            current_uid = getattr(os, "getuid", lambda: None)()
            if current_uid is not None and st.st_uid != current_uid:
                return None, None

            if (st.st_mode & 0o002) != 0:
                print(
                    f"[IntentListener] Rejected intent file — world-writable: {path}",
                    file=sys.stderr,
                )
                return None, None

            # RB-4 / H5 FIX: Also reject group-writable files (bit 0o020).
            # The original check only blocked world-writable (0o002). Under
            # umask 0022 (standard Linux) files are 0644 — accepted. But
            # under non-standard group-write umasks (e.g. 0o002) files can be
            # 0664 (group-writable) and should be treated as potentially unsafe:
            # any member of the file's group can overwrite the intent and inject
            # an arbitrary task objective. atomic_write_intent() now always
            # chmods to 0o644 so this is defence-in-depth for external writers.
            if (st.st_mode & 0o020) != 0:
                print(
                    f"[IntentListener] Rejected intent file — group-writable: {path}",
                    file=sys.stderr,
                )
                return None, None

            if st.st_size <= 0 or st.st_size > self.INTENT_MAX_BYTES:
                return None, None

            raw = os.read(fd, self.INTENT_MAX_BYTES)
            data = raw.decode("utf-8", errors="strict")

        except Exception:
            return None, None

        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except Exception:
                    pass

        if data is None:
            return None, None

        intent = self._strip_arm_prefix(data.strip())
        if not intent:
            print(
                f"[IntentListener] Discarded intent file — content is empty or "
                f"whitespace-only after stripping ARM: prefix: {path!r}. "
                f"Re-write the file with a valid intent string. File NOT deleted.",
                file=sys.stderr,
            )
            return None, None

        return intent, path

    def _consume_intent(self, file_path: Optional[str]) -> None:
        """
        RB-6 FIX: Delete the intent file after a successful arm.

        Only called when mode.arm() has returned without raising, so the
        intent is guaranteed to have been acted upon before it disappears.
        """
        if file_path is None:
            return  # stdin input — nothing to delete
        try:
            os.remove(file_path)
        except Exception:
            pass  # Best-effort; file may already be absent

    def _write_arm_failure_sidecar(
        self, reason: str, intent: str
    ) -> None:
        """
        RB-6 FIX: Write a structured JSON sidecar on arm failure.

        Path: temp/arm_failure.json (sibling of the intent file directory).
        Operators watching temp/ can detect failures without tailing stderr.
        """
        import json as _json
        _sidecar_dir = os.path.dirname(self.INTENT_FILE)
        _sidecar_path = os.path.join(_sidecar_dir, "arm_failure.json")
        try:
            os.makedirs(_sidecar_dir, exist_ok=True)
            payload = {
                "event": "arm_failure",
                "reason": reason,
                "intent_preview": intent[:80],
                "timestamp": time.time(),
            }
            _tmp = _sidecar_path + ".tmp"
            with open(_tmp, "w", encoding="utf-8") as _f:
                _json.dump(payload, _f)
            os.replace(_tmp, _sidecar_path)
        except Exception:
            pass  # Sidecar write is best-effort

    def _write_arm_success_sidecar(self) -> None:
        """
        RB-6 FIX: Write a structured JSON sidecar on successful arm.

        Clears any prior arm_failure.json and writes arm_success.json so
        monitoring tools can distinguish a recovered system from a stuck one.
        """
        import json as _json
        _sidecar_dir = os.path.dirname(self.INTENT_FILE)
        _success_path = os.path.join(_sidecar_dir, "arm_success.json")
        _failure_path = os.path.join(_sidecar_dir, "arm_failure.json")
        try:
            os.makedirs(_sidecar_dir, exist_ok=True)
            payload = {"event": "arm_success", "timestamp": time.time()}
            _tmp = _success_path + ".tmp"
            with open(_tmp, "w", encoding="utf-8") as _f:
                _json.dump(payload, _f)
            os.replace(_tmp, _success_path)
            # Clear stale failure sidecar if present
            try:
                os.remove(_failure_path)
            except OSError:
                pass
        except Exception:
            pass  # Sidecar write is best-effort

    # ==================================================
    # Helpers
    # ==================================================

    def _strip_arm_prefix(self, text: str) -> str:
        
        stripped = text.strip()
        if stripped.upper().startswith(self._ARM_PREFIX):
            stripped = stripped[len(self._ARM_PREFIX):].strip()
        return stripped
                
