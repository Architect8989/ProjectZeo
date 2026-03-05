from __future__ import annotations

import threading
import time
import sys
import select
import os
import stat
import pathlib
import json
import collections
from typing import Optional, Deque


def atomic_write_intent(intent_text: str, intent_file: str = None) -> None:
    
    if intent_file is None:
        intent_file = IntentListener.INTENT_FILE

    # Ensure the containing directory exists with restricted permissions.
    _dir = os.path.dirname(intent_file)
    try:
        os.makedirs(_dir, mode=0o700, exist_ok=True)
        os.chmod(_dir, 0o700)
    except OSError:
        pass

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
        try:
            os.chmod(tmp_path, 0o600)
        except OSError:
            pass
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

    
    _SECURE_DIR: str = os.path.join(os.path.expanduser("~"), ".projectzeo")
    INTENT_FILE: str = os.path.join(_SECURE_DIR, "arm_system.intent")

    
    _SIDECAR_DIR: str = _SECURE_DIR

    _ARM_PREFIX = "ARM:"

    def __init__(self, mode_controller, snapshot_provider):
        self.mode = mode_controller
        self.snapshot_provider = snapshot_provider
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._task_queue: Deque[str] = collections.deque()

        # Ensure the secure directory exists on construction.
        try:
            os.makedirs(self._SECURE_DIR, mode=0o700, exist_ok=True)
            os.chmod(self._SECURE_DIR, 0o700)
        except OSError as _dir_err:
            print(
                f"[IntentListener] WARNING: Could not create secure intent dir "
                f"{self._SECURE_DIR!r}: {_dir_err}. "
                "Intent file security may be degraded.",
                file=sys.stderr,
            )

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
        _consecutive_arm_failures: int = 0
        _BACKOFF_BASE: float = 0.1
        _BACKOFF_MAX: float = 30.0

        while self._running:
            try:
                if self.mode.mode.name != "OBSERVER":
                    _consecutive_arm_failures = 0
                    time.sleep(self.POLL_INTERVAL)
                    continue

                intent: Optional[str] = None
                intent_file_path: Optional[str] = None

                if self._task_queue:
                    intent = self._task_queue.popleft()
                else:
                    intent, intent_file_path = self._peek_intent()

                    if intent and intent.lstrip().startswith("["):
                        try:
                            candidates = json.loads(intent)
                            if (
                                isinstance(candidates, list)
                                and candidates
                                and all(isinstance(i, str) for i in candidates)
                            ):
                                clean = [
                                    i.strip() for i in candidates
                                    if i.strip() and not self._contains_injection(i)
                                ]
                                if len(clean) < len(candidates):
                                    print(
                                        f"[IntentListener] Dropped "
                                        f"{len(candidates) - len(clean)} queued intent(s) "
                                        "that contained injection markers.",
                                        file=sys.stderr,
                                    )
                                if clean:
                                    print(
                                        f"[IntentListener] Queuing "
                                        f"{len(clean)} intent(s) for sequential execution: "
                                        + ", ".join(repr(i[:40]) for i in clean),
                                        file=sys.stderr,
                                    )
                                    self._task_queue.extend(clean[1:])
                                    intent = clean[0]
                                else:
                                    intent = None
                        except (json.JSONDecodeError, ValueError):
                            pass

                if not intent:
                    _consecutive_arm_failures = 0
                    time.sleep(self.POLL_INTERVAL)
                    continue

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
                    _consecutive_arm_failures += 1
                    _backoff = min(
                        _BACKOFF_BASE * (2 ** (_consecutive_arm_failures - 1)),
                        _BACKOFF_MAX,
                    )
                    self._write_arm_failure_sidecar(arm_failure_reason, intent)
                    print(
                        f"[IntentListener] Arm sequence failed ({arm_failure_reason}). "
                        f"Consecutive failures: {_consecutive_arm_failures}. "
                        f"Retrying in {_backoff:.1f}s.",
                        file=sys.stderr,
                    )
                    self._task_queue.appendleft(intent)
                    time.sleep(_backoff)
                    continue

                _consecutive_arm_failures = 0
                self._consume_intent(intent_file_path)
                self._write_arm_success_sidecar()

                if self._task_queue:
                    print(
                        f"[IntentListener] Armed intent {intent[:60]!r}. "
                        f"{len(self._task_queue)} task(s) remain in queue.",
                        file=sys.stderr,
                    )

            except Exception:
                time.sleep(self.POLL_INTERVAL)

    # ==================================================
    # Input sources
    # ==================================================

    def _peek_intent(self) -> tuple:
        
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
                if not intent:
                    return None, None

                if self._contains_injection(intent):
                    print(
                        "[IntentListener] SECURITY: Rejected stdin intent — "
                        "injection marker detected.  Intent NOT armed.",
                        file=sys.stderr,
                    )
                    return None, None

                return intent, None

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
                f"[IntentListener] Discarded intent file — empty after stripping "
                f"ARM: prefix: {path!r}. Re-write with a valid intent string.",
                file=sys.stderr,
            )
            return None, None

        if self._contains_injection(intent):
            preview = intent[:80].replace("\n", "\\n")
            print(
                f"[IntentListener] SECURITY: Rejected intent file {path!r} — "
                f"injection marker detected (preview: {preview!r}). "
                "Intent file preserved for inspection.  NOT armed.",
                file=sys.stderr,
            )
            self._write_arm_failure_sidecar(
                reason="injection_marker_detected",
                intent=intent,
            )
            return None, None

        return intent, path

    def _contains_injection(self, text: str) -> bool:
        try:
            from core.security.injection_markers import contains_injection_marker
            return contains_injection_marker(text)
        except ImportError:
            _lowered = text.lower()
            return (
                "ignore previous instructions" in _lowered
                or "ignore all previous" in _lowered
            )

    def _consume_intent(self, file_path: Optional[str]) -> None:
        if file_path is None:
            return
        try:
            os.remove(file_path)
        except Exception:
            pass

    def _write_arm_failure_sidecar(self, reason: str, intent: str) -> None:
        _sidecar_path = os.path.join(self._SIDECAR_DIR, "arm_failure.json")
        try:
            os.makedirs(self._SIDECAR_DIR, exist_ok=True)
            payload = {
                "event": "arm_failure",
                "reason": reason,
                "intent_preview": intent[:80],
                "timestamp": time.time(),
            }
            _tmp = _sidecar_path + ".tmp"
            with open(_tmp, "w", encoding="utf-8") as _f:
                json.dump(payload, _f)
            os.replace(_tmp, _sidecar_path)
        except Exception:
            pass

    def _write_arm_success_sidecar(self) -> None:
        _success_path = os.path.join(self._SIDECAR_DIR, "arm_success.json")
        _failure_path = os.path.join(self._SIDECAR_DIR, "arm_failure.json")
        try:
            os.makedirs(self._SIDECAR_DIR, exist_ok=True)
            payload = {"event": "arm_success", "timestamp": time.time()}
            _tmp = _success_path + ".tmp"
            with open(_tmp, "w", encoding="utf-8") as _f:
                json.dump(payload, _f)
            os.replace(_tmp, _success_path)
            try:
                os.remove(_failure_path)
            except OSError:
                pass
        except Exception:
            pass

    # ==================================================
    # Helpers
    # ==================================================

    def _strip_arm_prefix(self, text: str) -> str:
        stripped = text.strip()
        if stripped.upper().startswith(self._ARM_PREFIX):
            stripped = stripped[len(self._ARM_PREFIX):].strip()
        return stripped
