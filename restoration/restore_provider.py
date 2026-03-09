"""
restoration/restore_provider.py  (patched — March 2026)

Orchestrates all restoration tiers differentially:
  Tier 0: Cursor + active application (always)
  Tier 1: Window layout via wmctrl/xdotool (always)
  Tier 2: Browser tabs via Playwright CDP (only if browser was open pre-task)
  Tier 3: Filesystem via BTRFS/rsync (only if task wrote files)
  Tier 4: Process checkpoint via CRIU (only for interrupted long-running procs)

Every tier is conditional. The provider asks "did the user have this before
the task?" and only restores what existed, cleaning up what the agent added.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional

from restoration.snapshot_types import (
    RestorationSnapshot,
    levenshtein_distance,
    title_match as _title_match_shared,
)
from restoration.snapshot_provider import SnapshotProvider
from core.mode_controller import ModeController, SystemMode

try:
    from restoration.window_state_provider import (
        capture as _win_capture,
        restore as _win_restore,
        close_agent_windows as _win_close_new,
        verify as _win_verify,
        WindowStateSnapshot,
    )
    _WIN_AVAILABLE = True
except ImportError:
    _WIN_AVAILABLE = False

try:
    from restoration.browser_snapshot_provider import (
        capture as _browser_capture,
        restore as _browser_restore,
        BrowserSnapshot,
    )
    _BROWSER_AVAILABLE = True
except ImportError:
    _BROWSER_AVAILABLE = False

try:
    from restoration.fs_snapshot_provider import (
        capture as _fs_capture,
        restore as _fs_restore,
        cleanup as _fs_cleanup,
        verify as _fs_verify,
        FsSnapshot,
    )
    _FS_AVAILABLE = True
except ImportError:
    _FS_AVAILABLE = False

try:
    from restoration.criu_provider import (
        capture as _criu_capture,
        restore as _criu_restore,
        cleanup as _criu_cleanup,
        CriuSnapshot,
    )
    _CRIU_AVAILABLE = True
except ImportError:
    _CRIU_AVAILABLE = False

_logger = logging.getLogger(__name__)


class RestorationError(RuntimeError):
    pass


class RestoreProvider:

    CURSOR_TOLERANCE_PX  = 5
    POST_ACTION_DELAY    = 0.50
    MAX_VERIFY_ATTEMPTS  = 10
    MAX_LEDGER_ENTRIES   = 10_000
    MAX_TITLE_DISTANCE   = 5

    _RESTORE_LEDGER_PATH: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "memory", "restore_ledger.json",
    )

    def __init__(
        self,
        *,
        os_backend,
        mode_controller: ModeController,
        snapshot_provider: SnapshotProvider,
        authority_state=None,
    ) -> None:
        self._os               = os_backend
        self._mode             = mode_controller
        self._snapshot_provider = snapshot_provider
        self._authority_state  = authority_state
        self._lock             = threading.Lock()

        self._ledger_available = True
        try:
            os.makedirs(os.path.dirname(self._RESTORE_LEDGER_PATH), exist_ok=True)
        except (PermissionError, OSError) as e:
            self._ledger_available = False
            _logger.warning("[RestoreProvider] Ledger dir unavailable: %s", e)

        self._completed_snapshots: dict = (
            self._load_ledger() if self._ledger_available else {}
        )

        # Extended tier snapshots — keyed by base snapshot_id
        self._win_snaps:  Dict[str, Any] = {}
        self._brow_snaps: Dict[str, Any] = {}
        self._fs_snaps:   Dict[str, Any] = {}
        self._criu_snaps: Dict[str, Any] = {}

        # Track per-task metadata: did agent open a browser? did it write files?
        self._task_opened_browser: Dict[str, bool] = {}
        self._task_wrote_files:    Dict[str, bool] = {}

    # =========================================================================
    # LEDGER
    # =========================================================================

    def _load_ledger(self) -> dict:
        if not os.path.exists(self._RESTORE_LEDGER_PATH):
            return {}
        try:
            with open(self._RESTORE_LEDGER_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {str(k): float(v) for k, v in list(data.items())[:self.MAX_LEDGER_ENTRIES] if isinstance(v, (int, float))}
            if isinstance(data, list):
                return {str(x): 0.0 for x in data[:self.MAX_LEDGER_ENTRIES]}
            raise RestorationError("Restore ledger corrupted")
        except RestorationError:
            raise
        except Exception as e:
            raise RestorationError(f"Ledger load failed: {e}") from e

    def _persist_ledger(self) -> None:
        if not self._ledger_available:
            return
        tmp = self._RESTORE_LEDGER_PATH + ".tmp"
        try:
            if len(self._completed_snapshots) > self.MAX_LEDGER_ENTRIES:
                sorted_entries = sorted(self._completed_snapshots.items(), key=lambda kv: kv[1])
                self._completed_snapshots = dict(sorted_entries[-self.MAX_LEDGER_ENTRIES:])
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._completed_snapshots, f, separators=(",", ":"))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self._RESTORE_LEDGER_PATH)
        except Exception as e:
            _logger.warning("[RestoreProvider] Ledger persist failed: %s", e)

    # =========================================================================
    # PRE-TASK CAPTURE (all tiers)
    # =========================================================================

    def capture_extended_state(
        self,
        snapshot_id: str,
        task_writes_files: bool = False,
        target_pids: Optional[List[int]] = None,
    ) -> None:
        """
        Capture all extended-tier state before a task starts.
        Call this immediately after the base RestorationSnapshot is taken.
        """
        # Tier 1: Window layout
        if _WIN_AVAILABLE:
            try:
                win_snap = _win_capture()
                if win_snap:
                    self._win_snaps[snapshot_id] = win_snap
            except Exception as e:
                _logger.debug("[RestoreProvider] Window capture error: %s", e)

        # Tier 2: Browser — record whether browser was running pre-task
        if _BROWSER_AVAILABLE:
            try:
                brow_snap = _browser_capture()
                self._brow_snaps[snapshot_id] = brow_snap
            except Exception as e:
                _logger.debug("[RestoreProvider] Browser capture error: %s", e)

        # Tier 3: Filesystem — only if task will write files
        if _FS_AVAILABLE and task_writes_files:
            try:
                fs_snap = _fs_capture(task_writes_files=True)
                if fs_snap:
                    self._fs_snaps[snapshot_id] = fs_snap
            except Exception as e:
                _logger.debug("[RestoreProvider] FS capture error: %s", e)

        # Tier 4: CRIU — only for explicit long-running process protection
        if _CRIU_AVAILABLE and target_pids:
            try:
                criu_snap = _criu_capture(target_pids=target_pids)
                self._criu_snaps[snapshot_id] = criu_snap
            except Exception as e:
                _logger.debug("[RestoreProvider] CRIU capture error: %s", e)

        self._task_wrote_files[snapshot_id]    = task_writes_files
        self._task_opened_browser[snapshot_id] = False  # updated during task

    def mark_agent_opened_browser(self, snapshot_id: str) -> None:
        self._task_opened_browser[snapshot_id] = True

    # =========================================================================
    # CORE RESTORE
    # =========================================================================

    def restore_snapshot(self, snapshot_id: str) -> None:
        snapshot = self._snapshot_provider.get_snapshot(snapshot_id)
        if snapshot is None:
            raise RestorationError(f"Snapshot not found: {snapshot_id!r}")
        self.restore(snapshot)

    def restore(self, snapshot: RestorationSnapshot) -> None:
        if not isinstance(snapshot, RestorationSnapshot):
            raise RestorationError(f"restore() requires RestorationSnapshot, got {type(snapshot).__name__}")

        sid = snapshot.snapshot_id

        with self._lock:
            if sid in self._completed_snapshots:
                _logger.warning("[RestoreProvider] Idempotency guard: snapshot %s already completed.", sid)
                auth = getattr(self, "_authority_state", None)
                if auth is not None and hasattr(auth, "verification_warning"):
                    try:
                        auth.verification_warning = True
                    except Exception:
                        pass
                return

        if self._mode.mode is not SystemMode.RESTORING:
            raise RestorationError(
                f"restore() in mode {self._mode.mode!r}; only RESTORING mode permitted."
            )

        try:
            self._os.stop_automated_input()
            self._os.force_release_all(reason="restoration")
            self._os.mark_automation_inactive()
        except Exception as e:
            raise RestorationError(f"Automation shutdown failed: {e}") from e

        # --- Tier 4: CRIU (restore long-running processes first) ---
        criu_snap = self._criu_snaps.get(sid)
        if criu_snap is not None and _CRIU_AVAILABLE:
            try:
                _criu_restore(criu_snap)
            except Exception as e:
                _logger.warning("[RestoreProvider] CRIU restore error (non-fatal): %s", e)

        # --- Tier 3: Filesystem ---
        fs_snap = self._fs_snaps.get(sid)
        if fs_snap is not None and _FS_AVAILABLE and self._task_wrote_files.get(sid, False):
            try:
                _fs_restore(fs_snap)
            except Exception as e:
                _logger.warning("[RestoreProvider] FS restore error (non-fatal): %s", e)

        # --- Tier 2: Browser (differential) ---
        brow_snap = self._brow_snaps.get(sid)
        if brow_snap is not None and _BROWSER_AVAILABLE:
            try:
                agent_opened = self._task_opened_browser.get(sid, False)
                _browser_restore(brow_snap, agent_opened_browser=agent_opened)
            except Exception as e:
                _logger.warning("[RestoreProvider] Browser restore error (non-fatal): %s", e)

        # --- Tier 1: Window layout ---
        win_snap = self._win_snaps.get(sid)
        if win_snap is not None and _WIN_AVAILABLE:
            try:
                post_win = _win_capture()
                _win_close_new(win_snap, post_win)
                _win_restore(win_snap)
            except Exception as e:
                _logger.warning("[RestoreProvider] Window restore error (non-fatal): %s", e)

        # --- Tier 0: Application + window focus + cursor (always) ---
        self._restore_application(snapshot)
        self._restore_window(snapshot)
        self._restore_cursor(snapshot)

        try:
            self._verify(snapshot)
        except RestorationError as err:
            self._handle_restore_failure(snapshot, str(err))
            raise

        self._report_unrestored_processes(snapshot)

        # Cleanup extended snapshots
        self._cleanup_extended(sid)

        with self._lock:
            if sid not in self._completed_snapshots:
                self._completed_snapshots[sid] = time.monotonic()
                self._persist_ledger()

    def _cleanup_extended(self, snapshot_id: str) -> None:
        if _FS_AVAILABLE:
            fs_snap = self._fs_snaps.pop(snapshot_id, None)
            if fs_snap:
                try:
                    _fs_cleanup(fs_snap)
                except Exception:
                    pass
        if _CRIU_AVAILABLE:
            criu_snap = self._criu_snaps.pop(snapshot_id, None)
            if criu_snap:
                try:
                    _criu_cleanup(criu_snap)
                except Exception:
                    pass
        self._win_snaps.pop(snapshot_id, None)
        self._brow_snaps.pop(snapshot_id, None)
        self._task_wrote_files.pop(snapshot_id, None)
        self._task_opened_browser.pop(snapshot_id, None)

    # =========================================================================
    # TIER 0 RESTORE HELPERS
    # =========================================================================

    def _restore_application(self, snapshot: RestorationSnapshot) -> None:
        if snapshot.application.process_name == "__bare_desktop__":
            return
        try:
            self._os.activate_application({"title": snapshot.application.process_name})
        except OSError as e:
            _logger.warning("[RestoreProvider] activate_application failed: %s", e)
        time.sleep(self.POST_ACTION_DELAY)

    def _restore_window(self, snapshot: RestorationSnapshot) -> None:
        wid = getattr(snapshot.focus, "window_id", None)
        if not isinstance(wid, str) or not wid.strip() or wid == "__bare_desktop__":
            return
        try:
            self._os.focus_window({"title": wid})
        except OSError as e:
            _logger.warning("[RestoreProvider] focus_window failed: %s", e)
        time.sleep(self.POST_ACTION_DELAY)

        try:
            cw = self._os.get_focused_window()
            if isinstance(cw, dict) and isinstance(cw.get("title"), str):
                dist = levenshtein_distance(
                    " ".join(wid.lower().split()),
                    " ".join(cw["title"].lower().split()),
                )
                if dist > self.MAX_TITLE_DISTANCE:
                    _logger.warning(
                        "[RestoreProvider] Window title mismatch: expected=%r actual=%r dist=%d",
                        wid, cw["title"], dist,
                    )
        except Exception:
            pass

    def _restore_cursor(self, snapshot: RestorationSnapshot) -> None:
        self._os.set_cursor_position({"x": snapshot.cursor.x, "y": snapshot.cursor.y})
        time.sleep(self.POST_ACTION_DELAY)

    # =========================================================================
    # VERIFICATION
    # =========================================================================

    def _verify(self, snapshot: RestorationSnapshot) -> None:
        if self._mode.mode is not SystemMode.RESTORING:
            raise RestorationError("Verification attempted outside RESTORING mode.")

        cursor_fail = window_fail = app_fail = 0
        for attempt in range(1, self.MAX_VERIFY_ATTEMPTS + 1):
            cursor      = self._os.get_cursor_position()
            cur_window  = self._os.get_focused_window()
            cur_app     = self._os.get_active_application()

            c_ok = self._validate_cursor(cursor, snapshot)
            w_ok = self._validate_window(cur_window, snapshot)
            a_ok = self._validate_application(cur_app, snapshot)

            if c_ok and w_ok and a_ok:
                return

            if not c_ok: cursor_fail += 1
            if not w_ok: window_fail += 1
            if not a_ok: app_fail    += 1

            if attempt < self.MAX_VERIFY_ATTEMPTS:
                time.sleep(self.POST_ACTION_DELAY)

        raise RestorationError(
            f"Verification failed after {self.MAX_VERIFY_ATTEMPTS} attempts. "
            f"cursor_fails={cursor_fail} window_fails={window_fail} app_fails={app_fail}."
        )

    def _validate_cursor(self, cursor, snapshot: RestorationSnapshot) -> bool:
        if not isinstance(cursor, dict):
            return False
        try:
            return (
                abs(int(cursor["x"]) - snapshot.cursor.x) <= self.CURSOR_TOLERANCE_PX
                and abs(int(cursor["y"]) - snapshot.cursor.y) <= self.CURSOR_TOLERANCE_PX
            )
        except (KeyError, TypeError, ValueError):
            return False

    def _normalize(self, text: str) -> str:
        return " ".join(text.lower().strip().split())

    def _strict_match(self, expected: str, actual: str) -> bool:
        if not expected or not actual:
            return False
        return _title_match_shared(expected, actual, max_distance=self.MAX_TITLE_DISTANCE)

    def _validate_window(self, current_window, snapshot: RestorationSnapshot) -> bool:
        if snapshot.focus.window_id in ("__bare_desktop__", "__wayland_unknown__"):
            return True
        if not isinstance(current_window, dict) or not isinstance(current_window.get("title"), str):
            return False
        return self._strict_match(
            self._normalize(snapshot.focus.window_id),
            self._normalize(current_window["title"]),
        )

    def _validate_application(self, current_app, snapshot: RestorationSnapshot) -> bool:
        if snapshot.application.process_name == "__bare_desktop__":
            return True
        if not isinstance(current_app, dict) or not isinstance(current_app.get("title"), str):
            return False
        return self._strict_match(
            self._normalize(snapshot.application.process_name),
            self._normalize(current_app["title"]),
        )

    # =========================================================================
    # FAILURE HANDLER
    # =========================================================================

    def _handle_restore_failure(self, snapshot: RestorationSnapshot, reason: str) -> None:
        try:
            self._os.set_cursor_position({"x": snapshot.cursor.x, "y": snapshot.cursor.y})
        except Exception:
            try:
                w, h = self._os.screen_size()
                self._os.set_cursor_position({"x": w // 2, "y": h // 2})
            except Exception:
                pass

        auth = getattr(self, "_authority_state", None)
        if auth is not None:
            try:
                if hasattr(auth, "restoration_incomplete"):
                    auth.restoration_incomplete = True
                elif hasattr(auth, "__dict__"):
                    auth.__dict__["restoration_incomplete"] = True
            except Exception:
                pass

        try:
            import pathlib
            flag = pathlib.Path(__file__).resolve().parent.parent / "temp" / "RESTORATION_INCOMPLETE"
            flag.parent.mkdir(parents=True, exist_ok=True)
            flag.write_text(
                f"RESTORATION INCOMPLETE\nReason: {reason}\n"
                f"Snapshot: {snapshot.snapshot_id}\n"
                f"Time: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n",
                encoding="utf-8",
            )
        except Exception:
            pass

    # =========================================================================
    # PROCESS CENSUS
    # =========================================================================

    def _report_unrestored_processes(self, snapshot: RestorationSnapshot) -> None:
        baseline = snapshot.metadata.get("extended", {}).get("processes")
        if not baseline:
            return
        try:
            import psutil
            current = {p.name() for p in psutil.process_iter(["name"]) if p.name()}
        except ImportError:
            return
        except Exception:
            return

        new_procs = sorted(current - set(baseline))
        if not new_procs:
            return

        _WHITELIST = frozenset({
            "systemd", "init", "dbus-daemon", "NetworkManager",
            "pulseaudio", "pipewire", "gnome-shell", "xorg", "ollama", "ydotoold",
        })
        import signal as _signal
        for name in new_procs:
            if name in _WHITELIST:
                continue
            try:
                import psutil
                for proc in psutil.process_iter(["pid", "name"]):
                    if proc.name() == name:
                        os.kill(proc.pid, _signal.SIGTERM)
            except Exception:
                pass

        _logger.info("[RestoreProvider] Cleaned up %d new process(es): %s", len(new_procs), new_procs)
