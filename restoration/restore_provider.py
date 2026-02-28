from __future__ import annotations

import time
import threading
import json
import os
import sys
from typing import Optional

from restoration.snapshot_types import RestorationSnapshot
from restoration.snapshot_provider import SnapshotProvider
from core.mode_controller import ModeController, SystemMode


class RestorationError(RuntimeError):
    """Raised when restoration or post-restore verification fails."""
    pass


class RestoreProvider:
    

    CURSOR_TOLERANCE_PX = 5

    
    POST_ACTION_DELAY: float = 0.50     # was 0.25s
    MAX_VERIFY_ATTEMPTS: int = 10       # was 5 → total window: 10 × 0.50 = 5.0s

    MAX_LEDGER_ENTRIES = 10_000
    
    MAX_TITLE_DISTANCE = 5

    _RESTORE_LEDGER_PATH: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "memory",
        "restore_ledger.json",
    )

    # ------------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        os_backend,
        mode_controller: ModeController,
        snapshot_provider: SnapshotProvider,
    ) -> None:
        self._os = os_backend
        self._mode = mode_controller
        self._snapshot_provider = snapshot_provider
        self._lock = threading.Lock()

        # Ledger availability — disabled on read-only filesystems without raising
        self._ledger_available: bool = True
        try:
            os.makedirs(os.path.dirname(self._RESTORE_LEDGER_PATH), exist_ok=True)
        except (PermissionError, OSError) as _ledger_dir_err:
            self._ledger_available = False
            print(
                f"[RestoreProvider] WARNING: Cannot create ledger directory "
                f"({os.path.dirname(self._RESTORE_LEDGER_PATH)!r}): "
                f"{_ledger_dir_err}. "
                "Duplicate-restore protection is DISABLED for this session. "
                "This is expected on read-only filesystems (containers, NFS, CI).",
                file=sys.stderr,
            )

        self._completed_snapshots: dict = (
            self._load_ledger() if self._ledger_available else {}
        )

    # =========================================================================
    # LEDGER
    # =========================================================================

    def _load_ledger(self) -> dict:
        """
        Load completed-snapshots ledger from disk.

        SI-4 / H9 FIX: Returns {snapshot_id: timestamp}.
        Backward-compatible: accepts the old JSON-list format (assigns
        timestamp=0.0 so legacy entries are evicted first on trim).
        """
        if not os.path.exists(self._RESTORE_LEDGER_PATH):
            return {}

        try:
            with open(self._RESTORE_LEDGER_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                # New format: {snapshot_id: float_timestamp}
                return {
                    str(k): float(v)
                    for k, v in list(data.items())[: self.MAX_LEDGER_ENTRIES]
                    if isinstance(v, (int, float))
                }

            if isinstance(data, list):
                # Old format: ["snapshot_id", ...] — upgrade in memory
                return {str(x): 0.0 for x in data[: self.MAX_LEDGER_ENTRIES]}

            raise RestorationError("Restore ledger corrupted: unexpected format")

        except RestorationError:
            raise
        except Exception as e:
            raise RestorationError(f"Restore ledger load failed: {e}") from e

    def _persist_ledger(self) -> None:
        """
        Persist completed-snapshots ledger atomically.

        SI-4 / H9 FIX: Trims by OLDEST timestamp (chronological eviction),
        not by lexicographic SHA-256 hex ID sort.

        Write is skipped when ledger directory is not writable (read-only FS).
        """
        if not self._ledger_available:
            return

        tmp_path = self._RESTORE_LEDGER_PATH + ".tmp"

        try:
            # SI-4 / H9: Evict oldest entries (lowest timestamps) to stay under cap
            if len(self._completed_snapshots) > self.MAX_LEDGER_ENTRIES:
                sorted_by_ts = sorted(
                    self._completed_snapshots.items(),
                    key=lambda kv: kv[1],           # sort ascending by timestamp
                )
                self._completed_snapshots = dict(
                    sorted_by_ts[-self.MAX_LEDGER_ENTRIES:]  # keep newest
                )

            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(
                    self._completed_snapshots,
                    f,
                    separators=(",", ":"),
                )
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp_path, self._RESTORE_LEDGER_PATH)

        except Exception as e:
            # Non-fatal: ledger persistence failure logs but does not abort restoration
            print(
                f"[RestoreProvider] WARNING: Ledger persist failed: {e}. "
                "Duplicate-restore protection may be incomplete for this session.",
                file=sys.stderr,
            )

    # =========================================================================
    # PUBLIC ENTRY
    # =========================================================================

    def restore_snapshot(self, snapshot_id: str) -> None:
        """
        Look up a snapshot by ID and restore it.  Delegates to restore().

        Raises RestorationError if snapshot_id is invalid or not found.
        """
        if not isinstance(snapshot_id, str) or not snapshot_id.strip():
            raise RestorationError("Invalid snapshot_id: must be a non-empty string")

        snapshot = self._snapshot_provider.get_snapshot(snapshot_id)
        if snapshot is None:
            raise RestorationError(f"Snapshot not found in provider: {snapshot_id!r}")

        self.restore(snapshot)

    # =========================================================================
    # CORE RESTORE
    # =========================================================================

    def restore(self, snapshot: RestorationSnapshot) -> None:
        
        if not isinstance(snapshot, RestorationSnapshot):
            raise RestorationError(
                f"restore() requires RestorationSnapshot, got {type(snapshot).__name__}"
            )

        snapshot_id = snapshot.snapshot_id

        with self._lock:

            # Idempotency: do not restore the same snapshot twice
            if snapshot_id in self._completed_snapshots:
                return

            if self._mode.mode is not SystemMode.RESTORING:
                raise RestorationError(
                    f"restore() attempted in mode {self._mode.mode!r}; "
                    "only RESTORING mode is permitted (fail-closed)."
                )

            # Stop all automated input before manipulating the workspace
            try:
                self._os.stop_automated_input()
                self._os.force_release_all(reason="restoration")
                self._os.mark_automation_inactive()
            except Exception as e:
                raise RestorationError(
                    f"Automation shutdown failed before restore: {e}"
                ) from e

            # Best-effort workspace restoration
            self._restore_application(snapshot)
            self._restore_window(snapshot)
            self._restore_cursor(snapshot)

            # Hard verification — raises RestorationError on failure
            self._verify(snapshot)

            
            self._report_unrestored_processes(snapshot)

            # Record successful completion with current timestamp
            # SI-4 / H9 FIX: timestamp (not just sentinel) enables oldest-first eviction
            self._completed_snapshots[snapshot_id] = time.time()
            self._persist_ledger()

    # =========================================================================
    # RESTORE STEPS (best-effort; log and continue on OSError)
    # =========================================================================

    def _restore_application(self, snapshot: RestorationSnapshot) -> None:
        
        if snapshot.application.process_name == "__bare_desktop__":
            return  # Bare desktop: no application to activate

        try:
            self._os.activate_application(
                {"title": snapshot.application.process_name}
            )
        except OSError as _app_err:
            # GAP-02: Log the specific failure reason before continuing
            print(
                f"[RestoreProvider] WARNING: _restore_application() — "
                f"activate_application({snapshot.application.process_name!r}) "
                f"raised OSError: {_app_err}. "
                "Target application may have been closed during task execution. "
                "Continuing with best-effort restoration (cursor will still be restored).",
                file=sys.stderr,
            )
            # Best-effort: continue; _verify() will detect and report the mismatch

        time.sleep(self.POST_ACTION_DELAY)

    def _restore_window(self, snapshot: RestorationSnapshot) -> None:
        """
        Focus the window captured in the snapshot.

        FIX-01 / SI-01: Guards __bare_desktop__ sentinel so focus_window() is
        not called with a sentinel value that would raise OSError.
        FIX-PATCH: Also guards missing/empty window_id (desktop was active at
        snapshot time with no window focused).
        """
        window_id = getattr(snapshot.focus, "window_id", None)
        if not isinstance(window_id, str) or not window_id.strip():
            return  # No window to focus

        if window_id == "__bare_desktop__":
            return  # Bare desktop sentinel — no window to focus

        try:
            self._os.focus_window({"title": window_id})
        except OSError:
            # Best-effort: window may have been closed during task execution
            print(
                f"[RestoreProvider] WARNING: _restore_window() — "
                f"focus_window({window_id!r}) raised OSError. "
                "Window may have been closed during task execution. Continuing.",
                file=sys.stderr,
            )

        time.sleep(self.POST_ACTION_DELAY)

    def _restore_cursor(self, snapshot: RestorationSnapshot) -> None:
        """
        Reposition cursor to the coordinates captured in the snapshot.

        This is a hard requirement: if cursor cannot be repositioned within
        CURSOR_TOLERANCE_PX, _verify() will raise RestorationError.
        """
        self._os.set_cursor_position(
            {"x": snapshot.cursor.x, "y": snapshot.cursor.y}
        )
        time.sleep(self.POST_ACTION_DELAY)

    # =========================================================================
    # VERIFICATION (hard; raises RestorationError on failure)
    # =========================================================================

    def _verify(self, snapshot: RestorationSnapshot) -> None:
        
        if self._mode.mode is not SystemMode.RESTORING:
            raise RestorationError(
                "Verification attempted outside RESTORING mode — "
                "this indicates a mode-controller bug or external mode mutation."
            )

        _verify_cursor_failures = 0
        _verify_window_failures = 0
        _verify_app_failures = 0

        for attempt in range(1, self.MAX_VERIFY_ATTEMPTS + 1):

            cursor = self._os.get_cursor_position()
            current_window = self._os.get_focused_window()
            current_app = self._os.get_active_application()

            _cursor_ok = self._validate_cursor(cursor, snapshot)
            _window_ok = self._validate_window(current_window, snapshot)
            _app_ok = self._validate_application(current_app, snapshot)

            if _cursor_ok and _window_ok and _app_ok:
                return  # All checks passed — restoration verified

            # Increment per-dimension failure counters for diagnostics
            if not _cursor_ok:
                _verify_cursor_failures += 1
            if not _window_ok:
                _verify_window_failures += 1
            if not _app_ok:
                _verify_app_failures += 1

            # Wait before retrying — gives WM time to propagate focus events
            if attempt < self.MAX_VERIFY_ATTEMPTS:
                time.sleep(self.POST_ACTION_DELAY)

        # All attempts exhausted — raise with full diagnostic context
        raise RestorationError(
            f"Post-restore verification failed after {self.MAX_VERIFY_ATTEMPTS} attempts "
            f"(window={self.MAX_VERIFY_ATTEMPTS * self.POST_ACTION_DELAY:.1f}s total). "
            f"Failures: cursor={_verify_cursor_failures}, "
            f"window={_verify_window_failures}, "
            f"app={_verify_app_failures}. "
            f"Expected: cursor=({snapshot.cursor.x},{snapshot.cursor.y}) "
            f"±{self.CURSOR_TOLERANCE_PX}px, "
            f"window={snapshot.focus.window_id!r}, "
            f"app={snapshot.application.process_name!r}. "
            "Possible causes: "
            "(1) Window manager did not propagate focus within the verification window — "
            f"increase POST_ACTION_DELAY (currently {self.POST_ACTION_DELAY}s) or "
            f"MAX_VERIFY_ATTEMPTS (currently {self.MAX_VERIFY_ATTEMPTS}). "
            "(2) Target application was closed during task execution (check window failures). "
            "(3) Display server is unresponsive (check cursor failures)."
        )

    # =========================================================================
    # VALIDATION HELPERS
    # =========================================================================

    def _validate_cursor(self, cursor, snapshot: RestorationSnapshot) -> bool:
        """True iff cursor is within CURSOR_TOLERANCE_PX of the expected position."""
        if not isinstance(cursor, dict):
            return False

        try:
            cx = int(cursor["x"])
            cy = int(cursor["y"])
        except (KeyError, TypeError, ValueError):
            return False

        return (
            abs(cx - snapshot.cursor.x) <= self.CURSOR_TOLERANCE_PX
            and abs(cy - snapshot.cursor.y) <= self.CURSOR_TOLERANCE_PX
        )

    def _normalize(self, text: str) -> str:
        """Normalise a window/app title for fuzzy comparison."""
        return " ".join(text.lower().strip().split())

    def _levenshtein(self, a: str, b: str) -> int:
        """Compute Levenshtein edit distance between two strings."""
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)

        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            curr = [i]
            for j, cb in enumerate(b, 1):
                insert = curr[j - 1] + 1
                delete = prev[j] + 1
                replace = prev[j - 1] + (ca != cb)
                curr.append(min(insert, delete, replace))
            prev = curr
        return prev[-1]

    def _strict_match(self, expected: str, actual: str) -> bool:
        
        if not expected or not actual:
            return False
        if expected == actual:
            return True
        if self._levenshtein(expected, actual) <= self.MAX_TITLE_DISTANCE:
            return True

        # Token-overlap fallback — handles dynamic-title browser windows (IH-5 fix).
        expected_tokens = set(expected.split())
        actual_tokens = set(actual.split())
        shared = expected_tokens & actual_tokens
        # Only count substantive tokens (length ≥ 4) to avoid trivial matches.
        substantive_shared = [t for t in shared if len(t) >= 4]
        if substantive_shared:
            longest = max(len(t) for t in substantive_shared)
            # Require the shared token to represent ≥ 40% of expected title length.
            if len(expected) > 0 and longest / len(expected) >= 0.40:
                return True

        return False

    def _validate_window(self, current_window, snapshot: RestorationSnapshot) -> bool:
        """True iff the currently focused window matches the snapshot's window title."""
        # Bare desktop: no window expected — always passes
        if snapshot.focus.window_id == "__bare_desktop__":
            return True

        if (
            not isinstance(current_window, dict)
            or not isinstance(current_window.get("title"), str)
        ):
            return False

        expected = self._normalize(snapshot.focus.window_id)
        actual = self._normalize(current_window["title"])
        return self._strict_match(expected, actual)

    def _report_unrestored_processes(self, snapshot: RestorationSnapshot) -> None:
        
        baseline_pids = snapshot.metadata.get("process_census_pids")
        if not baseline_pids:
            # Census was not captured at snapshot time (degraded mode or legacy
            # snapshot without census).  Cannot compute diff — skip silently.
            print(
                "[RestoreProvider] DEBUG: No process census in snapshot metadata — "
                "unrestored-process diff skipped (IH-6).",
                file=sys.stderr,
            )
            return

        try:
            # Collect current PID set using the same strategy as snapshot creation.
            current_pids: list = []
            try:
                import os as _os2
                current_pids = sorted(
                    int(e) for e in _os2.listdir("/proc") if e.isdigit()
                )
            except Exception:
                try:
                    import psutil as _psutil
                    current_pids = sorted(
                        p.pid for p in _psutil.process_iter(["pid"])
                    )
                except Exception:
                    print(
                        "[RestoreProvider] DEBUG: Cannot enumerate current PIDs "
                        "— unrestored-process diff skipped (IH-6).",
                        file=sys.stderr,
                    )
                    return

            baseline_set = set(baseline_pids)
            current_set = set(current_pids)
            new_pids = sorted(current_set - baseline_set)

            if not new_pids:
                print(
                    f"[RestoreProvider] IH-6: Process census diff: 0 new processes "
                    f"since snapshot {snapshot.snapshot_id!r}. No unrestored processes.",
                    file=sys.stderr,
                )
                return

            # Attempt to resolve PID → process name for actionable reporting.
            pid_names: dict = {}
            for pid in new_pids:
                try:
                    import os as _os3
                    _comm_path = f"/proc/{pid}/comm"
                    if _os3.path.exists(_comm_path):
                        with open(_comm_path, "r") as _f:
                            pid_names[pid] = _f.read().strip()
                except Exception:
                    pass
                if pid not in pid_names:
                    try:
                        import psutil as _psutil2
                        pid_names[pid] = _psutil2.Process(pid).name()
                    except Exception:
                        pid_names[pid] = "<unknown>"

            pid_summary = ", ".join(
                f"{pid}({pid_names.get(pid, '?')})" for pid in new_pids
            )

            print(
                f"[RestoreProvider] WARNING IH-6: {len(new_pids)} process(es) spawned "
                f"during task execution were NOT restored: {pid_summary}. "
                "These processes persist on the OS after restoration. "
                "For full process isolation, run ProjectZeo in a container or VM with snapshots.",
                file=sys.stderr,
            )

            # Store the diff in snapshot metadata for downstream consumers.
            # The snapshot is frozen (dataclass frozen=True) so we update the
            # dict that was already stored in metadata — this modifies the
            # in-memory dict object without reassigning the frozen field.
            try:
                snapshot.metadata["unrestored_pids"] = new_pids
                snapshot.metadata["unrestored_pid_names"] = pid_names
            except Exception:
                pass  # metadata dict may be immutable in some configurations

        except Exception as _diff_err:
            print(
                f"[RestoreProvider] DEBUG: Process census diff failed unexpectedly: "
                f"{_diff_err}. Restoration is still complete.",
                file=sys.stderr,
            )

    def _validate_application(
        self, current_app, snapshot: RestorationSnapshot
    ) -> bool:
        """True iff the currently active application matches the snapshot's app name."""
        # Bare desktop: no application expected — always passes
        if snapshot.application.process_name == "__bare_desktop__":
            return True

        if (
            not isinstance(current_app, dict)
            or not isinstance(current_app.get("title"), str)
        ):
            return False

        expected = self._normalize(snapshot.application.process_name)
        actual = self._normalize(current_app["title"])
        return self._strict_match(expected, actual)
