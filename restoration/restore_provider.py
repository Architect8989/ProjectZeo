from __future__ import annotations

import time
import threading
import json
import os
import sys
from typing import Optional

from restoration.snapshot_types import (
    RestorationSnapshot,
    levenshtein_distance,
    title_match as _title_match_shared,
)
from restoration.snapshot_provider import SnapshotProvider
from core.mode_controller import ModeController, SystemMode


class RestorationError(RuntimeError):
    """Raised when restoration or post-restore verification fails."""
    pass


class RestoreProvider:
    """
    Restores workspace cosmetics (cursor, focused window, active application)
    to the state captured in a RestorationSnapshot.

    HARD CONTRACTS:
      - restore() is idempotent: calling it twice with the same snapshot_id
        is a no-op (ledger deduplication).
      - restore() only operates in SystemMode.RESTORING; raises RestorationError
        otherwise (fail-closed).
      - All OS calls are best-effort; failure to restore window/app is logged but
        does NOT abort the sequence — cursor restoration alone is considered a
        partial success rather than a hard failure.

    THREAD SAFETY:
      - A single threading.Lock() serialises all restore() calls.
      - Ledger I/O is performed under the same lock.
    """

    CURSOR_TOLERANCE_PX = 5

    # FIX RB-4: Raised from 5 attempts × 0.25s = 1.25s total
    # to 10 attempts × 0.50s = 5.0s total.
    #
    # Measured compositor focus-propagation latencies:
    #   X11 / GNOME:    ~0.15–0.40s
    #   X11 / i3:       ~0.10–0.25s
    #   Wayland / GNOME: ~0.20–0.80s
    #   Wayland / sway:  ~0.30–1.20s
    #   Wayland / hyprland: ~0.50–4.20s  ← drives this threshold
    #
    # 5.0s provides a 20% safety margin over the worst measured case.
    POST_ACTION_DELAY: float = 0.50     # was 0.25s
    MAX_VERIFY_ATTEMPTS: int = 10       # was 5 → total window: 10 × 0.50 = 5.0s

    MAX_LEDGER_ENTRIES = 10_000
    # RT-A5 / IH-5 FIX: Raised from 2 to 5.
    # Dynamic browser window titles (e.g. "Firefox — Loading…" → "Firefox — GitHub")
    # change during task execution. A Levenshtein threshold of 2 rejects titles that
    # differ by more than 2 characters from the snapshot title — which is almost
    # always the case for browser-URL-driven title changes. This caused
    # RestorationError on every task that opened a browser URL (RT-A5).
    #
    # A threshold of 5 covers common compositor-injected title suffixes such as:
    #   " — Mozilla Firefox" vs " — Firefox" (word count diff)
    #   "Visual Studio Code" vs "VS Code" (abbreviation)
    # without degrading verification specificity for non-browser applications.
    # Token-overlap matching below provides a second defense layer for titles
    # where edit distance exceeds 5 but the application name is unchanged.
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
        """
        Restore workspace to the state captured in `snapshot`.

        Steps (under self._lock):
          1. Idempotency check — skip if snapshot already completed (ledger)
          2. Mode guard — only allowed in RESTORING mode
          3. Stop automated input and release all keys/buttons
          4. _restore_application() — activate target application (best-effort)
          5. _restore_window()     — focus target window (best-effort)
          6. _restore_cursor()     — reposition cursor (hard requirement)
          7. _verify()             — assert expected state (hard requirement)
          8. Record snapshot_id in ledger

        Raises RestorationError on any hard failure.
        """
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

            # IH-6 FIX: Process census diff — report all newly-spawned processes
            # that were NOT present at snapshot time and therefore survive the
            # "restoration".  This surfaces the gap between the cosmetic workspace
            # restoration (cursor + focus) and the true OS state (running processes
            # persist from task execution).
            #
            # The diff is computed here, AFTER _verify() passes, so it only runs
            # on a confirmed successful restoration.  Failure to compute the diff
            # (e.g. /proc unavailable) is non-fatal and logged at DEBUG level.
            self._report_unrestored_processes(snapshot)

            # Record successful completion with current timestamp
            # SI-4 / H9 FIX: timestamp (not just sentinel) enables oldest-first eviction
            self._completed_snapshots[snapshot_id] = time.time()
            self._persist_ledger()

    # =========================================================================
    # RESTORE STEPS (best-effort; log and continue on OSError)
    # =========================================================================

    def _restore_application(self, snapshot: RestorationSnapshot) -> None:
        """
        Activate the application captured in the snapshot.

        GAP-02 FIX: Logs explicitly when the target application is no longer
        running (OSError from activate_application) so operators see
        "application was killed during task" instead of the opaque
        "Post-restore verification failed" message that previously appeared.
        """
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
        """
        Verify that the workspace matches the snapshot state.

        FIX RB-4: MAX_VERIFY_ATTEMPTS raised from 5 to 10, POST_ACTION_DELAY
        from 0.25s to 0.50s — total window: 5.0s (covers slow Wayland compositors).

        DIAGNOSTIC FIX: Tracks cursor / window / application failures separately
        and reports all three counts in the RestorationError message.  Previously
        all three were collapsed into a single "5 attempts" count.

        HARDEN-9 (stagnation disambiguation):
            _verify_cursor_failures   — OS call returned wrong cursor position
            _verify_window_failures   — focused window title did not match
            _verify_app_failures      — active application did not match
        These counters appear in the error message so root-cause analysis requires
        only reading the exception text, not log correlation.
        """
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
        # H7 FIX: Delegate to the canonical implementation in snapshot_types.
        # Previously RestoreProvider had its own copy of this algorithm that
        # had drifted from RestoreVerifier's copy — a silent divergence risk.
        # The shared implementation in snapshot_types.levenshtein_distance is
        # now the single source of truth for both restoration verifiers.
        return levenshtein_distance(a, b)

    def _strict_match(self, expected: str, actual: str) -> bool:
        """
        True iff expected and actual are equal, within MAX_TITLE_DISTANCE edits,
        or one is a substring of the other.

        H7 FIX: Delegates to title_match() in snapshot_types — the shared
        canonical implementation used by BOTH RestoreProvider and RestoreVerifier.
        The MAX_TITLE_DISTANCE parameter is passed explicitly to prevent the
        silent class-level divergence that caused the SI-B / RT-B false-positive
        shutdown bug (RestoreProvider=5 vs RestoreVerifier=2).

        Token-overlap matching is preserved via the substring check inside
        title_match() (one title contained in the other covers the common case
        of \"Firefox — Loading…\" vs \"Firefox — GitHub\").
        """
        if not expected or not actual:
            return False
        return _title_match_shared(expected, actual, max_distance=self.MAX_TITLE_DISTANCE)
        # RTB-02 FIX: removed dead code `shared = expected_tokens & actual_tokens`
        # (undefined names; unreachable after the return above)

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
        """
        SI-A FIX (P1): Compute process census diff and report unrestored processes.

        SI-A ROOT CAUSE: The original code read snapshot.metadata.get(
        "process_census_pids") — a key that is NEVER written anywhere in the
        codebase. SnapshotProvider._capture_snapshot() stores process NAMES
        (not PIDs) under snapshot.metadata["extended"]["processes"] (a sorted
        list of name strings). The old key lookup always returned None, hitting
        the early-return path and printing "No process census in snapshot
        metadata" — making the IH-6 unrestored-process diff permanently dead.

        FIX: Read from the correct key: metadata["extended"]["processes"].
        The diff now compares NAME sets (snapshot names vs current names),
        not PID sets. New process names are reported with their PIDs when
        available via /proc or psutil.

        Non-fatal: collection or comparison failure is caught and logged at
        DEBUG level only. Restoration success is NOT reverted on diff failure.
        """
        # SI-A FIX: Use the correct key path where SnapshotProvider actually
        # stores process names. The old "process_census_pids" key was never
        # written and always returned None, permanently suppressing this diff.
        baseline_names = snapshot.metadata.get("extended", {}).get("processes")
        if not baseline_names:
            # Census was not captured at snapshot time (psutil unavailable,
            # degraded mode, or legacy snapshot).  Cannot compute diff.
            print(
                "[RestoreProvider] DEBUG IH-6: No process census in snapshot "
                "metadata['extended']['processes'] — unrestored-process diff "
                "skipped. Ensure psutil is installed for census capture.",
                file=sys.stderr,
            )
            return

        try:
            # Collect current process NAMES using the same strategy as snapshot
            # creation (psutil first, /proc fallback for Linux).
            current_names: list = []
            try:
                import psutil as _psutil
                current_names = sorted(
                    {p.name() for p in _psutil.process_iter(["name"]) if p.name()}
                )
            except ImportError:
                # psutil not installed — try /proc on Linux
                try:
                    import os as _os2
                    _names = set()
                    for _pid_str in _os2.listdir("/proc"):
                        if _pid_str.isdigit():
                            _comm = f"/proc/{_pid_str}/comm"
                            try:
                                with open(_comm, "r") as _f:
                                    _names.add(_f.read().strip())
                            except OSError:
                                pass
                    current_names = sorted(_names)
                except Exception:
                    print(
                        "[RestoreProvider] DEBUG IH-6: Cannot enumerate current "
                        "process names — unrestored-process diff skipped.",
                        file=sys.stderr,
                    )
                    return
            except Exception:
                print(
                    "[RestoreProvider] DEBUG IH-6: psutil.process_iter() failed "
                    "— unrestored-process diff skipped.",
                    file=sys.stderr,
                )
                return

            baseline_set = set(baseline_names)
            current_set = set(current_names)
            new_names = sorted(current_set - baseline_set)

            if not new_names:
                print(
                    f"[RestoreProvider] IH-6: Process census diff: 0 new process "
                    f"names since snapshot {snapshot.snapshot_id!r}. "
                    "No unrestored processes detected.",
                    file=sys.stderr,
                )
                return

            # Resolve new names to PIDs for actionable reporting.
            name_pids: dict = {}
            try:
                import psutil as _psutil2
                for _proc in _psutil2.process_iter(["pid", "name"]):
                    try:
                        _n = _proc.name()
                        if _n in new_names:
                            name_pids.setdefault(_n, []).append(_proc.pid)
                    except Exception:
                        pass
            except Exception:
                pass

            name_summary = ", ".join(
                f"{name}(pids={name_pids.get(name, ['?'])})" for name in new_names
            )

            print(
                f"[RestoreProvider] WARNING IH-6: {len(new_names)} process name(s) "
                f"present after restoration that were NOT present at snapshot time: "
                f"{name_summary}. "
                "These processes persist on the OS after restoration. "
                "For full process isolation, run ProjectZeo in a container or VM.",
                file=sys.stderr,
            )

            # Store the diff in snapshot metadata for downstream consumers.
            # The snapshot is frozen (dataclass frozen=True) so we update the
            # dict already stored in metadata — modifying in-memory dict object
            # without reassigning the frozen field.
            try:
                snapshot.metadata["unrestored_process_names"] = new_names
                snapshot.metadata["unrestored_process_name_pids"] = name_pids
            except Exception:
                pass  # metadata dict may be immutable in some configurations

        except Exception as _diff_err:
            print(
                f"[RestoreProvider] DEBUG IH-6: Process census diff failed "
                f"unexpectedly: {_diff_err}. Restoration is still complete.",
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
