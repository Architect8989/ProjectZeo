from __future__ import annotations

import time
import threading
import json
import os
from typing import Optional, Set

from restoration.snapshot_types import RestorationSnapshot
from restoration.snapshot_provider import SnapshotProvider
from core.mode_controller import ModeController, SystemMode


class RestorationError(RuntimeError):
    pass


class RestoreProvider:

    CURSOR_TOLERANCE_PX = 5
    POST_ACTION_DELAY = 0.08
    MAX_VERIFY_ATTEMPTS = 5

    MAX_LEDGER_ENTRIES = 10_000
    MAX_TITLE_DISTANCE = 2

    # HARD-6: Use an absolute path anchored to the directory containing this
    # file rather than os.path.join("memory", ...) which resolves relative to
    # os.getcwd(). If the process is started from a different working directory,
    # the relative path creates the ledger in a different location, causing
    # prior completed snapshots to not be found and allowing duplicate restorations.
    _RESTORE_LEDGER_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "memory",
        "restore_ledger.json",
    )

    def __init__(
        self,
        *,
        os_backend,
        mode_controller: ModeController,
        snapshot_provider: SnapshotProvider,
    ):
        self._os = os_backend
        self._mode = mode_controller
        self._snapshot_provider = snapshot_provider
        self._lock = threading.Lock()

        os.makedirs(os.path.dirname(self._RESTORE_LEDGER_PATH), exist_ok=True)
        self._completed_snapshots: Set[str] = self._load_ledger()

    # =========================================================
    # LEDGER
    # =========================================================

    def _load_ledger(self) -> Set[str]:
        if not os.path.exists(self._RESTORE_LEDGER_PATH):
            return set()

        try:
            with open(self._RESTORE_LEDGER_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise RestorationError("Restore ledger corrupted")

            # Enforce deterministic ordering
            return set(str(x) for x in data[: self.MAX_LEDGER_ENTRIES])

        except Exception as e:
            raise RestorationError(f"Restore ledger load failed: {e}") from e

    def _persist_ledger(self) -> None:
        tmp_path = self._RESTORE_LEDGER_PATH + ".tmp"

        try:
            # Enforce bounded size
            if len(self._completed_snapshots) > self.MAX_LEDGER_ENTRIES:
                trimmed = sorted(self._completed_snapshots)[
                    -self.MAX_LEDGER_ENTRIES :
                ]
                self._completed_snapshots = set(trimmed)

            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(
                    sorted(self._completed_snapshots),
                    f,
                    separators=(",", ":"),
                )
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp_path, self._RESTORE_LEDGER_PATH)

        except Exception as e:
            raise RestorationError(
                f"Restore ledger persist failed: {e}"
            ) from e

    # =========================================================
    # PUBLIC ENTRY
    # =========================================================

    def restore_snapshot(self, snapshot_id: str) -> None:
        if not isinstance(snapshot_id, str) or not snapshot_id.strip():
            raise RestorationError("Invalid snapshot_id")

        snapshot = self._snapshot_provider.get_snapshot(snapshot_id)
        if snapshot is None:
            raise RestorationError(f"Snapshot not found: {snapshot_id}")

        self.restore(snapshot)

    # =========================================================
    # CORE RESTORE
    # =========================================================

    def restore(self, snapshot: RestorationSnapshot) -> None:

        if not isinstance(snapshot, RestorationSnapshot):
            raise RestorationError("Invalid snapshot object")

        snapshot_id = snapshot.snapshot_id

        with self._lock:

            if snapshot_id in self._completed_snapshots:
                return

            if self._mode.mode is not SystemMode.RESTORING:
                raise RestorationError(
                    f"Restore attempted in invalid mode: {self._mode.mode}"
                )

            try:
                self._os.stop_automated_input()
                self._os.force_release_all(reason="restoration")
                self._os.mark_automation_inactive()
            except Exception as e:
                raise RestorationError(
                    f"Automation shutdown failed: {e}"
                ) from e

            self._restore_application(snapshot)
            self._restore_window(snapshot)
            self._restore_cursor(snapshot)

            self._verify(snapshot)

            self._completed_snapshots.add(snapshot_id)
            self._persist_ledger()

    # =========================================================
    # RESTORE STEPS
    # =========================================================

    def _restore_application(self, snapshot: RestorationSnapshot) -> None:
        self._os.activate_application(
            {"title": snapshot.application.process_name}
        )
        time.sleep(self.POST_ACTION_DELAY)

    def _restore_window(self, snapshot: RestorationSnapshot) -> None:
        # PATCH (audit): guard against missing/empty window_id.
        # If the snapshot captured no window title (e.g. desktop was active),
        # silently skip rather than crashing focus_window() with an empty title.
        window_id = getattr(snapshot.focus, "window_id", None)
        if not isinstance(window_id, str) or not window_id.strip():
            return

        try:
            self._os.focus_window({"title": window_id})
        except OSError:
            # Best-effort: if the window is no longer present (app was closed
            # during task execution), log and continue rather than failing
            # the entire restoration sequence.
            pass

        time.sleep(self.POST_ACTION_DELAY)

    def _restore_cursor(self, snapshot: RestorationSnapshot) -> None:
        self._os.set_cursor_position(
            {"x": snapshot.cursor.x, "y": snapshot.cursor.y}
        )
        time.sleep(self.POST_ACTION_DELAY)

    # =========================================================
    # VERIFICATION
    # =========================================================

    def _verify(self, snapshot: RestorationSnapshot) -> None:

        if self._mode.mode is not SystemMode.RESTORING:
            raise RestorationError("Verification outside RESTORING mode")

        for _ in range(self.MAX_VERIFY_ATTEMPTS):

            cursor = self._os.get_cursor_position()
            current_window = self._os.get_focused_window()
            current_app = self._os.get_active_application()

            if not self._validate_cursor(cursor, snapshot):
                time.sleep(self.POST_ACTION_DELAY)
                continue

            if not self._validate_window(current_window, snapshot):
                time.sleep(self.POST_ACTION_DELAY)
                continue

            if not self._validate_application(current_app, snapshot):
                time.sleep(self.POST_ACTION_DELAY)
                continue

            return

        raise RestorationError("Post-restore verification failed")

    # =========================================================
    # VALIDATION
    # =========================================================

    def _validate_cursor(self, cursor, snapshot) -> bool:
        if not isinstance(cursor, dict):
            return False

        try:
            cx = int(cursor["x"])
            cy = int(cursor["y"])
        except Exception:
            return False

        return (
            abs(cx - snapshot.cursor.x) <= self.CURSOR_TOLERANCE_PX
            and abs(cy - snapshot.cursor.y) <= self.CURSOR_TOLERANCE_PX
        )

    def _normalize(self, text: str) -> str:
        return " ".join(text.lower().strip().split())

    def _levenshtein(self, a: str, b: str) -> int:
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

        return self._levenshtein(expected, actual) <= self.MAX_TITLE_DISTANCE

    def _validate_window(self, current_window, snapshot) -> bool:
        if (
            not isinstance(current_window, dict)
            or not isinstance(current_window.get("title"), str)
        ):
            return False

        expected = self._normalize(snapshot.focus.window_id)
        actual = self._normalize(current_window["title"])

        return self._strict_match(expected, actual)

    def _validate_application(self, current_app, snapshot) -> bool:
        # FIX RTB-02: When the snapshot was taken on a bare desktop (no focused
        # application), application.process_name is "__bare_desktop__". No
        # application focus can be verified; skip the fuzzy match and return True
        # so restoration can complete without a spurious verification failure.
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
