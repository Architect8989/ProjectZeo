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
    """
    Restoration provider.

    Guarantees:
    - Idempotent per snapshot (persisted across restarts)
    - Concurrency-safe
    - Fail-closed
    - Strict mode enforcement
    - Deterministic restore order
    - No internal mode transitions
    """

    CURSOR_TOLERANCE_PX = 5
    POST_ACTION_DELAY = 0.08
    MAX_VERIFY_ATTEMPTS = 5

    _RESTORE_LEDGER_PATH = os.path.join("memory", "restore_ledger.json")

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

        os.makedirs("memory", exist_ok=True)
        self._completed_snapshots: Set[str] = self._load_ledger()

    # =========================================================
    # LEDGER (PERSISTED IDEMPOTENCY)
    # =========================================================

    def _load_ledger(self) -> Set[str]:
        if not os.path.exists(self._RESTORE_LEDGER_PATH):
            return set()

        try:
            with open(self._RESTORE_LEDGER_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise RestorationError("Restore ledger corrupted")

            return set(str(x) for x in data)

        except Exception as e:
            raise RestorationError(f"Restore ledger load failed: {e}") from e

    def _persist_ledger(self) -> None:
        tmp_path = self._RESTORE_LEDGER_PATH + ".tmp"

        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(sorted(self._completed_snapshots), f)
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp_path, self._RESTORE_LEDGER_PATH)

        except Exception as e:
            raise RestorationError(f"Restore ledger persist failed: {e}") from e

    # =========================================================
    # PUBLIC ENTRYPOINT
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
                return  # persisted idempotency

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
        try:
            self._os.activate_application(
                {"title": snapshot.application.process_name}
            )
            time.sleep(self.POST_ACTION_DELAY)
        except Exception as e:
            raise RestorationError(
                f"Application activation failed: {e}"
            ) from e

    def _restore_window(self, snapshot: RestorationSnapshot) -> None:
        try:
            self._os.focus_window(
                {"title": snapshot.focus.window_id}
            )
            time.sleep(self.POST_ACTION_DELAY)
        except Exception as e:
            raise RestorationError(
                f"Window focus restore failed: {e}"
            ) from e

    def _restore_cursor(self, snapshot: RestorationSnapshot) -> None:
        try:
            self._os.set_cursor_position(
                {"x": snapshot.cursor.x, "y": snapshot.cursor.y}
            )
            time.sleep(self.POST_ACTION_DELAY)
        except Exception as e:
            raise RestorationError(
                f"Cursor restore failed: {e}"
            ) from e

    # =========================================================
    # VERIFICATION
    # =========================================================

    def _verify(self, snapshot: RestorationSnapshot) -> None:

        if self._mode.mode is not SystemMode.RESTORING:
            raise RestorationError(
                f"Verification outside RESTORING mode: {self._mode.mode}"
            )

        for _ in range(self.MAX_VERIFY_ATTEMPTS):

            try:
                cursor = self._os.get_cursor_position()
                current_window = self._os.get_focused_window()
                current_app = self._os.get_active_application()
            except Exception as e:
                raise RestorationError(
                    f"Verification read failed: {e}"
                ) from e

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
    # VALIDATION HELPERS
    # =========================================================

    def _validate_cursor(self, cursor, snapshot) -> bool:
        if not isinstance(cursor, dict):
            return False

        try:
            current_x = int(cursor["x"])
            current_y = int(cursor["y"])
        except Exception:
            return False

        return (
            abs(current_x - snapshot.cursor.x)
            <= self.CURSOR_TOLERANCE_PX
            and abs(current_y - snapshot.cursor.y)
            <= self.CURSOR_TOLERANCE_PX
        )

    def _normalize(self, text: str) -> str:
        return " ".join(text.strip().lower().split())

    def _fuzzy_match(self, expected: str, actual: str) -> bool:
        if not expected or not actual:
            return False
        if expected == actual:
            return True
        if expected in actual:
            return True
        if actual in expected:
            return True
        return False

    def _validate_window(self, current_window, snapshot) -> bool:
        if (
            not isinstance(current_window, dict)
            or not isinstance(current_window.get("title"), str)
        ):
            return False

        expected = self._normalize(snapshot.focus.window_id)
        actual = self._normalize(current_window["title"])

        return self._fuzzy_match(expected, actual)

    def _validate_application(self, current_app, snapshot) -> bool:
        if (
            not isinstance(current_app, dict)
            or not isinstance(current_app.get("title"), str)
        ):
            return False

        expected = self._normalize(snapshot.application.process_name)
        actual = self._normalize(current_app["title"])

        return self._fuzzy_match(expected, actual)
