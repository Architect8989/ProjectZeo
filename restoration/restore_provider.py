from __future__ import annotations

import time
import threading
from typing import Optional

from restoration.snapshot_types import RestorationSnapshot
from core.mode_controller import ModeController, SystemMode


class RestorationError(RuntimeError):
    pass


class RestoreProvider:
    """
    Restoration provider.

    Guarantees:
    - Idempotent per snapshot
    - Concurrency-safe
    - Fail-closed
    - Strict mode enforcement
    - Deterministic restore order
    """

    CURSOR_TOLERANCE_PX = 5
    POST_ACTION_DELAY = 0.08

    def __init__(self, *, os_backend, mode_controller: ModeController):
        self._os = os_backend
        self._mode = mode_controller
        self._completed_snapshot_id: Optional[str] = None
        self._lock = threading.Lock()

    # =========================================================
    # PUBLIC
    # =========================================================

    def restore_snapshot(self, snapshot_id: str) -> None:
        from restoration.snapshot_provider import SnapshotProvider

        snapshot = SnapshotProvider.get_snapshot(snapshot_id)
        if snapshot is None:
            raise RestorationError(f"Snapshot not found: {snapshot_id}")

        self.restore(snapshot)

    # =========================================================
    # CORE RESTORE
    # =========================================================

    def restore(self, snapshot: RestorationSnapshot) -> None:
        snapshot_id = snapshot.snapshot_id

        with self._lock:

            if self._completed_snapshot_id == snapshot_id:
                return

            # STRICT MODE REQUIREMENT
            if self._mode.mode is not SystemMode.RESTORING:
                raise RestorationError(
                    f"Restore attempted in invalid mode: {self._mode.mode}"
                )

            # HARD STOP AUTOMATION FIRST
            try:
                self._os.stop_automated_input()
                self._os.force_release_all(reason="restoration")
                self._os.mark_automation_inactive()
            except Exception as e:
                raise RestorationError(
                    f"Automation shutdown failed: {e}"
                ) from e

            # Deterministic restore order:
            # 1) Application
            # 2) Window focus
            # 3) Cursor
            self._restore_application(snapshot)
            self._restore_window(snapshot)
            self._restore_cursor(snapshot)

            # Mode reset
            try:
                self._mode.force_observer()
            except Exception as e:
                raise RestorationError(
                    f"Mode reset failed: {e}"
                ) from e

            # Verification
            self._verify(snapshot)

            self._completed_snapshot_id = snapshot_id

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
    # VERIFICATION (FAIL-CLOSED)
    # =========================================================

    def _verify(self, snapshot: RestorationSnapshot) -> None:

        if self._mode.mode is not SystemMode.OBSERVER:
            raise RestorationError(
                f"Post-restore mode invalid: {self._mode.mode}"
            )

        try:
            cursor = self._os.get_cursor_position()
            current_window = self._os.get_focused_window()
            current_app = self._os.get_active_application()
        except Exception as e:
            raise RestorationError(
                f"Verification read failed: {e}"
            ) from e

        # -------------------------
        # Cursor validation
        # -------------------------

        if not isinstance(cursor, dict):
            raise RestorationError("Cursor read invalid")

        try:
            current_x = int(cursor["x"])
            current_y = int(cursor["y"])
        except Exception:
            raise RestorationError("Cursor coordinates invalid")

        if (
            abs(current_x - snapshot.cursor.x)
            > self.CURSOR_TOLERANCE_PX
            or abs(current_y - snapshot.cursor.y)
            > self.CURSOR_TOLERANCE_PX
        ):
            raise RestorationError(
                "Cursor position mismatch after restore"
            )

        # -------------------------
        # Focus validation
        # -------------------------

        if (
            not isinstance(current_window, dict)
            or not isinstance(current_window.get("title"), str)
            or snapshot.focus.window_id.lower()
            not in current_window["title"].lower()
        ):
            raise RestorationError(
                "Focused window mismatch after restore"
            )

        # -------------------------
        # Active application validation
        # -------------------------

        if (
            not isinstance(current_app, dict)
            or not isinstance(current_app.get("title"), str)
            or snapshot.application.process_name.lower()
            not in current_app["title"].lower()
        ):
            raise RestorationError(
                "Active application mismatch after restore"
        )
