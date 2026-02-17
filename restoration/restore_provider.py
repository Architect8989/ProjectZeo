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
    - No internal mode transitions
    """

    CURSOR_TOLERANCE_PX = 5
    POST_ACTION_DELAY = 0.08
    MAX_VERIFY_ATTEMPTS = 3

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

            # Idempotency guard
            if self._completed_snapshot_id == snapshot_id:
                return

            # Strict mode enforcement — must already be RESTORING
            if self._mode.mode is not SystemMode.RESTORING:
                raise RestorationError(
                    f"Restore attempted in invalid mode: {self._mode.mode}"
                )

            # HARD STOP AUTOMATION (fail-closed)
            try:
                self._os.stop_automated_input()
                self._os.force_release_all(reason="restoration")
                self._os.mark_automation_inactive()
            except Exception as e:
                raise RestorationError(
                    f"Automation shutdown failed: {e}"
                ) from e

            # Deterministic restore order
            self._restore_application(snapshot)
            self._restore_window(snapshot)
            self._restore_cursor(snapshot)

            # Strict verification (still in RESTORING state)
            self._verify(snapshot)

            # DO NOT mutate mode here.
            # Caller (main.py) must call mode.complete_execution()
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
    # VERIFICATION (STRICT + DETERMINISTIC)
    # =========================================================

    def _verify(self, snapshot: RestorationSnapshot) -> None:

        # Mode must still be RESTORING during verification
        if self._mode.mode is not SystemMode.RESTORING:
            raise RestorationError(
                f"Verification attempted outside RESTORING mode: {self._mode.mode}"
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

            return  # success

        raise RestorationError("Post-restore verification failed")

    # =========================================================
    # STRICT VALIDATION HELPERS
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
        return text.strip().lower()

    def _validate_window(self, current_window, snapshot) -> bool:

        if (
            not isinstance(current_window, dict)
            or not isinstance(current_window.get("title"), str)
        ):
            return False

        expected = self._normalize(snapshot.focus.window_id)
        actual = self._normalize(current_window["title"])

        return expected == actual

    def _validate_application(self, current_app, snapshot) -> bool:

        if (
            not isinstance(current_app, dict)
            or not isinstance(current_app.get("title"), str)
        ):
            return False

        expected = self._normalize(snapshot.application.process_name)
        actual = self._normalize(current_app["title"])

        return expected == actual
