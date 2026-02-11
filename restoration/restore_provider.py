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
    - Single authority for mode reset
    """

    CURSOR_TOLERANCE_PX = 5

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

            # -------------------------------------------------
            # HARD SAFETY: deterministic shutdown
            # -------------------------------------------------
            try:
                self._os.stop_automated_input()
                self._os.force_release_all(reason="restoration")
                self._os.mark_automation_inactive()
            except Exception as e:
                raise RestorationError(
                    f"Automation shutdown failed: {e}"
                ) from e

            # -------------------------------------------------
            # CORE STATE RESTORE
            # -------------------------------------------------
            self._restore_core_state(snapshot)

            # -------------------------------------------------
            # MODE RESET (authoritative)
            # -------------------------------------------------
            try:
                if self._mode.mode is not SystemMode.OBSERVER:
                    self._mode.force_observer()
            except Exception as e:
                raise RestorationError(
                    f"Mode reset failed: {e}"
                ) from e

            # -------------------------------------------------
            # VERIFY RESTORATION
            # -------------------------------------------------
            self._verify(snapshot)

            # -------------------------------------------------
            # COMMIT (idempotency seal)
            # -------------------------------------------------
            self._completed_snapshot_id = snapshot_id

    # =========================================================
    # CORE STATE
    # =========================================================

    def _restore_core_state(
        self,
        snapshot: RestorationSnapshot,
    ) -> None:

        # ---- Cursor ----
        try:
            self._os.set_cursor_position(
                {"x": snapshot.cursor.x, "y": snapshot.cursor.y}
            )
        except Exception as e:
            raise RestorationError(
                f"Cursor restore failed: {e}"
            ) from e

        # ---- Focus Window ----
        try:
            self._os.focus_window(
                {"title": snapshot.focus.window_id}
            )
        except Exception as e:
            raise RestorationError(
                f"Window focus restore failed: {e}"
            ) from e

    # =========================================================
    # VERIFICATION (FAIL-CLOSED)
    # =========================================================

    def _verify(self, snapshot: RestorationSnapshot) -> None:

        # allow OS settle
        time.sleep(0.05)

        if self._mode.mode is not SystemMode.OBSERVER:
            raise RestorationError(
                f"Post-restore mode invalid: {self._mode.mode}"
            )

        try:
            cursor = self._os.get_cursor_position()
            current_window = self._os.get_focused_window()
        except Exception as e:
            raise RestorationError(
                f"Verification read failed: {e}"
            ) from e

        # ---- Cursor validation ----
        if (
            not isinstance(cursor, dict)
            or "x" not in cursor
            or "y" not in cursor
        ):
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

        # ---- Focus validation ----
        if (
            not isinstance(current_window, dict)
            or current_window.get("title") != snapshot.focus.window_id
        ):
            raise RestorationError(
                "Focused window mismatch after restore"
    )
