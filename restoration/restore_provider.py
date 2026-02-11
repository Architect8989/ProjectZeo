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

    # -------------------------------------------------

    def restore_snapshot(self, snapshot_id: str) -> None:
        from restoration.snapshot_provider import SnapshotProvider

        snapshot = SnapshotProvider.get_snapshot(snapshot_id)
        if snapshot is None:
            raise RestorationError(f"Snapshot not found: {snapshot_id}")

        self.restore(snapshot)

    # -------------------------------------------------

    def restore(self, snapshot: RestorationSnapshot) -> None:
        snapshot_id = snapshot.snapshot_id

        with self._lock:
            if self._completed_snapshot_id == snapshot_id:
                return

            # HARD SAFETY
            try:
                self._os.mark_automation_inactive()
                self._os.force_release_all(reason="restoration")
                self._os.stop_automated_input()
            except Exception as e:
                raise RestorationError(
                    f"Automation shutdown failed: {e}"
                ) from e

            if getattr(self._os, "_automation_active", False):
                raise RestorationError(
                    "Automation still active after shutdown"
                )

            # EXTENDED (best effort only)
            meta = snapshot.metadata or {}
            extended = meta.get("extended") or {}
            self._restore_extended_state(snapshot, extended)

            # CORE
            self._restore_core_state(snapshot)

            # MODE RESET
            try:
                if self._mode.mode is not SystemMode.OBSERVER:
                    self._mode.force_observer()
            except Exception as e:
                raise RestorationError(
                    f"Mode reset failed: {e}"
                ) from e

            # VERIFY
            self._verify(snapshot)

            # COMMIT
            self._completed_snapshot_id = snapshot_id

    # -------------------------------------------------

    def _restore_extended_state(
        self,
        snapshot: RestorationSnapshot,
        extended: dict,
    ) -> None:
        # Disabled unless OS backend implements them
        try:
            if (
                extended.get("window_geometry") is not None
                and hasattr(self._os, "set_window_geometry")
            ):
                self._os.set_window_geometry(
                    snapshot.focus.window_id,
                    extended["window_geometry"],
                )
        except Exception:
            pass

        try:
            if (
                extended.get("window_z_order") is not None
                and hasattr(self._os, "set_window_z_order")
            ):
                self._os.set_window_z_order(
                    snapshot.focus.window_id,
                    extended["window_z_order"],
                )
        except Exception:
            pass

    # -------------------------------------------------

    def _restore_core_state(
        self,
        snapshot: RestorationSnapshot,
    ) -> None:
        # Cursor
        try:
            self._os.set_cursor_position(
                {"x": snapshot.cursor.x, "y": snapshot.cursor.y}
            )
        except Exception as e:
            raise RestorationError(
                f"Cursor restore failed: {e}"
            ) from e

        # Focus window
        try:
            self._os.focus_window(
                {"title": snapshot.focus.window_id}
            )
        except Exception as e:
            raise RestorationError(
                f"Window focus restore failed: {e}"
            ) from e

    # -------------------------------------------------

    def _verify(self, snapshot: RestorationSnapshot) -> None:
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

        if not isinstance(cursor, dict):
            raise RestorationError("Cursor read invalid")

        if (
            abs(cursor["x"] - snapshot.cursor.x)
            > self.CURSOR_TOLERANCE_PX
            or abs(cursor["y"] - snapshot.cursor.y)
            > self.CURSOR_TOLERANCE_PX
        ):
            raise RestorationError(
                "Cursor position mismatch after restore"
            )

        if (
            not isinstance(current_window, dict)
            or current_window.get("title")
            != snapshot.focus.window_id
        ):
            raise RestorationError(
                "Focused window mismatch after restore"
        )
