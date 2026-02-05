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
    - Fail-closed (never lies about success)
    - Single authority for mode reset
    """

    CURSOR_TOLERANCE_PX = 2

    def __init__(self, *, os_backend, mode_controller: ModeController):
        self._os = os_backend
        self._mode = mode_controller

        self._completed_snapshot_id: Optional[str] = None
        self._lock = threading.Lock()

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def restore_snapshot(self, snapshot_id: str) -> None:
        from restoration.snapshot_provider import SnapshotProvider

        snapshot = SnapshotProvider.get_snapshot(snapshot_id)
        if snapshot is None:
            raise RestorationError(
                f"Snapshot not found: {snapshot_id}"
            )

        self.restore(snapshot)

    def restore(self, snapshot: RestorationSnapshot) -> None:
        snapshot_id = snapshot.snapshot_id

        # -------------------------------------------------
        # GLOBAL IDEMPOTENCY + CONCURRENCY LOCK
        # -------------------------------------------------
        with self._lock:
            if self._completed_snapshot_id == snapshot_id:
                return

            # -------------------------------------------------
            # PHASE 0 — HARD SAFETY (FAIL-OPEN TO HUMAN)
            # -------------------------------------------------
            try:
                self._os.mark_automation_inactive()
                self._os.force_release_all()
                self._os.stop_automated_input()
                self._os.enable_user_input()
            except Exception:
                # safety best-effort, never fatal
                pass

            meta = snapshot.metadata or {}

            # -------------------------------------------------
            # PHASE 1 — EXTENDED STATE (BEST-EFFORT)
            # -------------------------------------------------
            try:
                if meta.get("window_geometry") and hasattr(
                    self._os, "set_window_geometry"
                ):
                    self._os.set_window_geometry(
                        snapshot.focus.window_id,
                        meta["window_geometry"],
                    )
            except Exception:
                pass

            try:
                if meta.get("window_z_order") is not None and hasattr(
                    self._os, "set_window_z_order"
                ):
                    self._os.set_window_z_order(
                        snapshot.focus.window_id,
                        meta["window_z_order"],
                    )
            except Exception:
                pass

            try:
                if meta.get("browser_state") and hasattr(
                    self._os, "restore_browser_state"
                ):
                    self._os.restore_browser_state(
                        meta["browser_state"]
                    )
            except Exception:
                pass

            try:
                if meta.get("media_playback_position") is not None and hasattr(
                    self._os, "set_media_playback_position"
                ):
                    self._os.set_media_playback_position(
                        meta["media_playback_position"]
                    )
            except Exception:
                pass

            # -------------------------------------------------
            # PHASE 2 — CORE STATE (FAIL-CLOSED)
            # -------------------------------------------------
            try:
                self._os.set_cursor_position(
                    snapshot.cursor.x,
                    snapshot.cursor.y,
                )
            except Exception as e:
                raise RestorationError(
                    f"Cursor restore failed: {e}"
                ) from e

            focused = False
            try:
                focused = self._os.focus_window(
                    snapshot.focus.window_id
                )
            except Exception:
                focused = False

            if not focused:
                try:
                    self._os.activate_application(
                        snapshot.application.process_name,
                        snapshot.application.pid,
                    )
                except Exception:
                    pass

            # -------------------------------------------------
            # PHASE 3 — AUTHORITY RESET (SINGLE SOURCE)
            # -------------------------------------------------
            try:
                if self._mode.mode != SystemMode.OBSERVER:
                    self._mode.complete_execution(
                        reason="restoration complete"
                    )
            except Exception as e:
                raise RestorationError(
                    f"Mode reset failed: {e}"
                ) from e

            # -------------------------------------------------
            # PHASE 4 — VERIFICATION (TRUTHFUL)
            # -------------------------------------------------
            self._verify(snapshot)

            # -------------------------------------------------
            # PHASE 5 — COMMIT (ATOMIC)
            # -------------------------------------------------
            self._completed_snapshot_id = snapshot_id

    # -------------------------------------------------
    # Verification
    # -------------------------------------------------

    def _verify(self, snapshot: RestorationSnapshot) -> None:
        # settle time for compositor / input stack
        time.sleep(0.05)

        if self._mode.mode != SystemMode.OBSERVER:
            raise RestorationError(
                f"Post-restore mode invalid: {self._mode.mode}"
            )

        try:
            x, y = self._os.get_cursor_position()
        except Exception as e:
            raise RestorationError(
                f"Cursor verification failed: {e}"
            ) from e

        if (
            abs(x - snapshot.cursor.x) > self.CURSOR_TOLERANCE_PX
            or abs(y - snapshot.cursor.y) > self.CURSOR_TOLERANCE_PX
        ):
            raise RestorationError(
                "Cursor position mismatch after restore"
    )
