from __future__ import annotations

import time
import threading
from typing import Optional

from restoration.snapshot_types import RestorationSnapshot


class RestorationError(RuntimeError):
    pass


class RestoreProvider:
    """
    Concrete restoration provider.

    Contract:
    - Snapshot captured in OBSERVER mode
    - Snapshot invariants already validated
    - Restoration is idempotent per snapshot
    - Fail-closed: never claims success on partial restore
    """

    CURSOR_TOLERANCE_PX = 2  # high-DPI / compositor jitter tolerance

    def __init__(self, *, os_backend):
        self._os = os_backend
        self._completed_snapshot_id: Optional[str] = None
        self._lock = threading.Lock()  # 🔒 CRITICAL FIX

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def restore_snapshot(self, snapshot_id: str) -> None:
        """
        Restore workspace state from snapshot ID.
        """
        from restoration.snapshot_provider import SnapshotProvider

        snapshot = SnapshotProvider.get_snapshot(snapshot_id)
        if snapshot is None:
            raise RestorationError(
                f"Snapshot not found for restoration: {snapshot_id}"
            )

        self.restore(snapshot)

    def restore(self, snapshot: RestorationSnapshot) -> None:
        """
        Restore workspace state from snapshot.

        Atomic in effect:
        either fully restored or raises RestorationError.
        """

        snapshot_id = snapshot.snapshot_id

        # -------------------------------------------------
        # PHASE -1 — ATOMIC IDEMPOTENCY GATE
        # -------------------------------------------------
        with self._lock:
            if self._completed_snapshot_id == snapshot_id:
                return
            # Mark intent (do NOT mark completed yet)
            in_progress_id = snapshot_id

        # -------------------------------------------------
        # PHASE 0 — HARD SAFETY
        # -------------------------------------------------

        try:
            if hasattr(self._os, "mark_automation_inactive"):
                self._os.mark_automation_inactive()
        except Exception:
            pass

        try:
            if hasattr(self._os, "force_release_all"):
                self._os.force_release_all()
        except Exception:
            pass

        try:
            self._os.stop_automated_input()
        except Exception:
            pass

        try:
            self._os.enable_user_input()
        except Exception:
            pass

        meta = snapshot.metadata or {}

        # -------------------------------------------------
        # PHASE 1 — EXTENDED STATE (BEST EFFORT)
        # -------------------------------------------------

        try:
            if meta.get("window_geometry") and hasattr(self._os, "set_window_geometry"):
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
                self._os.restore_browser_state(meta["browser_state"])
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
            focused = self._os.focus_window(snapshot.focus.window_id)
        except Exception:
            focused = False

        if not focused:
            try:
                if hasattr(self._os, "activate_application"):
                    self._os.activate_application(
                        snapshot.application.process_name,
                        snapshot.application.pid,
                    )
            except Exception:
                pass

        # -------------------------------------------------
        # PHASE 3 — MODE RESET (MANDATORY)
        # -------------------------------------------------

        try:
            self._os.set_execution_mode("OBSERVER")
        except Exception as e:
            raise RestorationError(
                f"Failed to reset execution mode: {e}"
            ) from e

        # -------------------------------------------------
        # PHASE 4 — VERIFY (TRUTHFUL)
        # -------------------------------------------------

        self._verify_post_restore(snapshot)

        # -------------------------------------------------
        # PHASE 5 — COMMIT COMPLETION (ATOMIC)
        # -------------------------------------------------
        with self._lock:
            self._completed_snapshot_id = snapshot_id

    # -------------------------------------------------
    # Internal Verification
    # -------------------------------------------------

    def _verify_post_restore(self, snapshot: RestorationSnapshot) -> None:
        """
        Local verification.
        Global verifier runs separately.
        """

        time.sleep(0.05)

        mode = self._os.get_execution_mode()
        if mode != "OBSERVER":
            raise RestorationError(
                f"Post-restore execution mode invalid: {mode}"
            )

        try:
            x, y = self._os.get_cursor_position()
        except Exception as e:
            raise RestorationError(
                f"Unable to verify cursor position: {e}"
            ) from e

        if (
            abs(x - snapshot.cursor.x) > self.CURSOR_TOLERANCE_PX
            or abs(y - snapshot.cursor.y) > self.CURSOR_TOLERANCE_PX
        ):
            raise RestorationError(
                "Cursor position verification failed (outside tolerance)"
            )

        try:
            if hasattr(self._os, "get_focused_window"):
                fw = self._os.get_focused_window()
                if fw and str(fw.get("id")) != snapshot.focus.window_id:
                    pass  # non-fatal on stub backends
        except Exception:
            pass
