from __future__ import annotations

import time
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
    - Restoration is idempotent, ordered, and fail-closed
    """

    def __init__(self, *, os_backend):
        """
        os_backend MUST provide:
          - stop_automated_input()
          - enable_user_input()
          - set_cursor_position(x, y)
          - focus_window(window_id) -> bool
          - activate_application(process_name, pid) -> bool
          - get_execution_mode()
          - set_execution_mode(mode)

        OPTIONAL:
          - force_release_all()
          - set_window_geometry(window_id, geom)
          - set_window_z_order(window_id, z)
          - restore_browser_state(state)
          - set_media_playback_position(seconds)
          - get_focused_window()
        """
        self._os = os_backend
        self._restore_completed = False

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def restore(self, snapshot: RestorationSnapshot) -> None:
        """
        Restore workspace state from snapshot.

        Atomic in effect:
        either fully restored or raises RestorationError.
        """

        if self._restore_completed:
            return

        # -------------------------------------------------
        # PHASE 0 — HARD SAFETY
        # -------------------------------------------------

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
        # PHASE 1 — EXTENDED STATE
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
            if meta.get("window_z_order") is not None and hasattr(self._os, "set_window_z_order"):
                self._os.set_window_z_order(
                    snapshot.focus.window_id,
                    meta["window_z_order"],
                )
        except Exception:
            pass

        try:
            if meta.get("browser_state") and hasattr(self._os, "restore_browser_state"):
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
        # PHASE 2 — CORE STATE
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
                activated = self._os.activate_application(
                    snapshot.application.process_name,
                    snapshot.application.pid,
                )
                if not activated:
                    raise RestorationError(
                        "Unable to restore focus or activate application"
                    )
            except Exception as e:
                raise RestorationError(
                    f"Application restore failed: {e}"
                ) from e

        # -------------------------------------------------
        # PHASE 3 — MODE RESET
        # -------------------------------------------------

        try:
            self._os.set_execution_mode("OBSERVER")
        except Exception as e:
            raise RestorationError(
                f"Failed to reset execution mode: {e}"
            ) from e

        # -------------------------------------------------
        # PHASE 4 — VERIFY
        # -------------------------------------------------

        self._verify_post_restore(snapshot)

        self._restore_completed = True

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

        if (x, y) != (snapshot.cursor.x, snapshot.cursor.y):
            raise RestorationError(
                "Cursor position verification failed"
            )

        try:
            if hasattr(self._os, "get_focused_window"):
                fw = self._os.get_focused_window()
                if str(fw.get("id")) != snapshot.focus.window_id:
                    raise RestorationError(
                        "Focused window verification failed"
                    )
        except Exception:
            pass
