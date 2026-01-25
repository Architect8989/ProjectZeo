from __future__ import annotations

import time
from typing import Optional

from restoration.snapshot_types import RestorationSnapshot


class RestorationError(RuntimeError):
    pass


class RestoreProvider:
    """
    Concrete restoration provider.

    This provider assumes:
    - Snapshot was captured in OBSERVER mode
    - Snapshot invariants are valid
    """

    def __init__(self, *, os_backend):
        """
        os_backend MUST provide:
          - stop_automated_input() -> None
          - enable_user_input() -> None
          - set_cursor_position(x, y) -> None
          - focus_window(window_id) -> bool
          - activate_application(process_name, pid) -> bool
          - get_execution_mode() -> str
          - set_execution_mode(mode: str) -> None

        OPTIONAL (used if present):
          - set_window_geometry(window_id, geom) -> None
          - set_window_z_order(window_id, z) -> None
          - restore_browser_state(state) -> None
          - set_media_playback_position(seconds) -> None
        """
        self._os = os_backend
        self._restore_completed = False  # idempotency guard

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def restore(self, snapshot: RestorationSnapshot) -> None:
        """
        Restore workspace state from snapshot.

        This method must never throw partial success.
        Either restoration completes to contract or fails loudly.
        """

        if self._restore_completed:
            return

        # 0. Absolute safety: reassert user dominance
        try:
            if hasattr(self._os, "force_release_all"):
                self._os.force_release_all()
        except Exception:
            pass

        # 1. Cease all automated control immediately
        try:
            self._os.stop_automated_input()
        except Exception:
            pass

        # 2. Reassert user input availability
        try:
            self._os.enable_user_input()
        except Exception:
            pass

        # -------------------------------------------------
        # ADDITIONS — EXTENDED RESTORATION
        # -------------------------------------------------

        meta = snapshot.metadata or {}

        try:
            if meta.get("window_geometry") and hasattr(self._os, "set_window_geometry"):
                self._os.set_window_geometry(
                    snapshot.focus.window_id,
                    meta.get("window_geometry"),
                )
        except Exception:
            pass

        try:
            if meta.get("window_z_order") is not None and hasattr(self._os, "set_window_z_order"):
                self._os.set_window_z_order(
                    snapshot.focus.window_id,
                    meta.get("window_z_order"),
                )
        except Exception:
            pass

        try:
            if meta.get("browser_state") and hasattr(self._os, "restore_browser_state"):
                self._os.restore_browser_state(meta.get("browser_state"))
        except Exception:
            pass

        try:
            if meta.get("media_playback_position") is not None and hasattr(
                self._os, "set_media_playback_position"
            ):
                self._os.set_media_playback_position(
                    meta.get("media_playback_position")
                )
        except Exception:
            pass

        # -------------------------------------------------

        # 3. Restore cursor position
        try:
            self._os.set_cursor_position(
                snapshot.cursor.x,
                snapshot.cursor.y,
            )
        except Exception as e:
            raise RestorationError(
                f"Failed to restore cursor position: {e}"
            ) from e

        # 4. Restore window focus
        focused = False
        try:
            focused = self._os.focus_window(snapshot.focus.window_id)
        except Exception:
            focused = False

        # 5. Restore active application if focus failed
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
                    f"Failed to restore application focus: {e}"
                ) from e

        # 6. Force execution mode back to OBSERVER
        try:
            self._os.set_execution_mode("OBSERVER")
        except Exception as e:
            raise RestorationError(
                f"Failed to reset execution mode: {e}"
            ) from e

        # 7. Final verification
        self._verify_post_restore(snapshot)

        self._restore_completed = True

    # -------------------------------------------------
    # Internal Verification
    # -------------------------------------------------

    def _verify_post_restore(self, snapshot: RestorationSnapshot) -> None:
        """
        Verifies restoration success according to contract.
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

        # ADDITION — focus verification if supported
        try:
            if hasattr(self._os, "get_focused_window"):
                fw = self._os.get_focused_window()
                if str(fw.get("id")) != snapshot.focus.window_id:
                    raise RestorationError(
                        "Focused window verification failed"
                    )
        except Exception:
            pass
