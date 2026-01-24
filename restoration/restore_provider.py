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
        """
        self._os = os_backend
        self._restore_completed = False  # NEW: idempotency guard

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def restore(self, snapshot: RestorationSnapshot) -> None:
        """
        Restore workspace state from snapshot.

        This method must never throw partial success.
        Either restoration completes to contract or fails loudly.
        """

        # NEW: idempotency guarantee
        if self._restore_completed:
            return

        # 0. Absolute safety: reassert user dominance
        try:
            # If available, this is stronger than stop_automated_input
            if hasattr(self._os, "force_release_all"):
                self._os.force_release_all()
        except Exception:
            pass

        # 1. Cease all automated control immediately
        try:
            self._os.stop_automated_input()
        except Exception:
            # This must never block restoration
            pass

        # 2. Reassert user input availability
        try:
            self._os.enable_user_input()
        except Exception:
            pass

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

        # 4. Restore window focus (best-effort, validated later)
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

        # 7. Final verification (minimal, contract-bound)
        self._verify_post_restore(snapshot)

        # NEW: mark restoration complete
        self._restore_completed = True

    # -------------------------------------------------
    # Internal Verification
    # -------------------------------------------------

    def _verify_post_restore(self, snapshot: RestorationSnapshot) -> None:
        """
        Verifies restoration success according to contract.
        """

        # Allow OS a brief moment to settle
        time.sleep(0.05)

        # Execution mode must be OBSERVER
        mode = self._os.get_execution_mode()
        if mode != "OBSERVER":
            raise RestorationError(
                f"Post-restore execution mode invalid: {mode}"
            )

        # Cursor position must match (tolerance allowed upstream if needed)
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
