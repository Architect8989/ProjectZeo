from __future__ import annotations

from typing import Tuple

from restoration.snapshot_types import RestorationSnapshot


class RestorationVerificationError(RuntimeError):
    pass


class RestoreVerifier:
    """
    Verifies that workspace restoration satisfied the Restoration Contract.

    This verifier does NOT attempt to fix anything.
    It only proves whether restoration succeeded.
    """

    def __init__(self, *, os_backend, cursor_tolerance_px: int = 0):
        """
        os_backend MUST provide:
          - get_cursor_position() -> (x, y)
          - get_focused_window_id() -> str | None
          - get_execution_mode() -> str

        cursor_tolerance_px:
          Allowed pixel deviation for cursor verification.
          Default = 0 (exact match).
        """
        self._os = os_backend
        self._cursor_tol = int(cursor_tolerance_px)

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def verify(self, snapshot: RestorationSnapshot) -> None:
        """
        Verifies restoration against snapshot.

        Raises RestorationVerificationError on failure.
        Returns None on success.
        """

        self._verify_execution_mode()
        self._verify_cursor(snapshot)
        self._verify_focus(snapshot)

        # NEW: explicit success path
        return

    # -------------------------------------------------
    # Verification Steps
    # -------------------------------------------------

    def _verify_execution_mode(self) -> None:
        mode = self._os.get_execution_mode()
        if mode != "OBSERVER":
            raise RestorationVerificationError(
                f"Execution mode verification failed: {mode}"
            )

    def _verify_cursor(self, snapshot: RestorationSnapshot) -> None:
        try:
            x, y = self._os.get_cursor_position()
        except Exception as e:
            raise RestorationVerificationError(
                f"Unable to read cursor position: {e}"
            ) from e

        if not self._within_tolerance(
            (x, y),
            (snapshot.cursor.x, snapshot.cursor.y),
        ):
            raise RestorationVerificationError(
                f"Cursor position mismatch: "
                f"expected=({snapshot.cursor.x},{snapshot.cursor.y}) "
                f"actual=({x},{y})"
            )

    def _verify_focus(self, snapshot: RestorationSnapshot) -> None:
        try:
            focused_id = self._os.get_focused_window_id()
        except Exception as e:
            raise RestorationVerificationError(
                f"Unable to read focused window: {e}"
            ) from e

        if not focused_id:
            raise RestorationVerificationError(
                "No focused window present after restoration"
            )

        if focused_id != snapshot.focus.window_id:
            raise RestorationVerificationError(
                f"Focused window mismatch: "
                f"expected={snapshot.focus.window_id} "
                f"actual={focused_id}"
            )

    # -------------------------------------------------
    # Utilities
    # -------------------------------------------------

    def _within_tolerance(
        self,
        actual: Tuple[int, int],
        expected: Tuple[int, int],
    ) -> bool:
        dx = abs(actual[0] - expected[0])
        dy = abs(actual[1] - expected[1])
        return dx <= self._cursor_tol and dy <= self._cursor_tol
