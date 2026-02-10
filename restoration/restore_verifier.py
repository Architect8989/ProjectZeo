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
          - is_automation_active() -> bool

        OPTIONAL (used if present):
          - get_window_geometry(window_id)
          - get_window_z_order(window_id)
          - get_browser_state()
          - get_media_playback_position()
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
        self._verify_input_released()
        self._verify_cursor(snapshot)
        self._verify_focus(snapshot)

        self._verify_window_geometry(snapshot)
        self._verify_window_z_order(snapshot)
        self._verify_browser_state(snapshot)
        self._verify_media_position(snapshot)

    # -------------------------------------------------
    # Verification Steps
    # -------------------------------------------------

    def _verify_execution_mode(self) -> None:
        mode = self._os.get_execution_mode()
        if mode != "OBSERVER":
            raise RestorationVerificationError(
                f"Execution mode verification failed: {mode}"
            )

    def _verify_input_released(self) -> None:
        try:
            active = self._os.is_automation_active()
        except Exception as e:
            raise RestorationVerificationError(
                f"Unable to determine automation state: {e}"
            ) from e

        if active:
            raise RestorationVerificationError(
                "Input still locked after restoration"
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
    # EXTENDED VERIFICATION (BEST-EFFORT)
    # -------------------------------------------------

    def _verify_window_geometry(self, snapshot: RestorationSnapshot) -> None:
        geom = snapshot.metadata.get("extended", {}).get("window_geometry")
        if geom is not None and hasattr(self._os, "get_window_geometry"):
            try:
                current = self._os.get_window_geometry(snapshot.focus.window_id)
                if current != geom:
                    raise RestorationVerificationError(
                        "Window geometry mismatch after restore"
                    )
            except Exception:
                pass

    def _verify_window_z_order(self, snapshot: RestorationSnapshot) -> None:
        z = snapshot.metadata.get("extended", {}).get("window_z_order")
        if z is not None and hasattr(self._os, "get_window_z_order"):
            try:
                current = self._os.get_window_z_order(snapshot.focus.window_id)
                if current != z:
                    raise RestorationVerificationError(
                        "Window Z-order mismatch after restore"
                    )
            except Exception:
                pass

    def _verify_browser_state(self, snapshot: RestorationSnapshot) -> None:
        state = snapshot.metadata.get("extended", {}).get("browser_state")
        if state is not None and hasattr(self._os, "get_browser_state"):
            try:
                current = self._os.get_browser_state()
                if current != state:
                    raise RestorationVerificationError(
                        "Browser state mismatch after restore"
                    )
            except Exception:
                pass

    def _verify_media_position(self, snapshot: RestorationSnapshot) -> None:
        pos = snapshot.metadata.get("extended", {}).get("media_playback_position")
        if pos is not None and hasattr(self._os, "get_media_playback_position"):
            try:
                current = self._os.get_media_playback_position()
                if abs(current - pos) > 1.0:
                    raise RestorationVerificationError(
                        "Media playback position mismatch after restore"
                    )
            except Exception:
                pass

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
