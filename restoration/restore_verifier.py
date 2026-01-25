from __future__ import annotations

from typing import Tuple, Optional, Dict, Any

from restoration.snapshot_types import RestorationSnapshot


class RestorationVerificationError(RuntimeError):
    pass


class RestoreVerifier:
    """
    Verifies that workspace restoration satisfied the Restoration Contract.

    This verifier does NOT attempt to fix anything.
    It only proves whether restoration succeeded.
    """

    def __init__(self, *, os_backend, screenpipe=None, cursor_tolerance_px: int = 0):
        """
        os_backend MUST provide:
          - get_cursor_position() -> (x, y)
          - get_focused_window_id() -> str | None
          - get_execution_mode() -> str

        OPTIONAL (used if present):
          - get_window_geometry(window_id) -> {x,y,width,height}
          - get_window_z_order(window_id) -> int
          - get_browser_state() -> {url, tab_index}
          - get_media_playback_position() -> float

        screenpipe (optional):
          ScreenpipeAdapter for visual hash verification
        """
        self._os = os_backend
        self._screenpipe = screenpipe
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

        # ADDITIONS
        self._verify_window_geometry(snapshot)
        self._verify_window_z_order(snapshot)
        self._verify_browser_state(snapshot)
        self._verify_media_position(snapshot)
        self._verify_screen_hash(snapshot)

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
    # ADDITIONS — EXTENDED VERIFICATION
    # -------------------------------------------------

    def _verify_window_geometry(self, snapshot: RestorationSnapshot) -> None:
        geom = snapshot.metadata.get("window_geometry")
        if geom and hasattr(self._os, "get_window_geometry"):
            try:
                current = self._os.get_window_geometry(snapshot.focus.window_id)
                if current != geom:
                    raise RestorationVerificationError(
                        "Window geometry mismatch after restore"
                    )
            except Exception:
                pass

    def _verify_window_z_order(self, snapshot: RestorationSnapshot) -> None:
        z = snapshot.metadata.get("window_z_order")
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
        state = snapshot.metadata.get("browser_state")
        if state and hasattr(self._os, "get_browser_state"):
            try:
                current = self._os.get_browser_state()
                if current != state:
                    raise RestorationVerificationError(
                        "Browser state mismatch after restore"
                    )
            except Exception:
                pass

    def _verify_media_position(self, snapshot: RestorationSnapshot) -> None:
        pos = snapshot.metadata.get("media_playback_position")
        if pos is not None and hasattr(self._os, "get_media_playback_position"):
            try:
                current = self._os.get_media_playback_position()
                if abs(current - pos) > 1.0:
                    raise RestorationVerificationError(
                        "Media playback position mismatch after restore"
                    )
            except Exception:
                pass

    def _verify_screen_hash(self, snapshot: RestorationSnapshot) -> None:
        """
        Final authority: pixels must match snapshot evidence.
        """
        if not self._screenpipe:
            return

        meta = snapshot.metadata.get("screenpipe")
        if not meta:
            return

        try:
            state = self._screenpipe.read()
        except Exception as e:
            raise RestorationVerificationError(
                f"Unable to read screen for restore verification: {e}"
            ) from e

        expected_hash = meta.get("screen_text_hash")
        actual_hash = state.get("screen_text_hash")

        if expected_hash and actual_hash != expected_hash:
            raise RestorationVerificationError(
                "Post-restore screen hash mismatch"
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
