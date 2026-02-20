"""
restore_verifier.py — Post-restoration verification against snapshot contract.

PATCH (audit Bug #3):
  _verify_focus() previously used an exact string match on the window ID,
  while RestoreProvider._validate_window() uses Levenshtein fuzzy matching
  (distance ≤ MAX_TITLE_DISTANCE = 2).  This asymmetry meant a window that
  passed RestoreProvider validation could then fail RestoreVerifier
  verification — producing false "restoration failed" errors.

  Fix: _verify_focus() now uses the same Levenshtein fuzzy logic with
  MAX_TITLE_DISTANCE = 2 so the two classes are consistent.  Exact match
  still passes as a special case (distance = 0).

  A new helper _levenshtein() and _normalize_title() mirror the
  RestoreProvider implementations.
"""

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

    # PATCH (audit Bug #3): maximum Levenshtein distance to accept as a
    # "matching" window title.  Must equal RestoreProvider.MAX_TITLE_DISTANCE.
    MAX_TITLE_DISTANCE: int = 2

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
        """
        PATCH (audit Bug #3): use Levenshtein fuzzy matching (distance ≤
        MAX_TITLE_DISTANCE) instead of exact equality so that this check is
        consistent with RestoreProvider._validate_window().

        Window IDs / titles that differ only by minor whitespace differences,
        trailing version strings, etc. (up to 2 edit-distance chars) are
        accepted.  Empty or None focused-window is still a hard failure.
        """
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

        expected = self._normalize_title(snapshot.focus.window_id)
        actual = self._normalize_title(focused_id)

        distance = self._levenshtein(expected, actual)
        if distance > self.MAX_TITLE_DISTANCE:
            raise RestorationVerificationError(
                f"Focused window mismatch (edit-distance={distance}, "
                f"max={self.MAX_TITLE_DISTANCE}): "
                f"expected={snapshot.focus.window_id!r} "
                f"actual={focused_id!r}"
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
    # PATCH helpers (audit Bug #3)
    # -------------------------------------------------

    @staticmethod
    def _normalize_title(text: str) -> str:
        """Lowercase, strip, collapse internal whitespace."""
        if not text:
            return ""
        return " ".join(text.lower().strip().split())

    @staticmethod
    def _levenshtein(a: str, b: str) -> int:
        """Standard O(m·n) Levenshtein edit distance."""
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)

        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            curr = [i]
            for j, cb in enumerate(b, 1):
                insert = curr[j - 1] + 1
                delete = prev[j] + 1
                replace = prev[j - 1] + (ca != cb)
                curr.append(min(insert, delete, replace))
            prev = curr
        return prev[-1]

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
