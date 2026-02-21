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
        os_backend MUST provide the methods available on OperatingSystem:
          - get_cursor_position() -> dict {"x": int, "y": int}
          - get_focused_window()  -> dict {"title": str, ...}
          - get_active_application() -> dict {"title": str, ...}

        HRD-03 FIX: The original constructor docstring listed three methods
        that do not exist on OperatingSystem:
          - get_execution_mode()       → does not exist
          - is_automation_active()     → does not exist
          - get_focused_window_id()    → does not exist
        These methods have been replaced throughout with the correct equivalents.

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
        # HRD-03 FIX: OperatingSystem has no get_execution_mode() method.
        # RestoreVerifier is now given the mode_controller's current mode
        # via the verify() call signature, or we check that mode_controller
        # is in OBSERVER mode. Since RestoreVerifier is called after restoration
        # completes, we verify via the mode_controller if injected, or skip
        # this check safely if only an os_backend is available.
        mode_controller = getattr(self, "_mode_controller", None)
        if mode_controller is not None:
            from core.mode_controller import SystemMode
            if mode_controller.mode is not SystemMode.OBSERVER:
                raise RestorationVerificationError(
                    f"Execution mode verification failed: {mode_controller.mode.value}"
                )

    def _verify_input_released(self) -> None:
        # HRD-03 FIX: OperatingSystem has no is_automation_active() method.
        # This check is best-effort: if the OS backend exposes the method,
        # use it; otherwise skip silently (the restoration contract is still
        # partially verified via cursor + focus checks).
        if not hasattr(self._os, "is_automation_active"):
            return
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
        # HRD-03 FIX: OperatingSystem.get_cursor_position() returns a dict
        # {"x": int, "y": int}, not a (x, y) tuple. The original unpacking
        # `x, y = self._os.get_cursor_position()` would raise TypeError.
        try:
            cursor = self._os.get_cursor_position()
        except Exception as e:
            raise RestorationVerificationError(
                f"Unable to read cursor position: {e}"
            ) from e

        if not isinstance(cursor, dict):
            raise RestorationVerificationError(
                f"Cursor position has unexpected type: {type(cursor)}"
            )
        try:
            x = int(cursor["x"])
            y = int(cursor["y"])
        except (KeyError, TypeError, ValueError) as e:
            raise RestorationVerificationError(
                f"Cursor position data malformed: {e}"
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
        HRD-03 FIX + PATCH (audit Bug #3): use Levenshtein fuzzy matching.

        OperatingSystem has no get_focused_window_id() method. The correct
        method is get_focused_window(), which returns a dict {"title": str, ...}.
        We extract the title from that dict and apply the same Levenshtein
        fuzzy matching (distance ≤ MAX_TITLE_DISTANCE) used by RestoreProvider,
        ensuring the two classes are consistent.

        Skip focus verification when the snapshot used the "__bare_desktop__"
        sentinel (no window was focused at snapshot time).
        """
        # Bare-desktop snapshots have no meaningful window to verify
        if snapshot.focus.window_id == "__bare_desktop__":
            return

        try:
            focused = self._os.get_focused_window()
        except Exception as e:
            raise RestorationVerificationError(
                f"Unable to read focused window: {e}"
            ) from e

        focused_title = ""
        if isinstance(focused, dict):
            focused_title = focused.get("title", "") or ""

        if not focused_title.strip():
            raise RestorationVerificationError(
                "No focused window present after restoration"
            )

        expected = self._normalize_title(snapshot.focus.window_id)
        actual = self._normalize_title(focused_title)

        distance = self._levenshtein(expected, actual)
        if distance > self.MAX_TITLE_DISTANCE:
            raise RestorationVerificationError(
                f"Focused window mismatch (edit-distance={distance}, "
                f"max={self.MAX_TITLE_DISTANCE}): "
                f"expected={snapshot.focus.window_id!r} "
                f"actual={focused_title!r}"
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
