from __future__ import annotations

from typing import Optional, Tuple

from restoration.snapshot_types import RestorationSnapshot


class RestorationVerificationError(RuntimeError):
    pass


class RestoreVerifier:
    """
    Verifies that workspace restoration satisfied the Restoration Contract.

    This verifier does NOT attempt to fix anything.
    It only proves whether restoration succeeded.

    Constructor parameters
    ----------------------
    os_backend : object
        Must implement: get_cursor_position(), get_focused_window(),
        get_active_application(). Optional: get_window_geometry(),
        get_window_z_order(), get_browser_state(),
        get_media_playback_position(), is_automation_active().

    mode_controller : ModeController | None
        FIX H-04: Injected here rather than via hidden attribute assignment.
        When provided, _verify_execution_mode() confirms the controller is
        in OBSERVER mode after restoration. Pass None only in unit tests
        where mode state is managed externally.

    cursor_tolerance_px : int
        Maximum pixel distance from snapshot cursor position to pass
        cursor verification. Should match RestoreProvider.CURSOR_TOLERANCE_PX.
    """

    MAX_TITLE_DISTANCE: int = 2

    def __init__(
        self,
        *,
        os_backend,
        mode_controller=None,
        cursor_tolerance_px: int = 0,
    ):
        self._os = os_backend
        # FIX H-04: Accept mode_controller as an explicit constructor parameter.
        # The previous code only checked getattr(self, "_mode_controller", None)
        # which was never set, making _verify_execution_mode() permanently inert.
        self._mode_controller = mode_controller
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
        """
        FIX H-04: mode_controller is now properly injected via __init__.
        Verifies the system is in OBSERVER mode after restoration completes.
        Skipped only when mode_controller is explicitly None (test contexts).
        """
        if self._mode_controller is None:
            # Explicitly opted out — acceptable in test contexts only.
            return

        from core.mode_controller import SystemMode
        current_mode = self._mode_controller.mode
        if current_mode is not SystemMode.OBSERVER:
            raise RestorationVerificationError(
                f"Execution mode verification failed: expected OBSERVER, "
                f"got {current_mode.value}. Restoration may be incomplete."
            )

    def _verify_input_released(self) -> None:
        """
        Best-effort check: if the OS backend exposes is_automation_active(),
        verify automation is no longer active. Skipped silently if the method
        is absent (OperatingSystem does not expose it by default).
        """
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
                f"actual=({x},{y}) tolerance={self._cursor_tol}px"
            )

    def _verify_focus(self, snapshot: RestorationSnapshot) -> None:
        """
        Uses Levenshtein fuzzy matching (audit Bug #3 fix, preserved).
        Skips bare-desktop snapshots where no window was focused.
        """
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
            except RestorationVerificationError:
                raise
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
            except RestorationVerificationError:
                raise
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
            except RestorationVerificationError:
                raise
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
            except RestorationVerificationError:
                raise
            except Exception:
                pass

    # -------------------------------------------------
    # Helpers
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

    def _within_tolerance(
        self,
        actual: Tuple[int, int],
        expected: Tuple[int, int],
    ) -> bool:
        dx = abs(actual[0] - expected[0])
        dy = abs(actual[1] - expected[1])
        return dx <= self._cursor_tol and dy <= self._cursor_tol

