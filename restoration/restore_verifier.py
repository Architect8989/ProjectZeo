from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

from restoration.snapshot_types import RestorationSnapshot

_logger = logging.getLogger(__name__)


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

    authority_state : object | None
        HAR-07: Optional authority state object. When provided and verification
        fails, RestoreVerifier sets verification_warning=True on the object so
        that verification failures are surfaced in the authority audit record
        rather than only printed to stderr.
    """

    MAX_TITLE_DISTANCE: int = 2

    def __init__(
        self,
        *,
        os_backend,
        mode_controller=None,
        cursor_tolerance_px: int = 0,
        authority_state=None,
    ):
        self._os = os_backend
        # FIX H-04: Accept mode_controller as an explicit constructor parameter.
        # The previous code only checked getattr(self, "_mode_controller", None)
        # which was never set, making _verify_execution_mode() permanently inert.
        self._mode_controller = mode_controller
        self._cursor_tol = int(cursor_tolerance_px)
        # HAR-07: Optional authority state for structured audit event emission.
        self._authority_state = authority_state

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def verify(self, snapshot: RestorationSnapshot) -> None:
        """
        Verifies restoration against snapshot.

        Raises RestorationVerificationError on failure.
        Returns None on success.

        FIX H1 (fail-closed contract):
        --------------------------------
        This verifier raises RestorationVerificationError on any mismatch.
        The CALLER (main.py) is responsible for treating this as a hard failure.

        As of the H1 patch, main.py's except block re-raises
        RestorationVerificationError so it propagates to the outer cleanup
        handler and triggers _force_safe_shutdown(). The system is now
        actually fail-closed: verification failure shuts down execution rather
        than continuing with an unknown workspace state.

        Previously main.py caught RestorationVerificationError and logged a
        warning, then continued execution — a silent swallow that made the
        "fail-closed verification" claim false.

        Contract:
          - OBSERVER mode after restoration (requires mode_controller).
          - Input locks released (is_automation_active() if available).
          - Cursor within cursor_tolerance_px of snapshot position.
          - Focused window title within MAX_TITLE_DISTANCE edits of snapshot title.
          - Extended checks (geometry, z-order, browser, media) are best-effort.

        HAR-07: On verification failure, emits a structured audit event to the
        authority_state (if provided) by setting verification_warning=True so
        that verification failures are surfaced in the authority audit record
        rather than only printed to stderr.
        """
        try:
            self._verify_execution_mode()
            self._verify_input_released()
            self._verify_cursor(snapshot)
            self._verify_focus(snapshot)

            self._verify_window_geometry(snapshot)
            self._verify_window_z_order(snapshot)
            self._verify_browser_state(snapshot)
            self._verify_media_position(snapshot)

        except RestorationVerificationError as exc:
            # HAR-07: Emit structured audit event on verification failure.
            # Previously failures were only printed to stderr (effectively silent
            # in production) and execution proceeded. Now the authority_state is
            # marked with verification_warning=True so the audit record reflects
            # the mismatch and callers can inspect or block execution accordingly.
            self._emit_verification_warning(snapshot, exc)
            raise

    def _emit_verification_warning(
        self,
        snapshot: RestorationSnapshot,
        error: RestorationVerificationError,
    ) -> None:
        """
        HAR-07: Emit a structured verification failure audit event.

        Marks authority_state.verification_warning = True (when available)
        and logs a structured WARNING entry. The log entry includes the
        snapshot_id, captured_at, and the specific mismatch reason so that
        post-hoc audit review can identify which tasks had imperfect restoration.
        """
        event = {
            "event": "RESTORATION_VERIFICATION_FAILURE",
            "timestamp": time.time(),
            "snapshot_id": snapshot.snapshot_id,
            "snapshot_captured_at": snapshot.captured_at,
            "restoration_scope": "cursor_and_focus_only",
            "mismatch_reason": str(error),
        }

        _logger.warning(
            "RestoreVerifier: verification failure for snapshot %s — %s. "
            "Restoration scope is cursor and focus only; file/browser/clipboard "
            "state is NOT restored. Event: %s",
            snapshot.snapshot_id,
            error,
            event,
        )

        # Mark authority_state if injected — surfaces mismatch in audit record.
        if self._authority_state is not None:
            try:
                self._authority_state.verification_warning = True
            except Exception:
                # authority_state may be read-only or not support this attribute;
                # never let audit instrumentation crash the verification path.
                pass

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

    # HARDEN-3 FIX: _GEOMETRY_MISMATCH_IS_HARD_FAILURE controls whether a
    # window geometry mismatch raises RestorationVerificationError (hard failure)
    # or emits a WARNING log and continues (soft failure).
    #
    # Audit finding: window managers legitimately resize, snap, or tile windows
    # during task execution. A geometry mismatch after restoration is
    # informational — it does NOT prove incomplete restoration. The previous
    # implementation raised a hard RestorationVerificationError for any geometry
    # difference, causing false-positive shutdown on tiling WMs, Wayland
    # compositors, and any environment where window geometry is not stable.
    #
    # New behaviour:
    #   - If geometry change is within _GEOMETRY_TOLERANCE_PX pixels on all sides
    #     → treated as normal WM rounding, silently accepted.
    #   - If geometry change exceeds tolerance → WARNING logged, execution continues.
    #     Hard failure is NOT raised by default.
    #   - Set _GEOMETRY_MISMATCH_IS_HARD_FAILURE = True (e.g. in policy.yaml or
    #     subclass) to restore the previous strict behaviour for environments where
    #     geometry stability is guaranteed (e.g. locked-down kiosks).
    _GEOMETRY_MISMATCH_IS_HARD_FAILURE: bool = False
    _GEOMETRY_TOLERANCE_PX: int = 10  # pixels; covers WM rounding and DPI scaling

    def _verify_window_geometry(self, snapshot: RestorationSnapshot) -> None:
        geom = snapshot.metadata.get("extended", {}).get("window_geometry")
        if geom is not None and hasattr(self._os, "get_window_geometry"):
            try:
                current = self._os.get_window_geometry(snapshot.focus.window_id)

                def _to_dict(val):
                    if isinstance(val, dict):
                        return val
                    if isinstance(val, str):
                        parsed: dict = {}
                        for line in val.splitlines():
                            if "=" in line:
                                k, _, v = line.partition("=")
                                try:
                                    parsed[k.strip().lower()] = int(v.strip())
                                except ValueError:
                                    pass
                        return parsed if {"x", "y", "width", "height"}.issubset(parsed) else None
                    return None

                current_d = _to_dict(current)
                geom_d = _to_dict(geom)

                if current_d is not None and geom_d is not None and current_d != geom_d:
                    deltas = [
                        abs(current_d.get(k, 0) - geom_d.get(k, 0))
                        for k in ("x", "y", "width", "height")
                    ]
                    within_tolerance = all(d <= self._GEOMETRY_TOLERANCE_PX for d in deltas)

                    if not within_tolerance:
                        msg = (
                            f"Window geometry changed after restore: "
                            f"snapshot={geom_d!r} current={current_d!r}. "
                            "This may be a legitimate window manager resize. "
                            "Set RestoreVerifier._GEOMETRY_MISMATCH_IS_HARD_FAILURE=True "
                            "to treat this as a hard failure."
                        )
                        if self._GEOMETRY_MISMATCH_IS_HARD_FAILURE:
                            raise RestorationVerificationError(msg)
                        else:
                            _logger.warning(
                                "RestoreVerifier: %s (soft failure — execution continues)",
                                msg,
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
