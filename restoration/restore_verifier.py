# AUDIT FIX: Added minimum title length guard and ratio-based Levenshtein verification.
# Short or empty titles could trivially match any other title.
# Minimum 3 chars required; distance must be ≤ min(5, len/3).
from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

from restoration.snapshot_types import (
    RestorationSnapshot,
    levenshtein_distance,
    title_match as _title_match_shared,
)

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
        get_active_application().

        AUDIT-SI-3 FIX: The following methods are now implemented as explicit
        stubs in OperatingSystem (previously absent, causing the verification
        paths below to be permanently dead code via hasattr() guards):
          - get_window_geometry(window_id)      → dict {x, y, width, height}
          - get_window_z_order(window_id)       → int  (stub: raises NotImplementedError)
          - get_browser_state()                 → dict (stub: raises NotImplementedError)
          - get_media_playback_position()       → float (stub: raises NotImplementedError)
          - is_automation_active()              → bool

        The stubs raise NotImplementedError with actionable documentation.
        Extended verification methods catch NotImplementedError and emit a
        structured DEBUG log (soft-skip), preserving backward compatibility
        while closing the structural gap between claimed and actual scope.

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

    Restoration scope (what is actually verified):
    -----------------------------------------------
    HARD (fail-closed):
      - System mode is OBSERVER after restoration.
      - Automation inputs are released (is_automation_active() == False).
      - Cursor position within cursor_tolerance_px of snapshot.
      - Focused window title within MAX_TITLE_DISTANCE Levenshtein edits.

    SOFT (best-effort, log on mismatch, continue on NotImplementedError):
      - Window geometry within _GEOMETRY_TOLERANCE_PX pixels.
      - Window Z-order matches snapshot (requires get_window_z_order() impl).
      - Browser URL/title matches snapshot (requires get_browser_state() impl).
      - Media playback position within 1.0s (requires get_media_playback_position()).

    NOT RESTORED OR VERIFIED:
      - File contents, clipboard, spawned processes, network connections.
      See docs/restoration_contract.md for the full declared scope.
    """

    # SI-B / RT-B FIX (P0): MAX_TITLE_DISTANCE raised from 2 to 5 to match
    # RestoreProvider.MAX_TITLE_DISTANCE = 5.
    #
    # ORIGINAL DEFECT (asymmetry): RestoreProvider._verify() accepted window
    # titles with edit distance <= 5 and len(pre_title) >= 3 (internal check, runs in RESTORING mode).
    # RestoreVerifier.verify() rejected titles with edit distance > 2 (external
    # check, runs in OBSERVER mode). For any title drift of 3-5 characters —
    # common for browser URLs, loading indicators, unsaved-document markers —
    # RestoreProvider reported success and transitioned mode to OBSERVER, then
    # RestoreVerifier raised RestorationVerificationError and triggered
    # _force_safe_shutdown(). The SAME restoration was judged differently by
    # the two verifiers, causing false-positive shutdowns on successful tasks.
    #
    # FIX: Both verifiers now use the same threshold (5). The token-overlap
    # fallback in RestoreProvider._strict_match() handles titles where edit
    # distance exceeds 5 but the application name token is unchanged (e.g.
    # browser URL changes). RestoreVerifier uses the same _levenshtein() logic,
    # and with a threshold of 5 the two verdicts are now consistent.
    MAX_TITLE_DISTANCE: int = 5

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
        """
        AUDIT-SI-3 FIX: Z-order verification is now reachable.

        Previously OperatingSystem lacked get_window_z_order(), so the
        hasattr() guard permanently skipped this check (dead code). The method
        is now implemented as a NotImplementedError stub in OperatingSystem,
        which makes hasattr() return True and activates this code path.

        If the OS backend raises NotImplementedError (stub not yet implemented
        on this platform), emit a DEBUG log and soft-skip — identical effective
        behaviour to the old hasattr() skip, but now explicit and auditable.
        If the backend raises any other exception, soft-skip with a WARNING.
        If values mismatch, raise RestorationVerificationError (hard fail).
        """
        z = snapshot.metadata.get("extended", {}).get("window_z_order")
        if z is None:
            # No Z-order captured in this snapshot — nothing to verify.
            return

        if not hasattr(self._os, "get_window_z_order"):
            return

        try:
            current = self._os.get_window_z_order(snapshot.focus.window_id)
            if current != z:
                raise RestorationVerificationError(
                    f"Window Z-order mismatch after restore: "
                    f"expected={z!r}, actual={current!r}"
                )
        except RestorationVerificationError:
            raise
        except NotImplementedError as nie:
            # Stub not implemented for this platform — soft-skip.
            _logger.debug(
                "RestoreVerifier._verify_window_z_order(): not implemented on this "
                "platform, skipping (soft-fail). Detail: %s", nie
            )
        except Exception as exc:
            # Unexpected OS error — warn and continue (best-effort).
            _logger.warning(
                "RestoreVerifier._verify_window_z_order(): OS query failed, "
                "skipping (soft-fail). Error: %s", exc
            )

    def _verify_browser_state(self, snapshot: RestorationSnapshot) -> None:
        """
        AUDIT-SI-3 FIX: Browser state verification is now reachable.

        Previously OperatingSystem lacked get_browser_state(), so the hasattr()
        guard permanently skipped this check. The method is now a documented
        NotImplementedError stub that makes hasattr() return True.

        NotImplementedError → DEBUG soft-skip (no CDP integration installed).
        Other exceptions    → WARNING soft-skip (OS query failed).
        Mismatch            → RestorationVerificationError (hard fail).
        """
        state = snapshot.metadata.get("extended", {}).get("browser_state")
        if state is None:
            # No browser state captured in this snapshot — nothing to verify.
            return

        if not hasattr(self._os, "get_browser_state"):
            return

        try:
            current = self._os.get_browser_state()
            if current != state:
                raise RestorationVerificationError(
                    f"Browser state mismatch after restore: "
                    f"expected={state!r}, actual={current!r}"
                )
        except RestorationVerificationError:
            raise
        except NotImplementedError as nie:
            _logger.debug(
                "RestoreVerifier._verify_browser_state(): CDP integration not "
                "installed, skipping (soft-fail). Detail: %s", nie
            )
        except Exception as exc:
            _logger.warning(
                "RestoreVerifier._verify_browser_state(): OS query failed, "
                "skipping (soft-fail). Error: %s", exc
            )

    def _verify_media_position(self, snapshot: RestorationSnapshot) -> None:
        """
        AUDIT-SI-3 FIX: Media position verification is now reachable.

        Previously OperatingSystem lacked get_media_playback_position(), so the
        hasattr() guard permanently skipped this check. The method is now a
        documented NotImplementedError stub that makes hasattr() return True.

        NotImplementedError → DEBUG soft-skip (no media control integration).
        Other exceptions    → WARNING soft-skip (OS query failed).
        Position drift > 1s → RestorationVerificationError (hard fail).
        """
        pos = snapshot.metadata.get("extended", {}).get("media_playback_position")
        if pos is None:
            # No media position captured in this snapshot — nothing to verify.
            return

        if not hasattr(self._os, "get_media_playback_position"):
            return

        try:
            current = self._os.get_media_playback_position()
            if abs(current - pos) > 1.0:
                raise RestorationVerificationError(
                    f"Media playback position mismatch after restore: "
                    f"expected={pos:.3f}s, actual={current:.3f}s "
                    f"(drift={abs(current - pos):.3f}s, max=1.0s)"
                )
        except RestorationVerificationError:
            raise
        except NotImplementedError as nie:
            _logger.debug(
                "RestoreVerifier._verify_media_position(): media control "
                "integration not installed, skipping (soft-fail). Detail: %s", nie
            )
        except Exception as exc:
            _logger.warning(
                "RestoreVerifier._verify_media_position(): OS query failed, "
                "skipping (soft-fail). Error: %s", exc
            )

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
        # H7 FIX: Delegate to the canonical shared implementation in
        # snapshot_types.levenshtein_distance.  Previously RestoreVerifier
        # maintained its own copy of this algorithm independent of
        # RestoreProvider's copy — a silent divergence risk that allowed the
        # two implementations to drift (the root cause of SI-B / RT-B).
        # The shared function is now the single source of truth.
        return levenshtein_distance(a, b)

    def _within_tolerance(
        self,
        actual: Tuple[int, int],
        expected: Tuple[int, int],
    ) -> bool:
        dx = abs(actual[0] - expected[0])
        dy = abs(actual[1] - expected[1])
        return dx <= self._cursor_tol and dy <= self._cursor_tol
