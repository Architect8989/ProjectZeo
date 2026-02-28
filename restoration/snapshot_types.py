from __future__ import annotations

import hashlib
import json
import os
import time
import threading
import uuid
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


# ----------------------------
# Core State Primitives
# ----------------------------

@dataclass(frozen=True)
class CursorState:
    """Absolute cursor position in screen coordinates."""
    x: int
    y: int

    def validate(self) -> None:
        if self.x < 0 or self.y < 0:
            raise ValueError("Cursor coordinates must be non-negative")


@dataclass(frozen=True)
class FocusState:
    """Represents focused window identity."""
    window_id: str
    title: Optional[str] = None

    def validate(self) -> None:
        if not self.window_id:
            raise ValueError("Focused window must have a valid window_id")


@dataclass(frozen=True)
class ApplicationState:
    """Represents the active foreground application."""
    process_name: str
    pid: Optional[int] = None

    def validate(self) -> None:
        if not self.process_name:
            raise ValueError("Active application must have a process_name")
        # "__bare_desktop__" is a valid sentinel for no-window-focused snapshots
        if self.pid is not None and self.pid <= 0:
            raise ValueError("PID must be positive if provided")


# ----------------------------
# Snapshot Aggregate
# ----------------------------

@dataclass(frozen=True)
class RestorationSnapshot:
    """
    Immutable snapshot of pre-hijack workspace state.
    Defines minimum restorable contract.

    Restoration scope: cursor position, focused window, and active
    application ONLY.  File contents, browser state, clipboard,
    network connections, and spawned processes are NOT restored.
    See to_dict() → "restoration_scope" for the explicit enumeration.
    """

    snapshot_id: str
    captured_at: float

    cursor: CursorState
    focus: FocusState
    application: ApplicationState

    execution_mode: str  # MUST be 'OBSERVER'
    metadata: Dict[str, Any] = field(default_factory=dict)

    

    @staticmethod
    def _derive_snapshot_id(
        *,
        cursor: CursorState,
        focus: FocusState,
        application: ApplicationState,
        execution_mode: str,
        captured_at: float,
    ) -> str:
        # H-1 FIX: Replace the process-local _nonce_counter (which resets to 0
        # on every process restart, making sub-millisecond snapshot ID collisions
        # possible between a freshly loaded snapshot and a new one created within
        # 1ms of the crash) with uuid.uuid4().hex[:8].
        #
        # uuid4() generates a cryptographically random 128-bit value — the
        # collision probability is 1/(2^32) ≈ 2.3×10⁻¹⁰ per snapshot pair, which
        # is negligible for all practical use.  Unlike the counter-based approach,
        # it provides this guarantee across process restarts without any persistent
        # state.  The _nonce_counter and _nonce_lock class attributes are no longer
        # needed and have been removed.
        nonce = uuid.uuid4().hex[:8]

        canonical = json.dumps(
            {
                "cursor_x": cursor.x,
                "cursor_y": cursor.y,
                "window_id": focus.window_id,
                "process_name": application.process_name,
                "execution_mode": execution_mode,
                # Round to nearest millisecond to avoid cross-platform float drift
                "captured_at_ms": round(captured_at * 1000),
                # H-1 FIX: crypto-random nonce (uuid4 hex) instead of process-local counter
                "nonce": nonce,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        # Use first 32 hex chars (128 bits) — negligible collision risk
        return digest[:32]

    @staticmethod
    def create(
        *,
        cursor: CursorState,
        focus: FocusState,
        application: ApplicationState,
        execution_mode: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "RestorationSnapshot":
        """
        Construct and validate a new RestorationSnapshot.

        IH-4 NOTE: The returned ``snapshot_id`` is NOT reproducible across
        process restarts.  It incorporates ``time.time()`` and a
        process-local nonce — see ``_derive_snapshot_id()`` for the full
        limitation analysis.  Callers that need a stable cross-restart
        reference should store the semantic intent hash separately.
        """

        captured_at = time.time()

        # RTB-03 / MF-03 FIX: Removed PID-based process census from create().
        # _capture_snapshot() in snapshot_provider.py collects process NAMES via
        # psutil into metadata["extended"]["processes"], which is what
        # _report_unrestored_processes() actually reads.  The old PID list stored in
        # metadata["process_census_pids"] here was never consumed by any verification
        # path, created a schema conflict with the name-based census, and caused the
        # diff to be silently skipped whenever a snapshot was constructed via
        # create() directly (e.g. deserialized from from_dict()).  PID collection
        # belongs exclusively in _capture_snapshot() alongside name resolution.
        _metadata = dict(metadata or {})

        snapshot_id = RestorationSnapshot._derive_snapshot_id(
            cursor=cursor,
            focus=focus,
            application=application,
            execution_mode=execution_mode,
            captured_at=captured_at,
        )

        snapshot = RestorationSnapshot(
            snapshot_id=snapshot_id,
            captured_at=captured_at,
            cursor=cursor,
            focus=focus,
            application=application,
            execution_mode=execution_mode,
            metadata=_metadata,
        )

        snapshot.validate()
        return snapshot

    # ----------------------------
    # Validation
    # ----------------------------

    def validate(self) -> None:
        if not self.snapshot_id:
            raise ValueError("snapshot_id must be present")

        if self.captured_at <= 0:
            raise ValueError("captured_at must be a valid epoch timestamp")

        if self.execution_mode != "OBSERVER":
            raise ValueError(
                f"Invalid execution_mode '{self.execution_mode}'. "
                "Pre-hijack snapshots MUST be captured in OBSERVER mode."
            )

        self.cursor.validate()
        self.focus.validate()
        self.application.validate()

    # ----------------------------
    # Serialization
    # ----------------------------

    def to_dict(self) -> Dict[str, Any]:
        
        return {
            "snapshot_id": self.snapshot_id,
            "captured_at": self.captured_at,
            "execution_mode": self.execution_mode,
            # HAR-01: Explicit scope declaration — prevents callers from
            # treating restoration_success as a full-state rollback guarantee.
            "restoration_scope": "cursor_and_focus_only",
            "restoration_not_restored": [
                "file_contents",
                "browser_state",
                "clipboard",
                "network_connections",
                "spawned_processes",
                "window_geometry",
                "window_z_order",
                
                "keyboard_modifiers_partially",
            ],
            "cursor": {
                "x": self.cursor.x,
                "y": self.cursor.y,
            },
            "focus": {
                "window_id": self.focus.window_id,
                "title": self.focus.title,
            },
            "application": {
                "process_name": self.application.process_name,
                "pid": self.application.pid,
            },
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RestorationSnapshot":
        """
        H-01 FIX: Reconstruct a RestorationSnapshot from a to_dict() payload.

        Used by SnapshotProvider._reload_from_disk() to restore persisted
        snapshots across process restarts.  Raises ValueError on malformed input.
        """
        if not isinstance(data, dict):
            raise ValueError("from_dict(): data must be a dict")

        snapshot_id = str(data.get("snapshot_id", "")).strip()
        if not snapshot_id:
            raise ValueError("from_dict(): missing snapshot_id")

        try:
            captured_at = float(data["captured_at"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"from_dict(): invalid captured_at — {e}") from e

        execution_mode = str(data.get("execution_mode", "observer"))

       
        execution_mode = execution_mode.upper().strip() or "OBSERVER"

        cursor_d = data.get("cursor", {})
        if not isinstance(cursor_d, dict):
            raise ValueError("from_dict(): invalid cursor block")
        cursor = CursorState(x=int(cursor_d["x"]), y=int(cursor_d["y"]))

        focus_d = data.get("focus", {})
        if not isinstance(focus_d, dict):
            raise ValueError("from_dict(): invalid focus block")
        focus = FocusState(
            window_id=str(focus_d.get("window_id", "")),
            title=focus_d.get("title"),
        )

        app_d = data.get("application", {})
        if not isinstance(app_d, dict):
            raise ValueError("from_dict(): invalid application block")
        application = ApplicationState(
            process_name=str(app_d.get("process_name", "__bare_desktop__")),
            pid=app_d.get("pid"),
        )

        metadata = dict(data.get("metadata", {}))

        # Reconstruct with the ORIGINAL snapshot_id (do not re-derive).
        instance = cls(
            snapshot_id=snapshot_id,
            captured_at=captured_at,
            execution_mode=execution_mode,
            cursor=cursor,
            focus=focus,
            application=application,
            metadata=metadata,
        )

        
        try:
            instance.validate()
        except ValueError as _val_err:
            raise ValueError(
                f"from_dict(): snapshot {snapshot_id!r} failed integrity check — "
                f"{_val_err}"
            ) from _val_err

        return instance


# ---------------------------------------------------------------------------
# H7 FIX — Shared window-title matching utilities
# ---------------------------------------------------------------------------
#
# DEFECT (SI-B / H7): RestoreProvider and RestoreVerifier each maintained
# independent copies of _levenshtein() and _strict_match().  The two copies
# had diverged (MAX_TITLE_DISTANCE = 5 in RestoreProvider vs 2 in
# RestoreVerifier — the root cause of the false-positive shutdown bug fixed
# in SI-B/RT-B).  Divergence is only possible because the implementations
# were not shared.
#
# FIX: Move the canonical implementations here, into snapshot_types.py,
# which is the natural home for restoration data types and their associated
# matching semantics.  Both RestoreProvider and RestoreVerifier now import
# and delegate to these functions.  The parameter max_distance is explicit
# so callers must consciously choose the threshold — there is no longer a
# silent per-class default that can drift.
#
# Usage:
#   from restoration.snapshot_types import levenshtein_distance, title_match
#
#   # Soft match (max 5 chars drift — RestoreProvider threshold):
#   ok = title_match(expected, actual, max_distance=5)
#
#   # Strict match (same threshold after SI-B fix):
#   ok = title_match(expected, actual, max_distance=5)
#
# Both RestoreProvider and RestoreVerifier use max_distance=5 after the
# SI-B fix.  The parameter is kept explicit so future callers cannot
# accidentally inherit a wrong default.
# ---------------------------------------------------------------------------

def levenshtein_distance(a: str, b: str) -> int:
    """
    Compute the Levenshtein edit distance between two strings.

    Pure-Python iterative implementation with O(min(len(a), len(b))) space.
    Used for window title fuzzy matching during restoration verification.

    Parameters
    ----------
    a, b:
        Strings to compare.  Both are treated as-is (no normalisation is
        applied here; callers are responsible for lowercasing / stripping
        before calling).

    Returns
    -------
    int
        Edit distance ≥ 0.  Returns 0 when a == b.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # Keep only two rows to bound memory to O(min(|a|, |b|)).
    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            if ca == cb:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr

    return prev[len(b)]


def title_match(expected: str, actual: str, *, max_distance: int) -> bool:
    """
    Return True if *actual* is an acceptable match for *expected*.

    Matching rules (in priority order):
    1. Exact match after stripping leading/trailing whitespace.
    2. One string is a substring of the other (handles truncated titles).
    3. Levenshtein edit distance ≤ max_distance (handles minor drift such
       as unsaved-indicator asterisks, loading suffixes, tab count changes).

    Parameters
    ----------
    expected:
        The canonical title stored at snapshot time.
    actual:
        The title observed at restoration / verification time.
    max_distance:
        Maximum Levenshtein edit distance accepted as a match.  Callers
        must supply this explicitly — there is no module-level default.

        RestoreProvider uses 5 (soft match, RESTORING mode).
        RestoreVerifier uses 5 (aligned after SI-B fix; was incorrectly 2).

    Returns
    -------
    bool
        True if the titles are close enough to be considered equivalent.
    """
    e = expected.strip()
    a = actual.strip()

    if e == a:
        return True
    if e in a or a in e:
        return True
    return levenshtein_distance(e, a) <= max_distance
