from __future__ import annotations

import hashlib
import json
import os
import time
import threading
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

    # ----------------------------
    # Construction
    # ----------------------------

    # HARD-7 / FIX-04 (SI-02): Process-local monotonic counter to prevent ID
    # collisions when two snapshots are taken within the same millisecond.
    #
    # FIX-04: The counter was previously declared as a dataclass instance field
    # (_nonce_counter: int = 0), which made it a constructor parameter and part
    # of __eq__/__hash__. Because create() never passed _nonce_counter, all
    # frozen instances had _nonce_counter=0 in their fields while the actual
    # counter lived as a class variable mutated by _derive_snapshot_id().
    # The decoupling between the instance field (always 0) and the class counter
    # (correct) created latent risk for equality-based deduplication.
    #
    # Fix: declare the counter OUTSIDE the dataclass body as a true class
    # attribute, completely separate from the frozen instance fields.

    @staticmethod
    def _derive_snapshot_id(
        *,
        cursor: CursorState,
        focus: FocusState,
        application: ApplicationState,
        execution_mode: str,
        captured_at: float,
    ) -> str:
        """
        HARD-7: Generate a collision-resistant content-addressed snapshot ID.

        Includes a process-local monotonic nonce so that two snapshots taken
        within the same millisecond (common in replan sequences) produce
        distinct IDs.  Without this, round(captured_at * 1000) could collide
        and trigger SnapshotProviderError("Snapshot id collision").
        """
        # HARD-3 (RB-6): Thread-safe nonce increment.
        #
        # Bug: RestorationSnapshot._nonce_counter += 1 is a non-atomic
        # read-modify-write on a class-level integer. Under concurrent snapshot
        # creation (replan sequences that briefly overlap, or test threads),
        # two threads could read the same counter value and compute identical
        # nonces. Two snapshots created within the same millisecond would then
        # produce identical snapshot_ids, causing SnapshotProviderError("Snapshot
        # id collision") and blocking replan arming.
        #
        # Fix: protect the increment with a class-level threading.Lock.
        # The lock is a module-level singleton (defined after the class body
        # alongside the counter initialisation) so it is shared across all
        # calls to _derive_snapshot_id() regardless of when they occur.
        with RestorationSnapshot._nonce_lock:
            RestorationSnapshot._nonce_counter += 1
            nonce = RestorationSnapshot._nonce_counter

        canonical = json.dumps(
            {
                "cursor_x": cursor.x,
                "cursor_y": cursor.y,
                "window_id": focus.window_id,
                "process_name": application.process_name,
                "execution_mode": execution_mode,
                # Round to nearest millisecond to avoid cross-platform float drift
                "captured_at_ms": round(captured_at * 1000),
                # HARD-7: monotonic nonce prevents same-millisecond collisions
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

        captured_at = time.time()

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
            metadata=metadata or {},
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
        # HAR-01: Explicitly enumerate what IS and IS NOT restored so that
        # callers cannot interpret a restoration success as full state rollback.
        # The restoration_scope field is the authoritative declaration of the
        # shallow restoration guarantee: only cursor position, focused window,
        # and active application are restored.  All other state (file contents,
        # browser URL/tabs/forms, clipboard, network connections, spawned
        # processes, window geometry/z-order) persists from task execution.
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
                # H-6 FIX: keyboard_modifiers_partially was absent from this
                # list despite force_release_all() performing modifier-key
                # release during restoration. The omission implied the keyboard
                # was fully restored, which is misleading: only modifier keys
                # (Ctrl, Shift, Alt, Win) are explicitly released. Key-down state
                # for non-modifier keys (e.g. a held arrow key) is NOT restored.
                # Adding this entry with the _partially suffix makes the shallow
                # guarantee explicit and auditable.
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

        # SI-3 / P2-1 FIX: Normalize execution_mode to uppercase before use.
        #
        # Root cause: from_dict() used the raw string from disk JSON, defaulting
        # to "observer" (lowercase). validate() checks execution_mode != "OBSERVER"
        # (uppercase). Because "observer" != "OBSERVER", every snapshot reloaded
        # from disk failed validate() — but from_dict() never called validate(),
        # so corrupted or lowercase snapshots silently bypassed the mode guard
        # and were used by RestoreProvider without any integrity check.
        #
        # Two-part fix:
        #   1. Normalize to uppercase here so the field is always canonical.
        #   2. Call instance.validate() before returning (see below) so malformed
        #      snapshots raise ValueError rather than reaching RestoreProvider.
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

        # SI-3 FIX: Validate the reconstructed snapshot before returning.
        #
        # Root cause: from_dict() previously returned the instance without
        # calling validate(). Corrupted JSON on disk (wrong execution_mode,
        # negative captured_at, empty snapshot_id, etc.) would be loaded into
        # memory as a RestorationSnapshot and used by RestoreProvider without
        # any integrity check. The broken snapshot would survive until the first
        # actual restore attempt, where it might cause unpredictable behaviour.
        #
        # Fix: call validate() here and convert any ValueError into a structured
        # ValueError so SnapshotProvider._reload_from_disk() can log and discard
        # the corrupted file rather than crashing the reload loop.
        try:
            instance.validate()
        except ValueError as _val_err:
            raise ValueError(
                f"from_dict(): snapshot {snapshot_id!r} failed integrity check — "
                f"{_val_err}"
            ) from _val_err

        return instance


# FIX-04 (SI-02): Declare the nonce counter as a TRUE class-level attribute
# OUTSIDE the dataclass body.  Inside the @dataclass(frozen=True) body,
# any annotation becomes a dataclass field — a constructor parameter and part
# of __eq__/__hash__.  By placing the assignment here (after the class body),
# Python sees it as a plain class attribute: mutable, not frozen, not in
# __init__, and invisible to equality comparisons.
#
# This repairs the structural decoupling between the class-level counter
# (correctly incremented by _derive_snapshot_id) and the per-instance field
# (always 0 because create() never passed _nonce_counter to __init__).
RestorationSnapshot._nonce_counter = 0  # type: ignore[attr-defined]

# HARD-3 (RB-6): Class-level lock for thread-safe nonce increment.
# Must be initialised here (outside the frozen dataclass body) alongside
# the counter. The lock is shared across all _derive_snapshot_id() calls.
RestorationSnapshot._nonce_lock = threading.Lock()  # type: ignore[attr-defined]
