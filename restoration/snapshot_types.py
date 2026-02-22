from __future__ import annotations

import hashlib
import json
import os
import time
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

    # HARD-7: Process-local monotonic counter to prevent ID collisions when
    # two snapshots are taken within the same millisecond (common in rapid
    # replan sequences). The counter is process-local so it resets on restart,
    # but within a session guarantees unique IDs regardless of wall-clock speed.
    _nonce_counter: int = 0

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
        distinct IDs. Without this, round(captured_at * 1000) could collide
        and trigger SnapshotProviderError("Snapshot id collision").
        """
        # Atomically advance the nonce
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
        return {
            "snapshot_id": self.snapshot_id,
            "captured_at": self.captured_at,
            "execution_mode": self.execution_mode,
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

