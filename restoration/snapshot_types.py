from __future__ import annotations

import hashlib
import json
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
        FIX H-07: Generate a deterministic content-addressed snapshot ID.

        The ID is SHA-256 of the canonical JSON representation of the core
        snapshot fields. This replaces uuid.uuid4() which produced random IDs
        that broke commitment chain reproducibility across restarts.

        Fields included: cursor (x, y), focus (window_id), application
        (process_name), execution_mode, captured_at (rounded to millisecond
        to avoid float representation drift between Python versions).
        """
        canonical = json.dumps(
            {
                "cursor_x": cursor.x,
                "cursor_y": cursor.y,
                "window_id": focus.window_id,
                "process_name": application.process_name,
                "execution_mode": execution_mode,
                # Round to nearest millisecond to avoid cross-platform float drift
                "captured_at_ms": round(captured_at * 1000),
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

