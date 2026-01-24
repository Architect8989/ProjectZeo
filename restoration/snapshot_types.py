"""
Restoration Snapshot Types

This module defines the authoritative data structures used to
capture, transport, and validate pre-hijack workspace state.

There is NO OS logic here.
There is NO restoration logic here.

These structures are contract-bound and immutable in meaning.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


# ----------------------------
# Core State Primitives
# ----------------------------

@dataclass(frozen=True)
class CursorState:
    """
    Absolute cursor position in screen coordinates.
    """
    x: int
    y: int

    def validate(self) -> None:
        if self.x < 0 or self.y < 0:
            raise ValueError("Cursor coordinates must be non-negative")


@dataclass(frozen=True)
class FocusState:
    """
    Represents focused window identity.
    """
    window_id: str
    title: Optional[str] = None

    def validate(self) -> None:
        if not self.window_id:
            raise ValueError("Focused window must have a valid window_id")


@dataclass(frozen=True)
class ApplicationState:
    """
    Represents the active foreground application.
    """
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

    This snapshot defines the minimum restorable contract.
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
    def create(
        *,
        cursor: CursorState,
        focus: FocusState,
        application: ApplicationState,
        execution_mode: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "RestorationSnapshot":
        """
        Factory method enforcing invariants at creation time.
        """
        snapshot = RestorationSnapshot(
            snapshot_id=str(uuid.uuid4()),
            captured_at=time.time(),
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
        """
        Enforces Restoration Contract invariants.

        Any violation here is a fatal programming error.
        """
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
    # Serialization (Evidence)
    # ----------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Deterministic serialization for audit and evidence.
        """
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
  }"""
Restoration Snapshot Types

This module defines the authoritative data structures used to
capture, transport, and validate pre-hijack workspace state.

There is NO OS logic here.
There is NO restoration logic here.

These structures are contract-bound and immutable in meaning.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


# ----------------------------
# Core State Primitives
# ----------------------------

@dataclass(frozen=True)
class CursorState:
    """
    Absolute cursor position in screen coordinates.
    """
    x: int
    y: int

    def validate(self) -> None:
        if self.x < 0 or self.y < 0:
            raise ValueError("Cursor coordinates must be non-negative")


@dataclass(frozen=True)
class FocusState:
    """
    Represents focused window identity.
    """
    window_id: str
    title: Optional[str] = None

    def validate(self) -> None:
        if not self.window_id:
            raise ValueError("Focused window must have a valid window_id")


@dataclass(frozen=True)
class ApplicationState:
    """
    Represents the active foreground application.
    """
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

    This snapshot defines the minimum restorable contract.
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
    def create(
        *,
        cursor: CursorState,
        focus: FocusState,
        application: ApplicationState,
        execution_mode: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "RestorationSnapshot":
        """
        Factory method enforcing invariants at creation time.
        """
        snapshot = RestorationSnapshot(
            snapshot_id=str(uuid.uuid4()),
            captured_at=time.time(),
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
        """
        Enforces Restoration Contract invariants.

        Any violation here is a fatal programming error.
        """
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
    # Serialization (Evidence)
    # ----------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Deterministic serialization for audit and evidence.
        """
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
