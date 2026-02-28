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

    

    @staticmethod
    def _derive_snapshot_id(
        *,
        cursor: CursorState,
        focus: FocusState,
        application: ApplicationState,
        execution_mode: str,
        captured_at: float,
    ) -> str:
        
        
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
        """
        Construct and validate a new RestorationSnapshot.

        IH-4 NOTE: The returned ``snapshot_id`` is NOT reproducible across
        process restarts.  It incorporates ``time.time()`` and a
        process-local nonce — see ``_derive_snapshot_id()`` for the full
        limitation analysis.  Callers that need a stable cross-restart
        reference should store the semantic intent hash separately.
        """

        captured_at = time.time()

        
        _process_census: list = []
        try:
            import os as _os
            # Linux /proc enumeration — most reliable and zero-dependency.
            _proc_entries = _os.listdir("/proc")
            _process_census = sorted(
                int(e) for e in _proc_entries if e.isdigit()
            )
        except Exception:
            # Non-Linux (macOS, Windows) or permission error — try psutil.
            try:
                import psutil as _psutil
                _process_census = sorted(
                    p.pid for p in _psutil.process_iter(["pid"])
                )
            except Exception:
                _process_census = []  # census unavailable — degraded mode

        _metadata = dict(metadata or {})
        if _process_census:
            _metadata["process_census_pids"] = _process_census
            _metadata["process_census_at"] = captured_at

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

RestorationSnapshot._nonce_counter = 0  # type: ignore[attr-defined]

RestorationSnapshot._nonce_lock = threading.Lock()  # type: ignore[attr-defined]
