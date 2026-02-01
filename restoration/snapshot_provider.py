from __future__ import annotations

import time
import threading
from typing import Dict, Any, Optional

from restoration.snapshot_types import (
    CursorState,
    FocusState,
    ApplicationState,
    RestorationSnapshot,
)

# Existing system components (authoritative)
from observer.screenpipe_adapter import ScreenpipeAdapter
from observer.observer_core import ObserverCore


class SnapshotProviderError(RuntimeError):
    pass


class SnapshotProvider:
    """
    Concrete snapshot provider.

    This is NOT optional.
    If this fails, SOC must never run.
    """

    SNAPSHOT_SCHEMA_VERSION = "1.1"

    # -------------------------------------------------
    # SNAPSHOT REGISTRY (AUTHORITATIVE, THREAD-SAFE)
    # -------------------------------------------------

    _snapshots: Dict[str, RestorationSnapshot] = {}
    _lock = threading.Lock()

    @classmethod
    def store_snapshot(cls, snapshot: RestorationSnapshot) -> str:
        snapshot_id = snapshot.snapshot_id
        with cls._lock:
            cls._snapshots[snapshot_id] = snapshot
        return snapshot_id

    @classmethod
    def get_snapshot(cls, snapshot_id: str) -> Optional[RestorationSnapshot]:
        with cls._lock:
            return cls._snapshots.get(snapshot_id)

    # -------------------------------------------------
    # INSTANCE
    # -------------------------------------------------

    def __init__(
        self,
        *,
        observer: Optional[ObserverCore],
        screenpipe: Optional[ScreenpipeAdapter],
        os_backend,
    ):
        # Allow late wiring (Tier-1 requirement)
        self._observer = observer
        self._screenpipe = screenpipe
        self._os = os_backend

    # -------------------------------------------------
    # TIER-1 PUBLIC API (REQUIRED)
    # -------------------------------------------------

    def take_snapshot(self) -> str:
        """
        Public entrypoint expected by main/kernel.
        Returns snapshot_id.
        """
        snapshot = self.capture_pre_hijack_snapshot()
        return self.store_snapshot(snapshot)

    # -------------------------------------------------
    # INTERNAL SNAPSHOT LOGIC
    # -------------------------------------------------

    def capture_pre_hijack_snapshot(self) -> RestorationSnapshot:
        """
        Capture and validate pre-hijack snapshot.

        Hard gate.
        Any exception aborts execution.
        """

        if self._observer is None or self._screenpipe is None:
            raise SnapshotProviderError(
                "SnapshotProvider not fully wired (observer/screenpipe missing)"
            )

        # 1. Enforce execution mode
        execution_mode = self._os.get_execution_mode()
        if execution_mode != "OBSERVER":
            raise SnapshotProviderError(
                f"Snapshot capture attempted in mode '{execution_mode}'. "
                "Snapshots MUST be captured in OBSERVER mode."
            )

        # 2. Enforce live vision (pre-check blindness)
        if getattr(self._screenpipe, "blind", False):
            raise SnapshotProviderError("Screenpipe is blind")

        screen_state = self._screenpipe.read()
        if not screen_state.get("available") or screen_state.get("blind"):
            raise SnapshotProviderError(
                "Screenpipe vision unavailable or blind during snapshot capture"
            )

        # 3. Pull OS-authoritative state
        try:
            cursor_x, cursor_y = self._os.get_cursor_position()
            focused_window = self._os.get_focused_window()
            active_app = self._os.get_active_application()
        except Exception as e:
            raise SnapshotProviderError(
                f"Failed to retrieve OS state: {e}"
            ) from e

        # 4. Build state objects (VALIDATED)
        try:
            x = int(cursor_x)
            y = int(cursor_y)
            if x < 0 or y < 0:
                raise ValueError("Cursor coordinates must be non-negative")
            cursor_state = CursorState(x=x, y=y)
        except (TypeError, ValueError) as e:
            raise SnapshotProviderError(
                f"Invalid cursor position: {e}"
            ) from e

        focus_state = FocusState(
            window_id=str(focused_window.get("id")),
            title=focused_window.get("title"),
        )

        application_state = ApplicationState(
            process_name=str(active_app.get("process_name")),
            pid=active_app.get("pid"),
        )

        # -------------------------------------------------
        # EXTENDED SNAPSHOT DATA (BEST EFFORT)
        # -------------------------------------------------

        window_geometry: Optional[Dict[str, int]] = None
        window_z_order: Optional[int] = None
        browser_state: Optional[Dict[str, Any]] = None
        media_position: Optional[float] = None
        os_signature: Optional[Dict[str, Any]] = None

        try:
            if hasattr(self._os, "get_window_geometry"):
                window_geometry = self._os.get_window_geometry(
                    focused_window.get("id")
                )
        except Exception:
            pass

        try:
            if hasattr(self._os, "get_window_z_order"):
                window_z_order = self._os.get_window_z_order(
                    focused_window.get("id")
                )
        except Exception:
            pass

        try:
            if hasattr(self._os, "get_browser_state"):
                browser_state = self._os.get_browser_state()
        except Exception:
            pass

        try:
            if hasattr(self._os, "get_media_playback_position"):
                media_position = self._os.get_media_playback_position()
        except Exception:
            pass

        try:
            if hasattr(self._os, "get_os_signature"):
                os_signature = self._os.get_os_signature()
        except Exception:
            pass

        # 5. Bind metadata
        metadata: Dict[str, Any] = {
            "schema_version": self.SNAPSHOT_SCHEMA_VERSION,
            "screenpipe": {
                "frame_ts": screen_state.get("frame_ts"),
                "screen_text_hash": screen_state.get("screen_text_hash"),
                "captured_at": time.time(),
            },
            "window_geometry": window_geometry,
            "window_z_order": window_z_order,
            "browser_state": browser_state,
            "media_playback_position": media_position,
            "os_signature": os_signature,
        }

        # 6. Create immutable snapshot
        snapshot = RestorationSnapshot.create(
            cursor=cursor_state,
            focus=focus_state,
            application=application_state,
            execution_mode=execution_mode,
            metadata=metadata,
        )

        return snapshot
