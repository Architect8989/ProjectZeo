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

from observer.screenpipe_adapter import ScreenpipeAdapter
from observer.observer_core import ObserverCore
from core.mode_controller import ModeController, SystemMode


class SnapshotProviderError(RuntimeError):
    pass


class SnapshotProvider:
    """
    Concrete snapshot provider.

    HARD CONTRACT:
    - Snapshot MUST be taken in OBSERVER mode
    - Vision MUST be live
    - Any failure aborts SOC execution
    """

    SNAPSHOT_SCHEMA_VERSION = "1.1"

    # -------------------------------------------------
    # SNAPSHOT REGISTRY (PROCESS-LOCAL, THREAD-SAFE)
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
        mode_controller: ModeController,
    ):
        self._observer = observer
        self._screenpipe = screenpipe
        self._os = os_backend
        self._mode = mode_controller

    # -------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------

    def take_snapshot(self) -> str:
        snapshot = self.capture_pre_hijack_snapshot()
        return self.store_snapshot(snapshot)

    # -------------------------------------------------
    # INTERNAL SNAPSHOT LOGIC
    # -------------------------------------------------

    def capture_pre_hijack_snapshot(self) -> RestorationSnapshot:
        """
        Capture and validate pre-hijack snapshot.

        Any exception here MUST abort SOC execution.
        """

        # ---- wiring validation ----
        if self._observer is None or self._screenpipe is None:
            raise SnapshotProviderError(
                "SnapshotProvider not fully wired (observer/screenpipe missing)"
            )

        # -------------------------------------------------
        # 1. AUTHORITY CHECK (SINGLE SOURCE OF TRUTH)
        # -------------------------------------------------
        if self._mode.mode != SystemMode.OBSERVER:
            raise SnapshotProviderError(
                f"Snapshot capture attempted in mode '{self._mode.mode.value}'. "
                "Snapshots MUST be captured in OBSERVER mode."
            )

        # -------------------------------------------------
        # 2. VISION AVAILABILITY CHECK
        # -------------------------------------------------
        if getattr(self._screenpipe, "blind", False):
            raise SnapshotProviderError("Screenpipe is blind")

        screen_state = self._screenpipe.read()
        if not screen_state.get("available") or screen_state.get("blind"):
            raise SnapshotProviderError(
                "Screenpipe vision unavailable during snapshot capture"
            )

        # -------------------------------------------------
        # 3. OS STATE (READ-ONLY)
        # -------------------------------------------------
        try:
            cursor_x, cursor_y = self._os.get_cursor_position()
            focused_window = self._os.get_focused_window()
            active_app = self._os.get_active_application()
        except Exception as e:
            raise SnapshotProviderError(
                f"Failed to retrieve OS state: {e}"
            ) from e

        # -------------------------------------------------
        # 4. VALIDATE CORE STATE
        # -------------------------------------------------
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
        # 5. EXTENDED STATE (BEST EFFORT)
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

        # -------------------------------------------------
        # 6. METADATA BINDING
        # -------------------------------------------------
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

        # -------------------------------------------------
        # 7. IMMUTABLE SNAPSHOT CREATION
        # -------------------------------------------------
        snapshot = RestorationSnapshot.create(
            cursor=cursor_state,
            focus=focus_state,
            application=application_state,
            execution_mode=self._mode.mode.value,
            metadata=metadata,
        )

        return snapshot
