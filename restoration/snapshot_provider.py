from __future__ import annotations

import time
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

    def __init__(
        self,
        *,
        observer: ObserverCore,
        screenpipe: ScreenpipeAdapter,
        os_backend,
    ):
        self._observer = observer
        self._screenpipe = screenpipe
        self._os = os_backend

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def capture_pre_hijack_snapshot(self) -> RestorationSnapshot:
        """
        Capture and validate pre-hijack snapshot.

        Hard gate.
        Any exception aborts execution.
        """

        # 1. Enforce execution mode
        execution_mode = self._os.get_execution_mode()
        if execution_mode != "OBSERVER":
            raise SnapshotProviderError(
                f"Snapshot capture attempted in mode '{execution_mode}'. "
                "Snapshots MUST be captured in OBSERVER mode."
            )

        # 2. Enforce live vision
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

        # 4. Build state objects
        cursor_state = CursorState(x=int(cursor_x), y=int(cursor_y))

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

        # 5. Bind metadata (NO snapshot_id duplication)
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

        # 6. Create immutable snapshot (single authority)
        snapshot = RestorationSnapshot.create(
            cursor=cursor_state,
            focus=focus_state,
            application=application_state,
            execution_mode=execution_mode,
            metadata=metadata,
        )

        # 7. Notify observer (witness continuity)
        try:
            self._observer.attach_ui_snapshot(snapshot.to_dict())
        except Exception:
            pass

        return snapshot
