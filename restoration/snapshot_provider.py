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

from observer.screenpipe_adapter import ScreenpipeAdapter, ScreenpipeBlindnessError
from observer.observer_core import ObserverCore
from core.mode_controller import ModeController, SystemMode


class SnapshotProviderError(RuntimeError):
    pass


class SnapshotProvider:
    """
    Snapshot provider.

    HARD CONTRACT:
    - Snapshot MUST be taken in OBSERVER mode
    - Vision MUST be live at capture time
    - OS core state MUST be captured successfully
    - Any failure aborts execution immediately
    """

    SNAPSHOT_SCHEMA_VERSION = "1.3"

    # -------------------------------------------------
    # PROCESS-LOCAL SNAPSHOT REGISTRY (NON-PERSISTENT)
    # -------------------------------------------------

    _snapshots: Dict[str, RestorationSnapshot] = {}
    _lock = threading.Lock()

    @classmethod
    def store_snapshot(cls, snapshot: RestorationSnapshot) -> str:
        with cls._lock:
            if snapshot.snapshot_id in cls._snapshots:
                raise SnapshotProviderError(
                    f"Snapshot id collision: {snapshot.snapshot_id}"
                )
            cls._snapshots[snapshot.snapshot_id] = snapshot
        return snapshot.snapshot_id

    @classmethod
    def get_snapshot(
        cls, snapshot_id: str
    ) -> Optional[RestorationSnapshot]:
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
        snapshot = self._capture_snapshot()
        return self.store_snapshot(snapshot)

    # -------------------------------------------------
    # INTERNAL SNAPSHOT LOGIC
    # -------------------------------------------------

    def _capture_snapshot(self) -> RestorationSnapshot:
        # ---- wiring validation ----
        if self._observer is None or self._screenpipe is None:
            raise SnapshotProviderError(
                "SnapshotProvider not wired: observer or screenpipe missing"
            )

        # -------------------------------------------------
        # 1. MODE CHECK (ABSOLUTE)
        # -------------------------------------------------
        if self._mode.mode is not SystemMode.OBSERVER:
            raise SnapshotProviderError(
                f"Snapshot attempted in {self._mode.mode.value}; "
                "OBSERVER mode required"
            )

        # -------------------------------------------------
        # 2. VISION CHECK (AUTHORITATIVE)
        # -------------------------------------------------
        if self._screenpipe.blind:
            raise SnapshotProviderError("Screenpipe is blind")

        try:
            screen_state = self._screenpipe.read()
        except ScreenpipeBlindnessError as e:
            raise SnapshotProviderError(
                f"Screenpipe read failed: {e}"
            ) from e

        if not screen_state.get("available"):
            raise SnapshotProviderError(
                "Screen unavailable during snapshot"
            )

        # -------------------------------------------------
        # 3. OS CORE STATE (MANDATORY)
        # -------------------------------------------------
        try:
            cursor_x, cursor_y = self._os.get_cursor_position()
            focused_window = self._os.get_focused_window()
            active_app = self._os.get_active_application()
        except Exception as e:
            raise SnapshotProviderError(
                f"OS state capture failed: {e}"
            ) from e

        if not isinstance(focused_window, dict):
            raise SnapshotProviderError(
                "Focused window state invalid or missing"
            )

        if not isinstance(active_app, dict):
            raise SnapshotProviderError(
                "Active application state invalid or missing"
            )

        # -------------------------------------------------
        # 4. CORE STATE VALIDATION
        # -------------------------------------------------
        try:
            cursor_state = CursorState(
                x=int(cursor_x),
                y=int(cursor_y),
            )
        except Exception as e:
            raise SnapshotProviderError(
                f"Invalid cursor position: {e}"
            ) from e

        try:
            focus_state = FocusState(
                window_id=str(focused_window.get("id")),
                title=focused_window.get("title"),
            )
        except Exception as e:
            raise SnapshotProviderError(
                f"Invalid focused window state: {e}"
            ) from e

        try:
            application_state = ApplicationState(
                process_name=str(active_app.get("process_name")),
                pid=active_app.get("pid"),
            )
        except Exception as e:
            raise SnapshotProviderError(
                f"Invalid application state: {e}"
            ) from e

        # -------------------------------------------------
        # 5. EXTENDED STATE (BEST-EFFORT, NEVER FAILS)
        # -------------------------------------------------
        extended: Dict[str, Any] = {}

        for attr, key in (
            ("get_window_geometry", "window_geometry"),
            ("get_window_z_order", "window_z_order"),
            ("get_browser_state", "browser_state"),
            ("get_media_playback_position", "media_playback_position"),
            ("get_os_signature", "os_signature"),
        ):
            if hasattr(self._os, attr):
                try:
                    extended[key] = getattr(self._os, attr)()
                except Exception:
                    extended[key] = None

        # -------------------------------------------------
        # 6. METADATA (SELF-DESCRIBING)
        # -------------------------------------------------
        metadata = {
            "schema_version": self.SNAPSHOT_SCHEMA_VERSION,
            "captured_at": time.time(),
            "execution_mode": self._mode.mode.value,
            "screen": {
                "frame_ts": screen_state.get("frame_ts"),
                "screen_hash": screen_state.get("screen_hash"),
            },
            "extended": extended,
        }

        # -------------------------------------------------
        # 7. IMMUTABLE SNAPSHOT
        # -------------------------------------------------
        return RestorationSnapshot.create(
            cursor=cursor_state,
            focus=focus_state,
            application=application_state,
            execution_mode=self._mode.mode.value,
            metadata=metadata,
            )
