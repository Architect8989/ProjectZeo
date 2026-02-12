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

from observer.observer_core import ObserverCore
from core.mode_controller import ModeController, SystemMode


class SnapshotProviderError(RuntimeError):
    pass


class SnapshotProvider:
    """
    Snapshot provider.

    HARD CONTRACT:
    - Snapshot MUST be taken in OBSERVER mode
    - Observer MUST be healthy
    - Vision MUST be live at capture time
    - OS core state MUST be captured atomically
    - Any failure aborts execution immediately
    """

    SNAPSHOT_SCHEMA_VERSION = "1.8"

    _snapshots: Dict[str, RestorationSnapshot] = {}
    _lock = threading.Lock()

    ATOMIC_WINDOW_SECONDS = 0.02  # 20ms max capture window

    # =========================================================
    # SNAPSHOT REGISTRY
    # =========================================================

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

    # =========================================================
    # INIT
    # =========================================================

    def __init__(
        self,
        *,
        observer: Optional[ObserverCore],
        os_backend,
        mode_controller: ModeController,
    ):
        self._observer = observer
        self._os = os_backend
        self._mode = mode_controller

    # =========================================================
    # PUBLIC
    # =========================================================

    def take_snapshot(self) -> str:
        snapshot = self._capture_snapshot()
        snapshot_id = self.store_snapshot(snapshot)
        return snapshot_id

    # =========================================================
    # INTERNAL CAPTURE
    # =========================================================

    def _capture_snapshot(self) -> RestorationSnapshot:

        if self._observer is None:
            raise SnapshotProviderError(
                "SnapshotProvider not wired: observer missing"
            )

        # 1. MODE CHECK
        if self._mode.mode is not SystemMode.OBSERVER:
            raise SnapshotProviderError(
                f"Snapshot attempted in {self._mode.mode.value}; OBSERVER mode required"
            )

        # 2. OBSERVER HEALTH
        if not self._observer.is_healthy():
            raise SnapshotProviderError(
                "Observer unhealthy during snapshot"
            )

        # 3. VISION CHECK
        observer_state = self._observer.snapshot()

        if not isinstance(observer_state, dict):
            raise SnapshotProviderError(
                "Observer snapshot invalid structure"
            )

        if not observer_state.get("perception_available"):
            raise SnapshotProviderError(
                "Vision unavailable during snapshot"
            )

        frame_ts = observer_state.get("perception_frame_ts")
        if frame_ts is not None and not isinstance(frame_ts, (int, float)):
            raise SnapshotProviderError(
                "Invalid vision frame timestamp"
            )

        # 4. ATOMIC OS STATE CAPTURE (single-pass, bounded window)
        t_start = time.monotonic()

        try:
            cursor = self._os.get_cursor_position()
            focused_window = self._os.get_focused_window()
            active_app = self._os.get_active_application()
        except Exception as e:
            raise SnapshotProviderError(
                f"OS state capture failed: {e}"
            ) from e

        t_end = time.monotonic()

        if (t_end - t_start) > self.ATOMIC_WINDOW_SECONDS:
            raise SnapshotProviderError(
                "OS state capture exceeded atomic window"
            )

        # 5. STRICT VALIDATION

        if not isinstance(cursor, dict):
            raise SnapshotProviderError("Cursor state invalid")

        if "x" not in cursor or "y" not in cursor:
            raise SnapshotProviderError("Cursor coordinates missing")

        try:
            cursor_x = int(cursor["x"])
            cursor_y = int(cursor["y"])
        except Exception as e:
            raise SnapshotProviderError(
                f"Invalid cursor coordinates: {e}"
            ) from e

        if (
            not isinstance(focused_window, dict)
            or not isinstance(focused_window.get("title"), str)
            or not focused_window["title"].strip()
        ):
            raise SnapshotProviderError("Focused window invalid")

        window_title = focused_window["title"].strip()

        if (
            not isinstance(active_app, dict)
            or not isinstance(active_app.get("title"), str)
            or not active_app["title"].strip()
        ):
            raise SnapshotProviderError("Active application invalid")

        app_title = active_app["title"].strip()

        # 6. STATE OBJECTS

        cursor_state = CursorState(
            x=cursor_x,
            y=cursor_y,
        )

        focus_state = FocusState(
            window_id=window_title,
            title=window_title,
        )

        application_state = ApplicationState(
            process_name=app_title,
            pid=0,
        )

        # 7. METADATA

        metadata = {
            "schema_version": self.SNAPSHOT_SCHEMA_VERSION,
            "captured_at_monotonic": t_end,
            "captured_at_wallclock": time.time(),
            "execution_mode": self._mode.mode.value,
            "vision_frame_ts": frame_ts,
            "capture_duration_ms": round(
                (t_end - t_start) * 1000, 3
            ),
        }

        # 8. CREATE SNAPSHOT

        return RestorationSnapshot.create(
            cursor=cursor_state,
            focus=focus_state,
            application=application_state,
            execution_mode=self._mode.mode.value,
            metadata=metadata,
        )
