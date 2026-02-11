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
    - Vision MUST be live at capture time
    - OS core state MUST be captured successfully
    - Any failure aborts execution immediately
    """

    SNAPSHOT_SCHEMA_VERSION = "1.5"

    _snapshots: Dict[str, RestorationSnapshot] = {}
    _lock = threading.Lock()

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
        return self.store_snapshot(snapshot)

    # =========================================================
    # INTERNAL CAPTURE
    # =========================================================

    def _capture_snapshot(self) -> RestorationSnapshot:

        if self._observer is None:
            raise SnapshotProviderError(
                "SnapshotProvider not wired: observer missing"
            )

        # -----------------------------------------------------
        # 1. MODE CHECK (STRICT)
        # -----------------------------------------------------
        if self._mode.mode != SystemMode.OBSERVER:
            raise SnapshotProviderError(
                f"Snapshot attempted in {self._mode.mode.value}; "
                "OBSERVER mode required"
            )

        # -----------------------------------------------------
        # 2. VISION CHECK (REAL, NOT STUB)
        # -----------------------------------------------------
        vision_snapshot = self._observer.snapshot()

        if (
            not isinstance(vision_snapshot, dict)
            or not vision_snapshot.get("available")
        ):
            raise SnapshotProviderError(
                "Vision unavailable during snapshot"
            )

        frame_ts = vision_snapshot.get("frame_ts")

        # -----------------------------------------------------
        # 3. OS CORE STATE (RETRY SAFE)
        # -----------------------------------------------------
        last_error: Optional[Exception] = None

        for attempt in range(3):
            try:
                cursor = self._os.get_cursor_position()
                focused_window = self._os.get_focused_window()
                active_app = self._os.get_active_application()
                break
            except Exception as e:
                last_error = e
                if attempt == 2:
                    raise SnapshotProviderError(
                        f"OS state capture failed: {e}"
                    ) from e
                time.sleep(0.1)

        # -----------------------------------------------------
        # 4. STRICT SCHEMA VALIDATION
        # -----------------------------------------------------

        # Cursor must be {"x": int, "y": int}
        if (
            not isinstance(cursor, dict)
            or "x" not in cursor
            or "y" not in cursor
        ):
            raise SnapshotProviderError("Cursor state invalid")

        try:
            cursor_x = int(cursor["x"])
            cursor_y = int(cursor["y"])
        except Exception as e:
            raise SnapshotProviderError(
                f"Invalid cursor coordinates: {e}"
            ) from e

        # Focused window must provide title
        if (
            not isinstance(focused_window, dict)
            or not focused_window.get("title")
        ):
            raise SnapshotProviderError("Focused window invalid")

        window_title = str(focused_window["title"])

        # Active application must provide title (PID removed)
        if (
            not isinstance(active_app, dict)
            or not active_app.get("title")
        ):
            raise SnapshotProviderError("Active application invalid")

        app_title = str(active_app["title"])

        # -----------------------------------------------------
        # 5. STATE OBJECTS
        # -----------------------------------------------------

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
            pid=0,  # PID intentionally removed from OS contract
        )

        # -----------------------------------------------------
        # 6. METADATA (BOUNDARY FREEZE)
        # -----------------------------------------------------

        metadata = {
            "schema_version": self.SNAPSHOT_SCHEMA_VERSION,
            "captured_at": time.time(),
            "execution_mode": self._mode.mode.value,
            "vision_frame_ts": frame_ts,
        }

        # -----------------------------------------------------
        # 7. CREATE SNAPSHOT
        # -----------------------------------------------------

        return RestorationSnapshot.create(
            cursor=cursor_state,
            focus=focus_state,
            application=application_state,
            execution_mode=self._mode.mode.value,
            metadata=metadata,
        )
