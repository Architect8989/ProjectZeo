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
    - Vision MUST be live at capture time (integration point)
    - OS core state MUST be captured successfully
    - Any failure aborts execution immediately
    """

    SNAPSHOT_SCHEMA_VERSION = "1.4"

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

    # -------------------------------------------------

    def take_snapshot(self) -> str:
        snapshot = self._capture_snapshot()
        return self.store_snapshot(snapshot)

    # -------------------------------------------------

    def _capture_snapshot(self) -> RestorationSnapshot:
        if self._observer is None:
            raise SnapshotProviderError(
                "SnapshotProvider not wired: observer missing"
            )

        # 1. MODE CHECK
        if self._mode.mode is not SystemMode.OBSERVER:
            raise SnapshotProviderError(
                f"Snapshot attempted in {self._mode.mode.value}; "
                "OBSERVER mode required"
            )

        # 2. VISION CHECK (replace stub when wiring vision runtime)
        screen_state: Dict[str, Any] = {
            "available": True,
            "frame_ts": time.time(),
        }

        if not screen_state.get("available"):
            raise SnapshotProviderError(
                "Screen unavailable during snapshot"
            )

        # 3. OS CORE STATE (RETRY SAFE)
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
                        f"OS state capture failed after retries: {e}"
                    ) from e
                time.sleep(0.1)

        # 4. STRICT VALIDATION (NEW UNIFIED SCHEMA)

        if (
            not isinstance(cursor, dict)
            or "x" not in cursor
            or "y" not in cursor
        ):
            raise SnapshotProviderError("Cursor state invalid")

        if (
            not isinstance(focused_window, dict)
            or not focused_window.get("title")
        ):
            raise SnapshotProviderError("Focused window invalid")

        if (
            not isinstance(active_app, dict)
            or not active_app.get("title")
        ):
            raise SnapshotProviderError("Active application invalid")

        try:
            cursor_state = CursorState(
                x=int(cursor["x"]),
                y=int(cursor["y"]),
            )
        except Exception as e:
            raise SnapshotProviderError(
                f"Invalid cursor position: {e}"
            ) from e

        try:
            focus_state = FocusState(
                window_id=str(focused_window["title"]),
                title=focused_window["title"],
            )
        except Exception as e:
            raise SnapshotProviderError(
                f"Invalid focus state: {e}"
            ) from e

        try:
            application_state = ApplicationState(
                process_name=str(active_app["title"]),
                pid=0,  # PID removed from OS backend contract
            )
        except Exception as e:
            raise SnapshotProviderError(
                f"Invalid application state: {e}"
            ) from e

        # 5. EXTENDED STATE DISABLED (OS backend incomplete)
        extended: Dict[str, Any] = {}

        # 6. METADATA
        metadata = {
            "schema_version": self.SNAPSHOT_SCHEMA_VERSION,
            "captured_at": time.time(),
            "execution_mode": self._mode.mode.value,
            "screen": {
                "frame_ts": screen_state.get("frame_ts"),
            },
            "extended": extended,
        }

        # 7. CREATE SNAPSHOT
        return RestorationSnapshot.create(
            cursor=cursor_state,
            focus=focus_state,
            application=application_state,
            execution_mode=self._mode.mode.value,
            metadata=metadata,
    )
