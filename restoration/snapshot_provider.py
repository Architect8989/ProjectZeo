from __future__ import annotations

import time
import threading
import json
from typing import Dict, Any, Optional
from collections import OrderedDict

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

    HARD GUARANTEES:
    - Snapshot only in OBSERVER mode
    - Observer must be healthy
    - Vision must be available
    - OS state captured atomically
    - Deterministic serialization
    - Bounded in-memory registry
    - No PID validation violation
    """

    SNAPSHOT_SCHEMA_VERSION = "2.0"

    # ---- BOUNDED SNAPSHOT REGISTRY (LRU discipline) ----
    _snapshots: "OrderedDict[str, RestorationSnapshot]" = OrderedDict()
    _lock = threading.Lock()

    MAX_SNAPSHOTS = 128
    MAX_SNAPSHOT_AGE_SECONDS = 3600

    ATOMIC_WINDOW_SECONDS = 0.02  # 20ms

    # =========================================================
    # SNAPSHOT REGISTRY
    # =========================================================

    @classmethod
    def store_snapshot(cls, snapshot: RestorationSnapshot) -> str:
        now = time.time()

        with cls._lock:
            # Evict stale
            stale_keys = [
                k for k, v in cls._snapshots.items()
                if (now - v.metadata.get("captured_at_wallclock", now))
                > cls.MAX_SNAPSHOT_AGE_SECONDS
            ]
            for k in stale_keys:
                cls._snapshots.pop(k, None)

            # Enforce LRU capacity
            if len(cls._snapshots) >= cls.MAX_SNAPSHOTS:
                cls._snapshots.popitem(last=False)

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
            snap = cls._snapshots.get(snapshot_id)
            if snap:
                cls._snapshots.move_to_end(snapshot_id)
            return snap

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

        # ---- Observer wiring ----
        if self._observer is None:
            raise SnapshotProviderError("Observer missing")

        # ---- Mode enforcement ----
        if self._mode.mode is not SystemMode.OBSERVER:
            raise SnapshotProviderError(
                f"Snapshot attempted in {self._mode.mode.value}"
            )

        # ---- Observer health ----
        if not self._observer.is_healthy():
            raise SnapshotProviderError("Observer unhealthy")

        observer_state = self._observer.snapshot()
        if not isinstance(observer_state, dict):
            raise SnapshotProviderError("Observer snapshot malformed")

        if not observer_state.get("perception_available"):
            raise SnapshotProviderError("Vision unavailable")

        frame_ts = observer_state.get("perception_frame_ts")

        # ---- Atomic OS capture ----
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
                "Atomic capture window exceeded"
            )

        # ---- Strict validation ----
        if not isinstance(cursor, dict):
            raise SnapshotProviderError("Cursor invalid")

        if "x" not in cursor or "y" not in cursor:
            raise SnapshotProviderError("Cursor coordinates missing")

        try:
            cursor_x = int(cursor["x"])
            cursor_y = int(cursor["y"])
        except Exception:
            raise SnapshotProviderError("Cursor coordinate coercion failed")

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

        # ---- State Objects ----
        cursor_state = CursorState(x=cursor_x, y=cursor_y)

        focus_state = FocusState(
            window_id=window_title,
            title=window_title,
        )

        # CRITICAL FIX: pid=None (no validation violation)
        application_state = ApplicationState(
            process_name=app_title,
            pid=None,
        )

        # ---- Deterministic Metadata ----
        metadata = {
            "schema_version": self.SNAPSHOT_SCHEMA_VERSION,
            "captured_at_monotonic": float(t_end),
            "captured_at_wallclock": float(time.time()),
            "execution_mode": self._mode.mode.value,
            "vision_frame_ts": frame_ts,
            "capture_duration_ms": round(
                (t_end - t_start) * 1000.0, 6
            ),
        }

        # Canonicalize metadata deterministically
        metadata = json.loads(
            json.dumps(metadata, sort_keys=True)
        )

        # ---- Create Snapshot ----
        snapshot = RestorationSnapshot.create(
            cursor=cursor_state,
            focus=focus_state,
            application=application_state,
            execution_mode=self._mode.mode.value,
            metadata=metadata,
        )

        return snapshot
