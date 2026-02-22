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
    - OS state captured within bounded window
    - Deterministic serialization
    - Instance-isolated LRU registry
    """

    SNAPSHOT_SCHEMA_VERSION = "2.2"

    MAX_SNAPSHOTS = 128
    MAX_SNAPSHOT_AGE_SECONDS = 3600
    # RTB-04: Increased from 0.25s to 0.5s. Under OS load, three consecutive
    # syscalls (cursor + focused_window + active_app) frequently exceeded 250ms,
    # causing permanent denial-of-service on task arming under adversarial load.
    # 500ms is still conservative enough to catch genuine OS hangs.
    ATOMIC_WINDOW_SECONDS = 0.5

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

        # instance-local registry
        self._snapshots: "OrderedDict[str, RestorationSnapshot]" = OrderedDict()
        self._lock = threading.Lock()

    # =========================================================
    # SNAPSHOT REGISTRY (LRU + TTL)
    # =========================================================

    def _evict_stale(self, now: float) -> None:
        stale_keys = []

        for k, v in self._snapshots.items():
            captured = v.metadata.get("captured_at_wallclock", now)
            try:
                captured = float(captured)
            except Exception:
                captured = now

            if (now - captured) > self.MAX_SNAPSHOT_AGE_SECONDS:
                stale_keys.append(k)

        for k in stale_keys:
            self._snapshots.pop(k, None)

    def _enforce_capacity(self) -> None:
        while len(self._snapshots) > self.MAX_SNAPSHOTS:
            self._snapshots.popitem(last=False)

    def store_snapshot(self, snapshot: RestorationSnapshot) -> str:
        if not isinstance(snapshot, RestorationSnapshot):
            raise SnapshotProviderError("Invalid snapshot object")

        now = time.time()

        with self._lock:
            self._evict_stale(now)

            if snapshot.snapshot_id in self._snapshots:
                raise SnapshotProviderError(
                    f"Snapshot id collision: {snapshot.snapshot_id}"
                )

            self._snapshots[snapshot.snapshot_id] = snapshot
            self._enforce_capacity()

        return snapshot.snapshot_id

    def get_snapshot(self, snapshot_id: str) -> Optional[RestorationSnapshot]:

        if not isinstance(snapshot_id, str) or not snapshot_id:
            return None

        with self._lock:
            snap = self._snapshots.get(snapshot_id)
            if snap:
                self._snapshots.move_to_end(snapshot_id)
            return snap

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
            raise SnapshotProviderError("Observer missing")

        if self._mode.mode is not SystemMode.OBSERVER:
            raise SnapshotProviderError(
                f"Snapshot attempted in {self._mode.mode.value}"
            )

        if not self._observer.is_healthy():
            raise SnapshotProviderError("Observer unhealthy")

        observer_state = self._observer.snapshot()
        if not isinstance(observer_state, dict):
            raise SnapshotProviderError("Observer snapshot malformed")

        if not observer_state.get("perception_available"):
            raise SnapshotProviderError("Vision unavailable")

        frame_ts = observer_state.get("perception_frame_ts")

        # ---------------- BOUNDED ATOMIC CAPTURE ----------------

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

        capture_duration = t_end - t_start

        if capture_duration > self.ATOMIC_WINDOW_SECONDS:
            raise SnapshotProviderError(
                f"Atomic capture window exceeded ({round(capture_duration,4)}s)"
            )

        # ---------------- VALIDATION ----------------

        if not isinstance(cursor, dict):
            raise SnapshotProviderError("Cursor invalid")

        if "x" not in cursor or "y" not in cursor:
            raise SnapshotProviderError("Cursor coordinates missing")

        try:
            cursor_x = int(cursor["x"])
            cursor_y = int(cursor["y"])
        except Exception:
            raise SnapshotProviderError(
                "Cursor coordinate coercion failed"
            )

        if (
            not isinstance(focused_window, dict)
            or not isinstance(focused_window.get("title"), str)
        ):
            raise SnapshotProviderError("Focused window invalid")

        window_title = focused_window["title"].strip()

        # FIX-05 (RTB-05): When the desktop is bare (no focused window),
        # get_focused_window() returns an empty title. The original guard
        # raised SnapshotProviderError("Focused window invalid") unconditionally,
        # permanently blocking task arming with no user diagnostic.
        #
        # Fallback: use the active application title as the window identity
        # sentinel. If that is also empty, use the "__bare_desktop__" sentinel
        # so snapshots can still be taken and restored (restoration will skip
        # window focus since no window was focused at snapshot time).
        if not window_title:
            if isinstance(active_app, dict) and isinstance(active_app.get("title"), str):
                window_title = active_app["title"].strip()
            if not window_title:
                window_title = "__bare_desktop__"

        if (
            not isinstance(active_app, dict)
            or not isinstance(active_app.get("title"), str)
        ):
            # FIX RTB-02: If active_app is entirely missing/malformed,
            # use the bare-desktop sentinel rather than raising an error.
            app_title = "__bare_desktop__"
        else:
            app_title = active_app["title"].strip() or "__bare_desktop__"

        # ---------------- STATE OBJECTS ----------------

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
            pid=None,  # deterministic & portable
        )

        # ---------------- METADATA (CANONICALIZED) ----------------

        metadata = {
            "schema_version": self.SNAPSHOT_SCHEMA_VERSION,
            "captured_at_monotonic": float(t_end),
            "captured_at_wallclock": float(time.time()),
            "execution_mode": self._mode.mode.value,
            "vision_frame_ts": frame_ts,
            "capture_duration_ms": round(
                capture_duration * 1000.0,
                6,
            ),
        }

        metadata = json.loads(
            json.dumps(
                metadata,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

        # ---------------- SNAPSHOT CREATION ----------------

        snapshot = RestorationSnapshot.create(
            cursor=cursor_state,
            focus=focus_state,
            application=application_state,
            execution_mode=self._mode.mode.value,
            metadata=metadata,
        )

        return snapshot
