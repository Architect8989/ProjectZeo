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
        """
        FIX H7: Enforce TTL at retrieval time, not only during store_snapshot().

        Bug: MAX_SNAPSHOT_AGE_SECONDS = 3600 was only applied inside
        _evict_stale(), which ran during store_snapshot(). get_snapshot()
        returned any snapshot regardless of age. A snapshot taken before a
        long planning warmup phase (up to 210s = planning 60s + warmup 150s)
        could still be retrieved and restored — but more critically, a snapshot
        taken before a multi-hour idle period could survive in memory and be
        restored from hours-old workspace state with no diagnostic.

        Fix: check TTL here. If the snapshot has aged past MAX_SNAPSHOT_AGE_SECONDS,
        evict it from the registry, print a diagnostic, and return None. The
        caller (main.py) treats None as a missing snapshot and proceeds to safe
        shutdown / skip restoration — the correct behaviour for expired state.

        The TTL check uses captured_at_wallclock from snapshot.metadata (set at
        capture time by _capture_snapshot). If the field is missing or unparseable,
        the snapshot is treated as non-expired (fail-open for the TTL check only;
        the restoration itself is still guarded by mode and verifier checks).
        """
        if not isinstance(snapshot_id, str) or not snapshot_id:
            return None

        now = time.time()

        with self._lock:
            snap = self._snapshots.get(snapshot_id)
            if snap is None:
                return None

            # FIX H7: Enforce TTL at retrieval time.
            captured = snap.metadata.get("captured_at_wallclock", now)
            try:
                captured = float(captured)
            except Exception:
                # Unparseable timestamp — treat as non-expired (fail-open for TTL).
                captured = now

            age_seconds = now - captured
            if age_seconds > self.MAX_SNAPSHOT_AGE_SECONDS:
                # Snapshot is stale — evict and return None.
                self._snapshots.pop(snapshot_id, None)
                import sys as _sys
                print(
                    f"[SnapshotProvider] Snapshot {snapshot_id[:12]}… expired "
                    f"({round(age_seconds, 1)}s old, max "
                    f"{self.MAX_SNAPSHOT_AGE_SECONDS}s). Evicted. Returning None.",
                    file=_sys.stderr,
                )
                return None

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

        # SI-05 FIX: Populate metadata['extended'] with observable OS state
        # so RestoreVerifier's extended checks can actually fire.
        #
        # Bug: metadata['extended'] was always set to {} — an empty dict.
        # RestoreVerifier._verify_extended() iterates over metadata['extended']
        # and only runs checks when keys are present. With an empty dict, ALL
        # extended checks (window geometry, process list, browser state) were
        # permanently skipped. The extended verification path was dead code.
        #
        # Fix: populate with the lightweight fields that can be captured
        # cross-platform without heavy dependencies:
        #   - window_geometry: via xdotool (Linux) or approximate
        #   - process_snapshot: names of running processes via psutil
        #
        # Browser state (CDP) and cryptographic file hashes are P3 items
        # and not included here; this patch covers the minimum viable
        # extended capture that makes verification non-trivially useful.
        extended: dict = {}

        # Window geometry (Linux/xdotool)
        try:
            import subprocess as _sp  # noqa: PLC0415
            _wid_result = _sp.run(
                ["xdotool", "getactivewindow"],
                capture_output=True,
                timeout=2,
            )
            if _wid_result.returncode == 0:
                _wid = _wid_result.stdout.strip().decode("utf-8", errors="replace")
                _geo_result = _sp.run(
                    ["xdotool", "getwindowgeometry", _wid],
                    capture_output=True,
                    timeout=2,
                )
                if _geo_result.returncode == 0:
                    extended["window_geometry"] = _geo_result.stdout.decode(
                        "utf-8", errors="replace"
                    ).strip()
        except Exception:
            pass  # xdotool absent or failed — non-fatal

        # Active process snapshot (psutil — cross-platform)
        try:
            import psutil as _psutil  # noqa: PLC0415
            extended["processes"] = sorted(
                {p.name() for p in _psutil.process_iter(["name"]) if p.name()}
            )
        except ImportError:
            pass  # psutil not installed — skip
        except Exception:
            pass  # process iteration failed — non-fatal

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
            "extended": extended,
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

