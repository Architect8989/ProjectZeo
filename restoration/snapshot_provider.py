from __future__ import annotations

import os
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
    
    MAX_SNAPSHOT_AGE_SECONDS = 10800
    # P0-A FIX (RT-01): Raised from 0.5s to 2.0s.  Under OS load, the three
    # sequential syscalls (cursor + focused_window + active_app) can each take
    # 200 ms+, totalling >600 ms even without genuine OS hangs.  The previous
    # 0.5 s window caused SnapshotProviderError → arm_failure.json → infinite
    # loop, permanently preventing task arming on loaded hosts.  2.0 s remains
    # conservative enough to detect true OS hangs (kernel freeze, X11 lockup).
    # Still overridable via PROJECTZEO_ATOMIC_WINDOW_SECONDS env var.
    ATOMIC_WINDOW_SECONDS = 2.0

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

        # FIX-4 (MEDIUM): ATOMIC_WINDOW_SECONDS is now runtime-configurable via
        # the PROJECTZEO_ATOMIC_WINDOW_SECONDS environment variable so that slow
        # VMs and disk-saturated hosts can relax the capture deadline without
        # requiring a code change.  The class constant (0.5s) remains as the
        # default.  Valid range is clamped to [0.1, 10.0] seconds to prevent
        # misconfiguration from masking genuine OS hangs.
        #
        # Usage:
        #   export PROJECTZEO_ATOMIC_WINDOW_SECONDS=2.0   # e.g. for slow VMs
        _env_val = os.environ.get("PROJECTZEO_ATOMIC_WINDOW_SECONDS")
        if _env_val is not None:
            try:
                _configured = float(_env_val)
                self._atomic_window_seconds: float = max(0.1, min(10.0, _configured))
            except ValueError:
                self._atomic_window_seconds = self.ATOMIC_WINDOW_SECONDS
        else:
            self._atomic_window_seconds = self.ATOMIC_WINDOW_SECONDS

        # instance-local registry
        self._snapshots: "OrderedDict[str, RestorationSnapshot]" = OrderedDict()
        self._lock = threading.Lock()

        # H-01 FIX (SI-01): Persist snapshots to disk so post-crash restoration
        # is possible.  On startup we reload all non-expired snapshots from
        # memory/snapshot_<id>.json files.  Each store_snapshot() call writes a
        # new file; each eviction or expiry removes the corresponding file.
        self._snapshot_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "memory",
            "snapshots",
        )
        try:
            os.makedirs(self._snapshot_dir, exist_ok=True)
            self._disk_persistence_available = True
        except (PermissionError, OSError):
            self._disk_persistence_available = False

        # Reload surviving snapshots from disk (skip expired ones).
        if self._disk_persistence_available:
            self._reload_from_disk()

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
            evicted_id, _ = self._snapshots.popitem(last=False)
            self._remove_snapshot_file(evicted_id)

    # =========================================================
    # DISK PERSISTENCE  (H-01 / SI-01 FIX)
    # =========================================================

    def _snapshot_path(self, snapshot_id: str) -> str:
        # Use only the first 64 chars of the hex ID as the filename to stay
        # well within filesystem path-length limits.
        safe_id = snapshot_id[:64].replace("/", "_").replace("\\", "_")
        return os.path.join(self._snapshot_dir, f"snapshot_{safe_id}.json")

    def _write_snapshot_file(self, snapshot: "RestorationSnapshot") -> None:
        """Persist snapshot to disk as JSON.  Non-fatal on failure."""
        if not self._disk_persistence_available:
            return
        try:
            path = self._snapshot_path(snapshot.snapshot_id)
            tmp = path + ".tmp"
            data = snapshot.to_dict()
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, separators=(",", ":"))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except Exception:
            pass  # Non-fatal; in-memory copy still valid

    def _remove_snapshot_file(self, snapshot_id: str) -> None:
        """Delete the on-disk snapshot file for the given ID.  Non-fatal."""
        if not self._disk_persistence_available:
            return
        try:
            os.remove(self._snapshot_path(snapshot_id))
        except FileNotFoundError:
            pass
        except Exception:
            pass

    def _reload_from_disk(self) -> None:
        """On startup, reload non-expired snapshots from disk into memory."""
        if not self._disk_persistence_available:
            return
        now = time.time()
        try:
            for fname in os.listdir(self._snapshot_dir):
                if not (fname.startswith("snapshot_") and fname.endswith(".json")):
                    continue
                fpath = os.path.join(self._snapshot_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    snap = RestorationSnapshot.from_dict(data)
                    captured = float(snap.metadata.get("captured_at_wallclock", 0))

                    
                    _max_age = self.MAX_SNAPSHOT_AGE_SECONDS * 2  # 12600s hard reject
                    _future_grace = 60.0  # tolerate up to 60s clock skew

                    if captured > 0 and (now - captured) > _max_age:
                        import sys as _sys
                        print(
                            f"[SnapshotProvider] Reload: snapshot {snap.snapshot_id[:12]}… "
                            f"is FAR PAST ({round((now - captured)/3600, 1)}h old, "
                            f"hard limit {_max_age/3600:.1f}h). Discarding.",
                            file=_sys.stderr,
                        )
                        os.remove(fpath)
                        continue

                    if captured > now + _future_grace:
                        import sys as _sys
                        print(
                            f"[SnapshotProvider] Reload: snapshot {snap.snapshot_id[:12]}… "
                            f"has a FUTURE timestamp ({round(captured - now, 1)}s ahead). "
                            "Possible NTP forward-step. Discarding to prevent stale restoration.",
                            file=_sys.stderr,
                        )
                        os.remove(fpath)
                        continue

                    if (now - captured) > self.MAX_SNAPSHOT_AGE_SECONDS:
                        os.remove(fpath)  # expired — discard
                        continue
                    if snap.snapshot_id not in self._snapshots:
                        self._snapshots[snap.snapshot_id] = snap
                except Exception:
                    # Corrupted or unreadable file — remove it
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass
        except Exception:
            pass  # Best-effort reload; failures are non-fatal

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
            # H-01: Persist to disk so post-crash restoration is possible.
            self._write_snapshot_file(snapshot)

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
                # Snapshot is stale — evict from memory and disk, return None.
                self._snapshots.pop(snapshot_id, None)
                self._remove_snapshot_file(snapshot_id)
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
        # GAP-1 DOCUMENTATION: What the snapshot captures and does NOT capture.
        #
        # CAPTURED (and restored):
        #   - Cursor XY position (pixel coordinates)
        #   - Focused window title (fuzzy-matched on restore, Levenshtein ≤5)
        #   - Active application name
        #   - Running process names (census for post-restore diff report)
        #
        # NOT CAPTURED — will NOT be restored:
        #   - Browser tab list, active tab, scroll position
        #   - Media playback position (YouTube timestamp, Spotify track position)
        #   - Application-internal state (document scroll, cursor position in editor)
        #   - Clipboard contents
        #   - Network connections or open sockets
        #   - In-memory application state (unsaved documents, form contents)
        #
        # ARCHITECTURAL LIMITATION (GAP-1):
        #   Full application-state restoration requires OS-level serialization APIs
        #   (e.g. CRIU on Linux, application checkpoint protocols) that are not
        #   generically available.  For media playback tasks, the caller should
        #   record the media timestamp before arming and seek back after restoration.
        #   For browser tasks, consider saving the URL before arming via xdotool
        #   or AT-SPI2 and re-navigating after restoration.
        #
        # This warning is printed once per snapshot to stderr so it appears in
        # log aggregators and is visible to operators reviewing task outcomes.
        import sys as _sys
        _sys.stderr.write(
            "[SnapshotProvider] GAP-1: Snapshot captures cursor+window+app ONLY. "
            "Media playback position, browser tabs/scroll, clipboard, and "
            "unsaved application state are NOT captured and will NOT be restored. "
            "For media tasks: record timestamp before arming and seek after restore.\n"
        )
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

        
        try:
            observer_state = self._observer.snapshot()
        except SnapshotProviderError:
            raise  # Already typed — propagate unchanged
        except Exception as _obs_exc:
            raise SnapshotProviderError(
                f"Observer.snapshot() raised during capture — observer may be "
                f"flapping (health check passed but snapshot failed): {_obs_exc}"
            ) from _obs_exc

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

        if capture_duration > self._atomic_window_seconds:
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
                    ["xdotool", "getwindowgeometry", "--shell", _wid],
                    capture_output=True,
                    timeout=2,
                )
                if _geo_result.returncode == 0:
                    _geo_dict: dict = {}
                    for _line in _geo_result.stdout.decode("utf-8", errors="replace").splitlines():
                        if "=" in _line:
                            _k, _, _v = _line.partition("=")
                            try:
                                _geo_dict[_k.strip().lower()] = int(_v.strip())
                            except ValueError:
                                pass
                    if {"x", "y", "width", "height"}.issubset(_geo_dict):
                        extended["window_geometry"] = {
                            "x": _geo_dict["x"],
                            "y": _geo_dict["y"],
                            "width": _geo_dict["width"],
                            "height": _geo_dict["height"],
                        }
        except Exception:
            pass

        # IH-1 FIX: Capture window Z-order so RestoreVerifier._verify_window_z_order()
        # has data to verify. Previously window_z_order was never populated in
        # metadata["extended"], making _verify_window_z_order() always short-circuit
        # at "if z is None: return" — the verification was dead code despite being reachable.
        # Fix: attempt to capture Z-order via xdotool getwindowstackingorder.
        # Non-fatal: Z-order capture is best-effort (xdotool may be absent).
        try:
            import subprocess as _sp_zo
            _zo_result = _sp_zo.run(
                ["xdotool", "getwindowstackingorder"],
                capture_output=True,
                timeout=2,
            )
            if _zo_result.returncode == 0:
                _stacking = _zo_result.stdout.strip().decode("utf-8", errors="replace").split()
                # Get the currently active window ID to find its Z-order position
                _active_result = _sp_zo.run(
                    ["xdotool", "getactivewindow"],
                    capture_output=True,
                    timeout=2,
                )
                if _active_result.returncode == 0:
                    _active_wid = _active_result.stdout.strip().decode("utf-8", errors="replace")
                    if _active_wid in _stacking:
                        # Z-order index: 0 = bottommost, len-1 = topmost
                        extended["window_z_order"] = _stacking.index(_active_wid)
        except Exception:
            pass  # xdotool absent or timed out — non-fatal

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

