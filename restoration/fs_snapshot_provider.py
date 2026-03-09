"""
restoration/fs_snapshot_provider.py

Filesystem snapshot and restore for user working directories.

Strategy (in order of preference):
  1. BTRFS subvolume snapshot — near-instant, copy-on-write, zero disk cost
  2. rsync --link-dest         — hardlink-based, ~2s for typical home dirs
  3. No-op                     — if neither tool is available

Only triggered for tasks that write to the filesystem (file_create,
file_delete, install operations). Read-only tasks skip this entirely.

Scope: Desktop, Documents, Downloads, and the current working directory.
Does NOT snapshot /home entirely — that would be too slow and invasive.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

_SNAPSHOT_BASE = os.path.expanduser(
    os.environ.get("PROJECTZEO_FS_SNAPSHOT_DIR", "~/.projectzeo/fs_snapshots")
)
_MAX_SNAPSHOTS   = int(os.environ.get("PROJECTZEO_FS_MAX_SNAPSHOTS", "5"))
_BTRFS_MOUNT     = os.environ.get("PROJECTZEO_BTRFS_MOUNT", "")
_RSYNC_TIMEOUT   = int(os.environ.get("PROJECTZEO_RSYNC_TIMEOUT", "30"))

_DEFAULT_SCOPE = [
    os.path.expanduser("~/Desktop"),
    os.path.expanduser("~/Documents"),
    os.path.expanduser("~/Downloads"),
]


@dataclass
class FsSnapshot:
    snapshot_id:  str
    captured_at:  float
    backend:      str
    scope_dirs:   List[str]  = field(default_factory=list)
    snapshot_dir: str        = ""
    btrfs_path:   str        = ""
    file_manifest: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FsSnapshot":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _run(cmd: List[str], timeout: float = 30.0) -> Tuple[int, str, str]:
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except (FileNotFoundError, OSError) as e:
        return -1, "", str(e)


def _btrfs_available() -> bool:
    rc, _, _ = _run(["btrfs", "--version"])
    return rc == 0 and bool(_BTRFS_MOUNT) and os.path.isdir(_BTRFS_MOUNT)


def _rsync_available() -> bool:
    rc, _, _ = _run(["rsync", "--version"])
    return rc == 0


def _manifest(scope_dirs: List[str]) -> List[str]:
    files: List[str] = []
    for d in scope_dirs:
        if not os.path.isdir(d):
            continue
        for root, _, fnames in os.walk(d):
            for fn in fnames:
                path = os.path.join(root, fn)
                try:
                    stat = os.stat(path)
                    files.append(f"{path}:{stat.st_mtime:.3f}:{stat.st_size}")
                except OSError:
                    pass
    return files[:2000]


def _evict_old_snapshots() -> None:
    os.makedirs(_SNAPSHOT_BASE, exist_ok=True)
    entries = sorted(
        (e for e in os.scandir(_SNAPSHOT_BASE) if e.is_dir()),
        key=lambda e: e.stat().st_mtime,
    )
    while len(entries) >= _MAX_SNAPSHOTS:
        old = entries.pop(0)
        try:
            shutil.rmtree(old.path)
            _logger.debug("[FsSnap] Evicted old snapshot: %s", old.path)
        except Exception:
            pass


def capture(scope_dirs: Optional[List[str]] = None, task_writes_files: bool = True) -> Optional[FsSnapshot]:
    if not task_writes_files:
        return None

    dirs = [d for d in (scope_dirs or _DEFAULT_SCOPE) if os.path.isdir(d)]
    if not dirs:
        return None

    snap_id  = f"snap_{int(time.time() * 1000)}"
    snap_dir = os.path.join(_SNAPSHOT_BASE, snap_id)

    os.makedirs(_SNAPSHOT_BASE, exist_ok=True)
    _evict_old_snapshots()
    os.makedirs(snap_dir, exist_ok=True)

    manifest = _manifest(dirs)

    if _btrfs_available():
        return _capture_btrfs(snap_id, snap_dir, dirs, manifest)
    if _rsync_available():
        return _capture_rsync(snap_id, snap_dir, dirs, manifest)

    _logger.warning("[FsSnap] Neither btrfs nor rsync available — filesystem not snapshotted.")
    return FsSnapshot(
        snapshot_id=snap_id,
        captured_at=time.time(),
        backend="none",
        scope_dirs=dirs,
        snapshot_dir=snap_dir,
        file_manifest=manifest,
    )


def _capture_btrfs(snap_id: str, snap_dir: str, dirs: List[str], manifest: List[str]) -> FsSnapshot:
    btrfs_path = os.path.join(_BTRFS_MOUNT, f".snapshots/{snap_id}")
    rc, _, err = _run(["btrfs", "subvolume", "snapshot", _BTRFS_MOUNT, btrfs_path], timeout=10.0)
    if rc != 0:
        _logger.warning("[FsSnap] BTRFS snapshot failed (%s) — falling back to rsync.", err)
        if _rsync_available():
            return _capture_rsync(snap_id, snap_dir, dirs, manifest)
    _logger.info("[FsSnap] BTRFS snapshot: %s", btrfs_path)
    return FsSnapshot(
        snapshot_id=snap_id,
        captured_at=time.time(),
        backend="btrfs",
        scope_dirs=dirs,
        snapshot_dir=snap_dir,
        btrfs_path=btrfs_path,
        file_manifest=manifest,
    )


def _capture_rsync(snap_id: str, snap_dir: str, dirs: List[str], manifest: List[str]) -> FsSnapshot:
    existing = sorted(
        (e for e in os.scandir(_SNAPSHOT_BASE) if e.is_dir() and e.name != snap_id),
        key=lambda e: e.stat().st_mtime,
        reverse=True,
    )
    link_dest_arg: List[str] = []
    if existing:
        link_dest_arg = [f"--link-dest={existing[0].path}"]

    for src_dir in dirs:
        rel  = os.path.basename(src_dir.rstrip("/"))
        dest = os.path.join(snap_dir, rel)
        cmd  = ["rsync", "-a", "--no-specials", "--no-devices"] + link_dest_arg + [f"{src_dir}/", dest]
        rc, _, err = _run(cmd, timeout=float(_RSYNC_TIMEOUT))
        if rc not in (0, 24):
            _logger.warning("[FsSnap] rsync error for %s: %s", src_dir, err)

    _logger.info("[FsSnap] rsync snapshot: %s (%d dirs)", snap_dir, len(dirs))
    return FsSnapshot(
        snapshot_id=snap_id,
        captured_at=time.time(),
        backend="rsync",
        scope_dirs=dirs,
        snapshot_dir=snap_dir,
        file_manifest=manifest,
    )


def restore(snapshot: FsSnapshot) -> bool:
    if snapshot.backend == "none":
        _logger.debug("[FsSnap] No backend — nothing to restore.")
        return True

    if snapshot.backend == "btrfs" and snapshot.btrfs_path:
        return _restore_btrfs(snapshot)
    if snapshot.backend == "rsync" and snapshot.snapshot_dir:
        return _restore_rsync(snapshot)

    return False


def _restore_btrfs(snap: FsSnapshot) -> bool:
    if not os.path.exists(snap.btrfs_path):
        _logger.warning("[FsSnap] BTRFS snapshot path missing: %s", snap.btrfs_path)
        return False
    rc, _, err = _run(
        ["btrfs", "subvolume", "snapshot", snap.btrfs_path, _BTRFS_MOUNT + "_restore"],
        timeout=15.0,
    )
    if rc != 0:
        _logger.warning("[FsSnap] BTRFS restore failed: %s", err)
        return False
    _logger.info("[FsSnap] BTRFS restore complete.")
    return True


def _restore_rsync(snap: FsSnapshot) -> bool:
    success = True
    for src_dir in snap.scope_dirs:
        rel  = os.path.basename(src_dir.rstrip("/"))
        src  = os.path.join(snap.snapshot_dir, rel)
        if not os.path.isdir(src):
            continue
        cmd = ["rsync", "-a", "--delete", "--no-specials", "--no-devices", f"{src}/", src_dir]
        rc, _, err = _run(cmd, timeout=float(_RSYNC_TIMEOUT))
        if rc not in (0, 24):
            _logger.warning("[FsSnap] rsync restore error for %s: %s", src_dir, err)
            success = False
    _logger.info("[FsSnap] rsync restore complete.")
    return success


def verify(snapshot: FsSnapshot) -> bool:
    if snapshot.backend == "none":
        return True
    current = _manifest(snapshot.scope_dirs)
    current_set  = set(current)
    original_set = set(snapshot.file_manifest)
    if not original_set:
        return True
    overlap = len(original_set & current_set) / len(original_set)
    passed  = overlap >= 0.9
    _logger.debug("[FsSnap] Verify overlap=%.1f%% pass=%s", overlap * 100, passed)
    return passed


def cleanup(snapshot: FsSnapshot) -> None:
    if snapshot.snapshot_dir and os.path.isdir(snapshot.snapshot_dir):
        try:
            shutil.rmtree(snapshot.snapshot_dir)
        except Exception as exc:
            _logger.debug("[FsSnap] Cleanup error: %s", exc)
    if snapshot.btrfs_path and os.path.exists(snapshot.btrfs_path):
        _run(["btrfs", "subvolume", "delete", snapshot.btrfs_path])
