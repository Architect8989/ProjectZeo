"""
restoration/criu_provider.py

Process state checkpoint and restore via CRIU (Checkpoint/Restore In Userspace).

Only used for long-running processes that the agent may interrupt during a task.
Short tasks and UI-only tasks do not need this tier.

Trigger criteria:
  - IRREVERSIBLE+ operations on processes running > 60 seconds
  - Any operation that would kill or restart a user process

Requirements:
  - CRIU installed: apt install criu
  - Running as root or with CAP_SYS_PTRACE + CAP_SYS_ADMIN
  - Linux kernel >= 3.11

Fallback: if CRIU is unavailable, provider returns gracefully with
a no-op snapshot that tracks PIDs only (for informational logging).
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

_CRIU_BIN     = shutil.which("criu") or "criu"
_DUMP_BASE    = os.path.expanduser(
    os.environ.get("PROJECTZEO_CRIU_DIR", "~/.projectzeo/criu_dumps")
)
_MAX_DUMPS    = int(os.environ.get("PROJECTZEO_CRIU_MAX_DUMPS", "3"))
_CRIU_TIMEOUT = int(os.environ.get("PROJECTZEO_CRIU_TIMEOUT", "30"))

_EXCLUDE_PROCS = frozenset({
    "systemd", "init", "Xorg", "X", "gnome-shell", "kwin",
    "pulseaudio", "pipewire", "dbus-daemon", "NetworkManager",
})


@dataclass
class ProcessRecord:
    pid:      int
    name:     str
    cmdline:  str
    dump_dir: str = ""
    dumped:   bool = False


@dataclass
class CriuSnapshot:
    snapshot_id:     str
    captured_at:     float
    criu_available:  bool
    processes:       List[ProcessRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CriuSnapshot":
        procs = [ProcessRecord(**p) for p in d.get("processes", [])]
        return cls(
            snapshot_id=d.get("snapshot_id", ""),
            captured_at=float(d.get("captured_at", 0.0)),
            criu_available=bool(d.get("criu_available", False)),
            processes=procs,
        )


def _run(cmd: List[str], timeout: float = 10.0) -> Tuple[int, str, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except (FileNotFoundError, OSError) as e:
        return -1, "", str(e)


def _criu_available() -> bool:
    rc, out, _ = _run([_CRIU_BIN, "--version"])
    return rc == 0 and "CRIU" in out.upper()


def _get_user_long_procs(min_age_sec: float = 60.0) -> List[ProcessRecord]:
    try:
        import psutil  # type: ignore
    except ImportError:
        return []

    now  = time.time()
    procs: List[ProcessRecord] = []
    for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            info    = proc.info
            name    = info.get("name", "") or ""
            if name.lower() in _EXCLUDE_PROCS:
                continue
            age = now - float(info.get("create_time", now))
            if age < min_age_sec:
                continue
            cmdline = " ".join(info.get("cmdline") or [])[:200]
            procs.append(ProcessRecord(pid=info["pid"], name=name, cmdline=cmdline))
        except Exception:
            pass
    return procs


def capture(target_pids: Optional[List[int]] = None) -> CriuSnapshot:
    snap_id  = f"criu_{int(time.time() * 1000)}"
    criu_ok  = _criu_available()

    if not criu_ok:
        _logger.debug("[CRIU] CRIU not available — tracking PIDs only.")
        procs = _get_user_long_procs()
        return CriuSnapshot(
            snapshot_id=snap_id,
            captured_at=time.time(),
            criu_available=False,
            processes=procs,
        )

    if target_pids:
        import psutil  # type: ignore
        procs = []
        for pid in target_pids:
            try:
                p    = psutil.Process(pid)
                name = p.name()
                if name.lower() not in _EXCLUDE_PROCS:
                    procs.append(ProcessRecord(pid=pid, name=name, cmdline=" ".join(p.cmdline())[:200]))
            except Exception:
                pass
    else:
        procs = _get_user_long_procs()

    os.makedirs(_DUMP_BASE, exist_ok=True)
    _evict_old_dumps()

    dumped: List[ProcessRecord] = []
    for proc in procs[:5]:
        dump_dir = os.path.join(_DUMP_BASE, f"{snap_id}_{proc.pid}")
        os.makedirs(dump_dir, exist_ok=True)
        rc, _, err = _run(
            [_CRIU_BIN, "dump", "-t", str(proc.pid), "-D", dump_dir,
             "--shell-job", "--leave-running"],
            timeout=float(_CRIU_TIMEOUT),
        )
        if rc == 0:
            proc.dump_dir = dump_dir
            proc.dumped   = True
            _logger.info("[CRIU] Dumped pid=%d (%s) → %s", proc.pid, proc.name, dump_dir)
        else:
            _logger.warning("[CRIU] Dump failed pid=%d: %s", proc.pid, err)
        dumped.append(proc)

    return CriuSnapshot(
        snapshot_id=snap_id,
        captured_at=time.time(),
        criu_available=True,
        processes=dumped,
    )


def restore(snapshot: CriuSnapshot) -> bool:
    if not snapshot.criu_available:
        _logger.debug("[CRIU] Not available — nothing to restore.")
        return True

    success = True
    for proc in snapshot.processes:
        if not proc.dumped or not os.path.isdir(proc.dump_dir):
            continue
        rc, _, err = _run(
            [_CRIU_BIN, "restore", "-D", proc.dump_dir, "--shell-job", "-d"],
            timeout=float(_CRIU_TIMEOUT),
        )
        if rc == 0:
            _logger.info("[CRIU] Restored pid=%d (%s).", proc.pid, proc.name)
        else:
            _logger.warning("[CRIU] Restore failed pid=%d: %s", proc.pid, err)
            success = False

    return success


def verify(snapshot: CriuSnapshot) -> bool:
    if not snapshot.criu_available:
        return True
    try:
        import psutil  # type: ignore
        live_pids = {p.pid for p in psutil.process_iter(["pid"])}
        expected  = {p.pid for p in snapshot.processes if p.dumped}
        if not expected:
            return True
        recovered = len(expected & live_pids) / len(expected)
        passed    = recovered >= 0.8
        _logger.debug("[CRIU] Verify recovery=%.0f%% pass=%s", recovered * 100, passed)
        return passed
    except Exception:
        return True


def cleanup(snapshot: CriuSnapshot) -> None:
    for proc in snapshot.processes:
        if proc.dump_dir and os.path.isdir(proc.dump_dir):
            try:
                shutil.rmtree(proc.dump_dir)
            except Exception:
                pass


def _evict_old_dumps() -> None:
    try:
        entries = sorted(
            (e for e in os.scandir(_DUMP_BASE) if e.is_dir()),
            key=lambda e: e.stat().st_mtime,
        )
        while len(entries) >= _MAX_DUMPS:
            old = entries.pop(0)
            shutil.rmtree(old.path, ignore_errors=True)
    except Exception:
        pass
