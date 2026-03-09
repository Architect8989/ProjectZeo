"""
restoration/window_state_provider.py

Captures and restores window layout using wmctrl and xdotool.

Differential restore design: only windows that EXISTED before the task
are restored. Windows the agent opened are closed. The user's pre-task
window arrangement is reconstructed without touching anything new.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set

_logger = logging.getLogger(__name__)

_WMCTRL  = "wmctrl"
_XDOTOOL = "xdotool"
_RESTORE_DELAY = 0.15


def _run(cmd: List[str], timeout: float = 5.0) -> str:
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""


def _tools_available() -> bool:
    return bool(_run([_WMCTRL, "--version"])) and bool(_run([_XDOTOOL, "version"]))


@dataclass
class WindowRecord:
    win_id:    str
    pid:       int
    title:     str
    x:         int
    y:         int
    w:         int
    h:         int
    desktop:   int
    minimised: bool = False


@dataclass
class WindowStateSnapshot:
    captured_at:   float
    windows:       List[WindowRecord] = field(default_factory=list)
    active_win_id: Optional[str]      = None
    desktop:       int                = 0

    def window_ids(self) -> Set[str]:
        return {w.win_id for w in self.windows}

    def to_dict(self) -> Dict:
        d = {
            "captured_at":   self.captured_at,
            "active_win_id": self.active_win_id,
            "desktop":       self.desktop,
            "windows": [asdict(w) for w in self.windows],
        }
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "WindowStateSnapshot":
        wins = [WindowRecord(**w) for w in d.get("windows", [])]
        return cls(
            captured_at=float(d.get("captured_at", 0.0)),
            windows=wins,
            active_win_id=d.get("active_win_id"),
            desktop=int(d.get("desktop", 0)),
        )


def _parse_wmctrl_line(line: str) -> Optional[WindowRecord]:
    parts = line.split(None, 4)
    if len(parts) < 5:
        return None
    try:
        win_id  = parts[0]
        desktop = int(parts[1])
        pid_str = parts[2]
        geo     = parts[3]
        title   = parts[4].strip()

        pid = int(pid_str) if pid_str.lstrip("-").isdigit() else -1
        gparts = geo.split(",")
        x, y, w, h = (int(g) for g in gparts) if len(gparts) == 4 else (0, 0, 0, 0)

        return WindowRecord(
            win_id=win_id, pid=pid, title=title,
            x=x, y=y, w=w, h=h, desktop=desktop,
        )
    except (ValueError, IndexError):
        return None


def capture() -> Optional[WindowStateSnapshot]:
    if not _tools_available():
        _logger.debug("[WindowState] wmctrl/xdotool not available — skipping.")
        return None

    raw = _run([_WMCTRL, "-lGp"])
    if not raw:
        return None

    windows: List[WindowRecord] = []
    for line in raw.splitlines():
        rec = _parse_wmctrl_line(line)
        if rec:
            windows.append(rec)

    active_id = _run([_XDOTOOL, "getactivewindow"]).strip() or None

    snap = WindowStateSnapshot(
        captured_at=time.time(),
        windows=windows,
        active_win_id=active_id,
    )
    _logger.debug("[WindowState] Captured %d windows.", len(windows))
    return snap


def close_agent_windows(
    pre_snapshot: WindowStateSnapshot,
    post_snapshot: Optional[WindowStateSnapshot],
) -> None:
    if post_snapshot is None:
        return

    pre_ids  = pre_snapshot.window_ids()
    post_ids = post_snapshot.window_ids()
    new_ids  = post_ids - pre_ids

    for win_id in new_ids:
        _logger.debug("[WindowState] Closing agent window %s", win_id)
        _run([_WMCTRL, "-ic", win_id])
        time.sleep(0.05)


def restore(snapshot: WindowStateSnapshot) -> bool:
    if not _tools_available():
        return False

    ok_count = 0

    for win in snapshot.windows:
        result = _run([
            _WMCTRL, "-ir", win.win_id,
            "-e", f"0,{win.x},{win.y},{win.w},{win.h}",
        ])
        if result is not None:
            ok_count += 1
        time.sleep(_RESTORE_DELAY)

    if snapshot.active_win_id:
        _run([_XDOTOOL, "windowfocus", "--sync", snapshot.active_win_id])
        _run([_XDOTOOL, "windowactivate", "--sync", snapshot.active_win_id])
        time.sleep(_RESTORE_DELAY)

    _logger.info("[WindowState] Restored %d/%d windows.", ok_count, len(snapshot.windows))
    return ok_count > 0


def verify(snapshot: WindowStateSnapshot) -> bool:
    live = capture()
    if live is None:
        return False
    live_ids = live.window_ids()
    original_ids = snapshot.window_ids()
    overlap = len(original_ids & live_ids)
    ratio   = overlap / max(len(original_ids), 1)
    passed  = ratio >= 0.7
    _logger.debug("[WindowState] Verify overlap=%.0f%% pass=%s", ratio * 100, passed)
    return passed
