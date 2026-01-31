import os
import platform
import socket
import getpass
from typing import Dict, Optional


def _safe_call(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _get_uid() -> Optional[int]:
    return _safe_call(lambda: os.getuid(), None)


def _get_euid() -> Optional[int]:
    return _safe_call(lambda: os.geteuid(), None)


def _is_root(uid: Optional[int], euid: Optional[int]) -> bool:
    """
    Cross-platform root/admin detection.
    Truthful, not optimistic.
    """
    system = platform.system()

    if system in ("Linux", "Darwin"):
        return bool(euid == 0)

    if system == "Windows":
        # No reliable, safe stdlib root check without ctypes.
        # Explicitly mark as False instead of guessing.
        return False

    return False


def collect_environment_fingerprint() -> Dict[str, object]:
    """
    One-time, read-only environment fingerprint.

    HARD GUARANTEES:
    - Never crashes
    - Never probes network
    - Never mutates state
    - Never assumes privilege APIs exist
    - Stable schema across OSes
    """

    uid = _get_uid()
    euid = _get_euid()

    fingerprint: Dict[str, object] = {
        # --- OS / PLATFORM ---
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),

        # --- IDENTITY ---
        "hostname": _safe_call(socket.gethostname),
        "username": _safe_call(getpass.getuser),

        # --- PRIVILEGE (OPTIONAL / NULLABLE) ---
        "uid": uid,
        "euid": euid,
        "is_root": _is_root(uid, euid),

        # --- DISPLAY / SESSION ---
        "display": os.environ.get("DISPLAY"),
        "wayland_display": os.environ.get("WAYLAND_DISPLAY"),
        "desktop_session": os.environ.get("XDG_SESSION_DESKTOP"),
        "session_type": os.environ.get("XDG_SESSION_TYPE"),

        # --- SHELL / TERMINAL ---
        "shell": os.environ.get("SHELL"),
        "term": os.environ.get("TERM"),

        # --- SAFETY FLAGS ---
        "running_in_container": bool(
            os.environ.get("container")
            or os.path.exists("/.dockerenv")
        ),
        "ci_environment": bool(os.environ.get("CI")),
    }

    return fingerprint
