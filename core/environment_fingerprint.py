import os
import platform
import socket
from typing import Dict


def collect_environment_fingerprint() -> Dict[str, object]:
    """
    One-time, read-only environment fingerprint.
    No probing, no mutation, no network calls.
    """

    fingerprint = {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "hostname": socket.gethostname(),
        "uid": os.getuid(),
        "euid": os.geteuid(),
        "is_root": os.geteuid() == 0,
        "display": os.environ.get("DISPLAY"),
        "wayland_display": os.environ.get("WAYLAND_DISPLAY"),
        "desktop_session": os.environ.get("XDG_SESSION_DESKTOP"),
        "session_type": os.environ.get("XDG_SESSION_TYPE"),
        "shell": os.environ.get("SHELL"),
        "term": os.environ.get("TERM"),
    }

    return fingerprint
