import os
import platform
import socket
import getpass
import shutil
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


def _is_root(uid: Optional[int], euid: Optional[int]) -> Optional[bool]:
    """
    Truthful privilege detection.
    Returns:
      - True / False where reliable
      - None where unknown (Windows stdlib limitation)
    """
    system = platform.system()

    if system in ("Linux", "Darwin"):
        return bool(euid == 0)

    if system == "Windows":
        return None  # explicitly unknown

    return None


def _which(cmd: str) -> bool:
    return bool(shutil.which(cmd))


def collect_environment_fingerprint() -> Dict[str, object]:
    """
    One-time, read-only environment fingerprint.

    GUARANTEES:
    - Never crashes
    - Never mutates system
    - Never assumes privileges
    - Never guesses capabilities
    """

    uid = _get_uid()
    euid = _get_euid()
    system = platform.system()

    # --- DISPLAY DETECTION ---
    display_available = False
    if system == "Linux":
        display_available = bool(
            os.environ.get("DISPLAY")
            or os.environ.get("WAYLAND_DISPLAY")
        )
    elif system == "Darwin":
        # macOS always has WindowServer if GUI session
        display_available = bool(os.environ.get("TERM_PROGRAM"))
    elif system == "Windows":
        display_available = True  # GUI always present

    # --- CONTAINER / VIRTUALIZATION ---
    in_container = bool(
        os.environ.get("container")
        or os.path.exists("/.dockerenv")
        or _safe_call(
            lambda: "docker" in open("/proc/1/cgroup").read(),
            False,
        )
    )

    in_wsl = bool(os.environ.get("WSL_DISTRO_NAME"))

    fingerprint: Dict[str, object] = {
        # -------------------------------------------------
        # PLATFORM
        # -------------------------------------------------
        "os": system,
        "os_release": platform.release(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "platform": platform.platform(),

        # -------------------------------------------------
        # RUNTIME
        # -------------------------------------------------
        "python_version": platform.python_version(),
        "python_executable": shutil.which("python")
        or shutil.which("python3"),

        # -------------------------------------------------
        # IDENTITY
        # -------------------------------------------------
        "hostname": _safe_call(socket.gethostname),
        "username": _safe_call(getpass.getuser),

        # -------------------------------------------------
        # PRIVILEGE
        # -------------------------------------------------
        "uid": uid,
        "euid": euid,
        "is_root": _is_root(uid, euid),

        # -------------------------------------------------
        # DISPLAY / SESSION
        # -------------------------------------------------
        "display_available": display_available,
        "display_env": {
            "DISPLAY": os.environ.get("DISPLAY"),
            "WAYLAND_DISPLAY": os.environ.get("WAYLAND_DISPLAY"),
            "XDG_SESSION_TYPE": os.environ.get("XDG_SESSION_TYPE"),
            "XDG_SESSION_DESKTOP": os.environ.get(
                "XDG_SESSION_DESKTOP"
            ),
        },

        # -------------------------------------------------
        # TOOL AVAILABILITY (NO EXECUTION)
        # -------------------------------------------------
        "tools": {
            "git": _which("git"),
            "docker": _which("docker"),
            "docker_compose": _which("docker-compose"),
            "node": _which("node"),
            "npm": _which("npm"),
            "pnpm": _which("pnpm"),
            "yarn": _which("yarn"),
            "pip": _which("pip") or _which("pip3"),
            "apt": _which("apt"),
            "brew": _which("brew"),
            "choco": _which("choco"),
        },

        # -------------------------------------------------
        # EXECUTION CONTEXT
        # -------------------------------------------------
        "running_in_container": in_container,
        "running_in_wsl": in_wsl,
        "ci_environment": bool(os.environ.get("CI")),

        # -------------------------------------------------
        # SAFETY FLAGS
        # -------------------------------------------------
        "network_assumed_available": None,  # explicitly unknown
        "can_install_packages": None,       # requires privilege + policy
    }

    return fingerprint
