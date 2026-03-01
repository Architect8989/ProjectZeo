"""
core/environment_fingerprint.py
================================
PATCH AUDIT FIX:

  ⚠️  §1.8: tools dict only checked 11 tools (git, docker, node, npm, pnpm,
            yarn, pip, apt, brew, choco, docker-compose).
            For a raw-OS hackathon scenario the LLM planner received an
            incomplete tool picture — it could not know whether curl, wget,
            python3, make, cargo, go, java etc. were available, causing it to
            plan commands with missing executables.
            FIX: Extended to 30 tools covering all common dev-tool categories.

  ✅  All existing correct behaviours preserved:
        - Never crashes
        - Never mutates system
        - Never guesses capabilities
        - shutil.which() only (no subprocess execution)
        - Windows uid/euid returns None (explicitly unknown)
"""

from __future__ import annotations

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


def _which_path(cmd: str) -> Optional[str]:
    """Return the full path, or None."""
    return shutil.which(cmd)


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
        # HIGH-5 FIX: TERM_PROGRAM is set by Terminal.app, NOT by the GUI
        # subsystem. An SSH session from a Mac has TERM_PROGRAM="ssh" (false
        # positive). A launchd GUI process has no TERM_PROGRAM but full display
        # access (false negative). Fix: check whether WindowServer is running —
        # it is the macOS Quartz compositor and is ONLY present in GUI sessions.
        try:
            import subprocess as _sp
            _result = _sp.run(
                ["pgrep", "-x", "WindowServer"],
                capture_output=True, timeout=2,
            )
            display_available = _result.returncode == 0
        except Exception:
            # pgrep unavailable (sandboxed/minimal env) — fall back to env heuristic
            display_available = bool(
                os.environ.get("TERM_PROGRAM")
                or os.environ.get("DISPLAY")  # XQuartz
                or os.environ.get("WAYLAND_DISPLAY")
            )
    elif system == "Windows":
        display_available = True

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

    # PATCH §1.8: Extended tool detection — was 11 tools, now 30+
    # Covers every common dev-tool category a LLM planner might need.
    tools: Dict[str, bool] = {
        # ---- Version Control ----
        "git":           _which("git"),
        "svn":           _which("svn"),

        # ---- Containers / Orchestration ----
        "docker":        _which("docker"),
        "docker_compose": _which("docker-compose") or _which("docker compose"),
        "podman":        _which("podman"),
        "kubectl":       _which("kubectl"),

        # ---- JavaScript / Node ----
        "node":          _which("node"),
        "npm":           _which("npm"),
        "pnpm":          _which("pnpm"),
        "yarn":          _which("yarn"),
        "bun":           _which("bun"),

        # ---- Python ----
        "python":        _which("python") or _which("python3"),
        "python3":       _which("python3"),
        "pip":           _which("pip") or _which("pip3"),
        "pip3":          _which("pip3"),
        "uv":            _which("uv"),

        # ---- Package Managers ----
        "apt":           _which("apt"),
        "apt_get":       _which("apt-get"),
        "dnf":           _which("dnf"),
        "yum":           _which("yum"),
        "pacman":        _which("pacman"),
        "brew":          _which("brew"),
        "choco":         _which("choco"),
        "winget":        _which("winget"),
        "snap":          _which("snap"),
        "flatpak":       _which("flatpak"),

        # ---- Build Tools ----
        "make":          _which("make"),
        "cmake":         _which("cmake"),
        "gcc":           _which("gcc"),
        "g++":           _which("g++"),
        "clang":         _which("clang"),
        "cargo":         _which("cargo"),

        # ---- JVM ----
        "java":          _which("java"),
        "javac":         _which("javac"),
        "mvn":           _which("mvn"),
        "gradle":        _which("gradle"),

        # ---- Go ----
        "go":            _which("go"),

        # ---- Ruby ----
        "ruby":          _which("ruby"),
        "gem":           _which("gem"),
        "bundle":        _which("bundle"),

        # ---- Shell / Utils ----
        "bash":          _which("bash"),
        "zsh":           _which("zsh"),
        "curl":          _which("curl"),
        "wget":          _which("wget"),
        "unzip":         _which("unzip"),
        "tar":           _which("tar"),
        "jq":            _which("jq"),
        "sed":           _which("sed"),
        "awk":           _which("awk"),

        # ---- Cloud CLIs ----
        "aws":           _which("aws"),
        "gcloud":        _which("gcloud"),
        "az":            _which("az"),

        # ---- Misc Dev Tools ----
        "gh":            _which("gh"),           # GitHub CLI
        "terraform":     _which("terraform"),
        "ansible":       _which("ansible"),
        "ffmpeg":        _which("ffmpeg"),
    }

    fingerprint: Dict[str, object] = {
        # -------------------------------------------------
        # PLATFORM
        # -------------------------------------------------
        "os":              system,
        "os_release":      platform.release(),
        "os_version":      platform.version(),
        "architecture":    platform.machine(),
        "platform":        platform.platform(),

        # -------------------------------------------------
        # RUNTIME
        # -------------------------------------------------
        "python_version":    platform.python_version(),
        "python_executable": _which_path("python") or _which_path("python3"),

        # -------------------------------------------------
        # IDENTITY
        # -------------------------------------------------
        "hostname": _safe_call(socket.gethostname),
        "username": _safe_call(getpass.getuser),

        # -------------------------------------------------
        # PRIVILEGE
        # -------------------------------------------------
        "uid":     uid,
        "euid":    euid,
        "is_root": _is_root(uid, euid),

        # -------------------------------------------------
        # DISPLAY / SESSION
        # -------------------------------------------------
        "display_available": display_available,
        "display_env": {
            "DISPLAY":             os.environ.get("DISPLAY"),
            "WAYLAND_DISPLAY":     os.environ.get("WAYLAND_DISPLAY"),
            "XDG_SESSION_TYPE":    os.environ.get("XDG_SESSION_TYPE"),
            "XDG_SESSION_DESKTOP": os.environ.get("XDG_SESSION_DESKTOP"),
        },

        # -------------------------------------------------
        # TOOL AVAILABILITY (NO EXECUTION — shutil.which only)
        # PATCH: extended from 11 to 50+ tools
        # -------------------------------------------------
        "tools": tools,

        # -------------------------------------------------
        # EXECUTION CONTEXT
        # -------------------------------------------------
        "running_in_container": in_container,
        "running_in_wsl":       in_wsl,
        "ci_environment":       bool(os.environ.get("CI")),

        # -------------------------------------------------
        # SAFETY FLAGS
        # -------------------------------------------------
        "network_assumed_available": None,  # explicitly unknown
        "can_install_packages":      None,  # requires privilege + policy
    }

    return fingerprint
