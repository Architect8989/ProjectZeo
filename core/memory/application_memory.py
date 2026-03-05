from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set

_logger = logging.getLogger(__name__)

_DEFAULT_APP_MEMORY_DIR = os.path.join(
    os.path.expanduser("~"), ".projectzeo", "app_memory"
)
_APP_MEMORY_FILE = "application_profiles.json"
_MAX_PROFILES = 500
_MAX_SHORTCUTS_PER_APP = 200
_MAX_WORKFLOWS_PER_APP = 50


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class InstallRecord:
    """Result of an installation attempt."""
    method: str           # e.g. "apt", "pip", "npm", "manual_download"
    command: str          # The actual install command used
    success: bool
    os_name: str          # "Linux", "Darwin", "Windows"
    os_version: str       # e.g. "Ubuntu 22.04"
    timestamp: float = field(default_factory=time.time)
    error_message: str = ""
    installed_version: str = ""


@dataclass
class WorkflowStep:
    """A single step in a known application workflow."""
    step_index: int
    description: str
    action_type: str      # "click", "hotkey", "command", "type", etc.
    action_detail: str    # The specific text/keys/command
    success_rate: float = 1.0


@dataclass
class ApplicationProfile:
    
    app_name: str                             # Normalised process name
    display_name: str = ""                    # Human-readable name (e.g. "Blender 4.1")
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    task_count: int = 0                       # Number of tasks run against this app

    # Shortcuts: {"action_name": {"keys": "Ctrl+T", "confidence": 0.9, "confirmed_count": 3}}
    shortcuts: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Menu paths: {"action": "File > Export > Export as PNG"}
    menu_paths: Dict[str, str] = field(default_factory=dict)

    # Known workflows: {"workflow_name": [WorkflowStep, ...]}
    workflows: Dict[str, List[WorkflowStep]] = field(default_factory=dict)

    # Installation history across OS/method combinations
    install_records: List[InstallRecord] = field(default_factory=list)

    # Free-text quirks and notes
    quirks: List[str] = field(default_factory=list)

    # Required setup steps (env vars, permissions, display setup, etc.)
    required_setup: List[str] = field(default_factory=list)

    # Error → solution mappings: {"error_text_snippet": "solution_description"}
    error_solutions: Dict[str, str] = field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_known(self) -> bool:
        """Return True if we have any substantive knowledge about this app."""
        return bool(
            self.shortcuts or self.menu_paths or self.workflows
            or self.install_records or self.quirks or self.required_setup
        )

    def best_install_method(self, os_name: str = "") -> Optional[InstallRecord]:
        """Return the most recent successful install record for the given OS."""
        candidates = [
            r for r in self.install_records
            if r.success and (not os_name or r.os_name == os_name)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.timestamp)


# ---------------------------------------------------------------------------
# ApplicationMemory
# ---------------------------------------------------------------------------

class ApplicationMemory:
    

    def __init__(
        self,
        memory_dir: Optional[str] = None,
        *,
        max_profiles: int = _MAX_PROFILES,
    ) -> None:
        self._memory_dir = memory_dir or _DEFAULT_APP_MEMORY_DIR
        self._memory_path = os.path.join(self._memory_dir, _APP_MEMORY_FILE)
        self._max_profiles = max_profiles

        self._profiles: Dict[str, ApplicationProfile] = {}
        self._lock = threading.RLock()
        self._dirty = False

        os.makedirs(self._memory_dir, exist_ok=True)
        self._load()

        _logger.info(
            "[ApplicationMemory] Initialised. profiles=%d", len(self._profiles)
        )

    # =========================================================================
    # Profile access
    # =========================================================================

    def get_profile(self, app_name: str) -> ApplicationProfile:
        
        key = self._normalize(app_name)
        with self._lock:
            if key not in self._profiles:
                self._profiles[key] = ApplicationProfile(
                    app_name=key, display_name=app_name
                )
                self._dirty = True
            return self._profiles[key]

    def has_profile(self, app_name: str) -> bool:
        """Return True if a profile with substantive knowledge exists."""
        key = self._normalize(app_name)
        with self._lock:
            p = self._profiles.get(key)
        return p is not None and p.is_known()

    # =========================================================================
    # Recording API
    # =========================================================================

    def record_shortcut(
        self,
        app_name: str,
        action: str,
        keys: str,
        *,
        confirmed: bool = True,
    ) -> None:
        """Record a keyboard shortcut for an application action."""
        profile = self.get_profile(app_name)
        action_key = action.strip().lower()

        if len(profile.shortcuts) >= _MAX_SHORTCUTS_PER_APP:
            return

        existing = profile.shortcuts.get(action_key, {})
        count = existing.get("confirmed_count", 0) + (1 if confirmed else 0)
        confidence = min(1.0, 0.5 + count * 0.1)

        profile.shortcuts[action_key] = {
            "keys": keys.strip(),
            "confidence": confidence,
            "confirmed_count": count,
            "last_seen": time.time(),
        }
        self._mark_dirty_and_update_seen(profile)
        _logger.debug("[AppMemory] Shortcut: %s → %s = %s", app_name, action, keys)

    def record_menu_path(
        self,
        app_name: str,
        action: str,
        menu_path: str,
    ) -> None:
        """Record a menu navigation path for an action."""
        profile = self.get_profile(app_name)
        profile.menu_paths[action.strip().lower()] = menu_path.strip()
        self._mark_dirty_and_update_seen(profile)

    def record_install_success(
        self,
        app_name: str,
        method: str,
        command: str,
        *,
        os_name: str = "",
        os_version: str = "",
        installed_version: str = "",
    ) -> None:
        """Record a successful installation."""
        import platform as _plat
        profile = self.get_profile(app_name)
        record = InstallRecord(
            method=method,
            command=command,
            success=True,
            os_name=os_name or _plat.system(),
            os_version=os_version or _plat.version(),
            installed_version=installed_version,
        )
        profile.install_records.append(record)
        self._mark_dirty_and_update_seen(profile)
        _logger.info("[AppMemory] Install success: %s via %s", app_name, method)

    def record_install_failure(
        self,
        app_name: str,
        method: str,
        command: str,
        error_message: str = "",
        *,
        os_name: str = "",
        os_version: str = "",
    ) -> None:
        """Record a failed installation attempt."""
        import platform as _plat
        profile = self.get_profile(app_name)
        record = InstallRecord(
            method=method,
            command=command,
            success=False,
            os_name=os_name or _plat.system(),
            os_version=os_version or _plat.version(),
            error_message=error_message[:500],
        )
        profile.install_records.append(record)
        self._mark_dirty_and_update_seen(profile)
        _logger.info(
            "[AppMemory] Install failure: %s via %s — %s",
            app_name, method, error_message[:80],
        )

    def record_quirk(self, app_name: str, quirk: str) -> None:
        """Record a UI quirk or gotcha for an application."""
        profile = self.get_profile(app_name)
        quirk_norm = quirk.strip()
        if quirk_norm and quirk_norm not in profile.quirks:
            profile.quirks.append(quirk_norm)
            self._mark_dirty_and_update_seen(profile)

    def record_error_solution(
        self, app_name: str, error_snippet: str, solution: str
    ) -> None:
        """Record an error → solution mapping."""
        profile = self.get_profile(app_name)
        profile.error_solutions[error_snippet.strip()[:200]] = solution.strip()[:500]
        self._mark_dirty_and_update_seen(profile)

    def record_required_setup(self, app_name: str, setup_step: str) -> None:
        """Record a required setup step (e.g. 'export DISPLAY=:0')."""
        profile = self.get_profile(app_name)
        step_norm = setup_step.strip()
        if step_norm and step_norm not in profile.required_setup:
            profile.required_setup.append(step_norm)
            self._mark_dirty_and_update_seen(profile)

    def increment_task_count(self, app_name: str) -> None:
        """Increment the task count for the given application."""
        profile = self.get_profile(app_name)
        profile.task_count += 1
        self._mark_dirty_and_update_seen(profile)

    # =========================================================================
    # Prompt formatting
    # =========================================================================

    def format_profile_for_prompt(
        self,
        app_name: str,
        *,
        max_shortcuts: int = 10,
        max_quirks: int = 5,
        max_errors: int = 5,
    ) -> str:
        
        if not self.has_profile(app_name):
            return ""

        profile = self.get_profile(app_name)
        lines = [f"Application knowledge for: {profile.display_name or app_name}"]

        # Install info
        best_install = profile.best_install_method()
        if best_install:
            lines.append(
                f"  Install: use '{best_install.command}' "
                f"(method={best_install.method}, "
                f"verified on {best_install.os_name})"
            )

        # Required setup
        if profile.required_setup:
            lines.append("  Required setup before use:")
            for step in profile.required_setup[:3]:
                lines.append(f"    - {step}")

        # Top shortcuts
        if profile.shortcuts:
            top_shortcuts = sorted(
                profile.shortcuts.items(),
                key=lambda x: x[1].get("confidence", 0),
                reverse=True,
            )[:max_shortcuts]
            lines.append("  Known shortcuts:")
            for action, info in top_shortcuts:
                lines.append(f"    {action}: {info['keys']}")

        # Menu paths
        if profile.menu_paths:
            lines.append("  Menu paths:")
            for action, path in list(profile.menu_paths.items())[:5]:
                lines.append(f"    {action}: {path}")

        # Quirks
        if profile.quirks:
            lines.append("  Known quirks:")
            for quirk in profile.quirks[:max_quirks]:
                lines.append(f"    - {quirk}")

        # Error solutions
        if profile.error_solutions:
            lines.append("  Known errors and solutions:")
            for error, solution in list(profile.error_solutions.items())[:max_errors]:
                lines.append(f"    If you see '{error}': {solution}")

        lines.append(
            f"  (Profile from {profile.task_count} prior task(s). "
            f"Last used: {_fmt_time(profile.last_seen)})"
        )
        return "\n".join(lines)

    def list_known_apps(self) -> List[str]:
        """Return sorted list of all apps with substantive profiles."""
        with self._lock:
            return sorted(
                key for key, p in self._profiles.items() if p.is_known()
            )

    def stats(self) -> dict:
        with self._lock:
            total = len(self._profiles)
            known = sum(1 for p in self._profiles.values() if p.is_known())
        return {
            "total_profiles": total,
            "known_apps": known,
            "memory_path": self._memory_path,
        }

    def save(self) -> None:
        """Force-save all profiles to disk."""
        self._persist()

    # =========================================================================
    # Persistence
    # =========================================================================

    def _load(self) -> None:
        if not os.path.exists(self._memory_path):
            return
        try:
            with open(self._memory_path, "rb") as f:
                data = json.loads(f.read().decode("utf-8"))
            for raw in data.get("profiles", []):
                try:
                    # Deserialize install_records sub-list
                    raw["install_records"] = [
                        InstallRecord(**r) for r in raw.get("install_records", [])
                    ]
                    # Deserialize workflows
                    workflows_raw = raw.pop("workflows", {})
                    workflows = {}
                    for wf_name, steps in workflows_raw.items():
                        workflows[wf_name] = [WorkflowStep(**s) for s in steps]
                    raw["workflows"] = workflows
                    profile = ApplicationProfile(**raw)
                    self._profiles[profile.app_name] = profile
                except Exception:
                    pass
            _logger.info(
                "[ApplicationMemory] Loaded %d profiles.", len(self._profiles)
            )
        except Exception as exc:
            _logger.warning("[ApplicationMemory] Load failed: %s", exc)

    def _persist(self) -> None:
        with self._lock:
            if not self._dirty:
                return

            profiles_data = []
            for profile in self._profiles.values():
                try:
                    d = asdict(profile)
                    profiles_data.append(d)
                except Exception:
                    pass

            payload = json.dumps(
                {"profiles": profiles_data, "saved_at": time.time()},
                ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")

            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=self._memory_dir, delete=False
                ) as tmp:
                    tmp.write(payload)
                    tmp.flush()
                    os.fsync(tmp.fileno())
                    tmp_path = tmp.name
                os.replace(tmp_path, self._memory_path)
                self._dirty = False
            except Exception as exc:
                _logger.error("[ApplicationMemory] Persist failed: %s", exc)

    def _mark_dirty_and_update_seen(self, profile: ApplicationProfile) -> None:
        profile.last_seen = time.time()
        self._dirty = True

    @staticmethod
    def _normalize(app_name: str) -> str:
        return app_name.strip().lower()[:100]

    def __del__(self):
        try:
            self._persist()
        except Exception:
            pass


def _fmt_time(ts: float) -> str:
    import datetime
    try:
        return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_global_app_memory: Optional[ApplicationMemory] = None
_global_lock = threading.Lock()


def get_global_application_memory(memory_dir: Optional[str] = None) -> ApplicationMemory:
    """Return the process-singleton ApplicationMemory instance."""
    global _global_app_memory
    with _global_lock:
        if _global_app_memory is None:
            _global_app_memory = ApplicationMemory(memory_dir=memory_dir)
    return _global_app_memory
