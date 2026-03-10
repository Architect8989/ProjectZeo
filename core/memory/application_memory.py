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

@dataclass
class InstallRecord:
    method: str
    command: str
    success: bool
    os_name: str
    os_version: str
    timestamp: float = field(default_factory=time.time)
    error_message: str = ""
    installed_version: str = ""

@dataclass
class WorkflowStep:
    step_index: int
    description: str
    action_type: str
    action_detail: str
    success_rate: float = 1.0

@dataclass
class ApplicationProfile:
    
    app_name: str
    display_name: str = ""
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    task_count: int = 0

    shortcuts: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    menu_paths: Dict[str, str] = field(default_factory=dict)

    workflows: Dict[str, List[WorkflowStep]] = field(default_factory=dict)

    install_records: List[InstallRecord] = field(default_factory=list)

    quirks: List[str] = field(default_factory=list)

    required_setup: List[str] = field(default_factory=list)

    error_solutions: Dict[str, str] = field(default_factory=dict)

    attempt_history: List[Dict[str, Any]] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_known(self) -> bool:
        return bool(
            self.shortcuts or self.menu_paths or self.workflows
            or self.install_records or self.quirks or self.required_setup
        )

    def best_install_method(self, os_name: str = "") -> Optional[InstallRecord]:
        candidates = [
            r for r in self.install_records
            if r.success and (not os_name or r.os_name == os_name)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.timestamp)

    def add_attempt(
        self,
        objective: str,
        success: bool,
        lesson: str = "",
        outcome_summary: str = "",
        *,
        max_history: int = 20,
    ) -> None:
        import time as _time
        entry = {
            "ts": _time.time(),
            "objective": objective[:200],
            "success": success,
            "lesson": lesson[:300],
            "outcome": outcome_summary[:200],
        }
        self.attempt_history.append(entry)
        if len(self.attempt_history) > max_history:
            self.attempt_history = self.attempt_history[-max_history:]

    def format_hindsight_for_prompt(self, max_entries: int = 5) -> str:
        if not self.attempt_history:
            return ""
        entries = self.attempt_history[-max_entries:]
        lines = ["[Chain of Hindsight — Prior Attempts]"]
        for i, e in enumerate(entries, 1):
            status = "✓ SUCCESS" if e.get("success") else "✗ FAILURE"
            obj = e.get("objective", "")[:80]
            lesson = e.get("lesson", "")[:120]
            lines.append(f"  {i}. {status}: {obj}")
            if lesson:
                lines.append(f"     → Lesson: {lesson}")
        return "\n".join(lines)

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
        key = self._normalize(app_name)
        with self._lock:
            p = self._profiles.get(key)
        return p is not None and p.is_known()

    def record_shortcut(
        self,
        app_name: str,
        action: str,
        keys: str,
        *,
        confirmed: bool = True,
    ) -> None:
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
        profile = self.get_profile(app_name)
        quirk_norm = quirk.strip()
        if quirk_norm and quirk_norm not in profile.quirks:
            profile.quirks.append(quirk_norm)
            self._mark_dirty_and_update_seen(profile)

    def record_error_solution(
        self, app_name: str, error_snippet: str, solution: str
    ) -> None:
        profile = self.get_profile(app_name)
        profile.error_solutions[error_snippet.strip()[:200]] = solution.strip()[:500]
        self._mark_dirty_and_update_seen(profile)

    def record_required_setup(self, app_name: str, setup_step: str) -> None:
        profile = self.get_profile(app_name)
        step_norm = setup_step.strip()
        if step_norm and step_norm not in profile.required_setup:
            profile.required_setup.append(step_norm)
            self._mark_dirty_and_update_seen(profile)

    def increment_task_count(self, app_name: str) -> None:
        profile = self.get_profile(app_name)
        profile.task_count += 1
        self._mark_dirty_and_update_seen(profile)

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

        best_install = profile.best_install_method()
        if best_install:
            lines.append(
                f"  Install: use '{best_install.command}' "
                f"(method={best_install.method}, "
                f"verified on {best_install.os_name})"
            )

        if profile.required_setup:
            lines.append("  Required setup before use:")
            for step in profile.required_setup[:3]:
                lines.append(f"    - {step}")

        if profile.shortcuts:
            top_shortcuts = sorted(
                profile.shortcuts.items(),
                key=lambda x: x[1].get("confidence", 0),
                reverse=True,
            )[:max_shortcuts]
            lines.append("  Known shortcuts:")
            for action, info in top_shortcuts:
                lines.append(f"    {action}: {info['keys']}")

        if profile.menu_paths:
            lines.append("  Menu paths:")
            for action, path in list(profile.menu_paths.items())[:5]:
                lines.append(f"    {action}: {path}")

        if profile.quirks:
            lines.append("  Known quirks:")
            for quirk in profile.quirks[:max_quirks]:
                lines.append(f"    - {quirk}")

        if profile.error_solutions:
            lines.append("  Known errors and solutions:")
            for error, solution in list(profile.error_solutions.items())[:max_errors]:
                lines.append(f"    If you see '{error}': {solution}")

        hindsight = profile.format_hindsight_for_prompt(max_entries=5)
        if hindsight:
            lines.append(hindsight)

        lines.append(
            f"  (Profile from {profile.task_count} prior task(s). "
            f"Last used: {_fmt_time(profile.last_seen)})"
        )
        return "\n".join(lines)

    def list_known_apps(self) -> List[str]:
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
        self._persist()

    def _load(self) -> None:
        if not os.path.exists(self._memory_path):
            return
        try:
            with open(self._memory_path, "rb") as f:
                data = json.loads(f.read().decode("utf-8"))
            for raw in data.get("profiles", []):
                try:
                    raw["install_records"] = [
                        InstallRecord(**r) for r in raw.get("install_records", [])
                    ]
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

_global_app_memory: Optional[ApplicationMemory] = None
_global_lock = threading.Lock()

def get_global_application_memory(memory_dir: Optional[str] = None) -> ApplicationMemory:
    global _global_app_memory
    with _global_lock:
        if _global_app_memory is None:
            _global_app_memory = ApplicationMemory(memory_dir=memory_dir)
    return _global_app_memory
