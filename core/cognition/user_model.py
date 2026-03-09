"""
core/cognition/user_model.py

Theory of Mind user model with online preference learning.

Tracks:
  - User working style (deliberate vs. fast)
  - Application expertise per-app
  - Interruption tolerance (how often user approves vs. denies)
  - Task domain preferences
  - Inferred goals from instruction patterns

Updates after every human-approval event and task outcome.
Used by GIIController to calibrate confirmation frequency and
communication style.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

_logger = logging.getLogger(__name__)

_STATE_PATH = os.path.expanduser(
    os.environ.get("PROJECTZEO_USER_MODEL_PATH", "~/.projectzeo/user_model.json")
)
_DECAY      = float(os.environ.get("PROJECTZEO_USER_MODEL_DECAY", "0.95"))
_MAX_HISTORY = 200


@dataclass
class AppExpertise:
    app_name:         str
    interactions:     int   = 0
    approvals:        int   = 0
    denials:          int   = 0
    avg_dwell_sec:    float = 0.0

    @property
    def approval_rate(self) -> float:
        total = self.approvals + self.denials
        return self.approvals / total if total > 0 else 0.5

    @property
    def expertise_score(self) -> float:
        return math.log1p(self.interactions) / 10.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UserPreferences:
    interruption_tolerance:  float = 0.5   # 0 = hates interruptions, 1 = fine with them
    explanation_verbosity:   float = 0.5   # 0 = terse, 1 = detailed
    autonomy_preference:     float = 0.5   # 0 = step-by-step approval, 1 = full auto
    speed_preference:        float = 0.5   # 0 = careful/slow, 1 = fast
    last_updated:            float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UserPreferences":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


class UserModel:
    """
    Online user model updated from approval/denial events and task outcomes.

    Provides:
      - should_auto_approve(app, operation) → bool
      - get_communication_style() → dict
      - on_approval(app, operation)
      - on_denial(app, operation)
      - on_task_complete(success, duration_s)
    """

    def __init__(self, state_path: Optional[str] = None) -> None:
        self._path = state_path or _STATE_PATH
        self._lock = threading.Lock()
        self._prefs      = UserPreferences()
        self._apps:       Dict[str, AppExpertise] = {}
        self._task_hist:  List[Dict[str, Any]]    = []
        self._goal_vocab: Dict[str, int]           = {}
        self._load()
        _logger.debug("[UserModel] Loaded. apps=%d", len(self._apps))

    # -------------------------------------------------------------------------
    # Events
    # -------------------------------------------------------------------------

    def on_approval(self, app: str, operation: str) -> None:
        with self._lock:
            exp = self._get_or_create_app(app)
            exp.interactions += 1
            exp.approvals    += 1
            # Increase autonomy and reduce interruption tolerance slightly
            self._update_pref("interruption_tolerance", -0.02)
            self._update_pref("autonomy_preference", +0.03)
        self._save_async()

    def on_denial(self, app: str, operation: str) -> None:
        with self._lock:
            exp = self._get_or_create_app(app)
            exp.interactions += 1
            exp.denials      += 1
            # User wants more control
            self._update_pref("interruption_tolerance", +0.03)
            self._update_pref("autonomy_preference", -0.04)
        self._save_async()

    def on_task_complete(
        self,
        success: bool,
        duration_s: float,
        app: str = "",
        objective: str = "",
    ) -> None:
        with self._lock:
            record = {
                "success":    success,
                "duration_s": round(duration_s, 1),
                "app":        app,
                "ts":         time.time(),
            }
            self._task_hist.append(record)
            if len(self._task_hist) > _MAX_HISTORY:
                self._task_hist = self._task_hist[-_MAX_HISTORY:]

            if success:
                self._update_pref("autonomy_preference", +0.01)
            else:
                self._update_pref("autonomy_preference", -0.02)

            if objective:
                for word in objective.lower().split()[:10]:
                    if len(word) > 3:
                        self._goal_vocab[word] = self._goal_vocab.get(word, 0) + 1

        self._save_async()

    def on_dwell(self, app: str, dwell_sec: float) -> None:
        with self._lock:
            exp = self._get_or_create_app(app)
            if exp.avg_dwell_sec == 0.0:
                exp.avg_dwell_sec = dwell_sec
            else:
                exp.avg_dwell_sec = 0.9 * exp.avg_dwell_sec + 0.1 * dwell_sec

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    def should_auto_approve(self, app: str, operation: str) -> bool:
        """
        Returns True if the agent should skip human confirmation for this
        app+operation based on learned user preferences.
        """
        with self._lock:
            auto_threshold = 0.7 + self._prefs.autonomy_preference * 0.3
            exp = self._apps.get(app.lower())
            if exp is None:
                return False
            return (
                exp.approval_rate >= auto_threshold
                and exp.interactions >= 5
                and operation not in ("install", "file_create", "command")
            )

    def get_communication_style(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "verbosity":    "detailed" if self._prefs.explanation_verbosity > 0.6 else "brief",
                "autonomy":     "supervised" if self._prefs.autonomy_preference < 0.4 else "autonomous",
                "speed":        "careful" if self._prefs.speed_preference < 0.4 else "fast",
                "tolerance":    round(self._prefs.interruption_tolerance, 2),
            }

    def get_app_expertise(self, app: str) -> Optional[AppExpertise]:
        with self._lock:
            return self._apps.get(app.lower())

    def get_common_goals(self, top_n: int = 5) -> List[str]:
        with self._lock:
            return sorted(self._goal_vocab, key=lambda w: -self._goal_vocab[w])[:top_n]

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "apps_tracked":  len(self._apps),
                "tasks_recorded": len(self._task_hist),
                "preferences":   self._prefs.to_dict(),
                "top_goals":     self.get_common_goals(),
            }

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _get_or_create_app(self, app: str) -> AppExpertise:
        key = app.lower()
        if key not in self._apps:
            self._apps[key] = AppExpertise(app_name=key)
        return self._apps[key]

    def _update_pref(self, name: str, delta: float) -> None:
        current = getattr(self._prefs, name)
        setattr(self._prefs, name, max(0.0, min(1.0, current + delta)))
        self._prefs.last_updated = time.time()

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            self._prefs = UserPreferences.from_dict(data.get("preferences", {}))
            for k, v in data.get("apps", {}).items():
                self._apps[k] = AppExpertise(**{
                    fk: fv for fk, fv in v.items()
                    if fk in AppExpertise.__dataclass_fields__
                })
            self._task_hist  = data.get("task_history", [])[-_MAX_HISTORY:]
            self._goal_vocab = data.get("goal_vocab", {})
        except Exception as e:
            _logger.warning("[UserModel] Load failed: %s", e)

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        tmp = self._path + ".tmp"
        try:
            with self._lock:
                data = {
                    "preferences":  self._prefs.to_dict(),
                    "apps":         {k: v.to_dict() for k, v in self._apps.items()},
                    "task_history": self._task_hist[-_MAX_HISTORY:],
                    "goal_vocab":   self._goal_vocab,
                }
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, separators=(",", ":"))
            os.replace(tmp, self._path)
        except Exception as e:
            _logger.debug("[UserModel] Save failed: %s", e)

    def _save_async(self) -> None:
        t = threading.Thread(target=self._save, daemon=True)
        t.start()


_instance: Optional[UserModel] = None
_instance_lock = threading.Lock()


def get_user_model() -> UserModel:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = UserModel()
    return _instance
