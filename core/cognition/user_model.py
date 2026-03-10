"""
core/cognition/user_model.py

Theory of Mind user model with online preference learning.

PATCH: March 2026 — Blueprint §17 Emotional & Social Modeling
  Added:
    - on_keystroke_event()  → stress inference from typing cadence
    - on_mouse_event()      → frustration inference from movement speed/jitter
    - infer_nlp_tone()      → NLP lexical tone analysis of user utterances
    - EmotionalState        → composite stress/engagement/sentiment dataclass
    - Emotional state persisted in JSON alongside preferences (cross-session)

  These three signals (Blueprint §17.2) complete the user ToM layer.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

_logger = logging.getLogger(__name__)

_STATE_PATH = os.path.expanduser(
    os.environ.get("PROJECTZEO_USER_MODEL_PATH", "~/.projectzeo/user_model.json")
)
_DECAY       = float(os.environ.get("PROJECTZEO_USER_MODEL_DECAY", "0.95"))
_MAX_HISTORY = 200

# Keystroke cadence thresholds (Blueprint §17.2)
_FAST_IKI_MS      = 80.0
_SLOW_IKI_MS      = 350.0
_KEYSTROKE_WINDOW = 20

# Mouse speed thresholds
_HIGH_SPEED_PX_S = 2500.0
_HIGH_JITTER_PX  = 15.0
_MOUSE_WINDOW    = 30

# NLP tone lexicons
_STRESS_WORDS = frozenset({
    "urgent","urgently","quickly","quick","asap","hurry","immediately",
    "now","fast","right away","as soon as","critical","emergency",
    "deadline","overdue","stuck","blocked","broken","failing",
})
_FRUSTRATION_WORDS = frozenset({
    "why","still","again","already","wrong","stop","ugh","terrible",
    "awful","hate","doesn't work","not working","failed","keeps",
    "ridiculous","useless","seriously","come on",
})
_POSITIVE_WORDS = frozenset({
    "good","great","perfect","done","thanks","thank you","excellent",
    "nice","awesome","works","correct","yes","proceed","continue",
})

@dataclass
class AppExpertise:
    app_name:      str
    interactions:  int   = 0
    approvals:     int   = 0
    denials:       int   = 0
    avg_dwell_sec: float = 0.0

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
    interruption_tolerance: float = 0.5
    explanation_verbosity:  float = 0.5
    autonomy_preference:    float = 0.5
    speed_preference:       float = 0.5
    last_updated:           float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UserPreferences":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


@dataclass
class EmotionalState:
    """Composite emotional state persisted cross-session (Blueprint §17.2)."""
    stress:       float = 0.0
    frustration:  float = 0.0
    engagement:   float = 0.5
    urgency:      float = 0.0
    sentiment:    float = 0.5
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EmotionalState":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


class UserModel:
    """
    Online user model — preferences, expertise, and emotional state.
    Blueprint §17: Theory of Mind + Social Modeling.
    """

    def __init__(
        self,
        state_path: Optional[str] = None,
        *,
        memory_dir: Optional[str] = None,  # GII-FIX: accept memory_dir from GIIController
    ) -> None:
        # GII-FIX: GIIController calls UserModel(memory_dir=memory_dir) but the
        # old __init__ only accepted state_path.  Derive state_path from memory_dir
        # if state_path is not explicitly given, so the file lands in the right
        # location alongside the other memory stores.
        if state_path is None and memory_dir is not None:
            import pathlib
            state_path = str(pathlib.Path(memory_dir) / "user_model.json")

        self._path = state_path or _STATE_PATH
        self._lock = threading.Lock()
        self._prefs      = UserPreferences()
        self._apps:       Dict[str, AppExpertise] = {}
        self._task_hist:  List[Dict[str, Any]]    = []
        self._goal_vocab: Dict[str, int]           = {}
        self._emotional   = EmotionalState()

        # Sliding windows for physiological proxies
        self._iki_window:   deque = deque(maxlen=_KEYSTROKE_WINDOW)
        self._mouse_window: deque = deque(maxlen=_MOUSE_WINDOW)
        self._last_mouse_ts: float = 0.0
        self._last_mouse_x:  float = 0.0
        self._last_mouse_y:  float = 0.0

        self._load()
        _logger.debug("[UserModel] Loaded. apps=%d", len(self._apps))

    # ── Standard Events ────────────────────────────────────────────────────

    def on_approval(self, app: str, operation: str) -> None:
        with self._lock:
            exp = self._get_or_create_app(app)
            exp.interactions += 1
            exp.approvals    += 1
            self._update_pref("interruption_tolerance", -0.02)
            self._update_pref("autonomy_preference", +0.03)
            self._emotional.frustration = max(0.0, self._emotional.frustration - 0.05)
            self._emotional.last_updated = time.time()
        self._save_async()

    def on_denial(self, app: str, operation: str) -> None:
        with self._lock:
            exp = self._get_or_create_app(app)
            exp.interactions += 1
            exp.denials      += 1
            self._update_pref("interruption_tolerance", +0.03)
            self._update_pref("autonomy_preference", -0.04)
            self._emotional.frustration = min(1.0, self._emotional.frustration + 0.1)
            self._emotional.last_updated = time.time()
        self._save_async()

    def on_objective_received(self, objective: str) -> None:
        """Infer urgency and NLP tone from objective language."""
        tone = self.infer_nlp_tone(objective)
        if tone["urgency"] > 0.3:
            with self._lock:
                self._emotional.urgency = min(1.0, self._emotional.urgency + tone["urgency"] * 0.6)
                self._emotional.last_updated = time.time()

    # ── NEW: Keystroke Timing (Blueprint §17.2) ────────────────────────────

    def on_keystroke_event(self, iki_ms: float) -> None:
        """
        Record inter-key interval (ms). Fast typing → stress/urgency.
        Slow typing → calm/deliberate.
        """
        if iki_ms <= 0:
            return
        with self._lock:
            self._iki_window.append(iki_ms)
            if len(self._iki_window) < 3:
                return
            avg_iki = sum(self._iki_window) / len(self._iki_window)
            if avg_iki < _FAST_IKI_MS:
                delta = 0.05 * (1.0 - avg_iki / _FAST_IKI_MS)
                self._emotional.stress   = min(1.0, self._emotional.stress + delta)
                self._emotional.urgency  = min(1.0, self._emotional.urgency + delta * 0.5)
                self._emotional.engagement = min(1.0, self._emotional.engagement + 0.02)
            elif avg_iki > _SLOW_IKI_MS:
                self._emotional.stress  = max(0.0, self._emotional.stress - 0.03)
                self._emotional.urgency = max(0.0, self._emotional.urgency - 0.015)
            self._emotional.last_updated = time.time()

    # ── NEW: Mouse Movement Analysis (Blueprint §17.2) ────────────────────

    def on_mouse_event(self, x: float, y: float, ts: Optional[float] = None) -> None:
        """
        Record mouse position. High speed → impatience. High jitter → frustration.
        """
        now = ts or time.time()
        with self._lock:
            if self._last_mouse_ts > 0:
                dt_s = max(0.001, now - self._last_mouse_ts)
                dx   = x - self._last_mouse_x
                dy   = y - self._last_mouse_y
                speed_px_s = math.hypot(dx, dy) / dt_s
                self._mouse_window.append((dx, dy, speed_px_s))

                if len(self._mouse_window) >= 5:
                    avg_speed = sum(s for _, _, s in self._mouse_window) / len(self._mouse_window)
                    if avg_speed > _HIGH_SPEED_PX_S:
                        d = 0.03 * min(1.0, avg_speed / (_HIGH_SPEED_PX_S * 2))
                        self._emotional.stress      = min(1.0, self._emotional.stress + d)
                        self._emotional.frustration = min(1.0, self._emotional.frustration + d * 0.5)

                    if len(self._mouse_window) >= 6:
                        xs = [dx_ for dx_, _, _ in list(self._mouse_window)[-6:]]
                        ys = [dy_ for _, dy_, _ in list(self._mouse_window)[-6:]]
                        if self._std(xs) > _HIGH_JITTER_PX or self._std(ys) > _HIGH_JITTER_PX:
                            self._emotional.frustration = min(1.0, self._emotional.frustration + 0.02)

            self._last_mouse_ts = now
            self._last_mouse_x  = x
            self._last_mouse_y  = y
            self._emotional.last_updated = time.time()

    # ── NEW: NLP Tone Analysis (Blueprint §17.2) ──────────────────────────

    def infer_nlp_tone(self, text: str) -> Dict[str, float]:
        """
        Lexical sentiment + stress analysis of user text.
        Returns dict with keys: sentiment [0,1], stress [0,1], urgency [0,1].
        No external library required. Override with VADER or distilbert for production.
        """
        if not text:
            return {"sentiment": 0.5, "stress": 0.0, "urgency": 0.0}

        tokens = set(re.sub(r"[^\w\s]", " ", text.lower()).split())
        n_stress   = len(tokens & _STRESS_WORDS)
        n_frustrate = len(tokens & _FRUSTRATION_WORDS)
        n_positive  = len(tokens & _POSITIVE_WORDS)
        n           = max(1, len(tokens))

        raw = (n_positive - n_stress - n_frustrate) / n
        sentiment = max(0.0, min(1.0, 0.5 + raw * 2.0))
        stress    = min(1.0, (n_stress + n_frustrate * 0.5) / n * 10.0)
        urgency   = min(1.0, n_stress / n * 12.0)

        alpha = 0.3
        with self._lock:
            self._emotional.sentiment   = (1 - alpha) * self._emotional.sentiment + alpha * sentiment
            self._emotional.stress      = (1 - alpha) * self._emotional.stress    + alpha * stress
            self._emotional.urgency     = max(self._emotional.urgency, urgency)
            if n_frustrate > 0:
                self._emotional.frustration = min(1.0, self._emotional.frustration + 0.08 * n_frustrate)
            if n_positive > 0:
                self._emotional.frustration = max(0.0, self._emotional.frustration - 0.05 * n_positive)
            self._emotional.last_updated = time.time()

        return {"sentiment": round(sentiment, 3), "stress": round(stress, 3), "urgency": round(urgency, 3)}

    # ── Decayed Emotional Properties ──────────────────────────────────────

    @property
    def urgency(self) -> float:
        with self._lock:
            elapsed_min = (time.time() - self._emotional.last_updated) / 60.0
            return max(0.0, self._emotional.urgency - 0.05 * elapsed_min)

    @property
    def frustration(self) -> float:
        with self._lock:
            elapsed_min = (time.time() - self._emotional.last_updated) / 60.0
            return max(0.0, self._emotional.frustration - 0.02 * elapsed_min)

    @property
    def stress(self) -> float:
        with self._lock:
            elapsed_min = (time.time() - self._emotional.last_updated) / 60.0
            return max(0.0, self._emotional.stress - 0.03 * elapsed_min)

    @property
    def engagement(self) -> float:
        with self._lock:
            elapsed_min = (time.time() - self._emotional.last_updated) / 60.0
            return max(0.2, self._emotional.engagement - 0.05 * elapsed_min)

    @property
    def sentiment(self) -> float:
        with self._lock:
            return self._emotional.sentiment

    def get_emotional_summary(self) -> Dict[str, float]:
        return {
            "urgency":     round(self.urgency, 2),
            "frustration": round(self.frustration, 2),
            "stress":      round(self.stress, 2),
            "engagement":  round(self.engagement, 2),
            "sentiment":   round(self.sentiment, 2),
        }

    # ── Task Events ───────────────────────────────────────────────────────

    def on_task_complete(self, success: bool, duration_s: float,
                         app: str = "", objective: str = "") -> None:
        with self._lock:
            self._task_hist.append({
                "success": success, "duration_s": round(duration_s, 1),
                "app": app, "ts": time.time(),
            })
            if len(self._task_hist) > _MAX_HISTORY:
                self._task_hist = self._task_hist[-_MAX_HISTORY:]
            if success:
                self._update_pref("autonomy_preference", +0.01)
                self._emotional.frustration = max(0.0, self._emotional.frustration - 0.3)
                self._emotional.urgency     = max(0.0, self._emotional.urgency - 0.5)
                self._emotional.stress      = max(0.0, self._emotional.stress - 0.2)
                self._emotional.sentiment   = min(1.0, self._emotional.sentiment + 0.1)
            else:
                self._update_pref("autonomy_preference", -0.02)
                self._emotional.frustration = min(1.0, self._emotional.frustration + 0.05)
                self._emotional.stress      = min(1.0, self._emotional.stress + 0.03)
            if objective:
                for word in objective.lower().split()[:10]:
                    if len(word) > 3:
                        self._goal_vocab[word] = self._goal_vocab.get(word, 0) + 1
            self._emotional.last_updated = time.time()
        self._save_async()

    def on_dwell(self, app: str, dwell_sec: float) -> None:
        with self._lock:
            exp = self._get_or_create_app(app)
            if exp.avg_dwell_sec == 0.0:
                exp.avg_dwell_sec = dwell_sec
            else:
                exp.avg_dwell_sec = 0.9 * exp.avg_dwell_sec + 0.1 * dwell_sec

    # ── Queries ───────────────────────────────────────────────────────────

    def should_auto_approve(self, app: str, operation: str) -> bool:
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
                "verbosity": "detailed" if self._prefs.explanation_verbosity > 0.6 else "brief",
                "autonomy":  "supervised" if self._prefs.autonomy_preference < 0.4 else "autonomous",
                "speed":     "careful" if self._prefs.speed_preference < 0.4 else "fast",
                "tolerance": round(self._prefs.interruption_tolerance, 2),
                "emotional": self.get_emotional_summary(),
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
                "apps_tracked":   len(self._apps),
                "tasks_recorded": len(self._task_hist),
                "preferences":    self._prefs.to_dict(),
                "top_goals":      self.get_common_goals(),
                "emotional":      self._emotional.to_dict(),
            }

    # ── Internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _std(values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5

    def _get_or_create_app(self, app: str) -> AppExpertise:
        key = app.lower()
        if key not in self._apps:
            self._apps[key] = AppExpertise(app_name=key)
        return self._apps[key]

    def _update_pref(self, name: str, delta: float) -> None:
        current = getattr(self._prefs, name)
        setattr(self._prefs, name, max(0.0, min(1.0, current + delta)))
        self._prefs.last_updated = time.time()

    # ── Persistence ───────────────────────────────────────────────────────

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
            if "emotional_state" in data:
                self._emotional = EmotionalState.from_dict(data["emotional_state"])
        except Exception as e:
            _logger.warning("[UserModel] Load failed: %s", e)

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        tmp = self._path + ".tmp"
        try:
            with self._lock:
                data = {
                    "preferences":   self._prefs.to_dict(),
                    "apps":          {k: v.to_dict() for k, v in self._apps.items()},
                    "task_history":  self._task_hist[-_MAX_HISTORY:],
                    "goal_vocab":    self._goal_vocab,
                    "emotional_state": self._emotional.to_dict(),
                }
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, separators=(",", ":"))
            os.replace(tmp, self._path)
        except Exception as e:
            _logger.debug("[UserModel] Save failed: %s", e)

    def _save_async(self) -> None:
        threading.Thread(target=self._save, daemon=True).start()


_instance: Optional[UserModel] = None
_instance_lock = threading.Lock()


def get_user_model() -> UserModel:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = UserModel()
    return _instance
