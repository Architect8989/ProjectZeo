from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

_TRAJECTORY_DIR = os.path.expanduser(
    os.environ.get("PROJECTZEO_TRAJECTORY_DIR", "~/.projectzeo/trajectories")
)
_MAX_CONTEXT_EPISODES    = int(os.environ.get("PROJECTZEO_AD_MAX_EPISODES", "5"))
_MAX_STEPS_PER_EPISODE   = int(os.environ.get("PROJECTZEO_AD_MAX_STEPS", "30"))
_AD_ENABLED              = os.environ.get("PROJECTZEO_AD_ENABLED", "1") == "1"


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrajectoryStep:
    """Single (observation, action, reward) tuple in a trajectory."""
    step_idx:    int
    observation: str             # Screen description / entity list summary
    action:      Dict[str, Any]  # The action taken
    reward:      float           # 1.0=success, 0.0=fail, 0.5=partial
    outcome:     str             # "success" | "failure" | "partial"
    timestamp:   float = field(default_factory=time.time)


@dataclass
class Episode:
    """A complete task execution trajectory."""
    episode_id:    str
    task_type:     str             # Normalized task description
    app_context:   str             # Which app was being used
    steps:         List[TrajectoryStep] = field(default_factory=list)
    final_reward:  float = 0.0     # 0.0-1.0 overall success
    duration_s:    float = 0.0
    created_at:    float = field(default_factory=time.time)
    metadata:      Dict[str, Any] = field(default_factory=dict)

    def add_step(
        self,
        observation: str,
        action: Dict[str, Any],
        reward: float,
        outcome: str = "unknown",
    ) -> None:
        self.steps.append(TrajectoryStep(
            step_idx=len(self.steps),
            observation=observation[:300],
            action={k: str(v)[:100] for k, v in action.items()},
            reward=reward,
            outcome=outcome,
        ))

    def to_context_string(self, max_steps: int = _MAX_STEPS_PER_EPISODE) -> str:
        """Format episode as AD context string."""
        lines = [
            f"=== Episode (reward={self.final_reward:.2f}) ===",
            f"Task: {self.task_type[:100]}",
            f"App: {self.app_context}",
        ]
        for step in self.steps[:max_steps]:
            op = step.action.get("operation", "?")
            detail = (
                step.action.get("command") or step.action.get("text") or
                step.action.get("content") or ""
            )[:60]
            lines.append(
                f"  Step {step.step_idx}: [{step.outcome.upper()}] "
                f"{op}: {detail} (r={step.reward:.1f})"
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory store
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryStore:
    """
    Persistent store for episode trajectories.
    Organizes episodes by task type so improving sequences can be retrieved.
    """

    def __init__(self, base_dir: str = _TRAJECTORY_DIR) -> None:
        self._dir = base_dir
        os.makedirs(self._dir, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: Dict[str, List[Episode]] = {}   # task_type → episodes

    def save_episode(self, episode: Episode) -> None:
        """Persist an episode to disk and update in-memory cache."""
        task_dir = os.path.join(self._dir, self._normalize_key(episode.task_type))
        os.makedirs(task_dir, exist_ok=True)

        fpath = os.path.join(task_dir, f"{episode.episode_id}.json")
        try:
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(asdict(episode), f, indent=2)
        except OSError as exc:
            _logger.warning("[AD] save_episode failed: %s", exc)
            return

        with self._lock:
            key = self._normalize_key(episode.task_type)
            if key not in self._cache:
                self._cache[key] = []
            # Insert in order of reward (improving sequence)
            self._cache[key].append(episode)
            self._cache[key].sort(key=lambda e: e.final_reward)

    def get_improving_sequence(
        self,
        task_type: str,
        max_episodes: int = _MAX_CONTEXT_EPISODES,
    ) -> List[Episode]:
        """
        Return the K most improving episodes for a task type,
        ordered from worst to best (AD requires improving order).
        """
        key = self._normalize_key(task_type)

        with self._lock:
            if key in self._cache:
                episodes = list(self._cache[key])
            else:
                episodes = self._load_from_disk(key)
                self._cache[key] = list(episodes)

        if not episodes:
            return []

        # Sort by reward ascending (worst → best = improving sequence)
        episodes.sort(key=lambda e: (e.final_reward, e.created_at))
        return episodes[-max_episodes:]  # last K are the improving tail

    def _load_from_disk(self, key: str) -> List[Episode]:
        task_dir = os.path.join(self._dir, key)
        if not os.path.isdir(task_dir):
            return []
        episodes = []
        for fname in sorted(os.listdir(task_dir)):
            if not fname.endswith(".json"):
                continue
            try:
                fpath = os.path.join(task_dir, fname)
                with open(fpath, encoding="utf-8") as f:
                    data = json.load(f)
                ep = self._dict_to_episode(data)
                if ep:
                    episodes.append(ep)
            except Exception as exc:
                _logger.debug("[AD] load episode failed: %s", exc)
        return episodes

    @staticmethod
    def _dict_to_episode(data: Dict[str, Any]) -> Optional[Episode]:
        try:
            steps = [
                TrajectoryStep(**s)
                for s in data.get("steps", [])
            ]
            return Episode(
                episode_id=data["episode_id"],
                task_type=data["task_type"],
                app_context=data.get("app_context", ""),
                steps=steps,
                final_reward=float(data.get("final_reward", 0.0)),
                duration_s=float(data.get("duration_s", 0.0)),
                created_at=float(data.get("created_at", 0.0)),
                metadata=data.get("metadata", {}),
            )
        except Exception:
            return None

    @staticmethod
    def _normalize_key(task_type: str) -> str:
        h = hashlib.sha1(task_type.lower().encode()).hexdigest()[:8]
        safe = "".join(c if c.isalnum() else "_" for c in task_type[:40])
        return f"{safe}_{h}"

    def list_task_types(self) -> List[str]:
        """Return all known task type keys from disk."""
        try:
            if os.path.isdir(self._dir):
                return [
                    d for d in os.listdir(self._dir)
                    if os.path.isdir(os.path.join(self._dir, d))
                ]
        except OSError:
            pass
        return list(self._cache.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm Distillation context builder
# ─────────────────────────────────────────────────────────────────────────────

_AD_SYSTEM = """\
You are an Algorithm Distillation (AD) policy (Laskin et al. 2023).

You will receive a history of IMPROVING past episodes — trajectories that
progressively get better at the same type of task. Your job: predict the
NEXT best action by continuing the improvement trend.

The final episode shows the best performance so far. Your prediction should
be BETTER than the final episode — apply lessons from the improving history.

RULES:
  - Learn from what worked in high-reward episodes
  - Avoid patterns that led to failures in low-reward episodes
  - Your action should continue the improvement trend
  - Consider the current world state carefully

Return ONLY a JSON object (one action):
{
  "operation": "<click|type|hotkey|command|scroll|wait|done>",
  "text": "<optional>",
  "command": "<optional>",
  "keys": ["<optional>"],
  "confidence": <0.0-1.0>,
  "reasoning": "<why this continues the improvement trend>"
}
"""

_AD_USER = """\
TASK TYPE: {task_type}
CURRENT OBJECTIVE: {objective}
APP: {app_context}

IMPROVING EPISODE HISTORY (worst → best):
{episode_context}

CURRENT WORLD STATE:
{world_state}

Based on the improving history above, predict the best next action.
"""


class AlgorithmDistiller:
    """
    Algorithm Distillation engine: formats improving trajectory sequences
    into LLM context and queries for the next best action.

    Zero-shot adaptation: no gradient update required.
    """

    def __init__(
        self,
        llm_call: Callable,
        trajectory_store: Optional[TrajectoryStore] = None,
    ) -> None:
        self._llm = llm_call
        self._store = trajectory_store or TrajectoryStore()

    def predict_action(
        self,
        task_type: str,
        objective: str,
        world_state: Dict[str, Any],
        app_context: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        Zero-shot action prediction via Algorithm Distillation.
        Returns None if insufficient trajectory history.
        """
        if not _AD_ENABLED:
            return None

        episodes = self._store.get_improving_sequence(task_type)
        if len(episodes) < 2:
            # Need at least 2 episodes to show an improving trend
            return None

        episode_context = "\n\n".join(
            ep.to_context_string() for ep in episodes
        )

        world_summary = self._format_world_state(world_state)

        messages = [
            {"role": "system", "content": _AD_SYSTEM},
            {"role": "user", "content": _AD_USER.format(
                task_type=task_type[:100],
                objective=objective[:300],
                app_context=app_context[:100],
                episode_context=episode_context[:6000],
                world_state=world_summary[:600],
            )},
        ]

        try:
            raw = self._llm(messages, objective=objective)
        except Exception as exc:
            _logger.warning("[AD] LLM call failed: %s", exc)
            return None

        return self._parse_action(raw)

    def create_episode(self, task_type: str, app_context: str = "") -> Episode:
        """Create a new episode for recording."""
        import uuid
        return Episode(
            episode_id=str(uuid.uuid4())[:12],
            task_type=task_type,
            app_context=app_context,
        )

    def finalize_episode(self, episode: Episode, success: bool) -> None:
        """Record final reward and persist."""
        episode.final_reward = 1.0 if success else 0.0
        episode.duration_s = time.time() - episode.created_at
        self._store.save_episode(episode)
        _logger.debug(
            "[AD] Episode saved: task=%r reward=%.1f steps=%d",
            episode.task_type[:40], episode.final_reward, len(episode.steps),
        )

    @staticmethod
    def _format_world_state(ws: Dict[str, Any]) -> str:
        entities = ws.get("entities", [])[:5]
        entity_labels = [e.get("label", e.get("text", "?"))[:30] for e in entities]
        return (
            f"App: {ws.get('focused_app', 'unknown')}\n"
            f"Entities: {entity_labels}"
        )

    @staticmethod
    def _parse_action(raw: Any) -> Optional[Dict[str, Any]]:
        if not raw:
            return None
        if isinstance(raw, dict) and "operation" in raw:
            return raw
        if isinstance(raw, str):
            cleaned = __import__("re").sub(r"```(?:json)?", "", raw).strip()
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end > start:
                try:
                    data = json.loads(cleaned[start:end + 1])
                    if isinstance(data, dict) and "operation" in data:
                        return data
                except Exception:
                    pass
        return None

    def get_stats(self) -> Dict[str, Any]:
        return {
            "enabled":          _AD_ENABLED,
            "max_episodes":     _MAX_CONTEXT_EPISODES,
            "trajectory_dir":   _TRAJECTORY_DIR,
        }

    def get_prompt_injection(
        self,
        task_type: str,
        app_context: str = "",
        max_episodes: int = 3,
    ) -> str:
        """
        GII-FIX: Extract cross-task learned patterns as a compact prompt string
        for injection into PerStepReasoner.

        This closes the Algorithm Distillation in-context RL feedback loop
        (Blueprint §9). The output is injected into the per-step decision prompt
        via set_algorithm_distillation_context() so that the current-task
        reasoner benefits from patterns learned across ALL prior tasks.

        Returns empty string if insufficient trajectory history.
        """
        if not _AD_ENABLED:
            return ""

        episodes = self._store.get_improving_sequence(task_type)
        if not episodes:
            # Also try to find relevant episodes from same app
            all_types = self._store.list_task_types() if hasattr(self._store, "list_task_types") else []
            for other_type in all_types:
                if app_context and app_context.lower() in other_type.lower():
                    episodes = self._store.get_improving_sequence(other_type)
                    if episodes:
                        break

        if not episodes:
            return ""

        # Select up to max_episodes best-performing episodes
        best = sorted(episodes, key=lambda e: e.final_reward, reverse=True)[:max_episodes]

        lines = ["[Algorithm Distillation — Learned Patterns]"]
        for ep in best:
            success_steps = [
                s for s in ep.steps if s.outcome == "success"
            ]
            if not success_steps:
                continue
            lines.append(
                f"Task: {ep.task_type[:60]} | App: {ep.app_context} "
                f"| Reward: {ep.final_reward:.1f}"
            )
            for s in success_steps[:3]:
                op = s.action.get("operation", "?")
                detail = (
                    s.action.get("command") or s.action.get("text") or
                    s.action.get("content") or ""
                )[:50]
                lines.append(f"  ✓ {op}: {detail}")

        return "\n".join(lines) if len(lines) > 1 else ""
