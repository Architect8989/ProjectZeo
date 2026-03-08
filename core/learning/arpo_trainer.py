from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional

_logger = logging.getLogger(__name__)

_TRAINING_DATA_DIR = os.path.expanduser(
    os.environ.get("PROJECTZEO_TRAINING_DIR", "~/.projectzeo/training")
)
_ARPO_ENABLED = os.environ.get("PROJECTZEO_ARPO_ENABLED", "0").strip() == "1"
_UI_EVOL_ENABLED = os.environ.get("PROJECTZEO_UI_EVOL_ENABLED", "1").strip() != "0"


# ─────────────────────────────────────────────────────────────────────────────
# ARPO Trajectory Recorder
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryRecord:
    """A single task trajectory for ARPO training."""

    def __init__(
        self,
        objective: str,
        app_name: str,
        temperature: float = 0.7,
    ) -> None:
        self.objective = objective
        self.app_name = app_name
        self.temperature = temperature
        self.steps: List[Dict[str, Any]] = []
        self.success: Optional[bool] = None
        self.reward: float = 0.0
        self.ts_start = time.time()
        self.ts_end: Optional[float] = None

    def add_step(
        self,
        action: Dict[str, Any],
        world_state: Dict[str, Any],
        outcome: bool,
        screenshot_b64: Optional[str] = None,
    ) -> None:
        """Record a single action step."""
        self.steps.append({
            "action": {k: str(v)[:200] for k, v in action.items()},
            "outcome": outcome,
            "has_screenshot": screenshot_b64 is not None,
            "screenshot_b64": screenshot_b64 if screenshot_b64 else None,
            "entity_count": len(world_state.get("entities", [])),
            "focused_app": world_state.get("focused_app", ""),
        })

    def finalize(self, success: bool, reason: str) -> None:
        """Mark trajectory as complete and compute ARPO reward."""
        self.success = success
        self.ts_end = time.time()
        # ARPO reward: binary task success + format penalty
        format_ok = all(
            s.get("action", {}).get("operation") for s in self.steps
        )
        self.reward = 1.0 if success else 0.0
        if not format_ok:
            self.reward -= 0.5
        self.reward = max(0.0, self.reward)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective": self.objective,
            "app_name": self.app_name,
            "temperature": self.temperature,
            "success": self.success,
            "reward": self.reward,
            "step_count": len(self.steps),
            "duration_sec": (self.ts_end or time.time()) - self.ts_start,
            "ts_start": self.ts_start,
            "steps": self.steps[:100],  # Cap for storage
        }


class ARPOTrainer:
    

    def __init__(
        self,
        memory_dir: Optional[str] = None,
        min_reward_threshold: float = 0.5,
    ) -> None:
        self._dir = memory_dir or _TRAINING_DATA_DIR
        os.makedirs(self._dir, exist_ok=True)
        self._min_reward = min_reward_threshold
        self._current: Optional[TrajectoryRecord] = None
        self._lock = threading.Lock()
        self._collected = 0

    def start_trajectory(self, objective: str, app_name: str, temperature: float = 0.7) -> None:
        """Begin recording a new trajectory."""
        with self._lock:
            self._current = TrajectoryRecord(objective, app_name, temperature)
        _logger.debug("[ARPO] Trajectory started: %s", objective[:60])

    def record_step(
        self,
        action: Dict[str, Any],
        world_state: Dict[str, Any],
        outcome: bool,
    ) -> None:
        """Record one step in the current trajectory."""
        with self._lock:
            if self._current is not None:
                self._current.add_step(action, world_state, outcome)

    def finalize_trajectory(self, success: bool, reason: str) -> None:
        """Finalize and persist the trajectory if it meets the reward threshold."""
        with self._lock:
            traj = self._current
            self._current = None

        if traj is None:
            return

        traj.finalize(success, reason)

        if not _ARPO_ENABLED:
            _logger.debug("[ARPO] ARPO disabled — trajectory not saved.")
            return

        if traj.reward < self._min_reward:
            _logger.debug(
                "[ARPO] Trajectory reward %.2f < threshold %.2f — skipping.",
                traj.reward, self._min_reward,
            )
            return

        self._save_trajectory(traj)

    def _save_trajectory(self, traj: TrajectoryRecord) -> None:
        """Persist trajectory to JSONL store."""
        try:
            filename = os.path.join(
                self._dir,
                f"traj_{int(time.time())}_{traj.app_name[:20]}.json",
            )
            tmp = filename + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(traj.to_dict(), f, default=str)
            os.replace(tmp, filename)
            with self._lock:
                self._collected += 1
            _logger.info(
                "[ARPO] Trajectory saved (reward=%.2f steps=%d): %s",
                traj.reward, len(traj.steps), filename,
            )
        except Exception as e:
            _logger.warning("[ARPO] Save error: %s", e)

    def get_stats(self) -> Dict[str, Any]:
        try:
            files = [f for f in os.listdir(self._dir) if f.endswith(".json")]
        except Exception:
            files = []
        return {
            "arpo_enabled": _ARPO_ENABLED,
            "collected_this_session": self._collected,
            "total_on_disk": len(files),
            "training_dir": self._dir,
        }


# ─────────────────────────────────────────────────────────────────────────────
# UI-Evol Post-Task Knowledge Refinement
# ─────────────────────────────────────────────────────────────────────────────

class UIEvol:
    

    def __init__(
        self,
        graphiti_store=None,
        llm_callable: Optional[Callable] = None,
    ) -> None:
        self._graphiti = graphiti_store
        self._llm = llm_callable

    def run_post_task(
        self,
        *,
        objective: str,
        app_name: str,
        planned_actions: List[Dict[str, Any]],
        actual_actions: List[Dict[str, Any]],
        success: bool,
        screenshot_pairs: Optional[List] = None,
    ) -> None:
        """
        Run UI-Evol analysis after task completion.
        Called by GIIController.on_task_complete().

        Args:
            objective: original task objective
            app_name: target application
            planned_actions: what the planner intended
            actual_actions: what was actually dispatched
            success: final task outcome
            screenshot_pairs: consecutive (before, after) PIL image pairs for diff analysis
        """
        if not _UI_EVOL_ENABLED:
            return

        def _run():
            try:
                self._analyze_and_update(
                    objective=objective,
                    app_name=app_name,
                    planned_actions=planned_actions,
                    actual_actions=actual_actions,
                    success=success,
                    screenshot_pairs=screenshot_pairs or [],
                )
            except Exception as e:
                _logger.warning("[UI-Evol] Analysis error: %s", e)

        t = threading.Thread(target=_run, name="ui_evol_analysis", daemon=True)
        t.start()

    def _analyze_and_update(
        self,
        *,
        objective: str,
        app_name: str,
        planned_actions: List[Dict[str, Any]],
        actual_actions: List[Dict[str, Any]],
        success: bool,
        screenshot_pairs: List,
    ) -> None:
        """Core UI-Evol analysis — runs in background thread."""
        # Compute plan-vs-actual divergence
        divergence_points: List[str] = []
        for i, (planned, actual) in enumerate(zip(planned_actions, actual_actions)):
            if planned.get("operation") != actual.get("operation"):
                divergence_points.append(
                    f"Step {i}: planned {planned.get('operation')} → "
                    f"actual {actual.get('operation')}"
                )

        # Screenshot diff analysis (if pairs available)
        visual_changes: List[str] = []
        for i, pair in enumerate(screenshot_pairs[:10]):
            try:
                before, after = pair
                change = self._compute_visual_change_description(before, after, i)
                if change:
                    visual_changes.append(change)
            except Exception:
                pass

        # LLM critique and knowledge update
        if self._llm is not None and (divergence_points or visual_changes):
            self._llm_critique_and_update(
                objective=objective,
                app_name=app_name,
                divergence_points=divergence_points,
                visual_changes=visual_changes,
                success=success,
            )

        # Update Graphiti/application memory with observations
        if self._graphiti is not None:
            from core.memory.graphiti_store import ApplicationEntity  # noqa
            entity = ApplicationEntity(app_name=app_name)
            if not success and divergence_points:
                entity.observed_failure_patterns.extend(divergence_points[:3])
            self._graphiti.store_application_entity(entity)
            _logger.debug("[UI-Evol] Application entity updated: %s", app_name)

    def _compute_visual_change_description(self, before, after, step_idx: int) -> str:
        """Describe what visually changed between two screenshots."""
        try:
            import imagehash  # type: ignore
            h1 = imagehash.phash(before.resize((64, 64)))
            h2 = imagehash.phash(after.resize((64, 64)))
            dist = h1 - h2
            if dist > 5:
                return f"Step {step_idx}: visual change (phash distance={dist})"
        except Exception:
            pass
        return ""

    def _llm_critique_and_update(
        self,
        *,
        objective: str,
        app_name: str,
        divergence_points: List[str],
        visual_changes: List[str],
        success: bool,
    ) -> None:
        """Use LLM to critique plan-vs-actual and extract application knowledge."""
        prompt = (
            f"Task: {objective[:300]}\n"
            f"Application: {app_name}\n"
            f"Outcome: {'SUCCESS' if success else 'FAILURE'}\n\n"
            f"Plan-vs-Actual divergences:\n" +
            "\n".join(divergence_points[:10]) + "\n\n"
            f"Visual changes observed:\n" +
            "\n".join(visual_changes[:10]) + "\n\n"
            "Based on this analysis, what did we learn about how this application works?\n"
            "List 1-3 specific insights that would help on the next similar task.\n"
            "Format: JSON array of insight strings."
        )
        result_holder: list = [None]
        def _call():
            try:
                raw = self._llm(
                    messages=[{"role": "user", "content": prompt}],
                    objective=None,
                    session_id="ui_evol_critique",
                )
                result_holder[0] = raw
            except Exception as e:
                _logger.debug("[UI-Evol] LLM call error: %s", e)

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        t.join(timeout=60.0)

        if result_holder[0] and self._graphiti is not None:
            try:
                raw = result_holder[0]
                if isinstance(raw, list) and raw:
                    raw = str(raw[0].get("content", "") if isinstance(raw[0], dict) else raw[0])
                import re
                clean = re.sub(r"```(?:json)?", "", str(raw)).strip()
                insights = json.loads(clean)
                if isinstance(insights, list):
                    for insight in insights[:3]:
                        self._graphiti.store_fact(
                            subject=f"app:{app_name}",
                            predicate="ui_evol_insight",
                            obj=str(insight)[:200],
                            confidence=0.7,
                        )
                    _logger.info("[UI-Evol] Stored %d insights for %s.", len(insights[:3]), app_name)
            except Exception as e:
                _logger.debug("[UI-Evol] Insight parse error: %s", e)
      
