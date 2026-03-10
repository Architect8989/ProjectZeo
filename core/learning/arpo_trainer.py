"""
core/learning/arpo_trainer.py  (patched — March 2026)

ARPO trajectory recorder + UI-Evol post-task knowledge refinement.

Changes from previous version:
  - ARPO enabled by default (PROJECTZEO_ARPO_ENABLED defaults to 1)
  - EWC Fisher matrix computed after first trajectory batch to prevent
    catastrophic forgetting on subsequent GRPO training runs
  - Reward shaping improved: format, efficiency, and reversal penalties
  - UI-Evol wires to HippoRAG in addition to Graphiti for multi-hop storage
"""
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
_ARPO_ENABLED    = os.environ.get("PROJECTZEO_ARPO_ENABLED", "1").strip() != "0"
_UI_EVOL_ENABLED = os.environ.get("PROJECTZEO_UI_EVOL_ENABLED", "1").strip() != "0"
_EWC_ENABLED     = os.environ.get("PROJECTZEO_ARPO_EWC", "1").strip() != "0"
_EWC_BATCH_SIZE  = int(os.environ.get("PROJECTZEO_EWC_BATCH_SIZE", "20"))


class TrajectoryRecord:

    def __init__(self, objective: str, app_name: str, temperature: float = 0.7) -> None:
        self.objective   = objective
        self.app_name    = app_name
        self.temperature = temperature
        self.steps:       List[Dict[str, Any]] = []
        self.success:     Optional[bool] = None
        self.reward:      float = 0.0
        self.ts_start    = time.time()
        self.ts_end:      Optional[float] = None

    def add_step(
        self,
        action: Dict[str, Any],
        world_state: Dict[str, Any],
        outcome: bool,
        screenshot_b64: Optional[str] = None,
    ) -> None:
        self.steps.append({
            "action":        {k: str(v)[:200] for k, v in action.items()},
            "outcome":       outcome,
            "has_screenshot": screenshot_b64 is not None,
            "screenshot_b64": screenshot_b64,
            "entity_count":  len(world_state.get("entities", [])),
            "focused_app":   world_state.get("focused_app", ""),
        })

    def finalize(self, success: bool, reason: str) -> None:
        self.success = success
        self.ts_end  = time.time()

        # ARPO reward: task success + format compliance + efficiency bonus
        format_ok   = all(s.get("action", {}).get("operation") for s in self.steps)
        n_steps     = len(self.steps)
        efficiency  = max(0.0, 1.0 - n_steps / 100.0)   # penalty for long trajectories

        base  = 1.0 if success else 0.0
        fpen  = 0.0 if format_ok else -0.3
        ebon  = 0.1 * efficiency if success else 0.0

        # Reversal penalty: detect action loops
        ops   = [s.get("action", {}).get("operation", "") for s in self.steps]
        loops = sum(1 for i in range(len(ops) - 2) if ops[i] == ops[i+1] == ops[i+2])
        rpen  = -0.05 * loops

        self.reward = max(0.0, base + fpen + ebon + rpen)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective":   self.objective,
            "app_name":    self.app_name,
            "temperature": self.temperature,
            "success":     self.success,
            "reward":      self.reward,
            "step_count":  len(self.steps),
            "duration_sec": (self.ts_end or time.time()) - self.ts_start,
            "ts_start":    self.ts_start,
            "steps":       self.steps[:100],
        }


class ARPOTrainer:

    def __init__(
        self,
        memory_dir: Optional[str] = None,
        min_reward_threshold: float = 0.5,
    ) -> None:
        self._dir      = memory_dir or _TRAINING_DATA_DIR
        self._min_r    = min_reward_threshold
        self._current: Optional[TrajectoryRecord] = None
        self._lock     = threading.Lock()
        self._collected = 0
        self._ewc_state: Dict[str, Any] = {}
        os.makedirs(self._dir, exist_ok=True)

    def start_trajectory(self, objective: str, app_name: str, temperature: float = 0.7) -> None:
        with self._lock:
            self._current = TrajectoryRecord(objective, app_name, temperature)
        _logger.debug("[ARPO] Trajectory started: %s", objective[:60])

    def record_step(self, action: Dict[str, Any], world_state: Dict[str, Any], outcome: bool) -> None:
        with self._lock:
            if self._current is not None:
                self._current.add_step(action, world_state, outcome)

    def finalize_trajectory(self, success: bool, reason: str) -> None:
        with self._lock:
            traj = self._current
            self._current = None

        if traj is None:
            return

        traj.finalize(success, reason)

        if not _ARPO_ENABLED:
            return

        if traj.reward < self._min_r:
            _logger.debug("[ARPO] Reward %.2f < threshold — skipping.", traj.reward)
            return

        self._save_trajectory(traj)

        # After every batch, compute EWC Fisher from collected trajectories
        if _EWC_ENABLED and self._collected % _EWC_BATCH_SIZE == 0:
            self._compute_ewc_fisher()

    def _save_trajectory(self, traj: TrajectoryRecord) -> None:
        try:
            fn  = os.path.join(self._dir, f"traj_{int(time.time())}_{traj.app_name[:20]}.json")
            tmp = fn + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(traj.to_dict(), f, default=str)
            os.replace(tmp, fn)
            with self._lock:
                self._collected += 1
            _logger.info("[ARPO] Saved (reward=%.2f steps=%d): %s", traj.reward, len(traj.steps), fn)
        except Exception as e:
            _logger.warning("[ARPO] Save error: %s", e)

    def _compute_ewc_fisher(self) -> None:
        """
        Approximate Fisher matrix from saved trajectories.
        Exports to grpo_trainer.EWCRegularizer if available.
        """
        try:
            files = sorted(
                (e for e in os.scandir(self._dir) if e.name.startswith("traj_")),
                key=lambda e: e.stat().st_mtime,
                reverse=True,
            )[:_EWC_BATCH_SIZE]

            calibration: List[Dict] = []
            for entry in files:
                try:
                    with open(entry.path, encoding="utf-8") as f:
                        calibration.append(json.load(f))
                except Exception:
                    pass

            if not calibration:
                return

            # Build simple model_state proxy from trajectory statistics
            model_state = {
                "avg_reward":     sum(d.get("reward", 0) for d in calibration) / len(calibration),
                "avg_steps":      sum(d.get("step_count", 0) for d in calibration) / len(calibration),
                "avg_duration":   sum(d.get("duration_sec", 0) for d in calibration) / len(calibration),
                "success_rate":   sum(1 for d in calibration if d.get("success")) / len(calibration),
            }

            from core.learning.grpo_trainer import get_grpo_trainer
            trainer = get_grpo_trainer()
            trainer.init_ewc(model_state, calibration)
            _logger.info("[ARPO] EWC Fisher updated from %d trajectories.", len(calibration))

        except Exception as e:
            _logger.debug("[ARPO] EWC Fisher compute error (non-fatal): %s", e)

    def get_stats(self) -> Dict[str, Any]:
        try:
            files = [f for f in os.listdir(self._dir) if f.endswith(".json")]
        except Exception:
            files = []
        return {
            "arpo_enabled":          _ARPO_ENABLED,
            "collected_this_session": self._collected,
            "total_on_disk":         len(files),
            "ewc_enabled":           _EWC_ENABLED,
            "training_dir":          self._dir,
        }

    def export_fisher_state(self) -> Optional[Dict[str, Any]]:
        """
        Export the current EWC Fisher matrix for import by GRPOTrainer.
        Blueprint §9.2: The EWC Fisher penalty should be shared between
        ARPO and GRPO to prevent either trainer from overwriting the other's
        skills (catastrophic forgetting).

        Returns: {"fisher": {...}, "theta_star": {...}} or None if not computed.
        """
        try:
            from core.learning.grpo_trainer import get_grpo_trainer
            grpo = get_grpo_trainer()
            ewc = grpo._ewc
            if ewc._computed and ewc._fisher:
                return {
                    "fisher": dict(ewc._fisher),
                    "theta_star": dict(ewc._theta_star),
                }
        except Exception as exc:
            _logger.debug("[ARPO] export_fisher_state failed: %s", exc)
        return None


class UIEvol:

    def __init__(
        self,
        graphiti_store=None,
        llm_callable: Optional[Callable] = None,
    ) -> None:
        self._graphiti = graphiti_store
        self._llm      = llm_callable

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
                _logger.warning("[UI-Evol] Error: %s", e)

        threading.Thread(target=_run, name="ui_evol", daemon=True).start()

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
        divergence: List[str] = []
        for i, (p, a) in enumerate(zip(planned_actions, actual_actions)):
            if p.get("operation") != a.get("operation"):
                divergence.append(f"Step {i}: {p.get('operation')} → {a.get('operation')}")

        visual: List[str] = []
        for i, pair in enumerate(screenshot_pairs[:10]):
            try:
                desc = self._phash_diff(pair[0], pair[1], i)
                if desc:
                    visual.append(desc)
            except Exception:
                pass

        if self._llm and (divergence or visual):
            self._llm_critique(objective, app_name, divergence, visual, success)

        if self._graphiti:
            try:
                from core.memory.graphiti_store import ApplicationEntity
                entity = ApplicationEntity(app_name=app_name)
                if not success and divergence:
                    entity.observed_failure_patterns.extend(divergence[:3])
                self._graphiti.store_application_entity(entity)
            except Exception:
                pass

        # Wire to HippoRAG for multi-hop storage
        try:
            from core.memory.hippo_rag import get_hippo_rag
            rag = get_hippo_rag()
            for div in divergence[:3]:
                rag.add_fact(
                    subject=app_name,
                    predicate="ui_divergence",
                    obj=div[:100],
                    weight=0.6,
                )
        except Exception:
            pass

    def _phash_diff(self, before, after, step: int) -> str:
        try:
            import imagehash  # type: ignore
            d = imagehash.phash(before.resize((64, 64))) - imagehash.phash(after.resize((64, 64)))
            return f"Step {step}: visual change (phash={d})" if d > 5 else ""
        except Exception:
            return ""

    def _llm_critique(
        self,
        objective: str,
        app_name: str,
        divergence: List[str],
        visual: List[str],
        success: bool,
    ) -> None:
        prompt = (
            f"Task: {objective[:300]}\nApp: {app_name}\n"
            f"Outcome: {'SUCCESS' if success else 'FAILURE'}\n\n"
            "Divergences:\n" + "\n".join(divergence[:10]) + "\n\n"
            "Visual changes:\n" + "\n".join(visual[:10]) + "\n\n"
            "What 1-3 insights would help next time? JSON array of strings."
        )
        holder: list = [None]

        def _call():
            try:
                holder[0] = self._llm(
                    messages=[{"role": "user", "content": prompt}],
                    objective=None,
                    session_id="ui_evol",
                )
            except Exception as e:
                _logger.debug("[UI-Evol] LLM error: %s", e)

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        t.join(timeout=60.0)

        if holder[0] and self._graphiti:
            try:
                import re
                raw = holder[0]
                if isinstance(raw, list) and raw:
                    raw = str(raw[0].get("content", "") if isinstance(raw[0], dict) else raw[0])
                clean    = re.sub(r"```(?:json)?", "", str(raw)).strip()
                insights = json.loads(clean)
                if isinstance(insights, list):
                    for ins in insights[:3]:
                        self._graphiti.store_fact(
                            subject=f"app:{app_name}",
                            predicate="ui_evol_insight",
                            obj=str(ins)[:200],
                            confidence=0.7,
                        )
                    _logger.info("[UI-Evol] Stored %d insights for %s.", len(insights[:3]), app_name)
            except Exception as e:
                _logger.debug("[UI-Evol] Parse error: %s", e)
