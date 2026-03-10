from __future__ import annotations

import json
import logging
import os
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

_ENABLED     = os.environ.get("PROJECTZEO_GRPO_ENABLED", "0").strip() == "1"
_GROUP_SIZE  = int(os.environ.get("PROJECTZEO_GRPO_GROUP_SIZE", "4"))
_BETA        = float(os.environ.get("PROJECTZEO_GRPO_BETA", "0.01"))
_OUTPUT_DIR  = os.path.expanduser(os.environ.get("PROJECTZEO_GRPO_OUTPUT_DIR", "~/.projectzeo/grpo"))
_MAX_TASKS   = int(os.environ.get("PROJECTZEO_GRPO_MAX_TASKS", "50"))

@dataclass
class RolloutResult:
    task_id:     str
    prompt:      str
    response:    str
    reward:      float
    success:     bool
    duration_s:  float

@dataclass
class GRPOSample:
    task_id:          str
    prompt:           str
    chosen:           str
    rejected:         str
    chosen_reward:    float
    rejected_reward:  float
    advantage:        float
    group_rewards:    List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class EWCRegularizer:

    def __init__(self) -> None:
        self._fisher:     Dict[str, Any] = {}
        self._theta_star: Dict[str, Any] = {}
        self._lambda      = float(os.environ.get("PROJECTZEO_EWC_LAMBDA", "5000"))
        self._computed    = False

    def compute_fisher(self, model_state: Dict[str, Any], calibration_data: List[Dict]) -> None:
        if not model_state or not calibration_data:
            _logger.debug("[EWC] No model state or calibration data — Fisher not computed.")
            return

        for name, value in model_state.items():
            if isinstance(value, (int, float)):
                self._fisher[name] = abs(float(value)) + 1e-8
                self._theta_star[name] = float(value)

        self._computed = True
        _logger.info("[EWC] Fisher matrix computed. %d parameters tracked.", len(self._fisher))

    def penalty(self, current_state: Dict[str, Any]) -> float:
        if not self._computed:
            return 0.0
        penalty = 0.0
        for name, f_val in self._fisher.items():
            curr = float(current_state.get(name, self._theta_star.get(name, 0.0)))
            star = self._theta_star.get(name, curr)
            penalty += f_val * ((curr - star) ** 2)
        return (self._lambda / 2.0) * penalty

    @property
    def ready(self) -> bool:
        return self._computed

class GRPOTrainer:

    def __init__(
        self,
        rollout_fn: Optional[Callable[[str, str], RolloutResult]] = None,
        output_dir: Optional[str] = None,
    ) -> None:
        self._dir      = output_dir or _OUTPUT_DIR
        self._lock     = threading.Lock()
        self._rollout  = rollout_fn or self._null_rollout
        self._ewc      = EWCRegularizer()
        self._samples_written = 0
        os.makedirs(self._dir, exist_ok=True)
        _logger.info("[GRPO] Trainer ready. enabled=%s group=%d beta=%.3f", _ENABLED, _GROUP_SIZE, _BETA)

    def init_ewc(self, model_state: Dict[str, Any], calibration_data: List[Dict]) -> None:
        self._ewc.compute_fisher(model_state, calibration_data)

    def run_training_pass(
        self,
        tasks: List[Dict[str, Any]],
        max_tasks: int = _MAX_TASKS,
    ) -> str:
        if not _ENABLED:
            _logger.info("[GRPO] Disabled — skipping training pass.")
            return ""

        tasks = tasks[:max_tasks]
        samples: List[GRPOSample] = []

        for task in tasks:
            sample = self._process_task(task)
            if sample is not None:
                samples.append(sample)

        if not samples:
            return ""

        path = self._write_samples(samples)
        _logger.info("[GRPO] Training pass complete. %d samples → %s", len(samples), path)
        return path

    def _process_task(self, task: Dict[str, Any]) -> Optional[GRPOSample]:
        prompt  = str(task.get("prompt", ""))
        task_id = str(task.get("task_id", f"t_{int(time.time())}"))

        if not prompt:
            return None

        rollouts: List[RolloutResult] = []
        for k in range(_GROUP_SIZE):
            t0     = time.monotonic()
            result = self._rollout(task_id, prompt)
            result.duration_s = time.monotonic() - t0
            rollouts.append(result)

        if len(rollouts) < 2:
            return None

        rewards = [r.reward for r in rollouts]
        mean_r  = statistics.mean(rewards)
        std_r   = statistics.stdev(rewards) if len(rewards) > 1 else 1.0

        norm_rewards = [(r - mean_r) / (std_r + 1e-8) for r in rewards]

        if self._ewc.ready:
            ewc_pen = self._ewc.penalty({})
            norm_rewards = [nr - ewc_pen * _BETA for nr in norm_rewards]

        best_idx  = max(range(len(rollouts)), key=lambda i: norm_rewards[i])
        worst_idx = min(range(len(rollouts)), key=lambda i: norm_rewards[i])

        if best_idx == worst_idx:
            return None

        advantage = norm_rewards[best_idx] - mean_r

        return GRPOSample(
            task_id=task_id,
            prompt=prompt,
            chosen=rollouts[best_idx].response,
            rejected=rollouts[worst_idx].response,
            chosen_reward=rollouts[best_idx].reward,
            rejected_reward=rollouts[worst_idx].reward,
            advantage=advantage,
            group_rewards=rewards,
        )

    def _write_samples(self, samples: List[GRPOSample]) -> str:
        ts   = int(time.time())
        path = os.path.join(self._dir, f"grpo_{ts}.jsonl")
        tmp  = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s.to_dict(), default=str) + "\n")
        os.replace(tmp, path)
        with self._lock:
            self._samples_written += len(samples)
        return path

    @staticmethod
    def _null_rollout(task_id: str, prompt: str) -> RolloutResult:
        return RolloutResult(
            task_id=task_id,
            prompt=prompt,
            response="",
            reward=0.0,
            success=False,
            duration_s=0.0,
        )

    def run_ewc_synthetic_replay(
        self,
        llm_callable,
        *,
        n_samples: int = 10,
        semantic_memory=None,
        trajectory_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if llm_callable is None:
            _logger.warning("[EWC-Replay] No LLM callable — synthetic replay skipped.")
            return []

        prior_tasks: List[str] = []

        if semantic_memory is not None:
            try:
                facts = semantic_memory.query("completed task", max_results=n_samples * 2)
                for f in facts:
                    obj = getattr(f, "object_", None) or (f.get("object") if isinstance(f, dict) else None)
                    if obj and len(str(obj)) > 10:
                        prior_tasks.append(str(obj)[:300])
            except Exception as exc:
                _logger.debug("[EWC-Replay] Semantic memory query failed: %s", exc)

        tdir = trajectory_dir or _OUTPUT_DIR
        try:
            for fname in os.listdir(tdir):
                if fname.endswith(".json") and len(prior_tasks) < n_samples * 2:
                    try:
                        with open(os.path.join(tdir, fname)) as f:
                            td = json.load(f)
                        obj = td.get("objective", "")
                        if obj and obj not in prior_tasks:
                            prior_tasks.append(str(obj)[:300])
                    except Exception:
                        pass
        except Exception:
            pass

        if not prior_tasks:
            _logger.info("[EWC-Replay] No prior tasks found — no synthetic samples generated.")
            return []

        prior_tasks = prior_tasks[:n_samples]
        synthetic_samples: List[Dict[str, Any]] = []

        _logger.info(
            "[EWC-Replay] Generating %d synthetic replay samples from %d prior tasks.",
            len(prior_tasks), len(prior_tasks),
        )

        for task_desc in prior_tasks:
            prompt = (
                "You are generating a synthetic training example for continual learning.\n\n"
                f"Prior task: {task_desc}\n\n"
                "Generate a realistic, successful action sequence for this task.\n"
                "Respond ONLY with JSON: "
                '{"objective": "...", "actions": [{"operation": "...", "thought": "...", '
                '"success": true}], "outcome": "success", "lesson": "..."}'
            )
            result_holder: List[Optional[str]] = [None]

            def _call(_p=prompt, _rh=result_holder):
                try:
                    raw = llm_callable(
                        messages=[{"role": "user", "content": _p}],
                        objective="ewc_replay",
                        session_id="ewc_replay_synthesis",
                    )
                    if isinstance(raw, list) and raw:
                        _rh[0] = str(raw[0].get("content", "") if isinstance(raw[0], dict) else raw[0])
                    elif isinstance(raw, str):
                        _rh[0] = raw
                except Exception as exc:
                    _logger.debug("[EWC-Replay] LLM call failed: %s", exc)

            t = threading.Thread(target=_call, daemon=True)
            t.start()
            t.join(timeout=20.0)

            if result_holder[0]:
                try:
                    import re as _re
                    clean = _re.sub(r"```(?:json)?", "", result_holder[0]).strip()
                    m = _re.search(r"\{.*\}", clean, _re.DOTALL)
                    if m:
                        sample = json.loads(m.group(0))
                        if isinstance(sample, dict) and sample.get("objective"):
                            sample["_synthetic"] = True
                            sample["_ewc_replay"] = True
                            synthetic_samples.append(sample)
                except Exception:
                    pass

        if synthetic_samples:
            ts = int(time.time())
            replay_path = os.path.join(tdir, f"ewc_replay_{ts}.jsonl")
            try:
                with open(replay_path, "w") as f:
                    for s in synthetic_samples:
                        f.write(json.dumps(s) + "\n")
                _logger.info(
                    "[EWC-Replay] Wrote %d synthetic samples → %s",
                    len(synthetic_samples), replay_path,
                )
            except OSError as exc:
                _logger.warning("[EWC-Replay] Failed to write replay file: %s", exc)

        return synthetic_samples

    def latest_dataset(self) -> Optional[str]:
        try:
            files = sorted(
                (e for e in os.scandir(self._dir) if e.name.startswith("grpo_") and e.name.endswith(".jsonl")),
                key=lambda e: e.stat().st_mtime,
                reverse=True,
            )
            return files[0].path if files else None
        except Exception:
            return None

    def get_stats(self) -> Dict[str, Any]:
        return {
            "enabled":          _ENABLED,
            "group_size":       _GROUP_SIZE,
            "beta":             _BETA,
            "ewc_ready":        self._ewc.ready,
            "samples_written":  self._samples_written,
            "output_dir":       self._dir,
        }

_instance: Optional[GRPOTrainer] = None
_lock = threading.Lock()

def get_grpo_trainer(rollout_fn: Optional[Callable] = None) -> GRPOTrainer:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = GRPOTrainer(rollout_fn=rollout_fn)
    return _instance
