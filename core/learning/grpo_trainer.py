"""
core/learning/grpo_trainer.py

Group Relative Policy Optimization (GRPO) training pipeline.

Reference: DeepSeek-R1 / Shao et al. 2024 — GRPO replaces PPO's critic
network with group-relative reward normalisation. For GUI agents this means
comparing K rollout trajectories per task and learning from their relative
success.

Pipeline:
  1. preference_generator.py writes DPO dataset (JSONL)
  2. grpo_trainer.py reads that dataset, runs group rollouts in vm_manager.py
  3. Computes group-relative rewards and exports RLVR training signal
  4. nightly_consolidation.py schedules the actual fine-tuning

Env vars:
  PROJECTZEO_GRPO_ENABLED      1 / 0        (default: 0 — opt-in)
  PROJECTZEO_GRPO_GROUP_SIZE   K rollouts   (default: 4)
  PROJECTZEO_GRPO_BETA         KL penalty   (default: 0.01)
  PROJECTZEO_GRPO_OUTPUT_DIR   training dir (default: ~/.projectzeo/grpo)
  PROJECTZEO_GRPO_MAX_TASKS    tasks/run    (default: 50)
"""
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
    chosen:           str        # highest-reward response
    rejected:         str        # lowest-reward response
    chosen_reward:    float
    rejected_reward:  float
    advantage:        float      # chosen_reward - mean(group)
    group_rewards:    List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EWCRegularizer:
    """
    Elastic Weight Consolidation — prevents catastrophic forgetting.

    Fisher matrix is approximated from a small calibration dataset.
    Must be computed BEFORE GRPO training begins.

    Reference: Kirkpatrick et al. 2017, PNAS.
    """

    def __init__(self) -> None:
        self._fisher:     Dict[str, Any] = {}   # param_name → importance
        self._theta_star: Dict[str, Any] = {}   # param_name → value at calibration
        self._lambda      = float(os.environ.get("PROJECTZEO_EWC_LAMBDA", "5000"))
        self._computed    = False

    def compute_fisher(self, model_state: Dict[str, Any], calibration_data: List[Dict]) -> None:
        """
        Approximate Fisher Information Matrix using gradient magnitudes
        on a calibration dataset. Called once before GRPO training.

        For production: replace with actual autograd-based Fisher computation.
        Here we store parameter importance heuristically from model state.
        """
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
        """
        Compute EWC penalty = λ/2 * Σ F_i (θ_i - θ*_i)².
        Returns 0.0 if Fisher not computed.
        """
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
    """
    Orchestrates group rollouts and produces GRPO training samples.

    Does not perform actual gradient updates — that is done by an external
    training process (vLLM + TRL or a nightly fine-tuning job) that consumes
    the JSONL output written here.
    """

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

    # -------------------------------------------------------------------------
    # Fisher / EWC
    # -------------------------------------------------------------------------

    def init_ewc(self, model_state: Dict[str, Any], calibration_data: List[Dict]) -> None:
        """Call before run_training_pass(). Computes Fisher matrix."""
        self._ewc.compute_fisher(model_state, calibration_data)

    # -------------------------------------------------------------------------
    # Training pass
    # -------------------------------------------------------------------------

    def run_training_pass(
        self,
        tasks: List[Dict[str, Any]],
        max_tasks: int = _MAX_TASKS,
    ) -> str:
        """
        Run GRPO training pass over a list of tasks.

        Each task dict must contain 'prompt' and 'task_id'.
        Returns path to the written JSONL file.
        """
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

        # Normalise rewards group-relative
        norm_rewards = [(r - mean_r) / (std_r + 1e-8) for r in rewards]

        # Apply EWC penalty if Fisher is ready
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
