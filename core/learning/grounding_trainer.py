"""
core/learning/grounding_trainer.py — UI-AGILE Continuous Grounding Reward Trainer

CREATED: March 2026 — Blueprint gap implementation.
This file was MISSING from the codebase (audit item: "grounding_trainer.py DOES NOT EXIST").

WHAT THIS IS:
  UI-AGILE (Adaptive GUI Instruction Learning with Execution feedback) is the
  continuous grounding reward loop that teaches the vision model to improve its
  GUI element grounding accuracy over time using execution feedback.

BLUEPRINT REFERENCE: §9.5 (Self-Improvement — Grounding Trainer)

ARCHITECTURE:
  1. GroundingStep: One data point — (screenshot, instruction, predicted_coord,
     ground_truth_coord or None, execution_outcome)
  2. GroundingTrainer: Collects steps, computes reward signals, generates
     (instruction, positive_coord, negative_coord) triplets for contrastive
     training. Writes training batches to JSONL for offline fine-tuning or
     for ARPO/GRPO trainers.
  3. Reward Function: Binary execution success + spatial proximity bonus.
     - Click lands within tolerance_px of a successful element: +1.0 reward
     - Click causes no visible change (wrong element): -0.5 reward
     - Task succeeds after click: +2.0 cumulative reward (back-propagated)

INTEGRATION:
  - GIIGoalDirectedLoop calls grounding_trainer.record_step() after each
    click/tap/navigate action with screenshot + predicted coords + outcome
  - NightlyConsolidation calls grounding_trainer.flush_batch() to write
    collected triplets to the training corpus at ~/.projectzeo/grounding/
  - ARPOTrainer reads from the same directory during nightly fine-tuning

GRACEFUL DEGRADATION:
  All heavy dependencies (PIL, torch) are optional-imported.
  Without them, the trainer still records steps to JSONL for offline use.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_GROUNDING_DIR = os.path.join(
    os.path.expanduser("~"), ".projectzeo", "grounding"
)

# Click tolerance: within this many normalised units is a "near miss" rather
# than a "wrong element" (0.05 ≈ 5% of screen width / height)
_CLICK_TOLERANCE_NORM = float(
    os.environ.get("PROJECTZEO_GROUNDING_TOLERANCE", "0.05")
)

# Batch size: write JSONL batch after this many steps
_BATCH_SIZE = int(os.environ.get("PROJECTZEO_GROUNDING_BATCH_SIZE", "50"))

# Reward values
_REWARD_SUCCESS_CLICK = 1.0    # Correct element clicked, action succeeded
_REWARD_FAILURE_CLICK = -0.5   # Wrong element / no change
_REWARD_TASK_COMPLETE = 2.0    # Task completed in same episode (back-prop)
_REWARD_NEAR_MISS     = 0.1    # Within tolerance but action failed


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GroundingStep:
    """
    One grounding data point collected during live execution.

    Fields:
        step_id:         Unique ID for deduplication
        screenshot_path: Path to saved screenshot (or "" if not saved)
        screenshot_b64:  Base64-encoded screenshot (if path not available)
        instruction:     Natural language instruction for this step
        predicted_x:     Normalised X coordinate predicted by vision model [0,1]
        predicted_y:     Normalised Y coordinate predicted by vision model [0,1]
        gt_x:            Ground truth X (from successful execution, if available)
        gt_y:            Ground truth Y (from successful execution, if available)
        outcome:         "success" | "failure" | "unknown"
        reward:          Computed reward signal
        app_name:        Focused application at time of action
        task_objective:  High-level task objective
        timestamp:       Unix timestamp
        episode_id:      Groups steps from the same task execution
    """
    step_id: str
    instruction: str
    predicted_x: float
    predicted_y: float
    gt_x: Optional[float]
    gt_y: Optional[float]
    outcome: str
    reward: float
    app_name: str = ""
    task_objective: str = ""
    screenshot_path: str = ""
    screenshot_b64: str = ""
    timestamp: float = field(default_factory=time.time)
    episode_id: str = ""


@dataclass
class GroundingBatch:
    """A batch of grounding triplets ready for training."""
    steps: List[GroundingStep]
    batch_id: str
    created_at: float = field(default_factory=time.time)
    total_reward: float = 0.0
    success_rate: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Reward Computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_grounding_reward(
    predicted_x: float,
    predicted_y: float,
    *,
    outcome: str,
    gt_x: Optional[float] = None,
    gt_y: Optional[float] = None,
    tolerance: float = _CLICK_TOLERANCE_NORM,
) -> float:
    """
    Compute grounding reward for a single click action.

    Formula:
        If outcome == "success":
            reward = REWARD_SUCCESS_CLICK + spatial_bonus
        If outcome == "failure":
            If near_miss (within tolerance): reward = REWARD_NEAR_MISS
            Else: reward = REWARD_FAILURE_CLICK
        Else (unknown):
            reward = 0.0

    Spatial bonus is awarded when gt_x/gt_y is known and the predicted
    coordinates are close — this rewards near-correct grounding even when
    the click happened to succeed through other means.
    """
    if outcome == "success":
        reward = _REWARD_SUCCESS_CLICK
        # Spatial bonus if ground truth is available
        if gt_x is not None and gt_y is not None:
            dist = math.sqrt(
                (predicted_x - gt_x) ** 2 + (predicted_y - gt_y) ** 2
            )
            # Linear bonus: 0.0 when dist=tolerance, 0.5 when dist=0
            if dist < tolerance:
                spatial_bonus = 0.5 * (1.0 - dist / tolerance)
                reward += spatial_bonus
        return round(reward, 4)

    elif outcome == "failure":
        if gt_x is not None and gt_y is not None:
            dist = math.sqrt(
                (predicted_x - gt_x) ** 2 + (predicted_y - gt_y) ** 2
            )
            if dist < tolerance:
                return _REWARD_NEAR_MISS  # near miss — gentle negative
        return _REWARD_FAILURE_CLICK

    return 0.0  # unknown outcome


# ─────────────────────────────────────────────────────────────────────────────
# GroundingTrainer
# ─────────────────────────────────────────────────────────────────────────────

class GroundingTrainer:
    """
    UI-AGILE continuous grounding reward trainer.

    Collects click-level grounding data during live task execution and
    generates training triplets for vision model fine-tuning.

    Usage (in GIIGoalDirectedLoop after each click):
        grounding_trainer.record_click(
            instruction="click the Submit button",
            predicted_x=0.72, predicted_y=0.85,
            screenshot_b64=screenshot_b64,
            outcome="success",
            app_name="Firefox",
            task_objective=self._objective,
        )

    Usage (at task completion):
        grounding_trainer.on_episode_complete(success=True)

    Usage (in NightlyConsolidation):
        grounding_trainer.flush_batch()
    """

    def __init__(
        self,
        grounding_dir: Optional[str] = None,
        *,
        batch_size: int = _BATCH_SIZE,
        save_screenshots: bool = False,
    ) -> None:
        self._dir = grounding_dir or _DEFAULT_GROUNDING_DIR
        os.makedirs(self._dir, exist_ok=True)

        self._batch_size = batch_size
        self._save_screenshots = save_screenshots

        self._steps: List[GroundingStep] = []
        self._lock = threading.Lock()

        self._current_episode_id: str = _new_id()
        self._episode_success: Optional[bool] = None

        # Stats
        self._total_steps = 0
        self._total_batches = 0
        self._success_steps = 0

        _logger.info(
            "[GroundingTrainer] Initialised. dir=%r batch_size=%d save_screenshots=%s",
            self._dir, self._batch_size, self._save_screenshots,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Data Collection API
    # ──────────────────────────────────────────────────────────────────────────

    def record_click(
        self,
        instruction: str,
        predicted_x: float,
        predicted_y: float,
        *,
        outcome: str = "unknown",
        gt_x: Optional[float] = None,
        gt_y: Optional[float] = None,
        screenshot_b64: str = "",
        app_name: str = "",
        task_objective: str = "",
    ) -> GroundingStep:
        """
        Record a single click grounding data point.

        Call this after every click/navigate action with the action outcome.
        For best training signal, also provide gt_x/gt_y when the true
        element coordinates are known (e.g. from AT-SPI accessibility tree).
        """
        reward = compute_grounding_reward(
            predicted_x=predicted_x,
            predicted_y=predicted_y,
            outcome=outcome,
            gt_x=gt_x,
            gt_y=gt_y,
        )

        step = GroundingStep(
            step_id=_new_id(),
            instruction=instruction[:500],
            predicted_x=round(float(predicted_x), 6),
            predicted_y=round(float(predicted_y), 6),
            gt_x=round(float(gt_x), 6) if gt_x is not None else None,
            gt_y=round(float(gt_y), 6) if gt_y is not None else None,
            outcome=outcome,
            reward=reward,
            app_name=app_name[:100],
            task_objective=task_objective[:200],
            screenshot_b64=(
                screenshot_b64[:200_000] if (self._save_screenshots and screenshot_b64)
                else ""
            ),
            episode_id=self._current_episode_id,
        )

        with self._lock:
            self._steps.append(step)
            self._total_steps += 1
            if outcome == "success":
                self._success_steps += 1

            # Auto-flush when batch is full
            if len(self._steps) >= self._batch_size:
                self._flush_locked()

        _logger.debug(
            "[GroundingTrainer] Recorded: outcome=%s reward=%.3f x=%.3f y=%.3f app=%r",
            outcome, reward, predicted_x, predicted_y, app_name[:30],
        )
        return step

    def on_episode_complete(self, success: bool) -> None:
        """
        Call when a task episode completes.

        Back-propagates a task-completion reward bonus to all steps in this
        episode — successful task completion means every grounding decision
        that contributed deserves positive reinforcement.
        """
        self._episode_success = success

        if success:
            with self._lock:
                # Apply task-completion bonus to all steps in current episode
                bonus = _REWARD_TASK_COMPLETE / max(len(self._steps), 1)
                for step in self._steps:
                    if step.episode_id == self._current_episode_id:
                        step.reward = round(step.reward + bonus, 4)
                _logger.info(
                    "[GroundingTrainer] Episode complete (success=True): "
                    "applied task bonus %.4f to %d steps.",
                    bonus, len(self._steps),
                )

        # Start new episode
        with self._lock:
            self._current_episode_id = _new_id()
            self._episode_success = None

    def flush_batch(self) -> Optional[str]:
        """
        Write current batch to JSONL and return the output path.
        Called by NightlyConsolidation. Returns None if no steps to flush.
        """
        with self._lock:
            return self._flush_locked()

    # ──────────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────────

    def _flush_locked(self) -> Optional[str]:
        """Must be called under self._lock."""
        if not self._steps:
            return None

        batch_id = _new_id()
        ts_str = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(self._dir, f"grounding_batch_{ts_str}_{batch_id}.jsonl")

        steps_data = []
        total_reward = 0.0
        successes = 0
        for step in self._steps:
            d = asdict(step)
            # Don't write large base64 screenshots to JSONL unless enabled
            if not self._save_screenshots:
                d.pop("screenshot_b64", None)
            steps_data.append(d)
            total_reward += step.reward
            if step.outcome == "success":
                successes += 1

        success_rate = successes / max(len(self._steps), 1)

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                for step_d in steps_data:
                    f.write(json.dumps(step_d, ensure_ascii=False) + "\n")

            # Also write a batch metadata file
            meta_path = out_path.replace(".jsonl", "_meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "batch_id": batch_id,
                    "created_at": time.time(),
                    "step_count": len(steps_data),
                    "total_reward": round(total_reward, 4),
                    "success_rate": round(success_rate, 4),
                    "batch_path": out_path,
                }, f, indent=2)

            _logger.info(
                "[GroundingTrainer] Flushed batch: steps=%d reward=%.3f success_rate=%.1%%  path=%s",
                len(steps_data), total_reward, success_rate * 100, out_path,
            )
            self._total_batches += 1

        except Exception as exc:
            _logger.error("[GroundingTrainer] Batch flush failed: %s", exc)
            return None

        self._steps = []
        return out_path

    # ──────────────────────────────────────────────────────────────────────────
    # Training Data Export (for ARPOTrainer / GRPOTrainer)
    # ──────────────────────────────────────────────────────────────────────────

    def export_training_pairs(
        self,
        min_reward: float = 0.5,
        max_pairs: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Export (positive, negative) grounding pairs from persisted batches.

        Returns list of dicts:
            {
                "instruction": str,
                "positive": {"x": float, "y": float},  # high-reward click
                "negative": {"x": float, "y": float},  # low-reward click
                "app_name": str,
            }

        Used by ARPOTrainer.generate_preference_pairs() to create DPO training data
        for fine-tuning the vision model's grounding head.
        """
        pairs: List[Dict[str, Any]] = []

        # Load all batch JSONL files
        try:
            jsonl_files = sorted(
                [f for f in os.listdir(self._dir) if f.endswith(".jsonl")],
                reverse=True,  # newest first
            )
        except Exception:
            return []

        # Group steps by instruction
        by_instruction: Dict[str, List[Dict]] = {}
        for jfile in jsonl_files[:20]:  # cap at 20 most recent batches
            try:
                with open(os.path.join(self._dir, jfile), "r", encoding="utf-8") as f:
                    for line in f:
                        step = json.loads(line.strip())
                        inst = step.get("instruction", "")
                        if inst:
                            by_instruction.setdefault(inst, []).append(step)
            except Exception:
                continue

        # Build contrastive pairs: (positive high-reward) vs (negative low-reward)
        for instruction, steps in by_instruction.items():
            positives = [s for s in steps if s.get("reward", 0) >= min_reward]
            negatives = [s for s in steps if s.get("reward", 0) < 0]
            if not positives or not negatives:
                continue
            best_pos = max(positives, key=lambda s: s.get("reward", 0))
            worst_neg = min(negatives, key=lambda s: s.get("reward", 0))
            pairs.append({
                "instruction": instruction,
                "positive": {
                    "x": best_pos.get("predicted_x", 0.5),
                    "y": best_pos.get("predicted_y", 0.5),
                    "reward": best_pos.get("reward", 0),
                },
                "negative": {
                    "x": worst_neg.get("predicted_x", 0.5),
                    "y": worst_neg.get("predicted_y", 0.5),
                    "reward": worst_neg.get("reward", 0),
                },
                "app_name": best_pos.get("app_name", ""),
            })
            if len(pairs) >= max_pairs:
                break

        _logger.info(
            "[GroundingTrainer] Exported %d training pairs (min_reward=%.2f).",
            len(pairs), min_reward,
        )
        return pairs

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            pending = len(self._steps)
        return {
            "total_steps": self._total_steps,
            "total_batches": self._total_batches,
            "success_steps": self._success_steps,
            "success_rate": self._success_steps / max(self._total_steps, 1),
            "pending_steps": pending,
            "grounding_dir": self._dir,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_instance: Optional[GroundingTrainer] = None
_instance_lock = threading.Lock()


def get_global_grounding_trainer(
    grounding_dir: Optional[str] = None,
) -> GroundingTrainer:
    """Return the process-singleton GroundingTrainer instance."""
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = GroundingTrainer(grounding_dir=grounding_dir)
    return _instance


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _new_id() -> str:
    import secrets
    return secrets.token_hex(8)
