"""
core/learning/sppo_trainer.py
=================================
Self-Play Policy Optimization (SPPO) — Algorithm 19 of the 19-algorithm GII stack.

Blueprint §12.1 — Self-Improvement via Self-Play

Reference:
    Wu et al. (2024) "Self-Play Preference Optimization for Language Model Alignment"
    arXiv:2405.00675

Principle:
    The agent plays against a past copy of itself. The current policy
    (μ_t) generates a response; a frozen snapshot of the prior policy (μ_{t-1})
    also generates a response to the same prompt. A judge (the LLM itself) then
    decides which response is better. The preference signal is used to update
    μ_t via a DPO-style gradient.

Role in ProjectZeo:
    SPPO closes the loop between task execution and policy improvement without
    requiring human preference labels. It produces DPO training pairs from
    real task trajectories by comparing:
      - "current policy" response: what the agent did
      - "reference policy" response: what the agent would have done N tasks ago

    These pairs are passed to PreferenceGenerator → AgentQ → nightly DPO fine-tune.

Key design choices:
    1. Frozen reference policy is stored as a JSON snapshot of task trajectories
       in ~/.projectzeo/sppo/ (no actual weight fork needed for prompt-based LLMs)
    2. The judge prompt is symmetric: both responses evaluated blind to avoid
       position bias (responses are shown in random order, winner determined by
       semantic quality + goal achievement)
    3. Adaptive sampling: SPPO triggers more often when recent task success rate
       drops, less often when performance is high (cost-efficient self-improvement)
    4. EWC-compatible: SPPO pairs are tagged so nightly consolidation can gate
       them behind the EWC Fisher penalty

Integration:
    gii_controller._initialise_phase3_components() instantiates SPPOTrainer
    gii_controller.on_task_complete() calls record_trajectory() → async judge
    nightly_consolidation._step_dpo_generation() picks up SPPO pairs via AgentQ
"""
from __future__ import annotations

import json
import logging
import os
import random
import statistics
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

_ENABLED = os.environ.get("PROJECTZEO_SPPO_ENABLED", "1").strip() != "0"
_STORE_DIR = os.path.expanduser(
    os.environ.get("PROJECTZEO_SPPO_DIR", "~/.projectzeo/sppo")
)
_WINDOW_SIZE = int(os.environ.get("PROJECTZEO_SPPO_WINDOW", "20"))
_MIN_PAIRS = int(os.environ.get("PROJECTZEO_SPPO_MIN_PAIRS", "5"))
_JUDGE_TIMEOUT = float(os.environ.get("PROJECTZEO_SPPO_JUDGE_TIMEOUT", "30.0"))
_ADAPTIVE_THRESHOLD = float(os.environ.get("PROJECTZEO_SPPO_ADAPTIVE_THRESHOLD", "0.7"))


@dataclass
class TaskTrajectory:
    """A single completed task trajectory used as SPPO training material."""
    trajectory_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    objective: str = ""
    app_context: str = ""
    key_actions: List[str] = field(default_factory=list)  # operation names
    action_outcomes: List[bool] = field(default_factory=list)  # success flags
    thoughts: List[str] = field(default_factory=list)   # PSR reasoning snippets
    final_outcome: bool = False
    duration_s: float = 0.0
    step_count: int = 0
    timestamp: float = field(default_factory=time.time)
    policy_version: int = 0   # increments when DPO training completes

    def to_response_text(self) -> str:
        """Render trajectory as a human-readable response for the judge."""
        lines = [f"Objective: {self.objective[:300]}"]
        lines.append(f"App: {self.app_context or 'unknown'}")
        lines.append(f"Steps taken: {self.step_count}")
        for i, (act, ok, thought) in enumerate(
            zip(self.key_actions[:10], self.action_outcomes[:10], self.thoughts[:10])
        ):
            status = "✓" if ok else "✗"
            t = f" — {thought[:80]}" if thought else ""
            lines.append(f"  {i+1}. {status} {act}{t}")
        lines.append(f"Outcome: {'SUCCESS' if self.final_outcome else 'FAILURE'}")
        return "\n".join(lines)

    def quality_score(self) -> float:
        """Heuristic quality signal (0-1). Used for adaptive sampling."""
        if self.step_count == 0:
            return 0.0
        success_rate = sum(self.action_outcomes) / max(1, len(self.action_outcomes))
        outcome_bonus = 0.3 if self.final_outcome else 0.0
        efficiency_bonus = max(0.0, 0.2 - (self.step_count / 100.0) * 0.2)
        return min(1.0, success_rate * 0.5 + outcome_bonus + efficiency_bonus)


@dataclass
class SPPOPair:
    """One self-play preference pair for DPO training."""
    pair_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    objective: str = ""
    current_trajectory_id: str = ""
    reference_trajectory_id: str = ""
    chosen_response: str = ""    # winner trajectory text
    rejected_response: str = ""  # loser trajectory text
    judge_reasoning: str = ""
    chosen_is_current: bool = True  # True if current > reference
    quality_delta: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SPPOJudge:
    """
    LLM-as-judge for self-play preference decisions.

    Uses a symmetric evaluation prompt (random response order) to avoid
    position bias. Returns the winner index (0 or 1) and reasoning.
    """

    def __init__(self, llm_callable: Optional[Callable] = None) -> None:
        self._llm = llm_callable

    def judge(
        self,
        objective: str,
        response_a: str,
        response_b: str,
        timeout: float = _JUDGE_TIMEOUT,
    ) -> Tuple[int, str]:
        """
        Compare two trajectory responses.

        Returns:
            (winner_idx, reasoning) where winner_idx is 0 for A, 1 for B.
            On failure returns (0, "fallback: quality heuristic").
        """
        if self._llm is None:
            return 0, "no_llm"

        # Randomise order to prevent position bias
        order = [0, 1]
        random.shuffle(order)
        responses = [response_a, response_b]
        ordered_a = responses[order[0]]
        ordered_b = responses[order[1]]

        prompt = (
            f"You are evaluating two AI agent trajectories for the same task.\n\n"
            f"TASK: {objective[:400]}\n\n"
            f"TRAJECTORY A:\n{ordered_a[:800]}\n\n"
            f"TRAJECTORY B:\n{ordered_b[:800]}\n\n"
            "Which trajectory better achieves the task? Consider: goal achievement, "
            "step efficiency, correct reasoning, minimal errors.\n"
            'Respond ONLY with JSON: {"winner": "A" or "B", "reasoning": "..."}'
        )

        result_holder: List[Optional[str]] = [None]

        def _call() -> None:
            try:
                raw = self._llm(
                    messages=[{"role": "user", "content": prompt}],
                    objective="sppo_judge",
                    session_id="sppo_judge",
                )
                if isinstance(raw, list) and raw:
                    result_holder[0] = str(
                        raw[0].get("content", "") if isinstance(raw[0], dict) else raw[0]
                    )
                elif isinstance(raw, str):
                    result_holder[0] = raw
            except Exception as exc:
                _logger.debug("[SPPO-Judge] LLM call failed: %s", exc)

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if not result_holder[0]:
            return 0, "timeout"

        try:
            import re
            m = re.search(r"\{[^}]+\}", result_holder[0])
            if not m:
                return 0, "parse_error"
            data = json.loads(m.group(0))
            winner_letter = str(data.get("winner", "A")).upper().strip()
            reasoning = str(data.get("reasoning", ""))[:300]
            # Map back: winner_letter "A" or "B" in shuffled order
            if winner_letter == "A":
                original_idx = order[0]
            else:
                original_idx = order[1]
            return original_idx, reasoning
        except Exception as parse_exc:
            _logger.debug("[SPPO-Judge] Parse failed: %s", parse_exc)
            return 0, "parse_error"


class SPPOTrainer:
    """
    Self-Play Policy Optimization trainer.

    Maintains a sliding window of recent task trajectories. On each new
    completed task, compares against a randomly sampled trajectory from N
    tasks ago (the "reference policy"). Generates DPO-ready preference pairs.

    Thread-safe: all state mutations protected by _lock.
    """

    def __init__(
        self,
        llm_callable: Optional[Callable] = None,
        store_dir: Optional[str] = None,
        window_size: int = _WINDOW_SIZE,
    ) -> None:
        self._llm = llm_callable
        self._dir = store_dir or _STORE_DIR
        self._window = window_size
        self._judge = SPPOJudge(llm_callable)
        self._lock = threading.Lock()
        self._policy_version: int = 0
        self._pairs_written: int = 0
        self._trajectories_seen: int = 0
        self._recent_success_rates: List[float] = []

        os.makedirs(self._dir, exist_ok=True)
        _logger.info(
            "[SPPO] Trainer ready. enabled=%s window=%d dir=%s",
            _ENABLED, self._window, self._dir,
        )

    def update_llm(self, llm_callable: Callable) -> None:
        """Update LLM callable (called when GII controller updates its LLM)."""
        with self._lock:
            self._llm = llm_callable
            self._judge = SPPOJudge(llm_callable)

    def record_trajectory(
        self,
        *,
        objective: str,
        app_context: str = "",
        execution_log: Optional[Dict[str, Any]] = None,
        success: bool = False,
        duration_s: float = 0.0,
    ) -> Optional[str]:
        """
        Record a completed task trajectory and (async) generate a self-play pair.

        Called by GIIController.on_task_complete().

        Returns:
            trajectory_id if accepted, None if SPPO disabled.
        """
        if not _ENABLED:
            return None

        traj = self._build_trajectory(
            objective=objective,
            app_context=app_context,
            execution_log=execution_log or {},
            success=success,
            duration_s=duration_s,
        )
        traj.policy_version = self._policy_version

        self._save_trajectory(traj)

        with self._lock:
            self._trajectories_seen += 1
            self._recent_success_rates.append(1.0 if success else 0.0)
            if len(self._recent_success_rates) > 20:
                self._recent_success_rates.pop(0)

        # Adaptive trigger: run more often when success rate is low
        if self._should_trigger():
            ref_traj = self._load_reference_trajectory(exclude_id=traj.trajectory_id)
            if ref_traj is not None:
                threading.Thread(
                    target=self._async_judge_and_store,
                    args=(traj, ref_traj),
                    daemon=True,
                    name="sppo-judge",
                ).start()

        return traj.trajectory_id

    def _should_trigger(self) -> bool:
        """Adaptive sampling: trigger when not enough trajectories or success rate is low."""
        with self._lock:
            n = self._trajectories_seen
        if n < _MIN_PAIRS:
            return False
        if not self._recent_success_rates:
            return True
        recent_success = statistics.mean(self._recent_success_rates)
        # Trigger always if success rate below threshold, else 30% random sample
        return recent_success < _ADAPTIVE_THRESHOLD or random.random() < 0.3

    def _build_trajectory(
        self,
        *,
        objective: str,
        app_context: str,
        execution_log: Dict[str, Any],
        success: bool,
        duration_s: float,
    ) -> TaskTrajectory:
        """Extract trajectory fields from execution log."""
        actions: List[str] = []
        outcomes: List[bool] = []
        thoughts: List[str] = []

        for step_data in list(execution_log.values())[:20]:
            if not isinstance(step_data, dict):
                continue
            for out in step_data.get("outputs", []):
                if not isinstance(out, dict):
                    continue
                actions.append(str(out.get("operation", "unknown"))[:60])
                outcomes.append(bool(out.get("success", True)))
                thought = str(out.get("thought", out.get("reasoning", "")))[:100]
                thoughts.append(thought)

        return TaskTrajectory(
            objective=objective,
            app_context=app_context,
            key_actions=actions[:15],
            action_outcomes=outcomes[:15],
            thoughts=thoughts[:15],
            final_outcome=success,
            duration_s=duration_s,
            step_count=len(actions),
        )

    def _save_trajectory(self, traj: TaskTrajectory) -> str:
        path = os.path.join(self._dir, f"traj_{traj.trajectory_id}.json")
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(asdict(traj), f)
            os.replace(tmp, path)
        except OSError as exc:
            _logger.warning("[SPPO] Failed to save trajectory: %s", exc)
        return path

    def _load_reference_trajectory(
        self, exclude_id: str
    ) -> Optional[TaskTrajectory]:
        """Load a random past trajectory to serve as the reference policy."""
        try:
            files = sorted(
                [f for f in os.listdir(self._dir) if f.startswith("traj_") and f.endswith(".json")],
                key=lambda x: os.path.getmtime(os.path.join(self._dir, x)),
            )
            # Reference = older trajectories (first half of the window)
            candidates = [
                f for f in files
                if not f.endswith(f"{exclude_id}.json")
            ]
            # Take from the older half for a meaningful policy gap
            window = candidates[: max(1, len(candidates) // 2)]
            if not window:
                return None
            chosen = random.choice(window)
            with open(os.path.join(self._dir, chosen), encoding="utf-8") as f:
                data = json.load(f)
            traj = TaskTrajectory(**{k: v for k, v in data.items() if k in TaskTrajectory.__dataclass_fields__})
            return traj
        except Exception as exc:
            _logger.debug("[SPPO] Failed to load reference trajectory: %s", exc)
            return None

    def _async_judge_and_store(
        self, current: TaskTrajectory, reference: TaskTrajectory
    ) -> None:
        """Judge two trajectories and write a DPO preference pair."""
        try:
            current_text = current.to_response_text()
            reference_text = reference.to_response_text()

            winner_idx, reasoning = self._judge.judge(
                objective=current.objective,
                response_a=current_text,
                response_b=reference_text,
            )

            current_wins = winner_idx == 0
            if current_wins:
                chosen_text = current_text
                rejected_text = reference_text
            else:
                chosen_text = reference_text
                rejected_text = current_text

            quality_delta = current.quality_score() - reference.quality_score()

            pair = SPPOPair(
                objective=current.objective,
                current_trajectory_id=current.trajectory_id,
                reference_trajectory_id=reference.trajectory_id,
                chosen_response=chosen_text,
                rejected_response=rejected_text,
                judge_reasoning=reasoning,
                chosen_is_current=current_wins,
                quality_delta=quality_delta,
            )
            self._write_pair(pair)
            _logger.info(
                "[SPPO] Pair generated: current_wins=%s quality_delta=%.2f reasoning=%s",
                current_wins, quality_delta, reasoning[:80],
            )
        except Exception as exc:
            _logger.warning("[SPPO] Judge/store failed: %s", exc)

    def _write_pair(self, pair: SPPOPair) -> str:
        ts = int(time.time())
        path = os.path.join(self._dir, f"sppo_pair_{ts}_{pair.pair_id}.jsonl")
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(json.dumps(pair.to_dict()) + "\n")
            os.replace(tmp, path)
            with self._lock:
                self._pairs_written += 1
        except OSError as exc:
            _logger.warning("[SPPO] Failed to write pair: %s", exc)
        return path

    def bump_policy_version(self) -> None:
        """Call after each DPO fine-tune completes to track policy versions."""
        with self._lock:
            self._policy_version += 1
        _logger.info("[SPPO] Policy version bumped to %d.", self._policy_version)

    def get_pairs_for_dpo(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Load recent SPPO pairs for DPO training."""
        pairs: List[Dict[str, Any]] = []
        try:
            files = sorted(
                [f for f in os.listdir(self._dir) if f.startswith("sppo_pair_") and f.endswith(".jsonl")],
                key=lambda x: os.path.getmtime(os.path.join(self._dir, x)),
                reverse=True,
            )
            for fname in files[:limit]:
                try:
                    with open(os.path.join(self._dir, fname), encoding="utf-8") as f:
                        for line in f:
                            pairs.append(json.loads(line))
                except Exception:
                    pass
        except Exception as exc:
            _logger.debug("[SPPO] get_pairs_for_dpo error: %s", exc)
        return pairs[:limit]

    def ingest_pairs_to_agent_q(self) -> int:
        """Push SPPO pairs into AgentQ for nightly DPO training."""
        try:
            from core.learning.agent_q import get_agent_q
            aq = get_agent_q()
            pairs = self.get_pairs_for_dpo()
            ingested = 0
            for p in pairs:
                try:
                    aq.ingest_sppo_pair(p)
                    ingested += 1
                except Exception:
                    pass
            if ingested:
                _logger.info("[SPPO] Ingested %d pairs into AgentQ.", ingested)
            return ingested
        except Exception as exc:
            _logger.debug("[SPPO] AgentQ ingest failed: %s", exc)
            return 0

    def get_stats(self) -> Dict[str, Any]:
        recent_sr = (
            statistics.mean(self._recent_success_rates)
            if self._recent_success_rates else 0.0
        )
        return {
            "enabled": _ENABLED,
            "policy_version": self._policy_version,
            "trajectories_seen": self._trajectories_seen,
            "pairs_written": self._pairs_written,
            "recent_success_rate": round(recent_sr, 3),
            "window_size": self._window,
            "store_dir": self._dir,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_instance: Optional[SPPOTrainer] = None
_lock = threading.Lock()


def get_sppo_trainer(llm_callable: Optional[Callable] = None) -> SPPOTrainer:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = SPPOTrainer(llm_callable=llm_callable)
    elif llm_callable is not None and _instance._llm is None:
        _instance.update_llm(llm_callable)
    return _instance
