# core/execution/progress_tracker.py

import time
import json
import os
import hashlib
from typing import Set, Optional

from core.schemas.execution_plan import ExecutionPlan


class ProgressTracker:
    """
    Deterministic execution progress authority.

    Purpose:
    - Track execution lifecycle
    - Enforce dependency correctness
    - Provide single source of truth for step state

    HARD CONTRACT:
    - No execution
    - No retries
    - No recovery logic
    - Side effects limited to progress persistence
    """

    PROGRESS_VERSION = 1

    def __init__(self, execution_plan: ExecutionPlan):
        if not isinstance(execution_plan, ExecutionPlan):
            raise ValueError("ProgressTracker requires ExecutionPlan")

        self._plan = execution_plan

        self._completed: Set[int] = set()
        self._failed: Set[int] = set()

        self._current_step: Optional[int] = None
        self._execution_start_ts: Optional[float] = None
        self._execution_finished: bool = False

        self._plan_hash = self._hash_plan(execution_plan)
        self._progress_path = f".progress_{self._plan_hash}.json"

        self._load()

    # ==================================================
    # INTERNALS
    # ==================================================

    @staticmethod
    def _hash_plan(plan: ExecutionPlan) -> str:
        """
        Stable hash binding progress to a specific execution plan.
        """
        h = hashlib.sha256()
        h.update(plan.json().encode("utf-8"))
        return h.hexdigest()[:16]

    # ==================================================
    # PERSISTENCE
    # ==================================================

    def _load(self) -> None:
        if not os.path.exists(self._progress_path):
            return

        try:
            with open(self._progress_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if data.get("version") != self.PROGRESS_VERSION:
                raise ValueError("Progress version mismatch")

            if data.get("plan_hash") != self._plan_hash:
                raise ValueError("Progress plan hash mismatch")

            self._completed = set(data.get("completed", []))
            self._failed = set(data.get("failed", []))
            self._current_step = data.get("current_step")
            self._execution_start_ts = data.get("execution_start_ts")
            self._execution_finished = data.get("execution_finished", False)

        except Exception:
            # Fail-closed: corrupted or mismatched progress is discarded
            self._completed.clear()
            self._failed.clear()
            self._current_step = None
            self._execution_start_ts = None
            self._execution_finished = False

    def _persist(self) -> None:
        tmp_path = f"{self._progress_path}.tmp"

        payload = {
            "version": self.PROGRESS_VERSION,
            "plan_hash": self._plan_hash,
            "completed": sorted(self._completed),
            "failed": sorted(self._failed),
            "current_step": self._current_step,
            "execution_start_ts": self._execution_start_ts,
            "execution_finished": self._execution_finished,
        }

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, self._progress_path)

    # ==================================================
    # EXECUTION LIFECYCLE
    # ==================================================

    def start_execution(self) -> None:
        if self._execution_start_ts is not None:
            raise RuntimeError("Execution already started")

        self._execution_start_ts = time.time()
        self._persist()

    def finish_execution(self) -> None:
        if self._execution_finished:
            return

        self._execution_finished = True
        self._current_step = None
        self._persist()

    # ==================================================
    # STEP STATE
    # ==================================================

    def start_step(self, step_id: int) -> None:
        if step_id in self._completed:
            raise RuntimeError(f"Step {step_id} already completed")

        if step_id in self._failed:
            raise RuntimeError(f"Step {step_id} already failed")

        if self._current_step is not None:
            raise RuntimeError(
                f"Cannot start step {step_id}; "
                f"step {self._current_step} still active"
            )

        self._current_step = step_id
        self._persist()

    def complete_step(self, step_id: int) -> None:
        if self._current_step != step_id:
            raise RuntimeError(
                f"Completing step {step_id} but current is {self._current_step}"
            )

        self._completed.add(step_id)
        self._current_step = None
        self._persist()

    def fail_step(self, step_id: int, reason: str) -> None:
        self._failed.add(step_id)

        if self._current_step == step_id:
            self._current_step = None

        self._persist()

    # ==================================================
    # QUERY INTERFACE
    # ==================================================

    def is_completed(self, step_id: int) -> bool:
        return step_id in self._completed

    def is_failed(self, step_id: int) -> bool:
        return step_id in self._failed

    def current_step(self) -> Optional[int]:
        return self._current_step

    def execution_started(self) -> bool:
        return self._execution_start_ts is not None

    def execution_finished(self) -> bool:
        return self._execution_finished

    def execution_runtime_seconds(self) -> Optional[float]:
        if self._execution_start_ts is None:
            return None
        end = (
            self._execution_start_ts
            if not self._execution_finished
            else time.time()
        )
        return time.time() - self._execution_start_ts
