# core/execution/progress_tracker.py

import time
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
    - No side effects outside memory
    """

    def __init__(self, execution_plan: ExecutionPlan):
        if not isinstance(execution_plan, ExecutionPlan):
            raise ValueError("ProgressTracker requires ExecutionPlan")

        self._plan = execution_plan

        self._completed: Set[int] = set()
        self._failed: Set[int] = set()

        self._current_step: Optional[int] = None
        self._execution_start_ts: Optional[float] = None

    # ==================================================
    # EXECUTION LIFECYCLE
    # ==================================================

    def start_execution(self) -> None:
        if self._execution_start_ts is not None:
            raise RuntimeError("Execution already started")

        self._execution_start_ts = time.time()

    # ==================================================
    # STEP STATE
    # ==================================================

    def start_step(self, step_id: int) -> None:
        if step_id in self._completed:
            raise RuntimeError(f"Step {step_id} already completed")

        if step_id in self._failed:
            raise RuntimeError(f"Step {step_id} already failed")

        self._current_step = step_id

    def complete_step(self, step_id: int) -> None:
        if self._current_step != step_id:
            raise RuntimeError(
                f"Completing step {step_id} but current is {self._current_step}"
            )

        self._completed.add(step_id)
        self._current_step = None

    def fail_step(self, step_id: int, reason: str) -> None:
        # reason is recorded by journal; tracker enforces state only
        self._failed.add(step_id)

        if self._current_step == step_id:
            self._current_step = None

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

    def execution_runtime_seconds(self) -> Optional[float]:
        if self._execution_start_ts is None:
            return None
        return time.time() - self._execution_start_ts
