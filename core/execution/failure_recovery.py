# core/execution/failure_recovery.py

"""
Failure Recovery Manager

Purpose:
Deterministic failure handling during execution.

This module:
- DOES NOT execute steps
- DOES NOT touch OS
- DOES NOT read screen
- DOES NOT make planning decisions

It is a pure policy engine that decides:
retry | alternative | abort
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List

from core.schemas.execution_plan import ExecutionStep


# ==================================================
# RECOVERY ACTION
# ==================================================

@dataclass(frozen=True)
class RecoveryAction:
    """
    Immutable recovery decision.

    action:
      - "retry"       → retry same step
      - "alternative" → execute alternative operations
      - "abort"       → terminate execution
    """
    action: str
    reason: Optional[str] = None
    delay: float = 0.0
    context: Optional[Dict[str, Any]] = None
    alternative_operations: Optional[List[Dict[str, Any]]] = None


# ==================================================
# FAILURE RECOVERY MANAGER
# ==================================================

class FailureRecoveryManager:
    """
    Deterministic, bounded failure recovery.

    HARD CONTRACT:
    - No execution
    - No randomness
    - No side effects
    - Same inputs → same outputs
    """

    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 2.0

    # -------------------------------------------------

    def handle_failure(
        self,
        step: ExecutionStep,
        error: Exception,
        attempt_ctx: Dict[str, Any],
    ) -> RecoveryAction:
        """
        Decide how to recover from a failed step.

        Inputs:
        - step: current ExecutionStep
        - error: raised exception
        - attempt_ctx: mutable retry context

        Returns:
        - RecoveryAction
        """

        attempt = int(attempt_ctx.get("attempt", 0))

        # ---- non-retryable steps (hard stop) ----
        if not getattr(step, "retryable", True):
            return RecoveryAction(
                action="abort",
                reason=f"Step {step.id} marked non-retryable: {error}",
            )

        # ---- bounded retry ----
        if attempt < self.MAX_RETRIES:
            return RecoveryAction(
                action="retry",
                reason=f"Retry {attempt + 1}/{self.MAX_RETRIES}: {error}",
                delay=self.RETRY_DELAY_SECONDS,
                context={"attempt": attempt + 1},
            )

        # ---- explicit alternatives only ----
        alternatives = step.action.get("alternatives")
        if isinstance(alternatives, list) and alternatives:
            return RecoveryAction(
                action="alternative",
                reason=(
                    f"Retry budget exhausted for step {step.id}, "
                    f"attempting alternatives: {error}"
                ),
                alternative_operations=alternatives,
            )

        # ---- fail-closed abort ----
        return RecoveryAction(
            action="abort",
            reason=f"Max retries exceeded for step {step.id}: {error}",
)
