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

    _VALID_ACTIONS = {"retry", "alternative", "abort"}

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
        - attempt_ctx: retry context (read-only for policy)

        Returns:
        - RecoveryAction
        """

        if not isinstance(step, ExecutionStep):
            raise ValueError("Invalid step for recovery")

        if not isinstance(attempt_ctx, dict):
            attempt_ctx = {}

        attempt = int(attempt_ctx.get("attempt", 0))

        # ---- non-retryable steps (hard stop) ----
        if not getattr(step, "retryable", True):
            return self._abort(
                reason=f"Step {step.id} marked non-retryable: {error}"
            )

        # ---- bounded retry ----
        if attempt < self.MAX_RETRIES:
            return self._retry(
                attempt=attempt,
                error=error,
            )

        # ---- explicit alternatives only ----
        alternatives = self._extract_alternatives(step)

        if alternatives:
            return RecoveryAction(
                action="alternative",
                reason=(
                    f"Retry budget exhausted for step {step.id}, "
                    f"attempting explicit alternatives"
                ),
                alternative_operations=alternatives,
            )

        # ---- fail-closed abort ----
        return self._abort(
            reason=f"Max retries exceeded for step {step.id}: {error}"
        )

    # ==================================================
    # INTERNAL HELPERS
    # ==================================================

    def _retry(
        self,
        *,
        attempt: int,
        error: Exception,
    ) -> RecoveryAction:

        return RecoveryAction(
            action="retry",
            reason=f"Retry {attempt + 1}/{self.MAX_RETRIES}: {error}",
            delay=self.RETRY_DELAY_SECONDS,
            context={"attempt": attempt + 1},
        )

    def _abort(self, *, reason: str) -> RecoveryAction:
        return RecoveryAction(
            action="abort",
            reason=reason,
        )

    def _extract_alternatives(
        self,
        step: ExecutionStep,
    ) -> Optional[List[Dict[str, Any]]]:

        action = getattr(step, "action", None)
        if not isinstance(action, dict):
            return None

        alternatives = action.get("alternatives")

        if not isinstance(alternatives, list) or not alternatives:
            return None

        # Strict validation: each alternative must be dict
        validated: List[Dict[str, Any]] = []
        for alt in alternatives:
            if isinstance(alt, dict):
                validated.append(alt)

        if not validated:
            return None

        return validated
