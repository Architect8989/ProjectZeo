# core/execution/failure_recovery.py

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import json

from core.schemas.execution_plan import ExecutionStep


# ==================================================
# RECOVERY ACTION
# ==================================================

@dataclass(frozen=True)
class RecoveryAction:
    action: str  # "retry" | "alternative" | "abort"
    reason: Optional[str] = None
    delay: float = 0.0
    context: Optional[Dict[str, Any]] = None
    alternative_operations: Optional[List[Dict[str, Any]]] = None


# ==================================================
# FAILURE RECOVERY MANAGER
# ==================================================

class FailureRecoveryManager:
    """
    Bounded failure recovery engine.

    HARD CONTRACT:
    - No execution
    - No OS access
    - No world mutation
    - Deterministic fallback always available
    - Retry ceiling strictly enforced
    """

    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 2.0
    MAX_LLM_DELAY_SECONDS = 5.0
    MAX_SNAPSHOT_ENTITIES = 10

    _VALID_ACTIONS = {"retry", "abort"}

    # ==================================================
    # PRIMARY ENTRYPOINT (Deterministic Core)
    # ==================================================

    def handle_failure(
        self,
        step: ExecutionStep,
        error: Exception,
        attempt_ctx: Dict[str, Any],
    ) -> RecoveryAction:

        if not isinstance(step, ExecutionStep):
            raise ValueError("Invalid step for recovery")

        if not isinstance(attempt_ctx, dict):
            attempt_ctx = {}

        attempt = int(attempt_ctx.get("attempt", 0))

        # ---- non-retryable ----
        if not getattr(step, "retryable", True):
            return self._abort(
                reason=f"Step {step.id} marked non-retryable: {error}"
            )

        # ---- bounded retry ----
        if attempt < self.MAX_RETRIES:
            return self._retry(attempt=attempt, error=error)

        # ---- explicit alternatives only if defined ----
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

        # ---- fail-closed ----
        return self._abort(
            reason=f"Max retries exceeded for step {step.id}: {error}"
        )

    # ==================================================
    # LLM-BOUNDED RECOVERY (SAFE EXTENSION)
    # ==================================================

    def handle_failure_with_perception(
        self,
        *,
        step: ExecutionStep,
        error: Exception,
        attempt_ctx: Dict[str, Any],
        world_graph=None,
        llm_callable=None,
    ) -> RecoveryAction:
        """
        Bounded LLM-assisted recovery.

        Strict guarantees:
        - Never exceeds MAX_RETRIES
        - Never injects new alternatives
        - Falls back deterministically on any failure
        - LLM may only choose: retry | abort
        """

        deterministic = self.handle_failure(step, error, attempt_ctx)

        attempt = int((attempt_ctx or {}).get("attempt", 0))

        # Respect retry ceiling strictly
        if attempt >= self.MAX_RETRIES:
            return deterministic

        # Only reason after first failure
        if attempt < 1:
            return deterministic

        if not world_graph or not llm_callable:
            return deterministic

        try:
            snapshot = world_graph.snapshot()

            prompt = self._build_llm_prompt(
                step=step,
                error=error,
                snapshot=snapshot,
            )

            raw = llm_callable(prompt)

            if not isinstance(raw, str):
                return deterministic

            decision = json.loads(raw.strip())

            action = decision.get("action")
            if action not in self._VALID_ACTIONS:
                return deterministic

            # ---- ABORT ----
            if action == "abort":
                return RecoveryAction(
                    action="abort",
                    reason=f"LLM advised abort: {decision.get('reason', '')}",
                )

            # ---- RETRY (STRICTLY BOUNDED) ----
            if action == "retry":
                if attempt + 1 > self.MAX_RETRIES:
                    return deterministic

                try:
                    delay = float(
                        decision.get("suggested_delay", self.RETRY_DELAY_SECONDS)
                    )
                except Exception:
                    delay = self.RETRY_DELAY_SECONDS

                delay = max(0.0, min(delay, self.MAX_LLM_DELAY_SECONDS))

                return RecoveryAction(
                    action="retry",
                    reason=f"LLM advised retry: {decision.get('reason', '')}",
                    delay=delay,
                    context={"attempt": attempt + 1},
                )

            return deterministic

        except Exception:
            return deterministic

    # ==================================================
    # INTERNAL HELPERS
    # ==================================================

    def _retry(self, *, attempt: int, error: Exception) -> RecoveryAction:
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

        validated: List[Dict[str, Any]] = []
        for alt in alternatives:
            if isinstance(alt, dict):
                validated.append(alt)

        return validated or None

    def _build_llm_prompt(
        self,
        *,
        step: ExecutionStep,
        error: Exception,
        snapshot: Dict[str, Any],
    ) -> str:

        entities = snapshot.get("entities", [])[: self.MAX_SNAPSHOT_ENTITIES]

        return f"""
You are a strictly bounded recovery policy engine.

FAILED STEP:
- id: {step.id}
- type: {step.type.value}
- description: {step.description}

ERROR:
{str(error)[:300]}

CURRENT WORLD STATE:
- focused_app: {snapshot.get("focused_app")}
- entity_count: {snapshot.get("entity_count")}
- visible_elements_sample:
{json.dumps(entities, indent=2)}

Return JSON only:
{{
  "action": "retry" | "abort",
  "reason": "short explanation",
  "suggested_delay": 0-5
}}

Rules:
- Retry only if temporary failure likely.
- Abort if impossible in current state.
- Never suggest alternatives.
- Never exceed retry ceiling.
- Be conservative.
"""
