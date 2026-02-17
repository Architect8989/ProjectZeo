# core/cognition/belief_state.py

from typing import Dict, Any, List
import time


class BeliefState:
    """
    Internal cognitive memory.

    Tracks:
        - Known entities
        - Action history
        - Failure patterns
        - Goal progress
        - Environment stability
    """

    def __init__(self):
        self.created_at = time.time()
        self.entities: Dict[str, Any] = {}
        self.action_history: List[Dict[str, Any]] = []
        self.failures: List[Dict[str, Any]] = []
        self.progress_score: float = 0.0
        self.environment_stability: float = 1.0

    # -------------------------------------------------

    def update_entities(self, world_snapshot: Dict[str, Any]) -> None:
        self.entities = world_snapshot or {}

    # -------------------------------------------------

    def record_action(self, action: Dict[str, Any], result: Dict[str, Any]) -> None:
        self.action_history.append({
            "action": action,
            "result": result,
            "timestamp": time.time(),
        })

    # -------------------------------------------------

    def record_failure(self, action: Dict[str, Any], reason: str) -> None:
        self.failures.append({
            "action": action,
            "reason": reason,
            "timestamp": time.time(),
        })

    # -------------------------------------------------

    def compute_environment_stability(self, delta: Dict[str, Any]) -> None:
        if not delta:
            return
        significant = delta.get("significant_change", False)
        if significant:
            self.environment_stability *= 0.8
        else:
            self.environment_stability = min(1.0, self.environment_stability + 0.05)

    # -------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        return {
            "progress_score": self.progress_score,
            "recent_failures": self.failures[-3:],
            "environment_stability": self.environment_stability,
            "action_count": len(self.action_history),
      }
