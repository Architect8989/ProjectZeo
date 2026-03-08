from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional

_logger = logging.getLogger(__name__)

# Number of parallel rollouts (N=1 disables bBoN, falls through to single run)
_BBON_N: int = int(os.environ.get("PROJECTZEO_BBON_N", "1"))

# Temperature variation for diversity
_BBON_TEMPERATURES: List[float] = [0.7, 1.0, 1.2]

# Max seconds each rollout is allowed before the judge selects
_BBON_ROLLOUT_TIMEOUT: int = int(os.environ.get("PROJECTZEO_BBON_ROLLOUT_TIMEOUT", "1800"))

_JUDGE_SYSTEM_PROMPT = """\
You are a Behavior Judge for a GUI agent. You will receive N Behavior Narratives
from parallel agent runs on the same task. Each narrative describes what the agent
attempted and what final state it reached.

Your job: Select the BEST narrative — the one most likely to represent a successful
or nearly-successful task completion.

Criteria:
  1. Completeness: Did the agent make substantial progress toward the objective?
  2. Safety: Did the agent avoid destructive or irreversible actions?
  3. Efficiency: Did the agent reach its goal state without excessive steps?
  4. Stability: Was the final state stable (not stuck in error dialogs)?

Respond with ONLY a JSON object:
{
  "selected_index": <0-based index of the best narrative>,
  "reason": "<one sentence>",
  "confidence": "HIGH" | "MEDIUM" | "LOW"
}
"""


class BehaviorNarrative:
    """Compact representation of an agent trajectory."""

    def __init__(self, rollout_index: int) -> None:
        self.rollout_index = rollout_index
        self.actions: List[Dict[str, Any]] = []
        self.outcomes: List[bool] = []
        self.final_world_state: Dict[str, Any] = {}
        self.success: bool = False
        self.reason: str = ""
        self.iterations: int = 0
        self.temperature: float = 1.0
        self.elapsed_sec: float = 0.0

    def to_summary(self) -> str:
        """Produce compact text summary for the Behavior Judge."""
        action_types = [a.get("operation", "?") for a in self.actions[-20:]]
        success_rate = (
            sum(self.outcomes) / len(self.outcomes) if self.outcomes else 0
        )
        return (
            f"Rollout {self.rollout_index} (T={self.temperature}):\n"
            f"  Iterations: {self.iterations}\n"
            f"  Final outcome: {'SUCCESS' if self.success else 'FAILURE'}\n"
            f"  Reason: {self.reason[:200]}\n"
            f"  Action success rate: {success_rate:.1%}\n"
            f"  Last 20 operations: {action_types}\n"
            f"  Duration: {self.elapsed_sec:.1f}s\n"
        )


class BBonOrchestrator:
    

    def __init__(
        self,
        *,
        loop_factory: Callable[..., Any],
        judge_llm_callable: Optional[Callable] = None,
        n: int = _BBON_N,
        rollout_timeout: int = _BBON_ROLLOUT_TIMEOUT,
    ) -> None:
        
        self._loop_factory = loop_factory
        self._judge_llm = judge_llm_callable
        self._n = max(1, n)
        self._rollout_timeout = rollout_timeout

    def run(self, start_ts: Optional[float] = None) -> Dict[str, Any]:
        """
        Run N parallel rollouts and return the best result.
        Falls back to single-run semantics when N=1.
        """
        if self._n == 1:
            # Single rollout — no bBoN overhead
            _logger.info("[bBoN] N=1 — single rollout (bBoN disabled).")
            loop = self._loop_factory(temperature=_BBON_TEMPERATURES[0])
            return loop.run(start_ts=start_ts)

        _logger.info("[bBoN] Starting %d parallel rollouts.", self._n)
        narratives: List[Optional[BehaviorNarrative]] = [None] * self._n
        threads: List[threading.Thread] = []
        lock = threading.Lock()

        def _run_rollout(idx: int) -> None:
            temp = _BBON_TEMPERATURES[idx % len(_BBON_TEMPERATURES)]
            narrative = BehaviorNarrative(rollout_index=idx)
            narrative.temperature = temp
            t0 = time.time()

            try:
                loop = self._loop_factory(temperature=temp, rollout_index=idx)

                # Intercept action recording for trajectory capture
                original_fn = getattr(loop, "_execute_decision_fn", None)

                def _tracked_exec(action, world_state):
                    narrative.actions.append(action)
                    if original_fn is not None:
                        result = original_fn(action, world_state)
                        ok = result.get("success", True) if isinstance(result, dict) else bool(result)
                        narrative.outcomes.append(ok)
                        return result
                    narrative.outcomes.append(True)
                    return {"success": True, "output": ""}

                loop._execute_decision_fn = _tracked_exec

                result = loop.run(start_ts=start_ts)
                narrative.success = result.get("success", False)
                narrative.reason = result.get("reason", "")
                narrative.iterations = result.get("iterations", 0)
                narrative.final_world_state = result.get("final_world_state", {})

            except Exception as e:
                _logger.warning("[bBoN] Rollout %d error: %s", idx, e)
                narrative.success = False
                narrative.reason = str(e)
            finally:
                narrative.elapsed_sec = time.time() - t0
                with lock:
                    narratives[idx] = narrative

        for i in range(self._n):
            t = threading.Thread(
                target=_run_rollout, args=(i,), name=f"bbon_rollout_{i}", daemon=True
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=self._rollout_timeout)

        valid = [n for n in narratives if n is not None]
        if not valid:
            return {"success": False, "reason": "All bBoN rollouts failed or timed out."}

        winner = self._judge_select(valid)
        _logger.info(
            "[bBoN] Winner: rollout %d (success=%s, iter=%d)",
            winner.rollout_index, winner.success, winner.iterations,
        )
        return {
            "success": winner.success,
            "reason": winner.reason,
            "iterations": winner.iterations,
            "final_world_state": winner.final_world_state,
            "bbon_rollouts": self._n,
            "bbon_winner_index": winner.rollout_index,
        }

    def _judge_select(self, narratives: List[BehaviorNarrative]) -> BehaviorNarrative:
        """Use Behavior Judge LLM to select the best narrative."""
        # Quick heuristic: if any rollout succeeded, prefer it
        successes = [n for n in narratives if n.success]
        if successes and self._judge_llm is None:
            # Prefer the most efficient success
            return min(successes, key=lambda n: n.iterations)

        if self._judge_llm is None:
            # No LLM — return the one with highest action success rate
            def _score(n):
                if not n.outcomes:
                    return 0.0
                return sum(n.outcomes) / len(n.outcomes)
            return max(narratives, key=_score)

        # Use Behavior Judge LLM
        narratives_text = "\n\n".join(n.to_summary() for n in narratives)
        result_holder: list = [None]

        def _call():
            try:
                raw = self._judge_llm(
                    messages=[
                        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": f"NARRATIVES:\n{narratives_text}"},
                    ],
                    objective=None,
                    session_id="bbon_judge",
                )
                if isinstance(raw, list) and raw:
                    result_holder[0] = str(raw[0].get("content", "") if isinstance(raw[0], dict) else raw[0])
                elif isinstance(raw, str):
                    result_holder[0] = raw
            except Exception as e:
                _logger.warning("[bBoN] Judge LLM error: %s", e)

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        t.join(timeout=60.0)

        try:
            if result_holder[0]:
                clean = re.sub(r"```(?:json)?", "", result_holder[0]).strip()
                parsed = json.loads(clean)
                idx = int(parsed.get("selected_index", 0))
                if 0 <= idx < len(narratives):
                    _logger.info("[bBoN] Judge selected rollout %d: %s", idx, parsed.get("reason", ""))
                    return narratives[idx]
        except Exception as e:
            _logger.warning("[bBoN] Judge parse error: %s", e)

        # Fallback: most successful
        return max(narratives, key=lambda n: (n.success, sum(n.outcomes) / max(len(n.outcomes), 1)))
      
