from __future__ import annotations

import logging
import sys
import time
import hashlib
from typing import Any, Callable, Dict, List, Optional

_logger = logging.getLogger(__name__)

# Sentinel returned by PerStepReasoner when goal is complete
_DONE_OP = "done"


class GIIGoalDirectedLoop:
    

    # Maximum iterations before forced termination (GII loop has no fixed plan length)
    MAX_ITERATIONS_DEFAULT = 500
    # Maximum consecutive stagnant iterations before declaring failure
    MAX_STAGNANT_DEFAULT = 25
    # Minimum seconds between consecutive LLM calls (anti-thrash on fast paths)
    MIN_LOOP_INTERVAL_SECONDS = 0.1

    def __init__(
        self,
        *,
        gii_controller,
        os_backend,
        world_graph,
        policy_engine,
        journal,
        objective: str,
        max_wallclock_seconds: int = 3600,
        max_iterations: int = MAX_ITERATIONS_DEFAULT,
        max_stagnant: int = MAX_STAGNANT_DEFAULT,
        watchdog=None,
        execute_decision_fn: Optional[Callable] = None,
        on_action_executed: Optional[Callable] = None,
    ) -> None:
        self._gii = gii_controller
        self._os_backend = os_backend
        self._world_graph = world_graph
        self._policy = policy_engine
        self._journal = journal
        self._objective = objective
        self._max_wallclock = max_wallclock_seconds
        self._max_iterations = max_iterations
        self._max_stagnant = max_stagnant
        self._watchdog = watchdog
        self._execute_decision_fn = execute_decision_fn
        self._on_action_executed = on_action_executed

        # State
        self._iteration = 0
        self._stagnant_count = 0
        self._last_action_key: Optional[str] = None
        self._visited_keys: Dict[str, int] = {}  # key → count
        self._permanent_deny: set = set()

    # =========================================================================
    # Main entry point
    # =========================================================================

    def run(self, start_ts: Optional[float] = None) -> Dict[str, Any]:
        
        start_ts = start_ts or time.time()
        goal_complete = False
        last_loop_ts = 0.0

        self._journal.record({
            "event": "gii_goal_directed_loop_start",
            "objective": self._objective[:200],
            "max_wallclock": self._max_wallclock,
            "max_iterations": self._max_iterations,
        })

        _logger.info(
            "[GIILoop] Starting goal-directed execution. objective=%r max_iter=%d",
            self._objective[:80], self._max_iterations,
        )

        while self._iteration < self._max_iterations:
            # ── Wall-clock timeout ────────────────────────────────────────
            elapsed = time.time() - start_ts
            if elapsed > self._max_wallclock:
                self._journal.record({"event": "gii_loop_timeout", "elapsed": elapsed})
                return self._result(False, "Wall-clock timeout exceeded")

            # ── Watchdog ──────────────────────────────────────────────────
            if self._watchdog is not None:
                try:
                    self._watchdog.check()
                except Exception as wd_exc:
                    return self._result(False, f"Watchdog violation: {wd_exc}")

            # ── Anti-thrash rate limit ────────────────────────────────────
            now = time.time()
            if now - last_loop_ts < self.MIN_LOOP_INTERVAL_SECONDS:
                time.sleep(self.MIN_LOOP_INTERVAL_SECONDS - (now - last_loop_ts))
            last_loop_ts = time.time()

            self._iteration += 1

            # ── Observe world state ───────────────────────────────────────
            try:
                world_state = self._get_world_state()
            except Exception as obs_exc:
                _logger.warning("[GIILoop] World state observation failed: %s", obs_exc)
                self._stagnant_count += 1
                if self._stagnant_count >= self._max_stagnant:
                    return self._result(False, "Stagnation limit reached (observation failures)")
                continue

            # ── GII: Decide next action ───────────────────────────────────
            try:
                action, reason = self._gii.decide_next_action(world_state)
            except Exception as gii_exc:
                _logger.warning("[GIILoop] GII decide_next_action error: %s", gii_exc)
                action, reason = None, str(gii_exc)

            if action is None:
                _logger.warning("[GIILoop] No action from GII reasoner: %s", reason)
                self._stagnant_count += 1
                if self._stagnant_count >= self._max_stagnant:
                    return self._result(False, f"Stagnation: GII returned no action. Last reason: {reason}")
                continue

            # ── Check for goal completion ─────────────────────────────────
            if action.get("operation") == _DONE_OP:
                summary = action.get("summary", "Task complete")
                self._journal.record({
                    "event": "gii_goal_complete",
                    "iteration": self._iteration,
                    "summary": summary,
                })
                _logger.info("[GIILoop] Goal complete: %s (iter=%d)", summary, self._iteration)
                return self._result(True, summary)

            # ── Deduplicate / stagnation detection ───────────────────────
            action_key = self._compute_action_key(action)
            if action_key in self._permanent_deny:
                _logger.warning("[GIILoop] Permanently denied action repeated: %s", action_key)
                self._stagnant_count += 1
                if self._stagnant_count >= self._max_stagnant:
                    return self._result(False, "Stagnation: permanently denied action repeated")
                continue

            visit_count = self._visited_keys.get(action_key, 0)
            self._visited_keys[action_key] = visit_count + 1
            if visit_count >= 3:
                _logger.warning(
                    "[GIILoop] Action key visited %d times — likely stagnation loop: %s",
                    visit_count + 1, action_key,
                )
                self._stagnant_count += 1
                if self._stagnant_count >= self._max_stagnant:
                    return self._result(False, "Stagnation: same action repeated too many times")
                # Force the GII to reconsider by injecting a note — don't just repeat
                world_state["_gii_loop_note"] = (
                    f"WARNING: You have chosen action {action_key!r} {visit_count+1} times. "
                    "This action is not making progress. Choose a DIFFERENT approach."
                )
                continue

            # ── Policy gate ───────────────────────────────────────────────
            focused_app = world_state.get("focused_app", "__unknown_app__")
            policy_decision, policy_reason = self._policy.validate_action_dict(
                action, focused_app=focused_app
            )

            if policy_decision == "DENY":
                self._permanent_deny.add(action_key)
                self._gii.record_denial(action_key)
                self._journal.record({
                    "event": "gii_loop_policy_deny",
                    "iteration": self._iteration,
                    "action_key": action_key,
                    "reason": policy_reason,
                })
                _logger.warning("[GIILoop] Policy DENY: %s | %s", action_key, policy_reason)
                self._stagnant_count += 1
                if self._stagnant_count >= self._max_stagnant:
                    return self._result(False, "Stagnation: repeated policy denials")
                continue

            # ── Execute action ────────────────────────────────────────────
            exec_success = False
            exec_output = ""
            try:
                if self._execute_decision_fn is not None:
                    exec_result = self._execute_decision_fn(action, world_state)
                    if isinstance(exec_result, dict):
                        exec_success = exec_result.get("success", True)
                        exec_output = str(exec_result.get("output", ""))
                    else:
                        exec_success = bool(exec_result)
                else:
                    # Minimal fallback: dispatch via os_backend directly
                    op = action.get("operation", "")
                    exec_success = self._minimal_dispatch(op, action)

                self._stagnant_count = 0  # Reset on successful execution

                self._journal.record({
                    "event": "gii_loop_action_executed",
                    "iteration": self._iteration,
                    "action_key": action_key,
                    "operation": action.get("operation"),
                    "success": exec_success,
                    "output_snippet": exec_output[:200],
                })

            except Exception as exec_exc:
                exec_success = False
                exec_output = str(exec_exc)
                _logger.warning("[GIILoop] Action execution failed: %s | %s", action_key, exec_exc)
                self._stagnant_count += 1
                if self._stagnant_count >= self._max_stagnant:
                    return self._result(False, f"Stagnation: repeated execution failures. Last: {exec_exc}")

            # ── Record outcome in GII controller ──────────────────────────
            try:
                self._gii.record_outcome(action, success=exec_success, output=exec_output)
            except Exception:
                pass

            # ── Callback hook ─────────────────────────────────────────────
            if self._on_action_executed is not None:
                try:
                    self._on_action_executed(action, exec_success, exec_output, self._iteration)
                except Exception:
                    pass

        return self._result(False, f"Maximum iterations ({self._max_iterations}) reached")

    # =========================================================================
    # Helpers
    # =========================================================================

    def _get_world_state(self) -> Dict[str, Any]:
        """Get current world state from world graph."""
        if self._world_graph is not None:
            try:
                snapshot = self._world_graph.snapshot()
                if isinstance(snapshot, dict):
                    return snapshot
            except Exception as exc:
                _logger.debug("[GIILoop] WorldGraph snapshot error: %s", exc)
        return {}

    def _compute_action_key(self, action: Dict[str, Any]) -> str:
        """Compute a stable hash key for an action for deduplication."""
        op = str(action.get("operation", ""))
        cmd = str(action.get("command", ""))
        text = str(action.get("text", ""))
        path = str(action.get("path", ""))
        x = str(action.get("x", ""))
        y = str(action.get("y", ""))
        raw = f"{op}:{cmd}:{text}:{path}:{x}:{y}"
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def _minimal_dispatch(self, op: str, action: Dict[str, Any]) -> bool:
        """Minimal action dispatch when execute_decision_fn is not provided."""
        if self._os_backend is None:
            return False
        try:
            if op == "click":
                x = float(action.get("x", 0.5))
                y = float(action.get("y", 0.5))
                self._os_backend.click(x, y)
            elif op in ("write", "type"):
                self._os_backend.write(str(action.get("content", "")))
            elif op == "press":
                self._os_backend.press(action.get("keys", []))
            elif op == "command":
                result = self._os_backend.exec(str(action.get("command", "")))
                return result.returncode == 0 if hasattr(result, "returncode") else True
            return True
        except Exception as exc:
            _logger.warning("[GIILoop] Minimal dispatch error: %s", exc)
            return False

    def _result(self, success: bool, reason: str) -> Dict[str, Any]:
        return {
            "success": success,
            "reason": reason,
            "iterations": self._iteration,
            "stagnant_count": self._stagnant_count,
            "final_world_state": self._get_world_state(),
        }
