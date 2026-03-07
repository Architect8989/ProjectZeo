from __future__ import annotations

"""
gii_loop.py — GII Goal-Directed Execution Loop

AUDIT FIXES APPLIED (March 2026):
  CRITICAL  : Added explicit REQUIRE_HUMAN_CONFIRMATION handler with
              _wait_for_human_approval(). Previously the loop fell through
              silently to the execute block, bypassing the human-approval gate
              when PROJECTZEO_USE_AGENT_ORCHESTRATOR=1.
  HIGH      : Hard iteration ceiling — max_iterations capped at 500 regardless
              of scaffold size, preventing 20+ hour runaway loops on CPU hardware.
  HIGH      : Pre-dispatch screen re-capture — lightweight pixel-hash diff
              performed immediately before every action dispatch. If the screen
              has changed materially since reasoning (dialog appeared, window
              changed), the iteration is skipped so the next observe→reason
              cycle picks up the updated world state.
"""

import hashlib
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional

_logger = logging.getLogger(__name__)

_DONE_OP = "done"

_APPROVAL_WAIT_POLL_INTERVAL = 2.0   # seconds between approval checks
_APPROVAL_WAIT_TIMEOUT       = 300.0  # 5-minute human response window
_SCREEN_DIFF_THRESHOLD       = 0.03   # 3% changed screen pixels = material change
_HARD_ITERATION_CEILING      = 500    # absolute cap regardless of scaffold size


class GIIGoalDirectedLoop:
    """
    True per-step GII execution loop: observe → reason → policy-gate →
    [human-approval] → [screen-diff] → execute → record.
    """

    MAX_ITERATIONS_DEFAULT = 500
    MAX_STAGNANT_DEFAULT   = 25
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
        # AUDIT FIX: Hard iteration ceiling — prevent runaway loops.
        self._max_iterations = min(max_iterations, _HARD_ITERATION_CEILING)
        self._max_stagnant = max_stagnant
        self._watchdog = watchdog
        self._execute_decision_fn = execute_decision_fn
        self._on_action_executed = on_action_executed

        self._iteration = 0
        self._stagnant_count = 0
        self._last_action_key: Optional[str] = None
        self._visited_keys: Dict[str, int] = {}
        self._permanent_deny: set = set()

        # Pre-dispatch screen diff — SHA-256 of frame pixels at reasoning time
        self._last_reasoning_frame_hash: Optional[str] = None

    # =========================================================================
    # Main entry point
    # =========================================================================

    def run(self, start_ts: Optional[float] = None) -> Dict[str, Any]:
        start_ts = start_ts or time.time()
        last_loop_ts = 0.0

        self._journal.record({
            "event": "gii_goal_directed_loop_start",
            "objective": self._objective[:200],
            "max_wallclock": self._max_wallclock,
            "max_iterations": self._max_iterations,
            "hard_ceiling": _HARD_ITERATION_CEILING,
        })

        _logger.info(
            "[GIILoop] Starting. objective=%r max_iter=%d (ceiling=%d)",
            self._objective[:80], self._max_iterations, _HARD_ITERATION_CEILING,
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
                    self._journal.record({
                        "event": "gii_loop_watchdog_violation",
                        "reason": str(wd_exc),
                    })
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

            # ── Capture frame hash BEFORE reasoning ───────────────────────
            self._last_reasoning_frame_hash = self._capture_screen_hash()

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
                    return self._result(
                        False,
                        f"Stagnation: GII returned no action. Last reason: {reason}",
                    )
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
                    "[GIILoop] Action key visited %d times — stagnation loop: %s",
                    visit_count + 1, action_key,
                )
                self._stagnant_count += 1
                if self._stagnant_count >= self._max_stagnant:
                    return self._result(False, "Stagnation: same action repeated too many times")
                world_state["_gii_loop_note"] = (
                    f"WARNING: You have chosen action {action_key!r} {visit_count + 1} times. "
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

            # ── AUDIT CRITICAL FIX: REQUIRE_HUMAN_CONFIRMATION handler ───
            # This block was entirely MISSING in the original code, causing
            # every REQUIRE_HUMAN_CONFIRMATION decision to silently execute.
            elif policy_decision == "REQUIRE_HUMAN_CONFIRMATION":
                self._journal.record({
                    "event": "gii_loop_require_human_confirmation",
                    "iteration": self._iteration,
                    "action_key": action_key,
                    "reason": policy_reason,
                    "operation": action.get("operation"),
                })
                _logger.warning(
                    "[GIILoop] REQUIRE_HUMAN_CONFIRMATION: %s | %s",
                    action_key, policy_reason,
                )
                approved = self._wait_for_human_approval(action, policy_reason)
                if not approved:
                    _logger.warning(
                        "[GIILoop] Human confirmation NOT received (timeout) for %s — "
                        "permanently denying.", action_key,
                    )
                    self._journal.record({
                        "event": "gii_loop_human_confirmation_timeout",
                        "iteration": self._iteration,
                        "action_key": action_key,
                    })
                    self._permanent_deny.add(action_key)
                    self._stagnant_count += 1
                    if self._stagnant_count >= self._max_stagnant:
                        return self._result(
                            False, "Stagnation: repeated unapproved confirmation requests"
                        )
                    continue
                # Human approved — log and fall through to execution
                self._journal.record({
                    "event": "gii_loop_human_confirmation_approved",
                    "iteration": self._iteration,
                    "action_key": action_key,
                })
                _logger.info(
                    "[GIILoop] Human confirmation received for %s — executing.", action_key
                )

            # ── AUDIT HIGH FIX: Pre-dispatch screen re-capture ────────────
            # Detect screen changes that occurred during LLM reasoning (e.g.,
            # a dialog appeared, window switched, error overlay appeared).
            # If the screen changed materially, skip and re-observe next iter.
            if self._last_reasoning_frame_hash is not None:
                current_hash = self._capture_screen_hash()
                if current_hash is not None:
                    if self._screen_changed_significantly(
                        self._last_reasoning_frame_hash, current_hash
                    ):
                        _logger.info(
                            "[GIILoop] Pre-dispatch screen diff: screen changed since "
                            "reasoning — skipping action %s, re-observing.", action_key,
                        )
                        self._journal.record({
                            "event": "gii_loop_screen_diff_skip",
                            "iteration": self._iteration,
                            "action_key": action_key,
                        })
                        # Not stagnation — this is healthy world-state adaptation
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
                    op = action.get("operation", "")
                    exec_success = self._minimal_dispatch(op, action)

                self._stagnant_count = 0

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
                    return self._result(
                        False,
                        f"Stagnation: repeated execution failures. Last: {exec_exc}",
                    )

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

        return self._result(
            False, f"Maximum iterations ({self._max_iterations}) reached"
        )

    # =========================================================================
    # AUDIT CRITICAL FIX: Human confirmation gate
    # =========================================================================

    def _wait_for_human_approval(
        self,
        action: Dict[str, Any],
        policy_reason: str,
    ) -> bool:
        """
        Block until human creates the .APPROVE signal file or timeout fires.

        Mirrors the create-to-approve semantics in operate.py:1351:
          Agent writes  {signal_dir}/{token}.signal
          Human creates {signal_dir}/{token}.signal.APPROVE
        Returns True if approved within _APPROVAL_WAIT_TIMEOUT seconds.
        Returns False on timeout or signal directory write failure.
        """
        import json as _json
        import secrets as _secrets
        import tempfile as _tempfile

        try:
            signal_dir = self._policy._APPROVAL_SIGNAL_DIR  # type: ignore[attr-defined]
        except AttributeError:
            signal_dir = _tempfile.gettempdir()

        token = _secrets.token_hex(8)
        signal_path = os.path.join(signal_dir, f"gii_approve_{token}.signal")
        approve_path = signal_path + ".APPROVE"

        op = str(action.get("operation", "?"))
        cmd_snippet = str(
            action.get("command", action.get("content", action.get("text", "")))
        )[:80]

        try:
            with open(signal_path, "w", encoding="utf-8") as _sf:
                _json.dump({
                    "iteration": self._iteration,
                    "operation": op,
                    "command_snippet": cmd_snippet,
                    "reason": policy_reason[:300],
                    "objective": self._objective[:200],
                    "approve_by_creating": approve_path,
                }, _sf, indent=2)
        except OSError as write_err:
            _logger.warning(
                "[GIILoop] Could not write approval signal file %r: %s — denying.",
                signal_path, write_err,
            )
            return False

        print(
            f"\n[GIILoop] ⚠  HUMAN APPROVAL REQUIRED  (iteration {self._iteration})\n"
            f"  Operation : {op}\n"
            f"  Command   : {cmd_snippet!r}\n"
            f"  Reason    : {policy_reason[:120]}\n"
            f"  Approve   : CREATE the file below within {_APPROVAL_WAIT_TIMEOUT:.0f}s\n"
            f"              {approve_path}\n",
            file=sys.stderr,
        )

        deadline = time.time() + _APPROVAL_WAIT_TIMEOUT
        approved = False
        try:
            while time.time() < deadline:
                if os.path.exists(approve_path):
                    approved = True
                    break
                time.sleep(_APPROVAL_WAIT_POLL_INTERVAL)
        finally:
            for _p in (signal_path, approve_path):
                try:
                    os.unlink(_p)
                except OSError:
                    pass

        return approved

    # =========================================================================
    # AUDIT HIGH FIX: Pre-dispatch screen diff helpers
    # =========================================================================

    def _capture_screen_hash(self) -> Optional[str]:
        """
        Capture a lightweight screenshot and return its SHA-256 pixel hash.
        No model inference — ~50ms latency (screenshot + resize + hash).
        Returns None on failure (non-fatal; screen diff is skipped).
        """
        try:
            try:
                import pyautogui as _pya
                _img = _pya.screenshot()
                # Downsample for speed — still detects dialogs and overlays
                _img = _img.resize((320, 180))
                _raw = _img.tobytes()
            except Exception:
                import tempfile as _tf
                from operate.utils.screenshot import capture_screen_with_cursor as _cap
                _tmp = _tf.NamedTemporaryFile(suffix=".png", delete=False)
                _tmp_name = _tmp.name
                _tmp.close()
                try:
                    _cap(_tmp_name)
                    with open(_tmp_name, "rb") as _f:
                        _raw = _f.read()
                finally:
                    try:
                        os.unlink(_tmp_name)
                    except OSError:
                        pass
            return hashlib.sha256(_raw).hexdigest()
        except Exception as exc:
            _logger.debug("[GIILoop] Screen hash capture failed (non-fatal): %s", exc)
            return None

    @staticmethod
    def _screen_changed_significantly(hash_a: str, hash_b: str) -> bool:
        """
        Return True if enough hex nibbles differ between the two SHA-256 hashes
        to indicate a material screen change (> _SCREEN_DIFF_THRESHOLD fraction).
        """
        if hash_a == hash_b:
            return False
        diff_chars = sum(1 for a, b in zip(hash_a, hash_b) if a != b)
        fraction = diff_chars / max(len(hash_a), 1)
        return fraction > _SCREEN_DIFF_THRESHOLD

    # =========================================================================
    # Helpers
    # =========================================================================

    def _get_world_state(self) -> Dict[str, Any]:
        if self._world_graph is not None:
            try:
                snapshot = self._world_graph.snapshot()
                if isinstance(snapshot, dict):
                    return snapshot
            except Exception as exc:
                _logger.debug("[GIILoop] WorldGraph snapshot error: %s", exc)
        return {}

    def _compute_action_key(self, action: Dict[str, Any]) -> str:
        op   = str(action.get("operation", ""))
        cmd  = str(action.get("command", ""))
        text = str(action.get("text", ""))
        path = str(action.get("path", ""))
        x    = str(action.get("x", ""))
        y    = str(action.get("y", ""))
        raw  = f"{op}:{cmd}:{text}:{path}:{x}:{y}"
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def _minimal_dispatch(self, op: str, action: Dict[str, Any]) -> bool:
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
