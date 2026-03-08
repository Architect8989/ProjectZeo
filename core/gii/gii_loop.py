from __future__ import annotations

"""
gii_loop.py — GII Goal-Directed Execution Loop

PATCH HISTORY
─────────────────────────────────────────────────────────────────────────────
March 2026 — Audit fixes applied:
  CRITICAL  : REQUIRE_HUMAN_CONFIRMATION handler added with _wait_for_human_approval()
  HIGH      : Hard iteration ceiling (500)
  HIGH      : Pre-dispatch screen diff

March 2026 — Research integration patch:
  GAP-1     : SHA-256 nibble compare → imagehash.phash() perceptual hashing
  LAYER-2   : VeriSafe Agent pre-action formal verification (Research §3)
  LAYER-5   : PIGuard external-content injection filter (Research §6.2)
  LAYER-5   : AT-SPI window registry for popup attack detection (Research §6.3)
  LAYER-7   : ProcessFence context manager wraps execute block (Research §8.1)
"""

import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Set

_logger = logging.getLogger(__name__)

_DONE_OP = "done"

_APPROVAL_WAIT_POLL_INTERVAL = 2.0
_APPROVAL_WAIT_TIMEOUT       = 300.0
_HARD_ITERATION_CEILING      = 500

# GAP-1: phash threshold — Hamming distance <= 10 = same screen (64-bit hash)
_PHASH_CHANGE_THRESHOLD: int = int(
    os.environ.get("PROJECTZEO_PHASH_THRESHOLD", "10")
)

# LAYER-5: window appearing <3s ago without AT-SPI event is SUSPICIOUS
_POPUP_SUSPICIOUS_WINDOW_AGE_SEC: float = float(
    os.environ.get("PROJECTZEO_POPUP_AGE_SEC", "3.0")
)


# ─────────────────────────────────────────────────────────────────────────────
# GAP-1: Perceptual hash helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_phash(img) -> Optional[int]:
    """
    Compute 64-bit perceptual hash via imagehash.phash().
    Falls back to None when imagehash is unavailable.
    """
    try:
        import imagehash  # type: ignore
        return int(imagehash.phash(img))
    except ImportError:
        return None
    except Exception:
        return None


def _phash_distance(h1: Optional[int], h2: Optional[int]) -> int:
    """Hamming distance between two phash values. 0=identical."""
    if h1 is None or h2 is None:
        return 0
    return bin(h1 ^ h2).count("1")


class GIIGoalDirectedLoop:
    """
    Per-step GII execution loop: observe → reason → policy-gate →
    [VSA verify] → [popup check] → [human-approval] → [phash diff] →
    execute (in ProcessFence) → record.
    """

    MAX_ITERATIONS_DEFAULT    = 500
    MAX_STAGNANT_DEFAULT      = 25
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
        # LAYER-2
        vsa_verifier=None,
        # LAYER-5
        piguard=None,
        atspi_window_registry: Optional[Set[int]] = None,
        # LAYER-7
        use_process_fence: bool = True,
    ) -> None:
        self._gii = gii_controller
        self._os_backend = os_backend
        self._world_graph = world_graph
        self._policy = policy_engine
        self._journal = journal
        self._objective = objective
        self._max_wallclock = max_wallclock_seconds
        self._max_iterations = min(max_iterations, _HARD_ITERATION_CEILING)
        self._max_stagnant = max_stagnant
        self._watchdog = watchdog
        self._execute_decision_fn = execute_decision_fn
        self._on_action_executed = on_action_executed

        self._vsa = vsa_verifier
        self._piguard = piguard
        self._atspi_window_registry: Set[int] = atspi_window_registry or set()
        self._use_process_fence = use_process_fence

        self._iteration = 0
        self._stagnant_count = 0
        self._visited_keys: Dict[str, int] = {}
        self._permanent_deny: set = set()

        # GAP-1: perceptual hash at reasoning time
        self._last_reasoning_phash: Optional[int] = None

        # Blueprint Phase 1: GoalRepresentation integration
        # Pulled from GIIController if available
        self._goal_repr = getattr(gii_controller, "_goal_repr", None)
        self._operator_cycle = getattr(gii_controller, "_operator_cycle", None)
        self._algorithm_distiller = getattr(gii_controller, "_algorithm_distiller", None)
        self._executed_operators: list = []   # Track for SOAR chunking

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
            "phash_threshold": _PHASH_CHANGE_THRESHOLD,
            "vsa_active": self._vsa is not None,
            "piguard_active": self._piguard is not None,
        })

        _logger.info(
            "[GIILoop] Starting. objective=%r max_iter=%d vsa=%s piguard=%s fence=%s",
            self._objective[:80], self._max_iterations,
            self._vsa is not None, self._piguard is not None, self._use_process_fence,
        )

        while self._iteration < self._max_iterations:

            elapsed = time.time() - start_ts
            if elapsed > self._max_wallclock:
                self._journal.record({"event": "gii_loop_timeout", "elapsed": elapsed})
                return self._result(False, "Wall-clock timeout exceeded")

            if self._watchdog is not None:
                try:
                    self._watchdog.check()
                except Exception as wd_exc:
                    self._journal.record({"event": "gii_loop_watchdog_violation", "reason": str(wd_exc)})
                    return self._result(False, f"Watchdog violation: {wd_exc}")

            now = time.time()
            if now - last_loop_ts < self.MIN_LOOP_INTERVAL_SECONDS:
                time.sleep(self.MIN_LOOP_INTERVAL_SECONDS - (now - last_loop_ts))
            last_loop_ts = time.time()

            self._iteration += 1

            try:
                world_state = self._get_world_state()
            except Exception as obs_exc:
                _logger.warning("[GIILoop] World state observation failed: %s", obs_exc)
                self._stagnant_count += 1
                if self._stagnant_count >= self._max_stagnant:
                    return self._result(False, "Stagnation limit reached (observation failures)")
                continue

            # GAP-1: capture phash BEFORE reasoning
            _, self._last_reasoning_phash = self._capture_phash()

            try:
                action, reason = self._gii.decide_next_action(world_state)
            except Exception as gii_exc:
                _logger.warning("[GIILoop] GII decide_next_action error: %s", gii_exc)
                action, reason = None, str(gii_exc)

            if action is None:
                self._stagnant_count += 1
                if self._stagnant_count >= self._max_stagnant:
                    return self._result(False, f"Stagnation: GII returned no action. Last: {reason}")
                continue

            if action.get("operation") == _DONE_OP:
                summary = action.get("summary", "Task complete")
                self._journal.record({"event": "gii_goal_complete", "iteration": self._iteration, "summary": summary})
                _logger.info("[GIILoop] Goal complete: %s (iter=%d)", summary, self._iteration)
                return self._result(True, summary)

            # LAYER-5: PIGuard on external content
            if self._piguard is not None and action.get("_external_content_source"):
                external_content = str(action.get("content") or action.get("text") or "")
                if external_content and self._piguard_check(external_content) == "INJECTION":
                    _logger.warning("[GIILoop] PIGuard: injection blocked in external content (iter=%d)", self._iteration)
                    self._journal.record({"event": "gii_loop_piguard_injection_blocked", "iteration": self._iteration})
                    self._stagnant_count += 1
                    if self._stagnant_count >= self._max_stagnant:
                        return self._result(False, "Stagnation: repeated PIGuard blocks")
                    continue

            # LAYER-5: Popup attack detection
            if self._is_suspicious_popup(action, world_state):
                _logger.warning("[GIILoop] SUSPICIOUS POPUP (iter=%d) — escalating", self._iteration)
                self._journal.record({"event": "gii_loop_suspicious_popup", "iteration": self._iteration})
                approved = self._wait_for_human_approval(
                    action,
                    "SUSPICIOUS: dialog appeared without AT-SPI window:create event (possible pop-up attack)"
                )
                if not approved:
                    self._permanent_deny.add(self._compute_action_key(action))
                    self._stagnant_count += 1
                    if self._stagnant_count >= self._max_stagnant:
                        return self._result(False, "Stagnation: repeated suspicious popup blocks")
                    continue

            action_key = self._compute_action_key(action)
            if action_key in self._permanent_deny:
                self._stagnant_count += 1
                if self._stagnant_count >= self._max_stagnant:
                    return self._result(False, "Stagnation: permanently denied action repeated")
                continue

            visit_count = self._visited_keys.get(action_key, 0)
            self._visited_keys[action_key] = visit_count + 1
            if visit_count >= 3:
                self._stagnant_count += 1
                if self._stagnant_count >= self._max_stagnant:
                    return self._result(False, "Stagnation: same action repeated too many times")
                world_state["_gii_loop_note"] = (
                    f"WARNING: action {action_key!r} repeated {visit_count + 1} times. Choose DIFFERENT approach."
                )
                continue

            focused_app = world_state.get("focused_app", "__unknown_app__")
            policy_decision, policy_reason = self._policy.validate_action_dict(action, focused_app=focused_app)

            if policy_decision == "DENY":
                self._permanent_deny.add(action_key)
                self._gii.record_denial(action_key)
                self._journal.record({"event": "gii_loop_policy_deny", "iteration": self._iteration, "action_key": action_key, "reason": policy_reason})
                _logger.warning("[GIILoop] Policy DENY: %s | %s", action_key, policy_reason)
                self._stagnant_count += 1
                if self._stagnant_count >= self._max_stagnant:
                    return self._result(False, "Stagnation: repeated policy denials")
                continue

            elif policy_decision == "REQUIRE_HUMAN_CONFIRMATION":
                self._journal.record({"event": "gii_loop_require_human_confirmation", "iteration": self._iteration, "action_key": action_key, "reason": policy_reason})
                _logger.warning("[GIILoop] REQUIRE_HUMAN_CONFIRMATION: %s | %s", action_key, policy_reason)
                approved = self._wait_for_human_approval(action, policy_reason)
                if not approved:
                    _logger.warning("[GIILoop] Human confirmation timeout for %s — denying.", action_key)
                    self._journal.record({"event": "gii_loop_human_confirmation_timeout", "iteration": self._iteration, "action_key": action_key})
                    self._permanent_deny.add(action_key)
                    self._stagnant_count += 1
                    if self._stagnant_count >= self._max_stagnant:
                        return self._result(False, "Stagnation: repeated unapproved confirmation requests")
                    continue
                self._journal.record({"event": "gii_loop_human_confirmation_approved", "iteration": self._iteration, "action_key": action_key})
                _logger.info("[GIILoop] Human confirmation approved for %s.", action_key)

            # LAYER-2: VeriSafe Agent verification
            if self._vsa is not None:
                vsa_result = self._vsa_verify(action)
                if vsa_result == "VIOLATION":
                    vsa_feedback = ""
                    try:
                        vsa_feedback = self._vsa.last_violation_reason()
                    except Exception:
                        pass
                    _logger.warning("[GIILoop] VSA VIOLATION (iter=%d): %s", self._iteration, vsa_feedback)
                    self._journal.record({"event": "gii_loop_vsa_violation", "iteration": self._iteration, "feedback": vsa_feedback})
                    world_state["_vsa_violation"] = (
                        f"FORMAL SPEC VIOLATION: {vsa_feedback}. "
                        "Your action violates a safety invariant from the task spec. "
                        "Choose a different action."
                    )
                    self._stagnant_count += 1
                    if self._stagnant_count >= self._max_stagnant:
                        return self._result(False, f"Stagnation: VSA violations. Last: {vsa_feedback}")
                    continue

            # GAP-1: Pre-dispatch phash diff
            if self._last_reasoning_phash is not None:
                _, current_phash = self._capture_phash()
                dist = _phash_distance(self._last_reasoning_phash, current_phash)
                if dist > _PHASH_CHANGE_THRESHOLD:
                    _logger.info("[GIILoop] phash diff=%d > threshold=%d — re-observing.", dist, _PHASH_CHANGE_THRESHOLD)
                    self._journal.record({"event": "gii_loop_screen_diff_skip", "iteration": self._iteration, "phash_distance": dist})
                    continue

            # LAYER-7: Execute inside ProcessFence
            exec_success = False
            exec_output = ""
            try:
                if self._use_process_fence:
                    try:
                        from core.safety.process_fence import process_fence  # noqa: PLC0415
                        with process_fence():
                            exec_success, exec_output = self._dispatch_action(action, world_state)
                    except ImportError:
                        exec_success, exec_output = self._dispatch_action(action, world_state)
                else:
                    exec_success, exec_output = self._dispatch_action(action, world_state)

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
                    return self._result(False, f"Stagnation: repeated execution failures. Last: {exec_exc}")

            try:
                self._gii.record_outcome(action, success=exec_success, output=exec_output)
            except Exception:
                pass

            # Blueprint §2.6.2: Algorithm Distillation — record trajectory step
            if self._algorithm_distiller is not None:
                try:
                    focused_app = world_state.get("focused_app", "unknown")
                    obs_summary = (
                        f"App: {focused_app} | "
                        f"Entities: {len(world_state.get('entities', []))}"
                    )
                    ad_episode = getattr(self, "_ad_current_episode", None)
                    if ad_episode is None:
                        ad_episode = self._algorithm_distiller.create_episode(
                            task_type=self._objective[:80],
                            app_context=focused_app,
                        )
                        self._ad_current_episode = ad_episode
                    ad_episode.add_step(
                        observation=obs_summary,
                        action=action,
                        reward=1.0 if exec_success else 0.0,
                        outcome="success" if exec_success else "failure",
                    )
                except Exception:
                    pass

            # GoalRepresentation: evaluate progress after each action
            if self._goal_repr is not None:
                try:
                    self._goal_repr.evaluate_from_screen(world_state)
                    if self._goal_repr.is_complete:
                        _logger.info("[GIILoop] GoalRepresentation: all conditions satisfied.")
                        self._journal.record({
                            "event": "gii_loop_goal_complete_via_goal_repr",
                            "iteration": self._iteration,
                            "progress": self._goal_repr.progress,
                        })
                        # Finalise AD episode on success
                        ad_ep = getattr(self, "_ad_current_episode", None)
                        if ad_ep and self._algorithm_distiller:
                            self._algorithm_distiller.finalize_episode(ad_ep, success=True)
                        return self._result(True, "All goal sub-conditions satisfied")
                except Exception:
                    pass

            if self._on_action_executed is not None:
                try:
                    self._on_action_executed(action, exec_success, exec_output, self._iteration)
                except Exception:
                    pass

        return self._result(False, f"Maximum iterations ({self._max_iterations}) reached")

    # =========================================================================
    # Helpers
    # =========================================================================

    def _dispatch_action(self, action: Dict[str, Any], world_state: Dict[str, Any]):
        if self._execute_decision_fn is not None:
            result = self._execute_decision_fn(action, world_state)
            if isinstance(result, dict):
                return result.get("success", True), str(result.get("output", ""))
            return bool(result), ""
        return self._minimal_dispatch(action.get("operation", ""), action), ""

    def _vsa_verify(self, action: Dict[str, Any]) -> str:
        try:
            return self._vsa.verify(action)
        except Exception as exc:
            _logger.warning("[GIILoop] VSA error (fail-open): %s", exc)
            return "OK"

    def _piguard_check(self, text: str) -> str:
        try:
            return self._piguard.classify(text)
        except Exception:
            return "SAFE"

    def _is_suspicious_popup(self, action: Dict[str, Any], world_state: Dict[str, Any]) -> bool:
        if not self._atspi_window_registry:
            return False
        target_win_id = action.get("_target_window_id")
        if target_win_id is None:
            return False
        if target_win_id in self._atspi_window_registry:
            return False
        win_age = action.get("_target_window_age_sec", 999)
        return isinstance(win_age, (int, float)) and win_age < _POPUP_SUSPICIOUS_WINDOW_AGE_SEC

    def _capture_phash(self):
        """GAP-1: Capture screenshot + compute imagehash.phash()."""
        try:
            try:
                import pyautogui as _pya  # type: ignore
                img = _pya.screenshot().resize((320, 180))
            except Exception:
                import tempfile as _tf
                from operate.utils.screenshot import capture_screen_with_cursor as _cap  # noqa
                from PIL import Image as _PILImage  # noqa
                tmp = _tf.NamedTemporaryFile(suffix=".png", delete=False)
                tmp_name = tmp.name; tmp.close()
                try:
                    _cap(tmp_name)
                    img = _PILImage.open(tmp_name).resize((320, 180))
                finally:
                    try: os.unlink(tmp_name)
                    except OSError: pass
            return img, _compute_phash(img)
        except Exception as exc:
            _logger.debug("[GIILoop] phash capture failed (non-fatal): %s", exc)
            return None, None

    def _wait_for_human_approval(self, action: Dict[str, Any], policy_reason: str) -> bool:
        import json as _json, secrets as _secrets, tempfile as _tempfile
        try:
            signal_dir = self._policy._APPROVAL_SIGNAL_DIR  # type: ignore[attr-defined]
        except AttributeError:
            signal_dir = _tempfile.gettempdir()
        token = _secrets.token_hex(8)
        signal_path = os.path.join(signal_dir, f"gii_approve_{token}.signal")
        approve_path = signal_path + ".APPROVE"
        op = str(action.get("operation", "?"))
        cmd_snippet = str(action.get("command", action.get("content", action.get("text", ""))))[:80]
        try:
            with open(signal_path, "w", encoding="utf-8") as sf:
                _json.dump({"iteration": self._iteration, "operation": op, "command_snippet": cmd_snippet, "reason": policy_reason[:300], "objective": self._objective[:200], "approve_by_creating": approve_path}, sf, indent=2)
        except OSError as e:
            _logger.warning("[GIILoop] Cannot write signal file: %s — denying.", e)
            return False
        print(f"\n[GIILoop] ⚠  HUMAN APPROVAL REQUIRED (iter {self._iteration})\n  Op: {op}\n  Cmd: {cmd_snippet!r}\n  Reason: {policy_reason[:120]}\n  Approve: CREATE {approve_path}\n", file=sys.stderr)
        deadline = time.time() + _APPROVAL_WAIT_TIMEOUT
        approved = False
        try:
            while time.time() < deadline:
                if os.path.exists(approve_path):
                    approved = True; break
                time.sleep(_APPROVAL_WAIT_POLL_INTERVAL)
        finally:
            for p in (signal_path, approve_path):
                try: os.unlink(p)
                except OSError: pass
        return approved

    def _get_world_state(self) -> Dict[str, Any]:
        if self._world_graph is not None:
            try:
                snap = self._world_graph.snapshot()
                if isinstance(snap, dict):
                    return snap
            except Exception as exc:
                _logger.debug("[GIILoop] WorldGraph snapshot error: %s", exc)
        return {}

    def _compute_action_key(self, action: Dict[str, Any]) -> str:
        import hashlib
        raw = ":".join(str(action.get(k, "")) for k in ("operation", "command", "text", "path", "x", "y"))
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def _minimal_dispatch(self, op: str, action: Dict[str, Any]) -> bool:
        if self._os_backend is None:
            return False
        try:
            if op == "click":
                self._os_backend.click(float(action.get("x", 0.5)), float(action.get("y", 0.5)))
            elif op in ("write", "type"):
                self._os_backend.write(str(action.get("content", "")))
            elif op == "press":
                self._os_backend.press(action.get("keys", []))
            elif op == "command":
                r = self._os_backend.exec(str(action.get("command", "")))
                return r.returncode == 0 if hasattr(r, "returncode") else True
            return True
        except Exception as exc:
            _logger.warning("[GIILoop] Minimal dispatch error: %s", exc)
            return False

    def _result(self, success: bool, reason: str) -> Dict[str, Any]:
        # Finalise Algorithm Distillation episode
        ad_ep = getattr(self, "_ad_current_episode", None)
        if ad_ep is not None and self._algorithm_distiller is not None:
            try:
                self._algorithm_distiller.finalize_episode(ad_ep, success=success)
            except Exception:
                pass

        # SOAR on_success callback
        if success and hasattr(self._gii, "on_operator_success"):
            try:
                focused_app = self._get_world_state().get("focused_app", "")
                self._gii.on_operator_success(
                    executed_operators=self._executed_operators,
                    focused_app=focused_app,
                )
            except Exception:
                pass

        return {
            "success": success,
            "reason": reason,
            "iterations": self._iteration,
            "stagnant_count": self._stagnant_count,
            "final_world_state": self._get_world_state(),
            "goal_progress": getattr(self._goal_repr, "progress", None),
        }
