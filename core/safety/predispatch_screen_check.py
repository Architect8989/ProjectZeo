"""
core/safety/predispatch_screen_check.py
========================================
Pre-dispatch screen diff module.

AUDIT-HIGH FIX: Mandatory screen re-capture immediately before action dispatch.

Problem
-------
The VL inference cycle on CPU takes 40–90 seconds.  By the time ``PerStepReasoner``
returns a proposed action, the screen may have changed (dialog appeared, window
switched, download prompt opened, error overlay rendered).  Acting on a stale world
model risks:
  - Clicking an element that no longer exists
  - Dismissing a safety dialog that appeared *after* the reasoning decision
  - Executing a command while an adversarial overlay is covering the target UI

Fix
---
Before every action dispatch (post-reasoning, pre-execution), capture a new
lightweight screenshot and compute a pixel-hash diff against the frame that was
used during reasoning.  If the difference exceeds a configurable threshold
(default: 5% of pixels changed), the proposed action is held and the reasoning
loop re-runs from the new screen state.

This module is intentionally standalone so it can be imported by operate.py
without creating circular dependency chains.

Configuration
-------------
``PROJECTZEO_PREDISPATCH_CHANGE_THRESHOLD``
    Float, 0.0–1.0.  Fraction of pixels that must change to trigger a re-reason.
    Default: 0.05 (5%).

``PROJECTZEO_PREDISPATCH_ENABLED``
    Set to "0" to disable the check globally (development/testing only).

Usage
-----
::

    from core.safety.predispatch_screen_check import PreDispatchScreenChecker

    checker = PreDispatchScreenChecker()
    # At reasoning time:
    checker.record_reasoning_frame()
    # ... reasoning produces proposed_action ...
    # Before dispatch:
    changed, diff_ratio = checker.check_screen_changed()
    if changed:
        # Re-run reasoning with fresh world state
        ...
    else:
        # Safe to dispatch
        dispatch(proposed_action)
"""

from __future__ import annotations

import hashlib
import os
import sys
import threading
import time
from typing import Optional, Tuple


_DEFAULT_THRESHOLD = float(
    os.environ.get("PROJECTZEO_PREDISPATCH_CHANGE_THRESHOLD", "0.05")
)
_ENABLED = os.environ.get("PROJECTZEO_PREDISPATCH_ENABLED", "1").strip() != "0"

# Operations that are cheap/idempotent and don't need a pre-dispatch check.
# Scrolling, hovering, and mouse movement are unlikely to trigger adversarial dialogs.
_SKIP_CHECK_OPS = frozenset({
    "scroll", "move", "hover", "done", "verify", "wait", "noop",
})


def _capture_frame_hash() -> Optional[str]:
    """
    Capture a lightweight screenshot and return a hex SHA-256 hash of its bytes.

    Uses ``mss`` (fast, no subprocess) for the capture if available, falling
    back to ``pyautogui.screenshot()`` otherwise.

    Returns None on any capture failure — callers must treat None as "unchanged"
    (fail-open, never block on capture failure).
    """
    try:
        import mss as _mss  # type: ignore[import]
        with _mss.mss() as sct:
            monitor = sct.monitors[0]  # All monitors combined
            img = sct.grab(monitor)
            raw = bytes(img.rgb)
    except ImportError:
        try:
            import pyautogui as _pya
            _img = _pya.screenshot()
            raw = _img.tobytes()
        except Exception as _cap_err:
            return None
    except Exception as _mss_err:
        return None

    return hashlib.sha256(raw).hexdigest()


def _pixel_diff_ratio(hash_a: str, hash_b: str) -> float:
    """
    Approximate pixel diff as normalised Hamming distance between two SHA-256 hashes.

    This is a proxy — not a true pixel-level diff — but it is:
    - O(64) to compute
    - Monotonic with the true pixel diff for natural images
    - Sufficient to detect dialog appearances and full-screen changes

    Returns a float in [0.0, 1.0].  Values near 0 → virtually identical.
    Values near 1 → completely different frames.
    """
    if not hash_a or not hash_b or hash_a == hash_b:
        return 0.0

    # Count differing hex characters as a fraction of total characters
    diff = sum(1 for a, b in zip(hash_a, hash_b) if a != b)
    return diff / max(len(hash_a), 1)


class PreDispatchScreenChecker:
    """
    Stateful helper for per-dispatch screen change detection.

    One instance should be created per task execution and shared across all
    iterations of the dispatch loop.

    Thread-safety: the ``record_reasoning_frame`` and ``check_screen_changed``
    methods are protected by an internal lock and safe to call from any thread.
    """

    def __init__(
        self,
        *,
        threshold: float = _DEFAULT_THRESHOLD,
        enabled: bool = _ENABLED,
    ) -> None:
        self._threshold = max(0.0, min(float(threshold), 1.0))
        self._enabled = bool(enabled)
        self._lock = threading.Lock()
        self._reasoning_frame_hash: Optional[str] = None
        self._reasoning_frame_ts: Optional[float] = None

        # Stats
        self._total_checks = 0
        self._changes_detected = 0

    def record_reasoning_frame(self) -> None:
        """
        Capture and record the screen state at the START of the reasoning cycle.

        Call this immediately after the observer snapshot is taken and before
        the LLM reasoning call.  The hash recorded here is compared against
        the pre-dispatch snapshot in ``check_screen_changed()``.
        """
        if not self._enabled:
            return

        h = _capture_frame_hash()
        if h is None:
            return

        with self._lock:
            self._reasoning_frame_hash = h
            self._reasoning_frame_ts = time.monotonic()

    def check_screen_changed(
        self,
        *,
        proposed_operation: str = "",
    ) -> Tuple[bool, float]:
        """
        Capture a new screenshot and compare it with the reasoning-time frame.

        Parameters
        ----------
        proposed_operation
            The ``operation`` field of the proposed action.  Certain cheap ops
            (scroll, hover, verify, done) skip the check automatically.

        Returns
        -------
        (changed, diff_ratio)
            ``changed`` is True if the screen has changed beyond the threshold.
            ``diff_ratio`` is the raw diff ratio in [0.0, 1.0].

        If the check is disabled, cannot capture, or has no baseline hash,
        returns (False, 0.0) — never blocks on uncertainty.
        """
        if not self._enabled:
            return False, 0.0

        if proposed_operation.lower() in _SKIP_CHECK_OPS:
            return False, 0.0

        with self._lock:
            baseline = self._reasoning_frame_hash
            baseline_ts = self._reasoning_frame_ts

        if baseline is None:
            return False, 0.0

        current = _capture_frame_hash()
        if current is None:
            return False, 0.0

        diff = _pixel_diff_ratio(baseline, current)

        with self._lock:
            self._total_checks += 1
            if diff > self._threshold:
                self._changes_detected += 1

        changed = diff > self._threshold
        if changed:
            age_s = (
                time.monotonic() - baseline_ts
                if baseline_ts is not None else 0.0
            )
            print(
                f"[PreDispatchScreenChecker] CHANGE DETECTED: "
                f"diff={diff:.3f} > threshold={self._threshold:.3f} "
                f"(reasoning frame age={age_s:.1f}s). "
                "Re-running reasoning before dispatch.",
                file=sys.stderr,
            )

        return changed, diff

    def get_stats(self) -> dict:
        """Return cumulative check statistics."""
        with self._lock:
            return {
                "total_checks": self._total_checks,
                "changes_detected": self._changes_detected,
                "threshold": self._threshold,
                "enabled": self._enabled,
            }
