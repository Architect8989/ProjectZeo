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


def _capture_frame() -> Optional[tuple]:
    
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
        except Exception:
            return None
    except Exception:
        return None

    return hashlib.sha256(raw).hexdigest(), raw


# Keep the old single-return-value wrapper for callers that only need the hash.
def _capture_frame_hash() -> Optional[str]:
    result = _capture_frame()
    return result[0] if result is not None else None


def _pixel_diff_ratio(hash_a: str, hash_b: str, raw_a: Optional[bytes] = None, raw_b: Optional[bytes] = None) -> float:
    
    if not hash_a or not hash_b:
        return 0.0
    if hash_a == hash_b:
        return 0.0

    # Strategy 1: true per-pixel diff (raw bytes available)
    if raw_a is not None and raw_b is not None and len(raw_a) == len(raw_b) and raw_a:
        try:
            # Count bytes that differ; divide by total to get fraction.
            # Each pixel is 3 bytes (RGB) so this is a slight over-count of
            # unique changed pixels but it's conservative (better to re-reason).
            n_diff = sum(1 for a, b in zip(raw_a, raw_b) if a != b)
            return n_diff / len(raw_a)
        except Exception:
            pass  # Fall through to hash proxy

    # Strategy 2: Hamming on SHA-256 hex chars (proxy)
    diff = sum(1 for a, b in zip(hash_a, hash_b) if a != b)
    return diff / max(len(hash_a), 1)


class PreDispatchScreenChecker:
    

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
        self._reasoning_frame_raw: Optional[bytes] = None   # for true pixel diff
        self._reasoning_frame_ts: Optional[float] = None

        # Stats
        self._total_checks = 0
        self._changes_detected = 0

    def record_reasoning_frame(self) -> None:
        
        if not self._enabled:
            return

        result = _capture_frame()
        if result is None:
            return

        h, raw = result
        with self._lock:
            self._reasoning_frame_hash = h
            self._reasoning_frame_raw = raw
            self._reasoning_frame_ts = time.monotonic()

    def check_screen_changed(
        self,
        *,
        proposed_operation: str = "",
    ) -> Tuple[bool, float]:
        
        if not self._enabled:
            return False, 0.0

        if proposed_operation.lower() in _SKIP_CHECK_OPS:
            return False, 0.0

        with self._lock:
            baseline_hash = self._reasoning_frame_hash
            baseline_raw = self._reasoning_frame_raw
            baseline_ts = self._reasoning_frame_ts

        if baseline_hash is None:
            return False, 0.0

        current = _capture_frame()
        if current is None:
            return False, 0.0

        current_hash, current_raw = current
        diff = _pixel_diff_ratio(
            baseline_hash, current_hash,
            raw_a=baseline_raw,
            raw_b=current_raw,
        )

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
                f"diff={diff:.4f} > threshold={self._threshold:.4f} "
                f"(reasoning frame age={age_s:.1f}s op={proposed_operation!r}). "
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
