from typing import Dict, Optional
import time
from observer.ui_schema import UISnapshot, UIElement, UIDialog, UIProgress
from observer.self_healing import PerceptionHealth


class PerceptionVerificationError(RuntimeError):
    """Raised when perception verification fails."""


class PerceptionEngine:
    """
    Semantic perception engine.
    READ-ONLY. NO DECISIONS. NO ACTIONS.

    Guarantees:
    - Deterministic verification
    - Bounded latency
    - No cross-task state leakage
    """

    MAX_VERIFICATION_LATENCY_SECONDS = 1.0

    def __init__(self):
        self.health = PerceptionHealth()
        self._reset_verification_state()

    # --------------------------------------------------
    # CORE PERCEPTION
    # --------------------------------------------------

    def process(self, screen_state: Dict[str, object]) -> UISnapshot:
        available = bool(screen_state.get("available", False))
        frame_ts = screen_state.get("frame_ts")

        stable = self.health.update(frame_ts, available)

        elements = []
        dialogs = []
        progress = []

        if available:
            elements.append(
                UIElement(
                    type="text",
                    label="screen content present",
                    confidence=0.6,
                )
            )

        if self.health.degraded():
            dialogs.append(
                UIDialog(
                    title="Perception degraded",
                    message="Screen observation unstable",
                    severity="warning",
                    blocking=False,
                    confidence=0.9,
                )
            )

        return UISnapshot(
            elements=elements,
            dialogs=dialogs,
            progress=progress,
            stable=stable,
        )

    # --------------------------------------------------
    # VERIFICATION
    # --------------------------------------------------

    def verify_task_completion(
        self,
        pre_state: Dict[str, object],
        post_state: Dict[str, object],
        *,
        expect_change: bool = True,
    ) -> bool:
        """
        Evidence-based verification.

        Rules:
        - Verification must complete within bounded time
        - Perception health must not be degraded
        - Change can be proven by hash OR frame timestamp
        """

        start = time.monotonic()
        self._reset_verification_state()

        if self.health.degraded():
            self.last_verification_reason = "perception health degraded"
            raise PerceptionVerificationError(
                self.last_verification_reason
            )

        pre_available = bool(pre_state.get("available"))
        post_available = bool(post_state.get("available"))

        if not pre_available or not post_available:
            self.last_verification_reason = (
                "screen unavailable during verification"
            )
            raise PerceptionVerificationError(
                self.last_verification_reason
            )

        pre_hash = pre_state.get("screen_text_hash")
        post_hash = post_state.get("screen_text_hash")
        pre_ts = pre_state.get("frame_ts")
        post_ts = post_state.get("frame_ts")

        if pre_ts is None or post_ts is None:
            self.last_verification_reason = "missing frame timestamp"
            raise PerceptionVerificationError(
                self.last_verification_reason
            )

        if post_ts < pre_ts:
            self.last_verification_reason = "frame time regression"
            raise PerceptionVerificationError(
                self.last_verification_reason
            )

        changed = (pre_hash != post_hash) or (post_ts > pre_ts)

        if expect_change and not changed:
            self.last_verification_reason = (
                "expected UI change not observed"
            )
            raise PerceptionVerificationError(
                self.last_verification_reason
            )

        if not expect_change and changed:
            self.last_verification_reason = (
                "unexpected UI change observed"
            )
            raise PerceptionVerificationError(
                self.last_verification_reason
            )

        latency = time.monotonic() - start
        if latency > self.MAX_VERIFICATION_LATENCY_SECONDS:
            self.last_verification_reason = "verification timeout"
            raise PerceptionVerificationError(
                self.last_verification_reason
            )

        self.last_verification_ts = time.monotonic()
        self.last_verification_reason = "verified"
        return True

    # --------------------------------------------------
    # FORENSICS
    # --------------------------------------------------

    def get_verification_snapshot(self) -> Dict[str, object]:
        return {
            "last_verification_ts": self.last_verification_ts,
            "last_verification_reason": self.last_verification_reason,
            "health_degraded": self.health.degraded(),
        }

    # --------------------------------------------------
    # INTERNAL
    # --------------------------------------------------

    def _reset_verification_state(self) -> None:
        self.last_verification_ts = None
        self.last_verification_reason = None
