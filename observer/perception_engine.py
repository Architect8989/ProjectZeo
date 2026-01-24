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
    """

    # === ADDITIONS ===
    MAX_VERIFICATION_LATENCY_SECONDS = 1.0

    def __init__(self):
        self.health = PerceptionHealth()

        # === ADDITIONS ===
        self.last_verification_reason: Optional[str] = None
        self.last_verification_ts: Optional[float] = None

    def process(self, screen_state: Dict[str, object]) -> UISnapshot:
        available = screen_state.get("available", False)
        frame_ts = screen_state.get("frame_ts")

        stable = self.health.update(frame_ts, available)

        elements = []
        dialogs = []
        progress = []

        # Phase 2B rule: Describe only what Screenpipe already provides
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

    # === ADDITIONS ===
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
        - If a task expects UI change, hashes or timestamps must differ
        - If no change expected, states must remain stable
        - Verification must occur within bounded time
        """

        self.last_verification_ts = time.monotonic()

        pre_available = pre_state.get("available")
        post_available = post_state.get("available")

        if not pre_available or not post_available:
            self.last_verification_reason = "screen unavailable during verification"
            raise PerceptionVerificationError(self.last_verification_reason)

        pre_hash = pre_state.get("screen_text_hash")
        post_hash = post_state.get("screen_text_hash")

        pre_ts = pre_state.get("frame_ts")
        post_ts = post_state.get("frame_ts")

        # Temporal sanity
        if pre_ts is None or post_ts is None:
            self.last_verification_reason = "missing frame timestamp"
            raise PerceptionVerificationError(self.last_verification_reason)

        if post_ts < pre_ts:
            self.last_verification_reason = "frame time went backwards"
            raise PerceptionVerificationError(self.last_verification_reason)

        # Expectation enforcement
        if expect_change:
            if pre_hash == post_hash:
                self.last_verification_reason = "expected UI change did not occur"
                raise PerceptionVerificationError(self.last_verification_reason)
        else:
            if pre_hash != post_hash:
                self.last_verification_reason = "unexpected UI change detected"
                raise PerceptionVerificationError(self.last_verification_reason)

        self.last_verification_reason = "verified"
        return True

    # === ADDITIONS ===
    def get_verification_snapshot(self) -> Dict[str, object]:
        """
        Forensic snapshot of last verification.
        """
        return {
            "last_verification_ts": self.last_verification_ts,
            "last_verification_reason": self.last_verification_reason,
            "health_degraded": self.health.degraded(),
                                     }
