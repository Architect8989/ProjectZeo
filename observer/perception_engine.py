from typing import Dict
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
    - FAIL-CLOSED semantics
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
                    label="screen available",
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
    # VERIFICATION (FAIL-CLOSED)
    # --------------------------------------------------

    def verify_task_completion(
        self,
        pre_state: Dict[str, object],
        post_state: Dict[str, object],
        *,
        expect_change: bool = True,
        semantic_proof: bool = False,
    ) -> bool:
        """
        Evidence-based verification.

        HARD RULES:
        - Timestamp change is NOT evidence
        - Verification REQUIRES explicit semantic proof
        - Health degradation aborts verification
        """

        start = time.monotonic()
        self._reset_verification_state()

        # ---- HEALTH GATE ----
        if self.health.degraded():
            self.last_verification_reason = "perception health degraded"
            raise PerceptionVerificationError(
                self.last_verification_reason
            )

        # ---- AVAILABILITY GATE ----
        if not pre_state.get("available") or not post_state.get("available"):
            self.last_verification_reason = (
                "screen unavailable during verification"
            )
            raise PerceptionVerificationError(
                self.last_verification_reason
            )

        # ---- SEMANTIC EVIDENCE GATE ----
        if expect_change and not semantic_proof:
            self.last_verification_reason = (
                "no semantic evidence of UI change"
            )
            raise PerceptionVerificationError(
                self.last_verification_reason
            )

        # ---- LATENCY BOUND ----
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
