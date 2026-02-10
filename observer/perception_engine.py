from typing import Dict, Set
import time
from observer.ui_schema import UISnapshot
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

        return screen_state.get("snapshot")  # authoritative snapshot only

    # --------------------------------------------------
    # VERIFICATION (SEMANTIC, FAIL-CLOSED)
    # --------------------------------------------------

    def verify_task_completion(
        self,
        pre_state: Dict[str, object],
        post_state: Dict[str, object],
        *,
        expect_change: bool = True,
    ) -> bool:
        """
        Evidence-based semantic verification.

        HARD RULES:
        - Timestamp change is NOT evidence
        - Semantic delta MUST be computed internally
        - No delta when change expected == failure
        - Health degradation aborts verification
        """

        start = time.monotonic()
        self._reset_verification_state()

        # ---- HEALTH GATE ----
        if self.health.degraded():
            self._fail("perception health degraded")

        # ---- AVAILABILITY GATE ----
        if not pre_state.get("available") or not post_state.get("available"):
            self._fail("screen unavailable during verification")

        pre_snapshot = pre_state.get("snapshot")
        post_snapshot = post_state.get("snapshot")

        if not isinstance(pre_snapshot, UISnapshot) or not isinstance(
            post_snapshot, UISnapshot
        ):
            self._fail("invalid or missing UISnapshot")

        semantic_changed = self._semantic_delta(
            pre_snapshot, post_snapshot
        )

        if expect_change and not semantic_changed:
            self._fail("no semantic UI change detected")

        # ---- LATENCY BOUND ----
        if (time.monotonic() - start) > self.MAX_VERIFICATION_LATENCY_SECONDS:
            self._fail("verification timeout")

        self.last_verification_ts = time.monotonic()
        self.last_verification_reason = "verified"
        return True

    # --------------------------------------------------
    # SEMANTIC DIFF (DETERMINISTIC)
    # --------------------------------------------------

    def _semantic_delta(
        self, pre: UISnapshot, post: UISnapshot
    ) -> bool:
        """
        Returns True iff a provable semantic change occurred.
        """

        if pre.stable != post.stable:
            return True

        if len(pre.elements) != len(post.elements):
            return True

        if len(pre.dialogs) != len(post.dialogs):
            return True

        if len(pre.progress) != len(post.progress):
            return True

        pre_text = self._extract_text(pre)
        post_text = self._extract_text(post)

        return pre_text != post_text

    def _extract_text(self, snap: UISnapshot) -> Set[str]:
        texts = set()
        for el in snap.elements:
            if el.label:
                texts.add(el.label)
        for dlg in snap.dialogs:
            if dlg.title:
                texts.add(dlg.title)
            if dlg.message:
                texts.add(dlg.message)
        for prog in snap.progress:
            if prog.label:
                texts.add(prog.label)
        return texts

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

    def _fail(self, reason: str):
        self.last_verification_reason = reason
        raise PerceptionVerificationError(reason)

    def _reset_verification_state(self) -> None:
        self.last_verification_ts = None
        self.last_verification_reason = None
