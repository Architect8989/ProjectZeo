from typing import Dict, Set, Tuple, List
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
    - Deterministic semantic verification
    - Constructive evidence (what changed, why)
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
        """
        Returns the authoritative UISnapshot produced upstream.
        PerceptionEngine does NOT enrich or mutate snapshots.
        """
        available = bool(screen_state.get("available", False))
        frame_ts = screen_state.get("frame_ts")

        self.health.update(frame_ts, available)

        snapshot = screen_state.get("snapshot")
        if not isinstance(snapshot, UISnapshot):
            raise PerceptionVerificationError("missing UISnapshot in screen_state")

        return snapshot

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
        - Timestamp is NEVER evidence
        - Delta is computed internally
        - No delta when change expected == failure
        - Ambiguity == failure
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

        if not isinstance(pre_snapshot, UISnapshot):
            self._fail("pre_state missing valid UISnapshot")
        if not isinstance(post_snapshot, UISnapshot):
            self._fail("post_state missing valid UISnapshot")

        changed, evidence = self._semantic_delta(pre_snapshot, post_snapshot)

        if expect_change and not changed:
            self._fail("no provable semantic UI change detected")

        # ---- LATENCY BOUND ----
        if (time.monotonic() - start) > self.MAX_VERIFICATION_LATENCY_SECONDS:
            self._fail("verification timeout")

        self.last_verification_ts = time.monotonic()
        self.last_verification_reason = "verified"
        self.last_verification_evidence = evidence

        return True

    # --------------------------------------------------
    # SEMANTIC DIFF (CONSTRUCTIVE)
    # --------------------------------------------------

    def _semantic_delta(
        self, pre: UISnapshot, post: UISnapshot
    ) -> Tuple[bool, Dict[str, object]]:
        """
        Computes a deterministic semantic delta.

        Returns:
            (changed: bool, evidence: dict)
        """

        evidence: Dict[str, object] = {}

        # ---- STABILITY CHANGE ----
        if pre.stable != post.stable:
            evidence["stable_changed"] = {
                "from": pre.stable,
                "to": post.stable,
            }

        # ---- STRUCTURAL COUNTS ----
        if len(pre.elements) != len(post.elements):
            evidence["elements_count"] = (len(pre.elements), len(post.elements))

        if len(pre.dialogs) != len(post.dialogs):
            evidence["dialogs_count"] = (len(pre.dialogs), len(post.dialogs))

        if len(pre.progress) != len(post.progress):
            evidence["progress_count"] = (len(pre.progress), len(post.progress))

        # ---- TEXTUAL SEMANTICS ----
        pre_text = self._extract_text(pre)
        post_text = self._extract_text(post)

        added = post_text - pre_text
        removed = pre_text - post_text

        if added:
            evidence["text_added"] = sorted(added)
        if removed:
            evidence["text_removed"] = sorted(removed)

        return bool(evidence), evidence

    def _extract_text(self, snap: UISnapshot) -> Set[str]:
        texts: Set[str] = set()

        for el in snap.elements:
            if isinstance(el.label, str) and el.label.strip():
                texts.add(el.label.strip())

        for dlg in snap.dialogs:
            if isinstance(dlg.title, str) and dlg.title.strip():
                texts.add(dlg.title.strip())
            if isinstance(dlg.message, str) and dlg.message.strip():
                texts.add(dlg.message.strip())

        for prog in snap.progress:
            if isinstance(prog.label, str) and prog.label.strip():
                texts.add(prog.label.strip())

        return texts

    # --------------------------------------------------
    # FORENSICS
    # --------------------------------------------------

    def get_verification_snapshot(self) -> Dict[str, object]:
        return {
            "last_verification_ts": self.last_verification_ts,
            "last_verification_reason": self.last_verification_reason,
            "last_verification_evidence": getattr(
                self, "last_verification_evidence", None
            ),
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
        self.last_verification_evidence = None
