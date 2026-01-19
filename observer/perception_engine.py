from typing import Dict
from observer.ui_schema import UISnapshot, UIElement, UIDialog, UIProgress
from observer.self_healing import PerceptionHealth


class PerceptionEngine:
    """
    Semantic perception engine.
    READ-ONLY. NO DECISIONS. NO ACTIONS.
    """

    def __init__(self):
        self.health = PerceptionHealth()

    def process(self, screen_state: Dict[str, object]) -> UISnapshot:
        available = screen_state.get("available", False)
        frame_ts = screen_state.get("frame_ts")

        stable = self.health.update(frame_ts, available)

        elements = []
        dialogs = []
        progress = []

        # Phase 2B rule:
        # We describe only what Screenpipe already provides.
        # No guessing. No planning.

        if available:
            # Placeholder mapping — real OCR/UI extraction stays in Screenpipe
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
