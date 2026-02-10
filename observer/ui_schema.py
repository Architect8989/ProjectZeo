from dataclasses import dataclass
from typing import Optional, FrozenSet, Dict


# --------------------------------------------------
# CORE UI PRIMITIVES (EVIDENCE CARRIERS)
# --------------------------------------------------

@dataclass(frozen=True)
class UIElement:
    """
    Atomic semantic UI unit.

    Evidence-only. No intent. No actionability.
    """
    id: str                          # stable semantic identifier (REQUIRED)
    type: str                        # button, icon, dialog, input, text
    text: Optional[str]              # visible label/text (raw)
    confidence: float                # 0.0 – 1.0
    enabled: Optional[bool] = None

    # Optional spatial grounding (NORMALIZED 0.0–1.0)
    x: Optional[float] = None
    y: Optional[float] = None

    def __post_init__(self):
        if not self.id or not isinstance(self.id, str):
            raise ValueError("UIElement.id must be a non-empty string")

        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("UIElement.confidence out of bounds")

        if self.x is not None and not (0.0 <= self.x <= 1.0):
            raise ValueError("UIElement.x must be normalized (0.0–1.0)")

        if self.y is not None and not (0.0 <= self.y <= 1.0):
            raise ValueError("UIElement.y must be normalized (0.0–1.0)")


@dataclass(frozen=True)
class UIDialog:
    """
    Modal or non-modal dialog evidence.
    """
    id: str
    title: Optional[str]
    message: Optional[str]
    severity: Optional[str]          # info, warning, error
    blocking: bool
    confidence: float

    def __post_init__(self):
        if not self.id:
            raise ValueError("UIDialog.id must be non-empty")

        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("UIDialog.confidence out of bounds")


@dataclass(frozen=True)
class UIProgress:
    """
    Progress indicator evidence.
    """
    id: str
    label: Optional[str]
    value: Optional[float]           # 0.0 – 1.0
    indeterminate: bool
    confidence: float

    def __post_init__(self):
        if not self.id:
            raise ValueError("UIProgress.id must be non-empty")

        if self.value is not None and not (0.0 <= self.value <= 1.0):
            raise ValueError("UIProgress.value out of bounds")

        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("UIProgress.confidence out of bounds")


# --------------------------------------------------
# SNAPSHOT (FRAME-LEVEL EVIDENCE)
# --------------------------------------------------

@dataclass(frozen=True)
class UISnapshot:
    """
    Immutable semantic snapshot of perceived UI state.

    HARD RULES:
    - Order-independent
    - No inference
    - No intent
    - Evidence-only
    """
    elements: FrozenSet[UIElement]
    dialogs: FrozenSet[UIDialog]
    progress: FrozenSet[UIProgress]

    stable: bool                     # perceptual stability flag

    # Non-authoritative raw surface (never used for decisions)
    evidence: Dict[str, object]

    def __post_init__(self):
        if not isinstance(self.elements, frozenset):
            raise TypeError("elements must be FrozenSet[UIElement]")

        if not isinstance(self.dialogs, frozenset):
            raise TypeError("dialogs must be FrozenSet[UIDialog]")

        if not isinstance(self.progress, frozenset):
            raise TypeError("progress must be FrozenSet[UIProgress]")
