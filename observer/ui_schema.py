from dataclasses import dataclass
from typing import Optional, List, Dict


# --------------------------------------------------
# CORE UI PRIMITIVES (EVIDENCE CARRIERS)
# --------------------------------------------------

@dataclass(frozen=True)
class UIElement:
    """
    Atomic semantic UI unit.

    NOTE:
    - This is NOT an action target
    - This is NOT inferred intent
    - This is evidence-only description
    """
    id: str                          # stable semantic identifier
    type: str                        # button, icon, dialog, input, text
    text: Optional[str]              # visible label/text (raw)
    confidence: float                # confidence this element exists
    enabled: Optional[bool] = None

    # Optional spatial grounding (normalized)
    x: Optional[float] = None
    y: Optional[float] = None


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


# --------------------------------------------------
# SNAPSHOT (FRAME-LEVEL EVIDENCE)
# --------------------------------------------------

@dataclass(frozen=True)
class UISnapshot:
    """
    Immutable semantic snapshot of perceived UI state.

    HARD RULES:
    - No inference
    - No intent
    - No temporal assumptions
    - Evidence only
    """
    elements: List[UIElement]
    dialogs: List[UIDialog]
    progress: List[UIProgress]

    stable: bool                     # perceptual stability flag
    evidence: Dict[str, object]      # raw evidence surface (text, hashes, etc.)
