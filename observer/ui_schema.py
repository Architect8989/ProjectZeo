from dataclasses import dataclass
from typing import Optional, List


@dataclass(frozen=True)
class UIElement:
    type: str                 # button, icon, dialog, input, text
    label: Optional[str]
    confidence: float         # 0.0 – 1.0
    enabled: Optional[bool] = None


@dataclass(frozen=True)
class UIDialog:
    title: Optional[str]
    message: Optional[str]
    severity: Optional[str]   # info, warning, error
    blocking: bool
    confidence: float


@dataclass(frozen=True)
class UIProgress:
    label: Optional[str]
    value: Optional[float]    # 0.0 – 1.0
    indeterminate: bool
    confidence: float


@dataclass(frozen=True)
class UISnapshot:
    elements: List[UIElement]
    dialogs: List[UIDialog]
    progress: List[UIProgress]
    stable: bool              # perceptual stability flag
