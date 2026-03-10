"""
core/learning/lats_planner.py
==============================
Re-export shim for LATSPlanner.

The canonical implementation lives in core/planner/lats_planner.py
(Blueprint §7.4 — Language Agent Tree Search, Zhou et al. ICML 2024).

gii_controller.py references core.learning.lats_planner:
    from core.learning.lats_planner import LATSPlanner

This file satisfies that import while keeping the implementation in the
correct planner package. Both paths are now valid.

Usage in GII:
  - gii_controller._on_task_complete() generates AgentQ DPO pairs via LATS
  - gii_loop.py triggers LATS recovery at stagnant_count >= 3
  - DPO pairs fed to AgentQStore for preference-based fine-tuning
"""
from __future__ import annotations

from core.planner.lats_planner import (  # noqa: F401
    LATSPlanner,
    LATSNode,
    LATSResult,
    NodeStatus,
    DPOPair,
)

__all__ = ["LATSPlanner", "LATSNode", "LATSResult", "NodeStatus", "DPOPair"]
