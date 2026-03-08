from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

_logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Broadcast proposal
# ─────────────────────────────────────────────────────────────────────────────

class ModuleType(str, Enum):
    PERCEPTION  = "perception"
    PLANNING    = "planning"
    MEMORY      = "memory"
    SAFETY      = "safety"
    MOTOR       = "motor"
    REFLECTION  = "reflection"


@dataclass
class BroadcastProposal:
    """A module's bid to broadcast to the global workspace."""
    module_type:  ModuleType
    content:      Dict[str, Any]          # What the module wants to broadcast
    activation:   float                   # 0.0-1.0; highest wins
    timestamp:    float = field(default_factory=time.time)
    metadata:     Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: "BroadcastProposal") -> bool:
        return self.activation < other.activation


@dataclass
class GlobalBroadcast:
    """Content broadcast to all modules after competition."""
    winner:       BroadcastProposal
    all_proposals: List[BroadcastProposal]
    cycle:        int
    broadcast_ts: float = field(default_factory=time.time)


# ─────────────────────────────────────────────────────────────────────────────
# Module base class
# ─────────────────────────────────────────────────────────────────────────────

class WorkspaceModule:
    

    module_type: ModuleType = ModuleType.PERCEPTION

    def propose(self, workspace_state: Dict[str, Any]) -> Optional[BroadcastProposal]:
        """Generate a proposal for the global workspace. Return None to abstain."""
        return None

    def receive(self, broadcast: GlobalBroadcast) -> None:
        """Process a broadcast from the global workspace."""
        pass

    def shutdown(self) -> None:
        """Clean up resources."""
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Concrete modules
# ─────────────────────────────────────────────────────────────────────────────

class PerceptionModule(WorkspaceModule):
    """Qwen3-VL perception: generates screen observations."""

    module_type = ModuleType.PERCEPTION

    def __init__(self, vision_runtime=None) -> None:
        self._vision = vision_runtime
        self._last_observation: Optional[Dict] = None
        self._screenshot = None

    def update_screenshot(self, screenshot) -> None:
        self._screenshot = screenshot

    def propose(self, workspace_state: Dict[str, Any]) -> Optional[BroadcastProposal]:
        if self._vision is None or self._screenshot is None:
            return None

        try:
            obs = self._vision.capture_and_analyze(self._screenshot)
            self._last_observation = obs
            # Perception activation: higher if new entities discovered
            prev_count = workspace_state.get("entity_count", 0)
            curr_count = len(obs.get("entities", []))
            activation = 0.85 if curr_count != prev_count else 0.65
            return BroadcastProposal(
                module_type=ModuleType.PERCEPTION,
                content={"observation": obs, "entity_count": curr_count},
                activation=activation,
            )
        except Exception as exc:
            _logger.debug("[GWS:Perception] propose error: %s", exc)
            return None

    def receive(self, broadcast: GlobalBroadcast) -> None:
        pass


class MemoryModule(WorkspaceModule):
    """OpenMemory retrieval: surfaces relevant memories for current context."""

    module_type = ModuleType.MEMORY

    def __init__(self, openmemory_store=None) -> None:
        self._memory = openmemory_store

    def propose(self, workspace_state: Dict[str, Any]) -> Optional[BroadcastProposal]:
        if self._memory is None:
            return None

        objective = workspace_state.get("objective", "")
        focused_app = workspace_state.get("focused_app", "")

        try:
            query = f"{objective} {focused_app}".strip()
            memories = self._memory.retrieve(query=query, top_k=5)
            if not memories:
                return None
            return BroadcastProposal(
                module_type=ModuleType.MEMORY,
                content={"memories": [m.content for m in memories[:5]]},
                activation=0.60,
            )
        except Exception as exc:
            _logger.debug("[GWS:Memory] propose error: %s", exc)
            return None

    def receive(self, broadcast: GlobalBroadcast) -> None:
        # Store successful action patterns if motor module won
        if broadcast.winner.module_type == ModuleType.MOTOR:
            action = broadcast.winner.content.get("action", {})
            if action and self._memory:
                try:
                    self._memory.store_episodic(
                        content=f"Executed action: {action}",
                        subject=broadcast.winner.metadata.get("app", "unknown"),
                        importance=0.5,
                    )
                except Exception:
                    pass


class ReflectionModule(WorkspaceModule):
    """GoalRepresentation evaluator: monitors goal progress."""

    module_type = ModuleType.REFLECTION

    def __init__(self, goal_repr=None) -> None:
        self._goal = goal_repr

    def propose(self, workspace_state: Dict[str, Any]) -> Optional[BroadcastProposal]:
        if self._goal is None:
            return None

        try:
            progress = self._goal.progress
            is_complete = self._goal.is_complete

            # High activation if goal is complete or stalled
            if is_complete:
                activation = 0.99  # highest — goal done, must broadcast
            elif progress > 0.0:
                activation = 0.55
            else:
                activation = 0.40

            return BroadcastProposal(
                module_type=ModuleType.REFLECTION,
                content={
                    "goal_progress":    progress,
                    "is_complete":      is_complete,
                    "progress_summary": self._goal.progress_summary,
                    "next_pending":     getattr(self._goal.next_pending(), "description", ""),
                },
                activation=activation,
            )
        except Exception as exc:
            _logger.debug("[GWS:Reflection] propose error: %s", exc)
            return None

    def receive(self, broadcast: GlobalBroadcast) -> None:
        # Update goal evaluation when new observation arrives
        if broadcast.winner.module_type == ModuleType.PERCEPTION and self._goal:
            obs = broadcast.winner.content.get("observation", {})
            if obs:
                try:
                    self._goal.evaluate_from_screen(obs)
                except Exception:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
# GlobalWorkspace
# ─────────────────────────────────────────────────────────────────────────────

_COLLECTION_WINDOW_SECONDS = 0.15   # time to collect proposals before arbitrating
_MAX_CYCLE_RATE_HZ         = 10     # cap at 10 arbitration cycles per second


class GlobalWorkspace:
    

    def __init__(self, objective: str = "") -> None:
        self._objective = objective
        self._modules:  List[WorkspaceModule] = []
        self._cycle:    int = 0
        self._lock:     threading.RLock = threading.RLock()
        self._last_broadcast: Optional[GlobalBroadcast] = None
        self._last_cycle_ts:  float = 0.0
        self._workspace_state: Dict[str, Any] = {"objective": objective}

        _logger.info("[GlobalWorkspace] Initialized. objective=%r", objective[:60])

    def register(self, module: WorkspaceModule) -> None:
        """Register a specialist module."""
        with self._lock:
            self._modules.append(module)
            _logger.debug("[GlobalWorkspace] Registered module: %s", module.module_type)

    def unregister(self, module_type: ModuleType) -> None:
        with self._lock:
            self._modules = [m for m in self._modules
                             if m.module_type != module_type]

    def update_workspace_state(self, updates: Dict[str, Any]) -> None:
        """Inject external state into workspace (e.g., world_state from GII loop)."""
        with self._lock:
            self._workspace_state.update(updates)

    def run_cycle(
        self,
        external_state: Optional[Dict[str, Any]] = None,
    ) -> Optional[GlobalBroadcast]:
        
        # Rate limiting
        now = time.time()
        min_interval = 1.0 / _MAX_CYCLE_RATE_HZ
        elapsed = now - self._last_cycle_ts
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        with self._lock:
            self._cycle += 1
            cycle = self._cycle
            state = dict(self._workspace_state)
            if external_state:
                state.update(external_state)
            modules = list(self._modules)

        if not modules:
            return None

        # Collect proposals in parallel
        proposals: List[Optional[BroadcastProposal]] = [None] * len(modules)
        threads = []

        def _propose(i: int, module: WorkspaceModule) -> None:
            try:
                proposals[i] = module.propose(state)
            except Exception as exc:
                _logger.debug("[GlobalWorkspace] Module %s propose error: %s",
                              module.module_type, exc)

        for i, module in enumerate(modules):
            t = threading.Thread(target=_propose, args=(i, module), daemon=True)
            threads.append(t)
            t.start()

        # Collect within window
        for t in threads:
            t.join(timeout=_COLLECTION_WINDOW_SECONDS)

        valid = [p for p in proposals if p is not None]
        if not valid:
            _logger.debug("[GlobalWorkspace] Cycle %d: no proposals.", cycle)
            return None

        # Arbitrate: highest activation wins
        winner = max(valid, key=lambda p: p.activation)

        broadcast = GlobalBroadcast(
            winner=winner,
            all_proposals=valid,
            cycle=cycle,
        )
        self._last_cycle_ts = time.time()

        _logger.debug(
            "[GlobalWorkspace] Cycle %d: winner=%s activation=%.2f",
            cycle, winner.module_type, winner.activation,
        )

        # Broadcast to all modules in parallel
        def _recv(module: WorkspaceModule) -> None:
            try:
                module.receive(broadcast)
            except Exception as exc:
                _logger.debug("[GlobalWorkspace] Module %s receive error: %s",
                              module.module_type, exc)

        recv_threads = [
            threading.Thread(target=_recv, args=(m,), daemon=True)
            for m in modules
        ]
        for t in recv_threads:
            t.start()
        for t in recv_threads:
            t.join(timeout=_COLLECTION_WINDOW_SECONDS)

        with self._lock:
            self._last_broadcast = broadcast

        return broadcast

    def get_last_broadcast(self) -> Optional[GlobalBroadcast]:
        with self._lock:
            return self._last_broadcast

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "cycle":           self._cycle,
                "modules":         [m.module_type.value for m in self._modules],
                "objective":       self._objective[:60],
                "last_winner":     (
                    self._last_broadcast.winner.module_type.value
                    if self._last_broadcast else None
                ),
            }

    def shutdown(self) -> None:
        with self._lock:
            for m in self._modules:
                try:
                    m.shutdown()
                except Exception:
                    pass
            self._modules.clear()
