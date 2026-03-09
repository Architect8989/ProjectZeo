"""
core/learning/agent_q.py

Agent Q: MCTS-based trajectory collection for DPO preference pair generation.

Reference: Putta et al., arXiv:2408.07199 — "Agent Q: Advanced Reasoning
and Learning for Autonomous AI Agents".

Collects best/worst rollout pairs from MCTS tree search. These pairs feed
the preference_generator.py which formats them for DPO fine-tuning.

Integration:
    - lats_planner.py exports DPO pairs via get_dpo_pairs()
    - agent_q.py collects those pairs and stores them persistently
    - nightly_consolidation.py reads the store and triggers DPO training
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

_STORE_DIR   = os.path.expanduser(os.environ.get("PROJECTZEO_AGENT_Q_DIR", "~/.projectzeo/agent_q"))
_MAX_PAIRS   = int(os.environ.get("PROJECTZEO_AGENT_Q_MAX_PAIRS", "500"))
_MIN_QUALITY = float(os.environ.get("PROJECTZEO_AGENT_Q_MIN_QUALITY", "0.3"))
_ENABLED     = os.environ.get("PROJECTZEO_AGENT_Q_ENABLED", "1").strip() != "0"


@dataclass
class TrajectoryNode:
    node_id:      str
    depth:        int
    observation:  str
    action:       Dict[str, Any]
    reward:       float
    visits:       int = 0
    children:     List[str] = field(default_factory=list)
    parent_id:    Optional[str] = None


@dataclass
class MCTSRollout:
    rollout_id:  str
    objective:   str
    app_name:    str
    nodes:       List[TrajectoryNode]
    final_reward: float
    outcome:     str  # "success" | "failure" | "partial"
    timestamp:   float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class PreferencePair:
    pair_id:     str
    objective:   str
    app_name:    str
    chosen:      List[Dict[str, Any]]    # winning trajectory steps
    rejected:    List[Dict[str, Any]]    # losing trajectory steps
    quality:     float                   # abs(chosen_reward - rejected_reward)
    timestamp:   float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _make_id(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class AgentQ:
    """
    Collects MCTS rollouts, identifies best/worst pairs per task,
    and persists them for DPO training.
    """

    def __init__(self, store_dir: Optional[str] = None) -> None:
        self._dir  = store_dir or _STORE_DIR
        self._lock = threading.Lock()
        self._pairs_this_session: List[PreferencePair] = []
        os.makedirs(self._dir, exist_ok=True)
        _logger.info("[AgentQ] Store: %s enabled=%s", self._dir, _ENABLED)

    # -------------------------------------------------------------------------
    # Ingest rollouts from LATS planner
    # -------------------------------------------------------------------------

    def ingest_lats_pairs(self, dpo_pairs: List[Dict[str, Any]]) -> int:
        """
        Accept DPO pairs exported by LATSPlanner.get_dpo_pairs().
        Returns number of pairs stored.
        """
        if not _ENABLED or not dpo_pairs:
            return 0
        stored = 0
        for raw in dpo_pairs:
            pair = self._parse_lats_pair(raw)
            if pair and pair.quality >= _MIN_QUALITY:
                self._store_pair(pair)
                stored += 1
        return stored

    def _parse_lats_pair(self, raw: Dict[str, Any]) -> Optional[PreferencePair]:
        try:
            chosen_steps   = raw.get("chosen", [])
            rejected_steps = raw.get("rejected", [])
            if not chosen_steps or not rejected_steps:
                return None
            quality = abs(
                float(raw.get("chosen_reward", 1.0)) -
                float(raw.get("rejected_reward", 0.0))
            )
            pair_id = _make_id(json.dumps(chosen_steps[:3], default=str))
            return PreferencePair(
                pair_id=pair_id,
                objective=str(raw.get("objective", ""))[:200],
                app_name=str(raw.get("app_name", "unknown"))[:50],
                chosen=chosen_steps[:30],
                rejected=rejected_steps[:30],
                quality=quality,
            )
        except Exception as e:
            _logger.debug("[AgentQ] Parse error: %s", e)
            return None

    # -------------------------------------------------------------------------
    # Direct rollout ingest (when LATS exports raw rollouts)
    # -------------------------------------------------------------------------

    def ingest_rollouts(
        self,
        rollouts: List[MCTSRollout],
        objective: str,
        app_name: str,
    ) -> int:
        if not _ENABLED or len(rollouts) < 2:
            return 0

        sorted_rolls = sorted(rollouts, key=lambda r: r.final_reward, reverse=True)
        best  = sorted_rolls[0]
        worst = sorted_rolls[-1]

        quality = best.final_reward - worst.final_reward
        if quality < _MIN_QUALITY:
            return 0

        pair_id = _make_id(f"{objective}{time.time()}")
        pair = PreferencePair(
            pair_id=pair_id,
            objective=objective[:200],
            app_name=app_name[:50],
            chosen=self._rollout_to_steps(best),
            rejected=self._rollout_to_steps(worst),
            quality=quality,
        )
        self._store_pair(pair)
        return 1

    def _rollout_to_steps(self, rollout: MCTSRollout) -> List[Dict[str, Any]]:
        return [
            {
                "observation": n.observation[:300],
                "action":      n.action,
                "reward":      n.reward,
            }
            for n in rollout.nodes
        ]

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def _store_pair(self, pair: PreferencePair) -> None:
        with self._lock:
            self._pairs_this_session.append(pair)
            self._evict_if_needed()

        path = os.path.join(self._dir, f"pair_{pair.pair_id}.json")
        tmp  = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(pair.to_dict(), f, default=str, separators=(",", ":"))
            os.replace(tmp, path)
            _logger.debug("[AgentQ] Stored pair %s quality=%.3f", pair.pair_id, pair.quality)
        except Exception as e:
            _logger.warning("[AgentQ] Store error: %s", e)

    def _evict_if_needed(self) -> None:
        try:
            files = sorted(
                (e for e in os.scandir(self._dir) if e.name.startswith("pair_")),
                key=lambda e: e.stat().st_mtime,
            )
            while len(files) >= _MAX_PAIRS:
                os.remove(files.pop(0).path)
        except Exception:
            pass

    def load_pairs(self, limit: int = 200) -> List[PreferencePair]:
        pairs: List[PreferencePair] = []
        try:
            files = sorted(
                (e for e in os.scandir(self._dir) if e.name.startswith("pair_")),
                key=lambda e: e.stat().st_mtime,
                reverse=True,
            )
            for entry in files[:limit]:
                try:
                    with open(entry.path, "r", encoding="utf-8") as f:
                        d = json.load(f)
                    pairs.append(PreferencePair(**{
                        k: v for k, v in d.items()
                        if k in PreferencePair.__dataclass_fields__
                    }))
                except Exception:
                    pass
        except Exception as e:
            _logger.debug("[AgentQ] Load error: %s", e)
        return pairs

    def get_stats(self) -> Dict[str, Any]:
        try:
            count = sum(1 for e in os.scandir(self._dir) if e.name.startswith("pair_"))
        except Exception:
            count = 0
        return {
            "enabled":             _ENABLED,
            "pairs_on_disk":       count,
            "pairs_this_session":  len(self._pairs_this_session),
            "min_quality":         _MIN_QUALITY,
        }


_instance: Optional[AgentQ] = None
_instance_lock = threading.Lock()


def get_agent_q() -> AgentQ:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = AgentQ()
    return _instance
