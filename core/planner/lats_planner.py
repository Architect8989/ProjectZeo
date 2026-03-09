"""
core/planner/lats_planner.py
==============================
LATS — Language Agent Tree Search for Failure Recovery.

Blueprint §7.4 — Zhou et al., ICML 2024 (arXiv:2310.04406)
Unifies: ReAct (reasoning+acting) + MCTS (tree search over action sequences)

Algorithm per decision node:
    1. Generate K=3 candidate actions via ReAct
    2. Score each with ConsequenceReasoner PRM (numeric 0-1)
    3. UCB1 selection: UCB_i = score_i + C * sqrt(ln(N) / n_i)
    4. Execute selected action → observe outcome
    5. Backpropagate reward to tree
    6. If subtree exhausted → backtrack to parent node
    Terminal: milestone success OR max_depth exceeded

Blueprint §7.6 — ReST-MCTS*: Process Reward Model (PRM)
    ConsequenceReasoner already evaluates partial trajectories.
    numeric_score from ConsequenceResult (added in this release) feeds
    the MCTS value function for UCB1 node selection.

Agent Q integration (Blueprint §7.5):
    MCTS generates best/worst rollout pairs → stored as DPO preference pairs.
    Called periodically: every 50 tasks, preference pairs feed DPO fine-tuning.

Integration:
    - gii_loop.py → call lats_planner.recover_via_lats() on milestone failure
    - consequence_reasoner.py → numeric_score now returned from evaluate()
    - core/learning/preference_generator.py (future) → consumes dpo_pairs
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Tunables
# ─────────────────────────────────────────────────────────────────────────────

_LATS_K_CANDIDATES   = int(os.environ.get("PROJECTZEO_LATS_K", "3"))
_LATS_MAX_DEPTH      = int(os.environ.get("PROJECTZEO_LATS_DEPTH", "4"))
_LATS_MAX_SIMS       = int(os.environ.get("PROJECTZEO_LATS_SIMS", "12"))
_UCB_EXPLORATION     = float(os.environ.get("PROJECTZEO_UCB_C", "1.4"))  # sqrt(2) ≈ 1.41
_LATS_NODE_TIMEOUT   = float(os.environ.get("PROJECTZEO_LATS_TIMEOUT", "90.0"))
_PRM_FALLBACK_SCORE  = 0.5   # score if PRM unavailable


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

class NodeStatus(str, Enum):
    UNEXPLORED  = "unexplored"
    EXPLORED    = "explored"
    EXHAUSTED   = "exhausted"
    TERMINAL    = "terminal"


@dataclass
class LATSNode:
    """A node in the LATS search tree."""
    node_id:        str
    depth:          int
    action:         Optional[Dict[str, Any]]    # None for root
    thought:        str                         # ReAct thought for this action
    parent_id:      Optional[str]
    children_ids:   List[str] = field(default_factory=list)
    status:         NodeStatus = NodeStatus.UNEXPLORED

    # MCTS statistics
    visit_count:    int   = 0
    total_reward:   float = 0.0
    prm_score:      float = _PRM_FALLBACK_SCORE  # ConsequenceReasoner numeric_score
    ucb_score:      float = 0.0

    # Outcome tracking
    executed:       bool  = False
    outcome:        str   = ""       # "success" | "failure" | "pending"
    created_at:     float = field(default_factory=time.time)

    @property
    def mean_reward(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_reward / self.visit_count

    def ucb1(self, parent_visits: int, c: float = _UCB_EXPLORATION) -> float:
        """UCB1 formula: mean_reward + C * sqrt(ln(N) / n_i)"""
        if self.visit_count == 0:
            return float("inf")
        exploitation = self.mean_reward
        exploration  = c * math.sqrt(math.log(max(1, parent_visits)) / self.visit_count)
        return exploitation + exploration

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id":     self.node_id,
            "depth":       self.depth,
            "action":      self.action,
            "thought":     self.thought[:200],
            "status":      self.status.value,
            "visits":      self.visit_count,
            "mean_reward": round(self.mean_reward, 3),
            "prm_score":   round(self.prm_score, 3),
            "outcome":     self.outcome,
        }


@dataclass
class DPOPair:
    """
    Agent Q style DPO preference pair (Blueprint §7.5).
    Chosen = best MCTS rollout; Rejected = worst rollout.
    """
    pair_id:   str
    milestone: str
    chosen:    List[Dict[str, Any]]   # Best trajectory
    rejected:  List[Dict[str, Any]]   # Worst trajectory
    chosen_score:  float
    rejected_score: float
    created_at: float = field(default_factory=time.time)


@dataclass
class LATSResult:
    """Result from a LATS recovery session."""
    success:         bool
    best_action:     Optional[Dict[str, Any]]
    best_thought:    str
    best_prm_score:  float
    tree_depth:      int
    simulations:     int
    dpo_pair:        Optional[DPOPair]
    reason:          str
    elapsed_ms:      float


# ─────────────────────────────────────────────────────────────────────────────
# LATSPlanner
# ─────────────────────────────────────────────────────────────────────────────

class LATSPlanner:
    """
    Language Agent Tree Search for milestone failure recovery.

    When a milestone fails after normal ReAct execution, LATS activates:
    1. Generates K alternative actions from current world state
    2. Scores each with ConsequenceReasoner (PRM numeric score)
    3. UCB1 selects the most promising node
    4. Simulates the action (or requests execution)
    5. Backpropagates reward
    6. Repeats until success or max_sims

    Usage:
        planner = LATSPlanner(
            llm_caller=my_llm,
            consequence_reasoner=cr,
        )
        result = planner.recover(
            milestone_desc="Open terminal",
            world_snapshot=current_world,
            objective="Install Python",
            reflection_context=engine.inject_context("Open terminal"),
        )
        if result.success:
            # Execute result.best_action
    """

    def __init__(
        self,
        *,
        llm_caller: Optional[Callable] = None,
        consequence_reasoner: Optional[Any] = None,
        k_candidates: int = _LATS_K_CANDIDATES,
        max_depth: int = _LATS_MAX_DEPTH,
        max_sims: int = _LATS_MAX_SIMS,
        ucb_c: float = _UCB_EXPLORATION,
    ) -> None:
        self._llm = llm_caller
        self._cr  = consequence_reasoner
        self._k   = k_candidates
        self._max_depth = max_depth
        self._max_sims  = max_sims
        self._ucb_c     = ucb_c

        # DPO pair accumulator (Blueprint §7.5)
        self._dpo_pairs: List[DPOPair] = []

        _logger.info(
            "[LATSPlanner] Init: K=%d depth=%d sims=%d UCB_C=%.2f",
            k_candidates, max_depth, max_sims, ucb_c,
        )

    # =========================================================================
    # Public API
    # =========================================================================

    def recover(
        self,
        milestone_desc: str,
        world_snapshot: Dict[str, Any],
        objective: str,
        reflection_context: str = "",
        previous_trajectory: Optional[List[Dict[str, Any]]] = None,
        execute_fn: Optional[Callable] = None,
    ) -> LATSResult:
        """
        Run LATS recovery for a failed milestone.

        Args:
            milestone_desc: Description of the failed milestone
            world_snapshot: Current world state dict
            objective: Overall task objective
            reflection_context: Injected Reflexion context
            previous_trajectory: Actions taken so far
            execute_fn: Optional callable(action) → outcome_str for simulation

        Returns:
            LATSResult with best_action to execute
        """
        t0 = time.monotonic()
        _logger.info("[LATSPlanner] Starting LATS recovery: %r", milestone_desc[:80])

        # Build root node
        root_id = _node_id()
        nodes: Dict[str, LATSNode] = {}
        root = LATSNode(
            node_id=root_id,
            depth=0,
            action=None,
            thought=f"Root: attempting to recover milestone '{milestone_desc[:80]}'",
            parent_id=None,
        )
        nodes[root_id] = root

        best_node: Optional[LATSNode] = None
        best_score: float = -1.0

        # Collect all evaluated nodes for DPO pairs
        evaluated_nodes: List[LATSNode] = []

        for sim_idx in range(self._max_sims):
            elapsed = (time.monotonic() - t0) * 1000
            if elapsed > _LATS_NODE_TIMEOUT * 1000:
                _logger.warning("[LATSPlanner] Timeout after %d sims", sim_idx)
                break

            # MCTS Selection
            node = self._select(root, nodes)
            if node.status == NodeStatus.EXHAUSTED:
                break

            # Expansion: generate K candidate actions
            if node.status == NodeStatus.UNEXPLORED:
                candidates = self._expand(
                    node, milestone_desc, world_snapshot, objective,
                    reflection_context, previous_trajectory
                )
                for c in candidates:
                    nodes[c.node_id] = c
                    node.children_ids.append(c.node_id)
                node.status = NodeStatus.EXPLORED
                node.visit_count += 1

                # Score each candidate with PRM
                for child in candidates:
                    child.prm_score = self._prm_score(child.action, objective)
                    evaluated_nodes.append(child)

                    # Update best
                    if child.prm_score > best_score:
                        best_score = child.prm_score
                        best_node  = child

            # Simulation: if execute_fn provided, simulate
            if execute_fn is not None and node.children_ids:
                child_id = self._ucb_select_child(node, nodes)
                if child_id:
                    child = nodes[child_id]
                    if not child.executed:
                        try:
                            outcome = execute_fn(child.action)
                            child.outcome = str(outcome)
                            child.executed = True
                            reward = 1.0 if "success" in child.outcome.lower() else 0.0
                        except Exception as exc:
                            child.outcome = f"error: {exc}"
                            child.executed = True
                            reward = 0.0

                        # Backpropagation
                        self._backpropagate(child_id, reward, nodes)
                        if reward > best_score:
                            best_score = reward
                            best_node  = child
                        if reward == 1.0:
                            break  # Found a success

        elapsed_ms = (time.monotonic() - t0) * 1000

        # Build DPO pair from best and worst evaluated nodes
        dpo_pair = self._make_dpo_pair(
            milestone_desc, evaluated_nodes
        ) if len(evaluated_nodes) >= 2 else None

        if dpo_pair:
            self._dpo_pairs.append(dpo_pair)

        if best_node is not None and best_node.action is not None:
            _logger.info(
                "[LATSPlanner] Recovery found: score=%.2f sims=%d elapsed=%.0fms",
                best_score, sim_idx + 1, elapsed_ms,
            )
            return LATSResult(
                success=True,
                best_action=best_node.action,
                best_thought=best_node.thought,
                best_prm_score=best_score,
                tree_depth=best_node.depth,
                simulations=sim_idx + 1,
                dpo_pair=dpo_pair,
                reason=f"LATS found action with PRM score {best_score:.2f}",
                elapsed_ms=elapsed_ms,
            )

        return LATSResult(
            success=False,
            best_action=None,
            best_thought="",
            best_prm_score=0.0,
            tree_depth=0,
            simulations=sim_idx + 1 if 'sim_idx' in dir() else 0,
            dpo_pair=dpo_pair,
            reason="LATS exhausted all candidates without finding viable action",
            elapsed_ms=elapsed_ms,
        )

    def get_dpo_pairs(self, flush: bool = False) -> List[DPOPair]:
        """Return accumulated DPO preference pairs for Agent Q training."""
        pairs = list(self._dpo_pairs)
        if flush:
            self._dpo_pairs.clear()
        return pairs

    def stats(self) -> Dict[str, Any]:
        return {
            "k_candidates": self._k,
            "max_depth":    self._max_depth,
            "max_sims":     self._max_sims,
            "dpo_pairs_accumulated": len(self._dpo_pairs),
        }

    # =========================================================================
    # Private — MCTS
    # =========================================================================

    def _select(self, root: LATSNode, nodes: Dict[str, LATSNode]) -> LATSNode:
        """Traverse from root to most promising unexplored node via UCB1."""
        node = root
        while node.status == NodeStatus.EXPLORED and node.children_ids:
            child_id = self._ucb_select_child(node, nodes)
            if child_id is None:
                node.status = NodeStatus.EXHAUSTED
                break
            node = nodes[child_id]
        return node

    def _ucb_select_child(
        self,
        node: LATSNode,
        nodes: Dict[str, LATSNode],
    ) -> Optional[str]:
        """Select child with highest UCB1 score."""
        best_id = None
        best_ucb = -float("inf")
        for child_id in node.children_ids:
            child = nodes.get(child_id)
            if child is None or child.status == NodeStatus.EXHAUSTED:
                continue
            score = child.ucb1(node.visit_count, self._ucb_c)
            # Incorporate PRM prior: UCB1 + PRM_score * 0.3
            score += child.prm_score * 0.3
            if score > best_ucb:
                best_ucb = score
                best_id  = child_id
        return best_id

    def _backpropagate(
        self,
        node_id: str,
        reward: float,
        nodes: Dict[str, LATSNode],
    ) -> None:
        """Propagate reward up to root."""
        nid: Optional[str] = node_id
        while nid is not None:
            node = nodes.get(nid)
            if node is None:
                break
            node.visit_count  += 1
            node.total_reward += reward
            nid = node.parent_id

    # =========================================================================
    # Private — Expansion (LLM)
    # =========================================================================

    def _expand(
        self,
        parent: LATSNode,
        milestone_desc: str,
        world_snapshot: Dict[str, Any],
        objective: str,
        reflection_context: str,
        previous_trajectory: Optional[List[Dict[str, Any]]],
    ) -> List[LATSNode]:
        """Generate K candidate actions from current state via LLM."""
        if self._llm is None:
            return self._fallback_expand(parent, milestone_desc)

        world_summary = _summarise_world(world_snapshot)
        traj_summary  = _summarise_trajectory(previous_trajectory or [])

        prompt = _EXPANSION_PROMPT.format(
            milestone_desc=milestone_desc[:300],
            world_summary=world_summary[:500],
            objective=objective[:300],
            reflection_context=reflection_context[:600],
            traj_summary=traj_summary[:400],
            k=self._k,
        )

        try:
            result = self._llm(
                prompt=prompt,
                timeout=_LATS_NODE_TIMEOUT,
                max_tokens=600,
                response_format="json",
            )
            text = result.get("text", "") if isinstance(result, dict) else str(result)
            candidates_raw = _parse_candidate_actions(text)
        except Exception as exc:
            _logger.warning("[LATSPlanner] LLM expansion failed: %s", exc)
            candidates_raw = self._fallback_candidates(milestone_desc)

        nodes = []
        for i, cand in enumerate(candidates_raw[:self._k]):
            nodes.append(LATSNode(
                node_id=_node_id(),
                depth=parent.depth + 1,
                action=cand.get("action", {}),
                thought=cand.get("thought", f"Alternative {i+1}"),
                parent_id=parent.node_id,
            ))
        return nodes

    def _fallback_expand(
        self, parent: LATSNode, milestone_desc: str
    ) -> List[LATSNode]:
        """Generate simple fallback candidates without LLM."""
        fallbacks = [
            {"operation": "screenshot", "thought": "Observe current state"},
            {"operation": "click", "thought": f"Try clicking primary element for: {milestone_desc[:60]}"},
            {"operation": "wait", "seconds": 2, "thought": "Wait for state to stabilize"},
        ]
        return [
            LATSNode(
                node_id=_node_id(),
                depth=parent.depth + 1,
                action=fb,
                thought=fb.get("thought", ""),
                parent_id=parent.node_id,
            )
            for fb in fallbacks[:self._k]
        ]

    def _fallback_candidates(self, milestone_desc: str) -> List[Dict[str, Any]]:
        return [
            {"thought": f"Try primary approach for {milestone_desc[:60]}", "action": {"operation": "screenshot"}},
            {"thought": "Check if element is visible", "action": {"operation": "observe"}},
        ]

    # =========================================================================
    # Private — PRM Scoring
    # =========================================================================

    def _prm_score(
        self, action: Optional[Dict[str, Any]], objective: str
    ) -> float:
        """
        Score action using ConsequenceReasoner as Process Reward Model.
        Returns numeric 0.0-1.0 (Blueprint §7.6 / §13).
        """
        if self._cr is None or action is None:
            return _PRM_FALLBACK_SCORE
        try:
            result = self._cr.evaluate(
                action=action,
                objective=objective,
                step_description=str(action.get("thought", "")),
            )
            return float(getattr(result, "numeric_score", _PRM_FALLBACK_SCORE))
        except Exception as exc:
            _logger.debug("[LATSPlanner] PRM score failed: %s", exc)
            return _PRM_FALLBACK_SCORE

    # =========================================================================
    # Private — DPO Pair Generation (Agent Q §7.5)
    # =========================================================================

    def _make_dpo_pair(
        self,
        milestone: str,
        nodes: List[LATSNode],
    ) -> Optional[DPOPair]:
        """
        Generate DPO preference pair from best vs worst MCTS rollout.
        chosen = highest PRM score; rejected = lowest PRM score.
        """
        if not nodes:
            return None
        sorted_nodes = sorted(nodes, key=lambda n: n.prm_score)
        worst = sorted_nodes[0]
        best  = sorted_nodes[-1]

        if abs(best.prm_score - worst.prm_score) < 0.1:
            return None  # Too similar — not useful for DPO

        return DPOPair(
            pair_id=str(uuid.uuid4())[:8],
            milestone=milestone[:200],
            chosen=[{"action": best.action, "thought": best.thought}],
            rejected=[{"action": worst.action, "thought": worst.thought}],
            chosen_score=best.prm_score,
            rejected_score=worst.prm_score,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _node_id() -> str:
    return str(uuid.uuid4())[:8]


def _summarise_world(world: Dict[str, Any]) -> str:
    app   = world.get("focused_app", "unknown")
    ents  = world.get("entities", []) or []
    n_ent = len(ents)
    labels = [str(e.get("label") or e.get("name") or "") for e in ents[:5]]
    return f"App={app!r}, entities={n_ent}, visible=[{', '.join(l for l in labels if l)}]"


def _summarise_trajectory(traj: List[Dict[str, Any]]) -> str:
    if not traj:
        return "No previous actions."
    lines = []
    for step in traj[-5:]:
        op = step.get("operation", "?")
        th = step.get("thought", "")[:80]
        lines.append(f"  {op}: {th}")
    return "\n".join(lines)


def _parse_candidate_actions(text: str) -> List[Dict[str, Any]]:
    """Parse LLM response into list of candidate action dicts."""
    import re
    # Try direct JSON array
    try:
        cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        if isinstance(data, dict) and "candidates" in data:
            return data["candidates"]
    except Exception:
        pass
    # Try extracting individual JSON objects
    candidates = []
    for match in re.finditer(r"\{[^{}]+\}", text):
        try:
            obj = json.loads(match.group())
            if "action" in obj or "operation" in obj:
                candidates.append(obj)
        except Exception:
            pass
    return candidates


_EXPANSION_PROMPT = """\
You are recovering from a failed milestone. Generate {k} alternative action candidates.

Milestone: {milestone_desc}
Objective: {objective}
Current screen state: {world_summary}
Previous trajectory:
{traj_summary}

{reflection_context}

Generate exactly {k} alternative actions as a JSON array. Each element:
{{"thought": "Why this action might work", "action": {{"operation": "...", ...}}}}

Operations: click, type, key_combo, scroll, command, screenshot, wait
Coordinates are (x, y) floats between 0-1 (normalized). Use labels when possible.
Focus on alternatives that address the failure reason from reflections above.

JSON array only, no preamble:"""
