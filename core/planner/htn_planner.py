from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

class TaskStatus(str, Enum):
    PENDING    = "pending"
    ACTIVE     = "active"
    COMPLETE   = "complete"
    FAILED     = "failed"
    SKIPPED    = "skipped"

class TaskType(str, Enum):
    ABSTRACT   = "abstract"
    PRIMITIVE  = "primitive"
    METHOD     = "method"

@dataclass
class Task:
    task_id:        str
    description:    str
    task_type:      TaskType
    status:         TaskStatus = TaskStatus.PENDING
    parent_id:      Optional[str] = None
    children:       List[str] = field(default_factory=list)
    preconditions:  List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    operator:       Optional[Dict[str, Any]] = None
    priority:       int = 50
    created_at:     float = field(default_factory=time.time)
    completed_at:   Optional[float] = None
    failure_reason: str = ""
    immutable:      bool = False

    def mark_complete(self) -> None:
        self.status = TaskStatus.COMPLETE
        self.completed_at = time.time()

    def mark_failed(self, reason: str = "") -> None:
        self.status = TaskStatus.FAILED
        self.failure_reason = reason

    def mark_active(self) -> None:
        self.status = TaskStatus.ACTIVE

    def is_terminal(self) -> bool:
        return self.status in (TaskStatus.COMPLETE, TaskStatus.FAILED, TaskStatus.SKIPPED)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id":     self.task_id,
            "description": self.description[:200],
            "type":        self.task_type.value,
            "status":      self.status.value,
            "parent_id":   self.parent_id,
            "children":    self.children,
            "priority":    self.priority,
            "operator":    self.operator,
            "failure":     self.failure_reason[:100] if self.failure_reason else "",
        }

_DECOMPOSE_SYSTEM = """\
You are an HTN (Hierarchical Task Network) planner for a desktop GUI agent.

Your task: decompose an abstract task into 2-6 ordered subtasks that together
achieve the parent task. Each subtask must be either:
  - PRIMITIVE: directly executable as a single GUI action
  - ABSTRACT: needs further decomposition (if complex)

RULES:
  - Order subtasks logically (earlier ones enable later ones)
  - Each subtask must have clear preconditions (what must be true before)
  - Each subtask must have clear postconditions (what will be true after)
  - Prefer primitive tasks — minimize depth
  - For GUI actions use: click|type|hotkey|scroll|command|verify|wait
  - Consider the CURRENT WORLD STATE when determining subtask feasibility

OUTPUT FORMAT — respond ONLY with a valid JSON array:
[
  {
    "description": "<what this subtask accomplishes>",
    "type": "primitive|abstract",
    "preconditions": ["<observable condition>"],
    "postconditions": ["<observable result>"],
    "operator": {
      "operation": "<click|type|hotkey|command|scroll|wait|verify>",
      "text": "<optional>",
      "command": "<optional>",
      "keys": ["<optional>"]
    },
    "priority": <50>
  }
]
Only include "operator" for primitive tasks.
"""

_GOALACT_SYSTEM = """\
You are a GoalAct global goal tracker (arXiv:2504.16563).

Your task: given the current task tree state and the user's original objective,
identify if any subtask is drifting away from the overall goal (local-branch-stall).

Return ONLY a JSON object:
{
  "stall_detected": true|false,
  "stalled_task_id": "<task_id or null>",
  "recommendation": "<redirect|continue|replan>",
  "reason": "<why>"
}
"""

_REPLAN_SYSTEM = """\
You are an HTN replanner for a GUI agent.
A subtask has FAILED or the world state has changed unexpectedly.
Propose an alternative decomposition for the failed subtask.

Return ONLY a JSON array of replacement subtasks (same format as decompose).
"""

class HTNPlanner:

    # ToT milestone stagnation threshold — number of consecutive failures on
    # the same milestone before Tree of Thoughts expansion is triggered.
    _TOT_STAGNATION_THRESHOLD: int = int(
        __import__("os").environ.get("PROJECTZEO_TOT_STAGNATION", "3")
    )

    def __init__(
        self,
        llm_call: Callable,
        *,
        objective: str,
        goalact_check_interval: int = 10,
        consequence_reasoner: Any = None,  # optional CR for PRM scoring
    ) -> None:
        self._llm = llm_call
        self._objective = objective
        self._goalact_interval = goalact_check_interval
        self._lock = threading.RLock()
        self._cycle_count = 0

        self._tasks: Dict[str, Task] = {}
        self._root_id: Optional[str] = None

        self._execution_queue: List[str] = []

        # ToT milestone stagnation tracking
        self._milestone_fail_counts: Dict[str, int] = {}   # task_id → fail count
        self._consequence_reasoner = consequence_reasoner  # for PRM scoring

        self._build_root(objective)

        _logger.info("[HTN] Initialized with objective: %r", objective[:60])

    def _build_root(self, objective: str) -> None:
        root_id = str(uuid.uuid4())[:8]
        root = Task(
            task_id=root_id,
            description=objective[:500],
            task_type=TaskType.ABSTRACT,
            status=TaskStatus.PENDING,
            immutable=True,
        )
        with self._lock:
            self._tasks[root_id] = root
            self._root_id = root_id

    def decompose(
        self,
        task_id: str,
        world_state: Optional[Dict[str, Any]] = None,
    ) -> List[Task]:
        with self._lock:
            parent = self._tasks.get(task_id)
        if parent is None:
            _logger.warning("[HTN] decompose: task_id %r not found", task_id)
            return []

        if parent.task_type == TaskType.PRIMITIVE:
            return []

        world_summary = ""
        if world_state:
            entities = world_state.get("entities", [])[:5]
            world_summary = (
                f"Focused app: {world_state.get('focused_app', 'unknown')}\n"
                f"Visible entities: {[e.get('label', '?') for e in entities]}"
            )

        messages = [
            {"role": "system", "content": _DECOMPOSE_SYSTEM},
            {"role": "user", "content": (
                f"PARENT TASK: {parent.description}\n"
                f"ORIGINAL OBJECTIVE: {self._objective[:300]}\n"
                f"WORLD STATE:\n{world_summary or '(unknown)'}\n"
                "Decompose this task into 2-6 ordered subtasks."
            )},
        ]

        try:
            raw = self._llm(messages, objective=self._objective)
        except Exception as exc:
            _logger.warning("[HTN] LLM decompose failed: %s", exc)
            return self._fallback_decompose(parent)

        subtasks_data = self._parse_task_array(raw)
        if not subtasks_data:
            return self._fallback_decompose(parent)

        child_tasks: List[Task] = []
        with self._lock:
            for item in subtasks_data[:6]:
                task_type_str = str(item.get("type", "primitive")).lower()
                task_type = (
                    TaskType.PRIMITIVE if task_type_str == "primitive"
                    else TaskType.ABSTRACT
                )
                child = Task(
                    task_id=str(uuid.uuid4())[:8],
                    description=str(item.get("description", ""))[:500],
                    task_type=task_type,
                    parent_id=task_id,
                    preconditions=[str(p) for p in item.get("preconditions", [])],
                    postconditions=[str(p) for p in item.get("postconditions", [])],
                    operator=item.get("operator"),
                    priority=int(item.get("priority", 50)),
                )
                self._tasks[child.task_id] = child
                parent.children.append(child.task_id)
                child_tasks.append(child)

            parent.status = TaskStatus.ACTIVE
            self._rebuild_execution_queue()

        _logger.info("[HTN] Decomposed '%s' into %d subtasks.",
                     parent.description[:60], len(child_tasks))
        return child_tasks

    def _rebuild_execution_queue(self) -> None:
        primitives = [
            t for t in self._tasks.values()
            if (t.task_type == TaskType.PRIMITIVE
                and t.status == TaskStatus.PENDING)
        ]
        primitives.sort(key=lambda t: t.priority, reverse=True)
        self._execution_queue = [t.task_id for t in primitives]

    def next_executable(self) -> Optional[Task]:
        with self._lock:
            self._rebuild_execution_queue()
            for task_id in self._execution_queue:
                task = self._tasks.get(task_id)
                if task is None or task.status != TaskStatus.PENDING:
                    continue
                if self._preconditions_met(task):
                    return task
        return None

    def _preconditions_met(self, task: Task) -> bool:
        if not task.parent_id:
            return True
        parent = self._tasks.get(task.parent_id)
        if parent is None:
            return True
        return parent.status in (TaskStatus.ACTIVE, TaskStatus.PENDING)

    def mark_task_complete(self, task_id: str) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.mark_complete()
                # Reset stagnation counter on success
                self._milestone_fail_counts.pop(task_id, None)
                self._propagate_completion(task)
                self._rebuild_execution_queue()

    def mark_task_failed(self, task_id: str, reason: str = "") -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.mark_failed(reason)
                # ToT: track stagnation count per milestone
                self._milestone_fail_counts[task_id] = (
                    self._milestone_fail_counts.get(task_id, 0) + 1
                )
                fail_count = self._milestone_fail_counts[task_id]
                self._rebuild_execution_queue()

        # ToT: trigger milestone-level expansion if stagnation threshold hit
        if fail_count >= self._TOT_STAGNATION_THRESHOLD:
            _logger.info(
                "[HTN] ToT milestone stagnation: task_id=%s failed %d times — expanding alternatives.",
                task_id, fail_count,
            )
            self._tot_expand_milestone(task_id, reason)

    def _tot_expand_milestone(
        self,
        stalled_task_id: str,
        failure_reason: str = "",
        n_branches: int = 3,
        world_state: Optional[Dict[str, Any]] = None,
    ) -> List["Task"]:
        """
        Tree of Thoughts expansion at milestone level.

        When a milestone (abstract task) stalls after _TOT_STAGNATION_THRESHOLD
        consecutive failures, generate n_branches alternative decompositions,
        score each with ConsequenceReasoner PRM (if available), and promote
        the highest-scoring branch as the replacement plan.

        This differs from replan() which handles individual primitive failures.
        tot_expand_milestone() operates at the abstract task level and generates
        structurally different strategies rather than alternative primitives.
        """
        with self._lock:
            stalled = self._tasks.get(stalled_task_id)
        if stalled is None:
            return []

        _BRANCH_SYSTEM = (
            "You are a strategic planner. Generate a DIFFERENT decomposition strategy "
            "for the given milestone — not a retry of the same approach. "
            "Return a JSON array of 2-4 subtasks with keys: description, type (primitive|abstract), "
            "preconditions (list), postconditions (list), operator (dict|null), priority (0-100)."
        )

        branch_results: List[tuple] = []  # (score, subtasks_data)
        for branch_idx in range(n_branches):
            prompt = (
                f"STALLED MILESTONE: {stalled.description}\n"
                f"FAILURE REASON: {failure_reason[:300]}\n"
                f"OBJECTIVE: {self._objective[:300]}\n"
                f"BRANCH {branch_idx + 1}/{n_branches}: "
                "Propose a structurally different approach. "
                "Imagine a completely different sequence of actions to reach the same outcome."
            )
            try:
                raw = self._llm(
                    [
                        {"role": "system", "content": _BRANCH_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    objective=self._objective,
                )
                parsed = self._parse_task_array(raw)
                if not parsed:
                    continue

                # Score this branch
                score = self._score_branch(parsed)
                branch_results.append((score, parsed))
                _logger.debug(
                    "[HTN] ToT branch %d/%d: %d subtasks, score=%.2f",
                    branch_idx + 1, n_branches, len(parsed), score,
                )
            except Exception as exc:
                _logger.debug("[HTN] ToT milestone branch %d error: %s", branch_idx + 1, exc)

        if not branch_results:
            _logger.warning("[HTN] ToT milestone expansion produced no viable branches.")
            return []

        branch_results.sort(key=lambda x: x[0], reverse=True)
        best_score, best_subtasks = branch_results[0]
        _logger.info(
            "[HTN] ToT milestone: %d branches evaluated, best score=%.2f, %d subtasks.",
            len(branch_results), best_score, len(best_subtasks),
        )

        # Install best branch as replacement for the stalled task
        new_tasks: List[Task] = []
        with self._lock:
            parent_id = stalled.parent_id
            if parent_id and parent_id in self._tasks:
                parent = self._tasks[parent_id]
                if stalled_task_id in parent.children:
                    idx = parent.children.index(stalled_task_id)
                    parent.children.pop(idx)
                    stalled.status = TaskStatus.SKIPPED

                for i, item in enumerate(best_subtasks[:4]):
                    task_type_str = str(item.get("type", "primitive")).lower()
                    task_type = (
                        TaskType.PRIMITIVE if task_type_str == "primitive"
                        else TaskType.ABSTRACT
                    )
                    t = Task(
                        task_id=str(uuid.uuid4())[:8],
                        description=str(item.get("description", ""))[:500],
                        task_type=task_type,
                        parent_id=parent_id,
                        preconditions=[str(p) for p in item.get("preconditions", [])],
                        postconditions=[str(p) for p in item.get("postconditions", [])],
                        operator=item.get("operator"),
                        priority=int(item.get("priority", 50)),
                    )
                    self._tasks[t.task_id] = t
                    parent.children.insert(idx + i, t.task_id)
                    new_tasks.append(t)

            # Reset stagnation counter after expansion
            self._milestone_fail_counts.pop(stalled_task_id, None)
            self._rebuild_execution_queue()

        return new_tasks

    def _score_branch(self, subtasks_data: List[Dict[str, Any]]) -> float:
        """
        Score a branch of subtasks for ToT selection.

        Uses ConsequenceReasoner PRM score if available; otherwise falls back
        to a heuristic based on task priority and description length.
        """
        if not subtasks_data:
            return 0.0

        base_score = sum(
            int(t.get("priority", 50)) for t in subtasks_data
        ) / max(len(subtasks_data), 1)

        # PRM scoring via ConsequenceReasoner if wired
        if self._consequence_reasoner is not None:
            try:
                # Ask CR to estimate reversibility/risk of the branch description
                branch_desc = " → ".join(
                    str(t.get("description", ""))[:80] for t in subtasks_data[:3]
                )
                prm_result = self._consequence_reasoner.estimate_reversibility(branch_desc)
                # reversibility ∈ {REVERSIBLE:1.0, RECOVERABLE:0.6, IRREVERSIBLE:0.0}
                reversibility_scores = {
                    "REVERSIBLE": 1.0,
                    "RECOVERABLE": 0.6,
                    "IRREVERSIBLE": 0.0,
                    "UNKNOWN": 0.5,
                }
                rev_label = str(getattr(prm_result, "label", "UNKNOWN")).upper()
                prm_score = reversibility_scores.get(rev_label, 0.5) * 50.0  # scale 0-50
                base_score = base_score * 0.5 + prm_score  # blend
            except Exception:
                pass  # CR unavailable — use heuristic only

        return base_score

    def _propagate_completion(self, task: Task) -> None:
        if not task.parent_id:
            return
        parent = self._tasks.get(task.parent_id)
        if parent is None:
            return
        sibling_tasks = [self._tasks.get(cid) for cid in parent.children]
        all_done = all(
            s is None or s.is_terminal()
            for s in sibling_tasks
        )
        if all_done and not parent.is_terminal():
            any_failed = any(
                s is not None and s.status == TaskStatus.FAILED
                for s in sibling_tasks
            )
            if any_failed:
                parent.mark_failed("A child task failed.")
            else:
                parent.mark_complete()
            self._propagate_completion(parent)

    def replan(
        self,
        failed_task_id: str,
        world_state: Optional[Dict[str, Any]] = None,
    ) -> List[Task]:
        with self._lock:
            failed = self._tasks.get(failed_task_id)
        if failed is None:
            return []

        world_summary = ""
        if world_state:
            entities = world_state.get("entities", [])[:5]
            world_summary = (
                f"Focused app: {world_state.get('focused_app', 'unknown')}\n"
                f"Entities: {[e.get('label', '?') for e in entities]}\n"
                f"Failure reason: {failed.failure_reason[:200]}"
            )

        _N_BRANCHES = 3
        _branch_results = []
        for _branch in range(_N_BRANCHES):
            messages = [
                {"role": "system", "content": _REPLAN_SYSTEM},
                {"role": "user", "content": (
                    f"FAILED TASK: {failed.description}\n"
                    f"FAILURE REASON: {failed.failure_reason[:300]}\n"
                    f"ORIGINAL OBJECTIVE: {self._objective[:300]}\n"
                    f"WORLD STATE:\n{world_summary or '(unknown)'}\n"
                    f"BRANCH {_branch + 1}/{_N_BRANCHES}: "
                    "Propose 1-3 alternative subtasks. "
                    "Think independently and creatively — prefer a different "
                    "approach from what may have failed before."
                )},
            ]
            try:
                raw = self._llm(messages, objective=self._objective)
                parsed = self._parse_task_array(raw)
                if parsed:
                    score = sum(int(t.get("priority", 50)) for t in parsed) / max(len(parsed), 1)
                    _branch_results.append((score, parsed))
            except Exception as exc:
                _logger.debug("[HTN] ToT branch %d failed: %s", _branch + 1, exc)

        if not _branch_results:
            return []

        _branch_results.sort(key=lambda x: x[0], reverse=True)
        subtasks_data = _branch_results[0][1]
        _logger.info(
            "[HTN] Tree of Thoughts replan: %d branches evaluated, best score=%.1f, %d subtasks.",
            len(_branch_results), _branch_results[0][0], len(subtasks_data),
        )

        new_tasks: List[Task] = []
        with self._lock:
            parent_id = failed.parent_id
            if parent_id and parent_id in self._tasks:
                parent = self._tasks[parent_id]
                if failed_task_id in parent.children:
                    idx = parent.children.index(failed_task_id)
                    parent.children.pop(idx)
                    failed.status = TaskStatus.SKIPPED

                for i, item in enumerate(subtasks_data[:3]):
                    task_type_str = str(item.get("type", "primitive")).lower()
                    task_type = (
                        TaskType.PRIMITIVE if task_type_str == "primitive"
                        else TaskType.ABSTRACT
                    )
                    t = Task(
                        task_id=str(uuid.uuid4())[:8],
                        description=str(item.get("description", ""))[:500],
                        task_type=task_type,
                        parent_id=parent_id,
                        preconditions=[str(p) for p in item.get("preconditions", [])],
                        postconditions=[str(p) for p in item.get("postconditions", [])],
                        operator=item.get("operator"),
                        priority=int(item.get("priority", 50)),
                    )
                    self._tasks[t.task_id] = t
                    parent.children.insert(idx + i, t.task_id)
                    new_tasks.append(t)

            self._rebuild_execution_queue()

        _logger.info("[HTN] Replan: replaced failed task with %d alternatives.", len(new_tasks))
        return new_tasks

    def goalact_check(self, world_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._cycle_count += 1
        if self._cycle_count % self._goalact_interval != 0:
            return {"stall_detected": False, "recommendation": "continue", "reason": ""}

        with self._lock:
            active = [t for t in self._tasks.values()
                      if t.status == TaskStatus.ACTIVE]
            tree_summary = self._format_tree_summary()

        messages = [
            {"role": "system", "content": _GOALACT_SYSTEM},
            {"role": "user", "content": (
                f"ORIGINAL OBJECTIVE: {self._objective[:300]}\n\n"
                f"CURRENT TASK TREE:\n{tree_summary}\n\n"
                f"ACTIVE TASKS: {[t.description[:80] for t in active]}\n"
                f"WORLD STATE: {str(world_state or '')[:300]}"
            )},
        ]

        try:
            raw = self._llm(messages, objective=self._objective)
            data = self._extract_json(raw)
            if isinstance(data, dict):
                return data
        except Exception as exc:
            _logger.debug("[HTN] GoalAct check error: %s", exc)

        return {"stall_detected": False, "recommendation": "continue", "reason": ""}

    def _format_tree_summary(self) -> str:
        lines = []

        def _format_node(task_id: str, depth: int = 0) -> None:
            task = self._tasks.get(task_id)
            if not task:
                return
            indent = "  " * depth
            icon = {"complete": "✓", "failed": "✗", "active": "►",
                    "pending": "○", "skipped": "~"}.get(task.status.value, "?")
            lines.append(f"{indent}{icon} [{task.task_id}] {task.description[:80]}")
            for child_id in task.children:
                _format_node(child_id, depth + 1)

        if self._root_id:
            _format_node(self._root_id)
        return "\n".join(lines)

    def get_tree_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "root_id":   self._root_id,
                "tasks":     {tid: t.to_dict() for tid, t in self._tasks.items()},
                "queue_len": len(self._execution_queue),
                "objective": self._objective[:100],
            }

    def is_complete(self) -> bool:
        with self._lock:
            if not self._root_id:
                return False
            root = self._tasks.get(self._root_id)
            return root is not None and root.status == TaskStatus.COMPLETE

    def is_failed(self) -> bool:
        with self._lock:
            if not self._root_id:
                return False
            root = self._tasks.get(self._root_id)
            return root is not None and root.status == TaskStatus.FAILED

    def _fallback_decompose(self, parent: Task) -> List[Task]:
        child = Task(
            task_id=str(uuid.uuid4())[:8],
            description=f"Execute: {parent.description[:200]}",
            task_type=TaskType.PRIMITIVE,
            parent_id=parent.task_id,
            operator={"operation": "verify", "method": "screenshot"},
            priority=50,
        )
        with self._lock:
            self._tasks[child.task_id] = child
            parent.children.append(child.task_id)
            parent.status = TaskStatus.ACTIVE
            self._rebuild_execution_queue()
        return [child]

    def _parse_task_array(self, raw: str) -> List[Dict[str, Any]]:
        if not raw:
            return []
        data = self._extract_json(raw)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return []

    @staticmethod
    def _extract_json(text: str) -> Any:
        if not text:
            return None
        cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
        for s, e in [("[", "]"), ("{", "}")]:
            start = cleaned.find(s)
            end = cleaned.rfind(e)
            if start != -1 and end > start:
                try:
                    return json.loads(cleaned[start:end + 1])
                except Exception:
                    pass
        return None
