from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Set
import time


# =================================================
# STEP TYPES
# =================================================

class StepType(str, Enum):
    UI_ACTION = "ui_action"          # click / write / press
    COMMAND = "command"              # shell / cli execution
    FILE_WRITE = "file_write"        # create or modify files
    TOOL_INSTALL = "tool_install"    # tool installation
    VERIFICATION = "verification"    # explicit verification step
    DONE = "done"


# =================================================
# ALLOWED UI ACTION SURFACE (SOC COMPAT)
# =================================================

ALLOWED_OPERATIONS = {"click", "write", "press", "done"}

COMMON_FIELDS = {"operation", "thought"}

REQUIRED_FIELDS = {
    "click": {"operation"},
    "write": {"operation", "content"},
    "press": {"operation", "keys"},
    "done": {"operation", "summary"},
}

OPTIONAL_FIELDS = {
    "click": {"x", "y", "label", "text", "thought"},
    "write": {"content", "thought"},
    "press": {"keys", "thought"},
    "done": {"summary", "thought"},
}


# =================================================
# EXECUTION STEP
# =================================================

@dataclass
class ExecutionStep:
    id: int
    type: StepType
    description: str

    # Action payload (SOC-compatible or extended)
    action: Dict[str, Any]

    # Verification contract (what proves success)
    verification: Dict[str, Any] = field(default_factory=dict)

    # Dependency graph
    depends_on: List[int] = field(default_factory=list)

    # Metadata
    estimated_duration: float = 0.0
    retryable: bool = True


# =================================================
# EXECUTION PLAN
# =================================================

@dataclass
class ExecutionPlan:
    objective: str
    steps: List[ExecutionStep]

    required_tools: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def validate(self) -> None:
        """
        HARD VALIDATION.
        Raises ValueError on any violation.
        """

        if not self.objective or not self.objective.strip():
            raise ValueError("ExecutionPlan.objective is empty")

        if not self.steps:
            raise ValueError("ExecutionPlan has no steps")

        self._validate_step_ids()
        self._validate_dependencies()
        self._validate_steps()
        self._validate_done_step()

    # -------------------------------------------------

    def _validate_step_ids(self) -> None:
        ids = [s.id for s in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate step IDs detected")

    def _validate_dependencies(self) -> None:
        ids: Set[int] = {s.id for s in self.steps}

        for step in self.steps:
            for dep in step.depends_on:
                if dep not in ids:
                    raise ValueError(
                        f"Step {step.id} depends on missing step {dep}"
                    )

        if self._has_cycles():
            raise ValueError("Circular dependency detected in execution plan")

    def _has_cycles(self) -> bool:
        graph = {s.id: set(s.depends_on) for s in self.steps}
        visited = set()
        stack = set()

        def visit(n):
            if n in stack:
                return True
            if n in visited:
                return False
            visited.add(n)
            stack.add(n)
            for d in graph.get(n, []):
                if visit(d):
                    return True
            stack.remove(n)
            return False

        return any(visit(n) for n in graph)

    def _validate_steps(self) -> None:
        for step in self.steps:
            if not isinstance(step.action, dict):
                raise ValueError(f"Step {step.id} action is not a dict")

            if step.type == StepType.UI_ACTION:
                if not validate_action(step.action):
                    raise ValueError(
                        f"Invalid UI action in step {step.id}: {step.action}"
                    )

            if step.type == StepType.DONE:
                if step.action.get("operation") != "done":
                    raise ValueError(
                        f"DONE step {step.id} must use operation=done"
                    )

    def _validate_done_step(self) -> None:
        done_steps = [s for s in self.steps if s.type == StepType.DONE]
        if len(done_steps) != 1:
            raise ValueError(
                f"ExecutionPlan must have exactly one DONE step, found {len(done_steps)}"
            )


# =================================================
# UI ACTION VALIDATION (SOC-COMPAT)
# =================================================

def _is_string(v):
    return isinstance(v, str) and len(v.strip()) > 0


def _is_list(v):
    return isinstance(v, list) and len(v) > 0


def _is_number(v):
    return isinstance(v, (int, float))


def validate_click(action: Dict[str, Any]) -> bool:
    modes = 0

    if "x" in action and "y" in action:
        if not (_is_number(action["x"]) and _is_number(action["y"])):
            return False
        modes += 1

    if "label" in action:
        if not _is_string(action["label"]):
            return False
        modes += 1

    if "text" in action:
        if not _is_string(action["text"]):
            return False
        modes += 1

    return modes == 1


def validate_write(action: Dict[str, Any]) -> bool:
    return _is_string(action.get("content"))


def validate_press(action: Dict[str, Any]) -> bool:
    keys = action.get("keys")
    return _is_list(keys) and all(_is_string(k) for k in keys)


def validate_done(action: Dict[str, Any]) -> bool:
    return _is_string(action.get("summary"))


def validate_action(action: Dict[str, Any]) -> bool:
    if not isinstance(action, dict):
        return False

    op = action.get("operation")
    if op not in ALLOWED_OPERATIONS:
        return False

    if not REQUIRED_FIELDS[op].issubset(action.keys()):
        return False

    allowed_fields = COMMON_FIELDS | OPTIONAL_FIELDS[op]
    if not set(action.keys()).issubset(allowed_fields):
        return False

    if op == "click":
        return validate_click(action)
    if op == "write":
        return validate_write(action)
    if op == "press":
        return validate_press(action)
    if op == "done":
        return validate_done(action)

    return False
