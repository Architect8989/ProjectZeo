from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Set
import time


# =================================================
# CANONICAL STEP TYPES (LOCKED)
# =================================================

class StepType(str, Enum):
    UI_INTERACTION = "ui_interaction"
    COMMAND_EXECUTION = "command_execution"
    FILE_CREATION = "file_creation"
    TOOL_INSTALLATION = "tool_installation"
    VERIFICATION = "verification"
    DONE = "done"


# =================================================
# UI ACTION SCHEMA (SOC-COMPAT, STRICT)
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
    "write": {"thought"},
    "press": {"thought"},
    "done": {"thought"},
}


# =================================================
# EXECUTION STEP
# =================================================

@dataclass
class ExecutionStep:
    id: int
    type: StepType
    description: str
    action: Dict[str, Any]
    verification: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[int] = field(default_factory=list)
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

    def validate(self) -> bool:
        try:
            if not isinstance(self.objective, str) or not self.objective.strip():
                return False

            if not self.steps or not isinstance(self.steps, list):
                return False

            self._validate_step_ids()
            self._validate_step_order()
            self._validate_dependencies()
            self._validate_steps()
            self._validate_done_step()

            return True
        except Exception:
            return False

    # -------------------------------------------------

    def _validate_step_ids(self) -> None:
        ids = [s.id for s in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate step IDs detected")

    def _validate_step_order(self) -> None:
        ids = [s.id for s in self.steps]
        if ids != sorted(ids):
            raise ValueError("Step IDs must be strictly increasing")

    def _validate_dependencies(self) -> None:
        ids: Set[int] = {s.id for s in self.steps}

        for step in self.steps:
            if not isinstance(step.dependencies, list):
                raise ValueError(f"Step {step.id} dependencies must be list")

            for dep in step.dependencies:
                if dep not in ids:
                    raise ValueError(
                        f"Step {step.id} depends on missing step {dep}"
                    )
                if dep >= step.id:
                    raise ValueError(
                        f"Step {step.id} has forward/self dependency {dep}"
                    )

        if self._has_cycles():
            raise ValueError("Circular dependency detected")

    def _has_cycles(self) -> bool:
        graph = {s.id: set(s.dependencies) for s in self.steps}
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
                raise ValueError(f"Step {step.id} action must be dict")

            if step.type == StepType.UI_INTERACTION:
                if not validate_action(step.action):
                    raise ValueError(
                        f"Invalid UI action in step {step.id}"
                    )

            if step.type == StepType.DONE:
                if step.action.get("operation") != "done":
                    raise ValueError(
                        f"DONE step {step.id} must use operation=done"
                    )

    def _validate_done_step(self) -> None:
        done_steps = [s for s in self.steps if s.type == StepType.DONE]
        if len(done_steps) != 1:
            raise ValueError("ExecutionPlan must contain exactly one DONE step")

        done_step = done_steps[0]
        if done_step.id != max(s.id for s in self.steps):
            raise ValueError("DONE step must be the final step")


# =================================================
# UI ACTION VALIDATION
# =================================================

def _is_string(v):
    return isinstance(v, str) and bool(v.strip())


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

    allowed_fields = (
        COMMON_FIELDS
        | REQUIRED_FIELDS[op]
        | OPTIONAL_FIELDS[op]
    )

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
