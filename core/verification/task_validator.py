"""
Task Validator

Purpose:
Determines whether a human objective is:
- understood
- feasible
- executable on this system

This module:
- DOES validate intent feasibility
- DOES analyze required capabilities
- DOES verify tool availability or installability
- DOES NOT install tools
- DOES NOT execute tasks
- DOES NOT touch UI or OS state

Hard rule:
No execution may proceed if validation fails.
"""

from typing import Dict, Any, List, Optional

from core.tools.tool_manager import ToolManager, ToolStatus


class TaskValidationError(RuntimeError):
    pass


class ValidationResult:
    """
    Immutable validation outcome.
    """

    def __init__(
        self,
        *,
        valid: bool,
        reason: Optional[str] = None,
        missing_tools: Optional[List[str]] = None,
        required_tools: Optional[List[str]] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.valid = valid
        self.reason = reason
        self.missing_tools = missing_tools or []
        self.required_tools = required_tools or []
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "reason": self.reason,
            "missing_tools": self.missing_tools,
            "required_tools": self.required_tools,
            "details": self.details,
        }


class TaskValidator:
    """
    Deterministic feasibility validator.

    HARD CONTRACT:
    - Must be called BEFORE planning completion
    - Must block execution on failure
    """

    def __init__(
        self,
        *,
        tool_manager: ToolManager,
        environment_fingerprint: Dict[str, Any],
    ):
        self._tools = tool_manager
        self._env = environment_fingerprint

    # -------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------

    def validate(self, objective: str) -> ValidationResult:
        """
        Validates whether the task can be executed on this system.
        """

        if not isinstance(objective, str) or not objective.strip():
            return ValidationResult(
                valid=False,
                reason="Objective must be non-empty string",
            )

        # ---- analyze required tools (semantic) ----
        try:
            required = self._tools.analyze_required_tools(objective)
        except Exception as e:
            return ValidationResult(
                valid=False,
                reason=f"Tool analysis failed: {e}",
            )

        required_tool_names = [t["name"] for t in required]

        missing: List[str] = []
        nonfunctional: List[str] = []

        tool_status: Dict[str, ToolStatus] = {}

        for tool in required_tool_names:
            status = self._tools.check_tool_status(tool)
            tool_status[tool] = status

            if not status.installed:
                missing.append(tool)
            elif not status.functional:
                nonfunctional.append(tool)

        if nonfunctional:
            return ValidationResult(
                valid=False,
                reason="Required tools installed but non-functional",
                required_tools=required_tool_names,
                details={
                    "nonfunctional_tools": nonfunctional,
                    "tool_status": {
                        k: v.to_dict() for k, v in tool_status.items()
                    },
                },
            )

        # Missing tools are allowed ONLY if installer exists later
        if missing:
            return ValidationResult(
                valid=False,
                reason="Required tools missing",
                missing_tools=missing,
                required_tools=required_tool_names,
                details={
                    "install_required": True,
                    "tool_status": {
                        k: v.to_dict() for k, v in tool_status.items()
                    },
                },
            )

        # ---- environment sanity checks ----
        if not self._environment_ok():
            return ValidationResult(
                valid=False,
                reason="Environment constraints violated",
                details={"environment": self._env},
            )

        return ValidationResult(
            valid=True,
            required_tools=required_tool_names,
            details={
                "tool_status": {
                    k: v.to_dict() for k, v in tool_status.items()
                }
            },
        )

    # -------------------------------------------------
    # INTERNAL CHECKS
    # -------------------------------------------------

    def _environment_ok(self) -> bool:
        """
        Minimal sanity checks only.
        No assumptions.
        """

        os_name = self._env.get("os")
        arch = self._env.get("architecture")

        if not os_name or not arch:
            return False

        # Example guardrails (conservative)
        if os_name.lower() not in ("linux", "darwin", "windows"):
            return False

        return True
