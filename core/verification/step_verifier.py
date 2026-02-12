import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional, Any, Dict, Tuple

from core.schemas.execution_plan import ExecutionStep, StepType


class VerificationError(RuntimeError):
    pass


@dataclass(frozen=True)
class VerificationResult:
    success: bool
    reason: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class StepVerifier:
    """
    Deterministic, evidence-based verifier.

    HARD RULES:
    - Evidence > vision
    - Absence of evidence == failure
    - Unknown step types == failure
    """

    # =================================================
    # PUBLIC API
    # =================================================

    def verify_step(
        self,
        step: ExecutionStep,
        execution_result: Optional[Any] = None,
        *,
        screenshot: Optional[Dict[str, Any]] = None,
        previous_screenshot: Optional[Dict[str, Any]] = None,
        world_graph=None,
    ) -> VerificationResult:

        if not isinstance(step, ExecutionStep):
            raise VerificationError("Invalid step object")

        try:
            if step.type == StepType.DONE:
                return VerificationResult(True, reason="done")

            if step.type == StepType.VERIFICATION:
                return VerificationResult(True, reason="verification-only")

            if step.type == StepType.COMMAND_EXECUTION:
                ok, reason, details = self._verify_command(
                    step, execution_result
                )
                return VerificationResult(ok, None if ok else reason, details)

            if step.type == StepType.FILE_CREATION:
                ok, reason, details = self._verify_file(step)
                return VerificationResult(ok, None if ok else reason, details)

            if step.type == StepType.TOOL_INSTALLATION:
                ok, reason, details = self._verify_tool(step)
                return VerificationResult(ok, None if ok else reason, details)

            if step.type == StepType.UI_INTERACTION:
                ok, reason, details = self._verify_ui_change(
                    step=step,
                    screenshot=screenshot,
                    previous_screenshot=previous_screenshot,
                    world_graph=world_graph,
                )
                return VerificationResult(ok, None if ok else reason, details)

        except Exception as e:
            return VerificationResult(
                success=False,
                reason=f"verification exception: {e}",
                details={"exception_type": type(e).__name__},
            )

        return VerificationResult(
            success=False,
            reason=f"Unhandled step type: {step.type}",
        )

    # =================================================
    # COMMAND VERIFICATION
    # =================================================

    def _verify_command(
        self,
        step: ExecutionStep,
        result: Any,
    ) -> Tuple[bool, str, Dict[str, Any]]:

        if result is None or not hasattr(result, "returncode"):
            return False, "missing command execution result", {}

        verification = step.verification or {}
        expected_codes = verification.get("expected_return_codes", [0])

        if not isinstance(expected_codes, list):
            return False, "expected_return_codes must be list", {}

        if result.returncode not in expected_codes:
            return (
                False,
                f"unexpected return code: {result.returncode}",
                {"returncode": result.returncode},
            )

        expected_output = verification.get("output_contains")
        if expected_output:
            if not isinstance(expected_output, list):
                return False, "output_contains must be list", {}

            stdout = getattr(result, "stdout", "") or ""
            stderr = getattr(result, "stderr", "") or ""
            combined = stdout + stderr

            for token in expected_output:
                if token not in combined:
                    return (
                        False,
                        f"expected output token missing: {token}",
                        {"stdout": stdout, "stderr": stderr},
                    )

        return True, "", {"returncode": result.returncode}

    # =================================================
    # FILE VERIFICATION
    # =================================================

    def _verify_file(
        self,
        step: ExecutionStep,
    ) -> Tuple[bool, str, Dict[str, Any]]:

        action = step.action or {}
        verification = step.verification or {}

        path = action.get("path")
        if not isinstance(path, str) or not path:
            return False, "file path missing or invalid", {}

        if not os.path.exists(path):
            return False, f"path does not exist: {path}", {}

        if verification.get("is_directory"):
            if not os.path.isdir(path):
                return False, "expected directory but found file", {}
            return True, "", {"path": path}

        if not os.path.isfile(path):
            return False, "expected file but path is not file", {}

        expected = verification.get("content_contains")
        if expected:
            if not isinstance(expected, list):
                return False, "content_contains must be list", {}

            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                for token in expected:
                    if token not in content:
                        return (
                            False,
                            f"expected file content missing: {token}",
                            {},
                        )
            except Exception as e:
                return False, f"file read failed: {e}", {}

        return True, "", {"path": path}

    # =================================================
    # TOOL VERIFICATION
    # =================================================

    def _verify_tool(
        self,
        step: ExecutionStep,
    ) -> Tuple[bool, str, Dict[str, Any]]:

        action = step.action or {}
        verification = step.verification or {}

        tool = action.get("tool")
        if not isinstance(tool, str) or not tool:
            return False, "tool name missing or invalid", {}

        tool_path = shutil.which(tool)
        if not tool_path:
            return False, f"tool not found in PATH: {tool}", {}

        version_cmd = verification.get("version_command")
        min_version = verification.get("min_version")

        if version_cmd:
            if not isinstance(version_cmd, (list, tuple)):
                return False, "version_command must be list/tuple", {}

            try:
                out = subprocess.check_output(
                    version_cmd,
                    stderr=subprocess.STDOUT,
                    timeout=5,
                ).decode(errors="ignore")
            except Exception as e:
                return False, f"version command failed: {e}", {}

            if min_version and min_version not in out:
                return (
                    False,
                    f"minimum version not satisfied: {min_version}",
                    {"version_output": out},
                )

        return True, "", {"tool_path": tool_path}

    # =================================================
    # UI VERIFICATION (HARDENED)
    # =================================================

    def _verify_ui_change(
        self,
        *,
        step: ExecutionStep,
        screenshot: Optional[Dict[str, Any]],
        previous_screenshot: Optional[Dict[str, Any]] = None,
        world_graph=None,
    ) -> Tuple[bool, str, Dict[str, Any]]:

        if not isinstance(screenshot, dict):
            return False, "invalid screenshot structure", {}

        if not screenshot.get("available"):
            return False, "no screenshot evidence available", {}

        verification = step.verification or {}
        action = step.action or {}

        # -------------------------------------------------
        # 1. World graph required for UI verification
        # -------------------------------------------------

        if world_graph is None:
            return False, "world_graph required for UI verification", {}

        # -------------------------------------------------
        # 2. Text grounding
        # -------------------------------------------------

        expected_text = verification.get("screen_contains")
        if expected_text:
            if not isinstance(expected_text, list):
                return False, "screen_contains must be list", {}

            for token in expected_text:
                matches = world_graph.find_by_text(contains=token)
                if not matches:
                    return False, f"text not found in world graph: {token}", {}

        # -------------------------------------------------
        # 3. Focused application validation
        # -------------------------------------------------

        expected_focus = verification.get("focused_app_equals")
        if expected_focus:
            current_focus = world_graph.focused_application()
            if current_focus != expected_focus:
                return (
                    False,
                    f"focused app mismatch: expected {expected_focus}, got {current_focus}",
                    {"focused_app": current_focus},
                )

        # -------------------------------------------------
        # 4. Entity type existence
        # -------------------------------------------------

        expected_type = verification.get("entity_type_exists")
        if expected_type:
            found = world_graph.find_by_type(expected_type)
            if not found:
                return (
                    False,
                    f"entity type not found: {expected_type}",
                    {},
                )

        # -------------------------------------------------
        # 5. Interactable grounding for clicks
        # -------------------------------------------------

        if action.get("operation") == "click":
            x = action.get("x")
            y = action.get("y")

            if x is None or y is None:
                return False, "click missing coordinates", {}

            nearby = world_graph.find_by_type("button") + world_graph.find_by_type("link")

            grounded = False
            for ent in nearby:
                ex = ent.get("x", 0.0)
                ey = ent.get("y", 0.0)
                if abs(ex - x) <= 0.05 and abs(ey - y) <= 0.05:
                    if ent.get("interactable") is False:
                        return False, "clicked element not interactable", {}
                    grounded = True
                    break

            if not grounded:
                return False, "click not grounded in world graph", {}

        # -------------------------------------------------
        # 6. Entity count delta check
        # -------------------------------------------------

        min_delta = verification.get("entity_count_delta_min")
        if min_delta is not None and previous_screenshot:
            before_count = previous_screenshot.get("entity_count")
            after_count = world_graph.entity_count()

            if isinstance(before_count, int):
                if (after_count - before_count) < min_delta:
                    return (
                        False,
                        "entity count delta below expectation",
                        {},
                    )

        # -------------------------------------------------
        # If explicit criteria provided, and none failed → success
        # -------------------------------------------------

        if verification:
            return True, "", {"world_graph_verified": True}

        # -------------------------------------------------
        # Fail closed if no verification criteria
        # -------------------------------------------------

        return (
            False,
            "ui verification requires explicit verification criteria",
            {},
    )
