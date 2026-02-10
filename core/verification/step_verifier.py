import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional, Any, Dict, Tuple

from core.schemas.execution_plan import ExecutionStep, StepType


class VerificationError(RuntimeError):
    """Authoritative verification failure."""


@dataclass(frozen=True)
class VerificationResult:
    """
    Immutable verification outcome.

    success: definitive truth value
    reason: human-readable failure reason
    details: optional forensic metadata
    """
    success: bool
    reason: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class StepVerifier:
    """
    Deterministic, evidence-based verifier.

    HARD RULES:
    - Evidence > vision
    - Vision is last-resort only
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
    ) -> VerificationResult:
        if not isinstance(step, ExecutionStep):
            raise VerificationError("Invalid step object")

        try:
            if step.type == StepType.DONE:
                return VerificationResult(True, reason="done")

            if step.type == StepType.VERIFICATION:
                return VerificationResult(True, reason="verification-only")

            if step.type == StepType.COMMAND_EXECUTION:
                ok, reason = self._verify_command(step, execution_result)
                return VerificationResult(ok, None if ok else reason)

            if step.type == StepType.FILE_CREATION:
                ok, reason = self._verify_file(step)
                return VerificationResult(ok, None if ok else reason)

            if step.type == StepType.TOOL_INSTALLATION:
                ok, reason = self._verify_tool(step)
                return VerificationResult(ok, None if ok else reason)

            if step.type == StepType.UI_INTERACTION:
                ok, reason = self._verify_ui_change(
                    step=step,
                    screenshot=screenshot,
                )
                return VerificationResult(ok, None if ok else reason)

        except Exception as e:
            return VerificationResult(
                success=False,
                reason=f"verification exception: {e}",
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
    ) -> Tuple[bool, str]:
        if result is None or not hasattr(result, "returncode"):
            return False, "missing command execution result"

        verification = step.verification or {}
        expected_codes = verification.get("expected_return_codes", [0])

        if result.returncode not in expected_codes:
            return (
                False,
                f"unexpected return code: {result.returncode}",
            )

        expected_output = verification.get("output_contains")
        if expected_output:
            combined = (result.stdout or "") + (result.stderr or "")
            for token in expected_output:
                if token not in combined:
                    return (
                        False,
                        f"expected output token missing: {token}",
                    )

        return True, ""

    # =================================================
    # FILE VERIFICATION
    # =================================================

    def _verify_file(self, step: ExecutionStep) -> Tuple[bool, str]:
        action = step.action or {}
        verification = step.verification or {}

        path = action.get("path")
        if not isinstance(path, str):
            return False, "file path missing or invalid"

        if not os.path.exists(path):
            return False, f"path does not exist: {path}"

        if verification.get("is_directory"):
            if not os.path.isdir(path):
                return False, "expected directory but found file"
            return True, ""

        if not os.path.isfile(path):
            return False, "expected file but path is not file"

        expected = verification.get("content_contains")
        if expected:
            try:
                with open(
                    path,
                    "r",
                    encoding="utf-8",
                    errors="ignore",
                ) as f:
                    content = f.read()
                for token in expected:
                    if token not in content:
                        return (
                            False,
                            f"expected file content missing: {token}",
                        )
            except Exception as e:
                return False, f"file read failed: {e}"

        return True, ""

    # =================================================
    # TOOL VERIFICATION
    # =================================================

    def _verify_tool(self, step: ExecutionStep) -> Tuple[bool, str]:
        action = step.action or {}
        verification = step.verification or {}

        tool = action.get("tool")
        if not isinstance(tool, str):
            return False, "tool name missing or invalid"

        tool_path = shutil.which(tool)
        if not tool_path:
            return False, f"tool not found in PATH: {tool}"

        version_cmd = verification.get("version_command")
        min_version = verification.get("min_version")

        if version_cmd:
            try:
                out = subprocess.check_output(
                    version_cmd,
                    stderr=subprocess.STDOUT,
                    shell=isinstance(version_cmd, str),
                    timeout=5,
                ).decode(errors="ignore")
            except Exception as e:
                return False, f"version command failed: {e}"

            if min_version and min_version not in out:
                return (
                    False,
                    f"minimum version not satisfied: {min_version}",
                )

        return True, ""

    # =================================================
    # UI VERIFICATION (FAIL-CLOSED)
    # =================================================

    def _verify_ui_change(
        self,
        *,
        step: ExecutionStep,
        screenshot: Optional[Dict[str, Any]],
    ) -> Tuple[bool, str]:
        """
        UI verification is ONLY allowed when an explicit
        semantic condition is provided.

        No condition == no evidence == failure.
        """

        if not screenshot or not screenshot.get("available"):
            return False, "no screenshot evidence available"

        verification = step.verification or {}

        expected_text = verification.get("screen_contains")
        if not expected_text:
            return (
                False,
                "ui verification requires explicit screen_contains condition",
            )

        screen_text = screenshot.get("text")
        if not isinstance(screen_text, str):
            return False, "screenshot text unavailable"

        for token in expected_text:
            if token not in screen_text:
                return (
                    False,
                    f"expected screen text not found: {token}",
                )

        return True, ""
