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
                return VerificationResult(ok, reason if not ok else None, details)

            if step.type == StepType.FILE_CREATION:
                ok, reason, details = self._verify_file(step)
                return VerificationResult(ok, reason if not ok else None, details)

            if step.type == StepType.TOOL_INSTALLATION:
                ok, reason, details = self._verify_tool(step)
                return VerificationResult(ok, reason if not ok else None, details)

            if step.type == StepType.UI_INTERACTION:
                ok, reason, details = self._verify_ui_change(
                    step=step,
                    screenshot=screenshot,
                    previous_screenshot=previous_screenshot,
                )
                return VerificationResult(ok, reason if not ok else None, details)

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

        if result.returncode not in expected_codes:
            return (
                False,
                f"unexpected return code: {result.returncode}",
                {"returncode": result.returncode},
            )

        expected_output = verification.get("output_contains")
        if expected_output:
            combined = (result.stdout or "") + (result.stderr or "")
            for token in expected_output:
                if token not in combined:
                    return (
                        False,
                        f"expected output token missing: {token}",
                        {"stdout": result.stdout, "stderr": result.stderr},
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
        if not isinstance(path, str):
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
        if not isinstance(tool, str):
            return False, "tool name missing or invalid", {}

        tool_path = shutil.which(tool)
        if not tool_path:
            return False, f"tool not found in PATH: {tool}", {}

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
                return False, f"version command failed: {e}", {}

            if min_version and min_version not in out:
                return (
                    False,
                    f"minimum version not satisfied: {min_version}",
                    {"version_output": out},
                )

        return True, "", {"tool_path": tool_path}

    # =================================================
    # UI VERIFICATION
    # =================================================

    def _verify_ui_change(
        self,
        *,
        step: ExecutionStep,
        screenshot: Optional[Dict[str, Any]],
        previous_screenshot: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:

        if not screenshot or not screenshot.get("available"):
            return False, "no screenshot evidence available", {}

        verification = step.verification or {}
        expected_text = verification.get("screen_contains")

        # Fallback: allow verification if screen hash changed
        if not expected_text:
            if previous_screenshot and screenshot:
                prev_hash = previous_screenshot.get("hash")
                curr_hash = screenshot.get("hash")
                if prev_hash and curr_hash and prev_hash != curr_hash:
                    return True, "", {"screen_changed": True}

            return (
                False,
                "ui verification requires screen_contains or detectable change",
                {},
            )

        screen_text = screenshot.get("text") or ""
        if not isinstance(screen_text, str):
            return False, "screenshot text unavailable", {}

        for token in expected_text:
            if token not in screen_text:
                return (
                    False,
                    f"expected screen text not found: {token}",
                    {},
                )

        return True, "", {"matched_tokens": expected_text}
