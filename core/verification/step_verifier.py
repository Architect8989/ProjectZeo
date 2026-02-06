import os
import shutil
import subprocess
from typing import Optional, Any, Dict

from core.schemas.execution_plan import ExecutionStep, StepType


class VerificationError(RuntimeError):
    """Authoritative verification failure."""


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
    ) -> bool:
        if not isinstance(step, ExecutionStep):
            raise VerificationError("Invalid step object")

        if step.type == StepType.DONE:
            return True

        if step.type == StepType.VERIFICATION:
            return True

        if step.type == StepType.COMMAND_EXECUTION:
            return self._verify_command(step, execution_result)

        if step.type == StepType.FILE_CREATION:
            return self._verify_file(step)

        if step.type == StepType.TOOL_INSTALLATION:
            return self._verify_tool(step)

        if step.type == StepType.UI_INTERACTION:
            return self._verify_ui_change(
                screenshot=screenshot,
                previous_screenshot=previous_screenshot,
            )

        raise VerificationError(f"Unhandled step type: {step.type}")

    # =================================================
    # COMMAND VERIFICATION
    # =================================================

    def _verify_command(
        self,
        step: ExecutionStep,
        result: Any,
    ) -> bool:
        if result is None or not hasattr(result, "returncode"):
            return False

        verification = step.verification or {}
        expected_codes = verification.get("expected_return_codes", [0])

        if result.returncode not in expected_codes:
            return False

        expected_output = verification.get("output_contains")
        if expected_output:
            combined = (result.stdout or "") + (result.stderr or "")
            for token in expected_output:
                if token not in combined:
                    return False

        return True

    # =================================================
    # FILE VERIFICATION
    # =================================================

    def _verify_file(self, step: ExecutionStep) -> bool:
        action = step.action or {}
        verification = step.verification or {}

        path = action.get("path")
        if not isinstance(path, str):
            return False

        if not os.path.exists(path):
            return False

        if verification.get("is_directory"):
            return os.path.isdir(path)

        if not os.path.isfile(path):
            return False

        expected = verification.get("content_contains")
        if expected:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                for token in expected:
                    if token not in content:
                        return False
            except Exception:
                return False

        return True

    # =================================================
    # TOOL VERIFICATION
    # =================================================

    def _verify_tool(self, step: ExecutionStep) -> bool:
        action = step.action or {}
        verification = step.verification or {}

        tool = action.get("tool")
        if not isinstance(tool, str):
            return False

        tool_path = shutil.which(tool)
        if not tool_path:
            return False

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
            except Exception:
                return False

            if min_version and min_version not in out:
                return False

        return True

    # =================================================
    # UI VERIFICATION (LAST RESORT)
    # =================================================

    def _verify_ui_change(
        self,
        *,
        screenshot: Optional[Dict[str, Any]],
        previous_screenshot: Optional[Dict[str, Any]],
    ) -> bool:
        if not screenshot or not screenshot.get("available"):
            return False

        if previous_screenshot is None:
            return True

        curr_hash = screenshot.get("screen_text_hash")
        prev_hash = previous_screenshot.get("screen_text_hash")

        if curr_hash and prev_hash and curr_hash != prev_hash:
            return True

        curr_ts = screenshot.get("frame_ts")
        prev_ts = previous_screenshot.get("frame_ts")

        if curr_ts and prev_ts and curr_ts > prev_ts:
            return True

        return False
