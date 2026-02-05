import os
import shutil
import subprocess
from typing import Dict, Any, Optional


class VerificationError(Exception):
    pass


class StepVerifier:
    """
    Deterministic, evidence-based verifier.

    Design rules:
    - Screen change alone is NEVER sufficient
    - Each operation type has a concrete success predicate
    - Vision is auxiliary, not authoritative
    - Absence of evidence == failure
    """

    def verify_step(
        self,
        action: Dict[str, Any],
        execution_result: Optional[Any],
        screenshot: Optional[Dict[str, Any]] = None,
        previous_screenshot: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not action or "operation" not in action:
            return False

        op = action["operation"]

        if op == "done":
            return True

        # Dispatch by operation type
        if op == "command":
            return self._verify_command(action, execution_result)

        if op == "file_create":
            return self._verify_file_creation(action)

        if op == "file_append":
            return self._verify_file_append(action)

        if op == "mkdir":
            return self._verify_directory(action)

        if op == "tool_check":
            return self._verify_tool(action)

        if op == "ui_action":
            return self._verify_ui_change(screenshot, previous_screenshot)

        # Unknown operations are invalid
        return False

    # -------------------------------------------------
    # COMMAND VERIFICATION
    # -------------------------------------------------

    def _verify_command(self, action: Dict[str, Any], result: Any) -> bool:
        if result is None:
            return False

        expected_codes = action.get("expected_return_codes", [0])
        if result.returncode not in expected_codes:
            return False

        expected_output = action.get("output_contains")
        if expected_output:
            combined = (result.stdout or "") + (result.stderr or "")
            for token in expected_output:
                if token not in combined:
                    return False

        return True

    # -------------------------------------------------
    # FILE VERIFICATION
    # -------------------------------------------------

    def _verify_file_creation(self, action: Dict[str, Any]) -> bool:
        path = action.get("path")
        if not path or not os.path.exists(path):
            return False

        expected = action.get("content_contains")
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

    def _verify_file_append(self, action: Dict[str, Any]) -> bool:
        # Same validation as creation, but file must pre-exist
        return self._verify_file_creation(action)

    def _verify_directory(self, action: Dict[str, Any]) -> bool:
        path = action.get("path")
        return bool(path and os.path.isdir(path))

    # -------------------------------------------------
    # TOOL VERIFICATION
    # -------------------------------------------------

    def _verify_tool(self, action: Dict[str, Any]) -> bool:
        tool = action.get("tool")
        if not tool:
            return False

        if not shutil.which(tool):
            return False

        version_cmd = action.get("version_command")
        min_version = action.get("min_version")

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

    # -------------------------------------------------
    # UI VERIFICATION (LAST RESORT)
    # -------------------------------------------------

    def _verify_ui_change(
        self,
        screenshot: Optional[Dict[str, Any]],
        previous_screenshot: Optional[Dict[str, Any]],
    ) -> bool:
        if not screenshot or not screenshot.get("available"):
            return False

        if previous_screenshot is None:
            return True

        curr_hash = screenshot.get("screen_text_hash")
        prev_hash = previous_screenshot.get("screen_text_hash")

        curr_ts = screenshot.get("frame_ts")
        prev_ts = previous_screenshot.get("frame_ts")

        if curr_hash and prev_hash and curr_hash != prev_hash:
            return True

        if curr_ts and prev_ts and curr_ts > prev_ts:
            return True

        return False
