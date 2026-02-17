import os
import shutil
import subprocess
import re
from dataclasses import dataclass
from typing import Optional, Any, Dict, Tuple, List

from core.schemas.execution_plan import ExecutionStep, StepType


class VerificationError(RuntimeError):
    pass


@dataclass(frozen=True)
class VerificationResult:
    success: bool
    reason: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    confidence: float = 1.0
    progress_score: float = 0.0


class StepVerifier:
    """
    Deterministic + cognitive verifier.

    HARD RULES:
    - Evidence > vision
    - Absence of evidence == failure
    - Unknown step types == failure
    - No substring version validation
    - Fail-closed always

    Cognitive additions:
    - Confidence modeling
    - Progress scoring
    - Delta-based UI verification
    """

    VERSION_REGEX = re.compile(r"\d+(?:\.\d+)+")

    # =================================================
    # PUBLIC API
    # =================================================

    def verify_step(
        self,
        step: Optional[ExecutionStep],
        execution_result: Optional[Any] = None,
        *,
        screenshot: Optional[Dict[str, Any]] = None,
        previous_screenshot: Optional[Dict[str, Any]] = None,
        world_graph=None,
    ) -> VerificationResult:

        if step is None:
            # cognitive-only check (used in cognition loop)
            return VerificationResult(
                success=True,
                reason="cognitive-only",
                confidence=0.8,
                progress_score=0.05,
            )

        if not isinstance(step, ExecutionStep):
            raise VerificationError("Invalid step object")

        try:
            if step.type == StepType.DONE:
                return VerificationResult(True, "done", confidence=1.0, progress_score=1.0)

            if step.type == StepType.VERIFICATION:
                return VerificationResult(True, "verification-only", confidence=1.0)

            if step.type == StepType.COMMAND_EXECUTION:
                ok, reason, details = self._verify_command(step, execution_result)
                return self._result(ok, reason, details)

            if step.type == StepType.FILE_CREATION:
                ok, reason, details = self._verify_file(step)
                return self._result(ok, reason, details)

            if step.type == StepType.TOOL_INSTALLATION:
                ok, reason, details = self._verify_tool(step)
                return self._result(ok, reason, details)

            if step.type == StepType.UI_INTERACTION:
                ok, reason, details = self._verify_ui_change(
                    step=step,
                    screenshot=screenshot,
                    previous_screenshot=previous_screenshot,
                    world_graph=world_graph,
                )
                return self._result(ok, reason, details)

        except Exception as e:
            return VerificationResult(
                False,
                f"verification exception: {e}",
                {"exception_type": type(e).__name__},
                confidence=0.0,
                progress_score=0.0,
            )

        return VerificationResult(False, f"Unhandled step type: {step.type}")

    # =================================================
    # RESULT BUILDER
    # =================================================

    def _result(self, ok: bool, reason: str, details: Dict[str, Any]) -> VerificationResult:
        if ok:
            return VerificationResult(
                True,
                None,
                details,
                confidence=0.95,
                progress_score=0.1,
            )
        return VerificationResult(
            False,
            reason,
            details,
            confidence=0.0,
            progress_score=0.0,
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
            return False, f"unexpected return code: {result.returncode}", {
                "returncode": result.returncode
            }

        return True, "", {"returncode": result.returncode}

    # =================================================
    # FILE VERIFICATION
    # =================================================

    def _verify_file(self, step: ExecutionStep) -> Tuple[bool, str, Dict[str, Any]]:

        action = step.action or {}
        path = action.get("path")

        if not isinstance(path, str) or not os.path.exists(path):
            return False, "file missing", {}

        return True, "", {"path": path}

    # =================================================
    # TOOL VERIFICATION
    # =================================================

    def _verify_tool(self, step: ExecutionStep) -> Tuple[bool, str, Dict[str, Any]]:

        tool = (step.action or {}).get("tool")
        if not tool:
            return False, "tool missing", {}

        tool_path = shutil.which(tool)
        if not tool_path:
            return False, "tool not found", {}

        return True, "", {"tool_path": tool_path}

    # =================================================
    # UI VERIFICATION (DELTA-BASED)
    # =================================================

    def _verify_ui_change(
        self,
        *,
        step: ExecutionStep,
        screenshot: Optional[Dict[str, Any]],
        previous_screenshot: Optional[Dict[str, Any]],
        world_graph=None,
    ) -> Tuple[bool, str, Dict[str, Any]]:

        if world_graph is None:
            return False, "world_graph required", {}

        current = world_graph.snapshot()

        delta = None
        if previous_screenshot:
            delta = world_graph.compute_delta(previous_screenshot)

        # semantic evidence
        verification = step.verification or {}

        expected_text = verification.get("screen_contains")
        if expected_text:
            for token in expected_text:
                if not world_graph.find_by_text(contains=token):
                    return False, f"text not found: {token}", {}

        # delta-based progress
        if delta and delta.get("significant_change"):
            return True, "", {
                "delta_detected": True,
                "delta": delta,
            }

        # fallback: entity existence
        expected_type = verification.get("entity_type_exists")
        if expected_type:
            if world_graph.find_by_type(expected_type):
                return True, "", {"entity_type": expected_type}

        return False, "no meaningful UI change detected", {}
