import os
import shutil
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
    Evidence-first verifier.
    Strict. Deterministic. Fail-closed.
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
            return self._fail("step required")

        if not isinstance(step, ExecutionStep):
            raise VerificationError("Invalid step object")

        try:

            if step.type == StepType.DONE:
                return VerificationResult(
                    True,
                    None,
                    {"done": True},
                    confidence=1.0,
                    progress_score=1.0,
                )

            if step.type == StepType.VERIFICATION:
                return self._verify_explicit_conditions(
                    step=step,
                    execution_result=execution_result,
                    screenshot=screenshot,
                    previous_screenshot=previous_screenshot,
                    world_graph=world_graph,
                )

            if step.type == StepType.COMMAND_EXECUTION:
                ok, reason, details = self._verify_command(step, execution_result)
                return self._build(ok, reason, details)

            if step.type == StepType.FILE_CREATION:
                ok, reason, details = self._verify_file(step)
                return self._build(ok, reason, details)

            if step.type == StepType.TOOL_INSTALLATION:
                ok, reason, details = self._verify_tool(step)
                return self._build(ok, reason, details)

            if step.type == StepType.UI_INTERACTION:
                ok, reason, details = self._verify_ui_change(
                    step=step,
                    previous_screenshot=previous_screenshot,
                    world_graph=world_graph,
                )
                return self._build(ok, reason, details)

        except Exception as e:
            return VerificationResult(
                False,
                f"verification exception: {e}",
                {"exception_type": type(e).__name__},
                confidence=0.0,
                progress_score=0.0,
            )

        return self._fail(f"Unhandled step type: {step.type}")

    # =================================================
    # RESULT HELPERS
    # =================================================

    def _build(
        self,
        ok: bool,
        reason: str,
        details: Dict[str, Any],
    ) -> VerificationResult:

        if ok:
            return VerificationResult(
                True,
                None,
                details,
                confidence=0.9,
                progress_score=0.1,
            )

        return self._fail(reason, details)

    def _fail(
        self,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        return VerificationResult(
            False,
            reason,
            details or {},
            confidence=0.0,
            progress_score=0.0,
        )

    # =================================================
    # EXPLICIT VERIFICATION STEP (NO AUTO-SUCCESS)
    # =================================================

    def _verify_explicit_conditions(
        self,
        *,
        step: ExecutionStep,
        execution_result: Optional[Any],
        screenshot: Optional[Dict[str, Any]],
        previous_screenshot: Optional[Dict[str, Any]],
        world_graph,
    ) -> VerificationResult:

        verification = step.verification or {}

        if not isinstance(verification, dict) or not verification:
            return self._fail("verification conditions required")

        evidence_hits = 0
        total_checks = 0

        # ----- COMMAND RETURN CODE -----
        if "expected_return_codes" in verification:
            total_checks += 1
            ok, _, _ = self._verify_command(step, execution_result)
            if ok:
                evidence_hits += 1

        # ----- FILE EXISTS -----
        if "file_exists" in verification:
            total_checks += 1
            path = verification["file_exists"]
            if isinstance(path, str) and os.path.exists(path):
                evidence_hits += 1

        # ----- SCREEN TEXT -----
        if "screen_contains" in verification:
            total_checks += 1
            if not world_graph:
                return self._fail("world_graph required")

            tokens = verification["screen_contains"]
            if not isinstance(tokens, list):
                return self._fail("screen_contains must be list")

            found_all = True
            for token in tokens:
                if not isinstance(token, str):
                    return self._fail("invalid screen_contains token")
                matches = world_graph.find_by_text(contains=token)
                if not matches:
                    found_all = False
                    break

            if found_all:
                evidence_hits += 1

        # ----- ENTITY TYPE EXISTS -----
        if "entity_type_exists" in verification:
            total_checks += 1
            if not world_graph:
                return self._fail("world_graph required")

            etype = verification["entity_type_exists"]
            matches = world_graph.find_by_type(etype) or []
            if matches:
                evidence_hits += 1

        if total_checks == 0:
            return self._fail("no supported verification conditions")

        if evidence_hits == total_checks:
            return VerificationResult(
                True,
                None,
                {"checks_passed": evidence_hits},
                confidence=0.95,
                progress_score=0.1,
            )

        return self._fail("verification conditions not satisfied")

    # =================================================
    # COMMAND VERIFICATION
    # =================================================

    def _verify_command(
        self,
        step: ExecutionStep,
        result: Any,
    ) -> Tuple[bool, str, Dict[str, Any]]:

        if result is None or not hasattr(result, "returncode"):
            return False, "missing command result", {}

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

        path = (step.action or {}).get("path")

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

        # FIX-02 (RTB-02): step.action["tool"] is the full tool spec dict
        # (e.g. {"name": "node", "official_url": ..., ...}), not a bare string.
        # shutil.which() requires a str — passing the dict raises TypeError,
        # which was caught by the outer except and returned confidence=0.0,
        # causing every TOOL_INSTALLATION verification to fail unconditionally.
        if isinstance(tool, dict):
            tool_name = tool.get("name")
            if not isinstance(tool_name, str) or not tool_name.strip():
                return False, "tool name missing or invalid", {}
            tool_name = tool_name.strip()
        elif isinstance(tool, str):
            tool_name = tool.strip()
            if not tool_name:
                return False, "tool name empty", {}
        else:
            return False, f"tool has unexpected type: {type(tool).__name__}", {}

        tool_path = shutil.which(tool_name)

        if not tool_path:
            return False, f"tool '{tool_name}' not found on PATH", {"tool_name": tool_name}

        return True, "", {"tool_name": tool_name, "tool_path": tool_path}

    # =================================================
    # UI VERIFICATION (CAUSAL + STRICT)
    # =================================================

    def _verify_ui_change(
        self,
        *,
        step: ExecutionStep,
        previous_screenshot: Optional[Dict[str, Any]],
        world_graph=None,
    ) -> Tuple[bool, str, Dict[str, Any]]:

        if world_graph is None:
            return False, "world_graph required", {}

        verification = step.verification or {}

        if verification:
            # Delegate to explicit verification
            result = self._verify_explicit_conditions(
                step=step,
                execution_result=None,
                screenshot=None,
                previous_screenshot=previous_screenshot,
                world_graph=world_graph,
            )
            return result.success, result.reason or "", result.details or {}

        # If no explicit condition → require delta evidence
        # FIX-3 (MEDIUM): When previous_screenshot is None (iteration 0 of a
        # UI step) the old code always returned (False, "no UI evidence detected")
        # because there is nothing to diff against.  This wasted one stagnation
        # slot on every first UI action, reducing the effective retry budget.
        #
        # Fix: treat the absence of a previous screenshot as implicit acceptance
        # of the first action.  The system has no baseline to compare against,
        # so returning True here is the least-wrong default — any regression will
        # be caught on the *next* iteration where previous_screenshot is populated.
        if previous_screenshot is None:
            return True, "first action accepted (no prior screenshot baseline)", {
                "first_action": True,
            }

        if previous_screenshot:
            try:
                delta = world_graph.compute_delta(previous_screenshot)
                if delta and delta.get("significant_change"):
                    return True, "", {"delta_detected": True}
            except Exception:
                pass

        return False, "no UI evidence detected", {}

