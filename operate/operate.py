import time
import asyncio
import math

from operate.exceptions import ModelNotRecognizedException
from operate.models.apis_openrouter import get_next_action

from authority.authority_policy import AuthorityDecision
from core.safety.action_timeout import action_timeout, ActionTimeout
from core.telemetry.logger import log_warn


# --------------------------------------------------
# PURE EXECUTION ENTRYPOINT
# --------------------------------------------------

def execute_soc(
    *,
    model,
    objective,
    observer,
    screenpipe,               # read-only, unused here by contract
    os_backend,
    accessibility_backend,
    journal,
    input_arbitrator,
    max_iterations=500,
):
    """
    SOC executor.

    HARD CONTRACT:
    - NO mode transitions
    - NO snapshot control
    - NO restoration
    - NO authority ownership
    """

    if not objective:
        raise ValueError("objective is required")

    messages = []
    session_id = None
    iteration = 0

    try:
        while True:
            os_backend.heartbeat()

            iteration += 1
            if iteration > max_iterations:
                raise RuntimeError("Iteration budget exceeded")

            # ----------------------------
            # PLANNING (LLM)
            # ----------------------------
            operations, session_id = asyncio.run(
                get_next_action(
                    model,
                    messages,
                    objective,
                    session_id,
                )
            )

            os_backend.heartbeat()

            # ----------------------------
            # VALIDATION (HARD GATE)
            # ----------------------------
            if not isinstance(operations, list):
                journal.record(
                    event="invalid_plan",
                    reason="operations_not_list",
                    value=str(type(operations)),
                )
                return

            # ----------------------------
            # EXECUTION
            # ----------------------------
            stop = _execute_operations(
                operations=operations,
                observer=observer,
                os_backend=os_backend,
                accessibility_backend=accessibility_backend,
                journal=journal,
                input_arbitrator=input_arbitrator,
            )

            if stop:
                return

    except ModelNotRecognizedException as e:
        journal.record(event="fatal_error", error=str(e))
        raise

    except Exception as e:
        journal.record(event="fatal_error", error=str(e))
        raise


# --------------------------------------------------
# EXECUTION + VERIFICATION
# --------------------------------------------------

def _execute_operations(
    *,
    operations,
    observer,
    os_backend,
    accessibility_backend,
    journal,
    input_arbitrator,
):
    if not operations:
        journal.record(event="no_operations")
        return True

    pre_state = observer.get_state()

    for operation in operations:
        time.sleep(0.25)
        os_backend.heartbeat()

        if not isinstance(operation, dict):
            journal.record(
                event="invalid_operation",
                reason="not_dict",
                value=str(operation),
            )
            return True

        op_type = operation.get("operation")
        thought = operation.get("thought")

        if not isinstance(op_type, str):
            journal.record(
                event="invalid_operation",
                reason="missing_operation_type",
                operation=operation,
            )
            return True

        op_type = op_type.lower()

        journal.record(
            event="operation_start",
            operation=op_type,
            thought=thought,
        )

        # ----------------------------
        # AUTHORITY CHECK
        # ----------------------------
        decision = input_arbitrator.evaluate(
            input_event_ts=time.monotonic(),
            high_risk=False,
            soc_confident=True,
        )

        if decision in (
            AuthorityDecision.YIELD,
            AuthorityDecision.ABORT,
        ):
            journal.record(event=f"authority_{decision.name.lower()}")
            return True

        try:
            input_arbitrator.soc_action_started()

            with action_timeout(5):
                if op_type in ("press", "hotkey"):
                    keys = operation.get("keys")
                    if not keys:
                        raise ValueError("missing keys")
                    os_backend.press(keys)

                elif op_type == "write":
                    content = operation.get("content")
                    if not isinstance(content, str):
                        raise ValueError("invalid write content")
                    os_backend.write(content)

                elif op_type == "click":
                    x = operation.get("x")
                    y = operation.get("y")

                    if not _valid_coord(x) or not _valid_coord(y):
                        raise ValueError(f"invalid click coordinates: {x}, {y}")

                    os_backend.mouse({"x": x, "y": y})

                elif op_type == "done":
                    journal.record(
                        event="objective_complete",
                        summary=operation.get("summary"),
                    )
                    return True

                else:
                    journal.record(
                        event="unknown_operation",
                        detail=operation,
                    )
                    return True

        except ActionTimeout:
            log_warn(f"[SOC] Action timeout: {operation}")
            journal.record(
                event="action_timeout",
                operation=op_type,
            )
            return True

        except Exception as e:
            journal.record(
                event="operation_abort",
                operation=op_type,
                error=str(e),
            )
            return True

        os_backend.heartbeat()

        # ----------------------------
        # VERIFICATION (MINIMUM SAFE)
        # ----------------------------
        post_state = observer.get_state()

        if not _state_changed(pre_state, post_state):
            journal.record(
                event="verification_failed",
                operation=op_type,
                detail="no observable change",
            )
            return True

        journal.record(
            event="operation_complete",
            operation=op_type,
        )

        pre_state = post_state

    return False


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def _valid_coord(v):
    return isinstance(v, (int, float)) and not math.isnan(v) and 0.0 <= v <= 1.0


def _state_changed(a, b):
    if a is b:
        return False
    if not isinstance(a, dict) or not isinstance(b, dict):
        return True
    return a.get("screen_hash") != b.get("screen_hash")
