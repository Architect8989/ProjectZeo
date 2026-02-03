import time
import asyncio

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
    screenpipe,
    os_backend,
    accessibility_backend,
    journal,
    input_arbitrator,
    max_iterations=500,
):
    """
    SOC executor.

    Responsibilities:
    - plan
    - act
    - verify
    - yield control

    NO authority ownership.
    NO lifecycle control.
    """

    messages = []
    session_id = None
    iteration = 0

    while True:
        os_backend.heartbeat()

        iteration += 1
        if iteration > max_iterations:
            raise RuntimeError("Iteration budget exceeded")

        operations, session_id = asyncio.run(
            get_next_action(model, messages, objective, session_id)
        )

        os_backend.heartbeat()

        stop = _execute_operations(
            operations=operations,
            model=model,
            observer=observer,
            os_backend=os_backend,
            accessibility_backend=accessibility_backend,
            journal=journal,
            input_arbitrator=input_arbitrator,
        )

        if stop:
            return


# --------------------------------------------------
# EXECUTION + VERIFICATION
# --------------------------------------------------

def _execute_operations(
    *,
    operations,
    model,
    observer,
    os_backend,
    accessibility_backend,
    journal,
    input_arbitrator,
):
    if not operations:
        journal.record(event="no_operations")
        return True

    # Snapshot perception BEFORE actions
    pre_state = observer.get_state()
    frozen_nodes = accessibility_backend.get_nodes()

    for operation in operations:
        time.sleep(0.5)
        os_backend.heartbeat()

        op_type = operation.get("operation", "").lower()
        thought = operation.get("thought")
        detail = None

        journal.record(
            event="operation_start",
            operation=op_type,
            thought=thought,
        )

        decision = input_arbitrator.evaluate(
            input_event_ts=time.monotonic(),
            high_risk=False,
            soc_confident=True,
        )

        if decision in (AuthorityDecision.YIELD, AuthorityDecision.ABORT):
            journal.record(event=f"authority_{decision.name.lower()}")
            return True

        try:
            input_arbitrator.soc_action_started()

            with action_timeout(5):
                if op_type in ("press", "hotkey"):
                    detail = operation.get("keys")
                    os_backend.press(detail)

                elif op_type == "write":
                    detail = operation.get("content")
                    os_backend.write(detail)

                elif op_type == "click":
                    detail = {
                        "x": operation.get("x"),
                        "y": operation.get("y"),
                    }
                    os_backend.mouse(detail)

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
            journal.record(event="action_timeout", operation=op_type)
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
        # VERIFICATION (CRITICAL)
        # ----------------------------

        post_state = observer.get_state()

        if pre_state == post_state:
            journal.record(
                event="verification_failed",
                operation=op_type,
                detail="no observable state change",
            )
            return True

        journal.record(
            event="operation_complete",
            operation=op_type,
            detail=detail,
        )

        pre_state = post_state

    return False
