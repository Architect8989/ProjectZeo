import sys
import os
import time
import asyncio
import platform
import uuid

from prompt_toolkit.shortcuts import message_dialog
from prompt_toolkit import prompt

from operate.exceptions import ModelNotRecognizedException
from operate.models.prompts import USER_QUESTION, get_system_prompt
from operate.config import Config
from operate.utils.style import (
    ANSI_GREEN,
    ANSI_RESET,
    ANSI_YELLOW,
    ANSI_RED,
    ANSI_BRIGHT_MAGENTA,
    ANSI_BLUE,
    style,
)
from operate.utils.operating_system import OperatingSystem
from operate.models.apis_openrouter import get_next_action

# Execution / control layers (frozen)
from utils.accessibility import AccessibilityBackend
from audit.journal import ActionJournal
from policy.engine import PolicyEngine

# Authority arbitration (frozen)
from authority.input_arbitrator import InputArbitrator
from authority.authority_policy import AuthorityDecision

# NEW: Restoration system (isolated, sovereign)
from restoration.snapshot_provider import SnapshotProvider
from restoration.restore_provider import RestoreProvider
from restoration.restore_verifier import RestoreVerifier


# ----------------------------
# GLOBAL SINGLETONS (ALLOWED)
# ----------------------------

config = Config()
operating_system = OperatingSystem()

accessibility_backend = AccessibilityBackend()
journal = ActionJournal()
policy_engine = PolicyEngine()

input_arbitrator = InputArbitrator()

EXECUTION_MODE = "ACTIVE"  # OBSERVER | ACTIVE


# ----------------------------
# ENTRYPOINT
# ----------------------------

def main(model, terminal_prompt, voice_mode=False, verbose_mode=False):
    mic = None
    config.verbose = verbose_mode
    config.validation(model, voice_mode)

    if voice_mode:
        try:
            from whisper_mic import WhisperMic
            mic = WhisperMic()
        except ImportError:
            sys.exit(1)

    if not terminal_prompt:
        message_dialog(
            title="Self-Operating Computer",
            text="An experimental framework to enable multimodal models to operate computers",
            style=style,
        ).run()

    if platform.system() == "Windows":
        os.system("cls")
    else:
        print("\033c", end="")

    if terminal_prompt:
        objective = terminal_prompt
    elif voice_mode:
        objective = mic.listen()
    else:
        print(
            f"[{ANSI_GREEN}SOC{ANSI_RESET}|{ANSI_BRIGHT_MAGENTA} {model}{ANSI_RESET}]\n"
            f"{USER_QUESTION}"
        )
        print(f"{ANSI_YELLOW}[User]{ANSI_RESET}")
        objective = prompt(style=style)

    system_prompt = get_system_prompt(model, objective)
    messages = [{"role": "system", "content": system_prompt}]

    execution_id = str(uuid.uuid4())
    journal.open(session_id=execution_id, reason="OBJECTIVE_START")

    # --------------------------------------------------
    # NEW: Pre-hijack snapshot (HARD GATE)
    # --------------------------------------------------

    snapshot = None
    snapshot_provider = SnapshotProvider(
        observer=accessibility_backend.observer,
        screenpipe=accessibility_backend.screenpipe,
        os_backend=operating_system,
    )

    restore_provider = RestoreProvider(os_backend=operating_system)
    restore_verifier = RestoreVerifier(os_backend=operating_system)

    try:
        snapshot = snapshot_provider.capture_pre_hijack_snapshot()
    except Exception as e:
        journal.record(
            event="snapshot_failed",
            execution_id=execution_id,
            error=str(e),
        )
        journal.seal(reason="SNAPSHOT_FAILURE")
        raise

    session_id = None

    try:
        while True:
            operations, session_id = asyncio.run(
                get_next_action(model, messages, objective, session_id)
            )

            stop = operate(
                operations=operations,
                model=model,
                execution_id=execution_id,
            )

            if stop:
                break

    except ModelNotRecognizedException as e:
        journal.record(event="fatal_error", detail=str(e))

    except Exception as e:
        journal.record(event="fatal_error", detail=str(e))

    finally:
        # --------------------------------------------------
        # NEW: Guaranteed restoration + verification
        # --------------------------------------------------
        if snapshot is not None:
            try:
                restore_provider.restore(snapshot)
                restore_verifier.verify(snapshot)
                journal.record(
                    event="restoration_verified",
                    execution_id=execution_id,
                )
            except Exception as e:
                journal.record(
                    event="restoration_failed",
                    execution_id=execution_id,
                    error=str(e),
                )

        journal.seal(reason="OBJECTIVE_COMPLETE")


# ----------------------------
# EXECUTION LOOP (UNCHANGED)
# ----------------------------

def operate(operations, model, execution_id: str):
    frozen_nodes = accessibility_backend.get_nodes()

    for operation in operations:
        time.sleep(1)

        op_type = operation.get("operation", "").lower()
        thought = operation.get("thought")
        detail = ""

        journal.record(
            event="operation_start",
            execution_id=execution_id,
            operation=op_type,
            thought=thought,
        )

        decision = input_arbitrator.evaluate(
            input_event_ts=time.monotonic(),
            high_risk=False,
            soc_confident=True,
        )

        if decision == AuthorityDecision.YIELD:
            journal.record(event="authority_yield", execution_id=execution_id)
            return True

        if decision == AuthorityDecision.ABORT:
            journal.record(event="authority_abort", execution_id=execution_id)
            return True

        try:
            input_arbitrator.soc_action_started()

            if op_type in ("press", "hotkey"):
                detail = operation.get("keys")
                operating_system.press(detail)

            elif op_type == "write":
                detail = operation.get("content")
                operating_system.write(detail)

            elif op_type == "click":
                detail = {"x": operation.get("x"), "y": operation.get("y")}
                operating_system.mouse(detail)

            elif op_type == "done":
                summary = operation.get("summary")
                journal.record(
                    event="objective_complete",
                    execution_id=execution_id,
                    summary=summary,
                )
                return True

            else:
                journal.record(
                    event="unknown_operation",
                    execution_id=execution_id,
                    detail=operation,
                )
                return True

        except Exception as e:
            journal.record(
                event="operation_abort",
                execution_id=execution_id,
                operation=op_type,
                error=str(e),
            )
            return True

        journal.record(
            event="operation_complete",
            execution_id=execution_id,
            operation=op_type,
            detail=detail,
        )

        print(
            f"[{ANSI_GREEN}SOC{ANSI_RESET}|{ANSI_BRIGHT_MAGENTA} {model}{ANSI_RESET}]"
        )
        print(thought)
        print(f"{ANSI_BLUE}Action:{ANSI_RESET} {op_type} {detail}\n")

    return False
