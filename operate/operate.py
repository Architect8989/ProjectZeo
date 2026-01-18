import sys
import os
import time
import asyncio
import platform

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

# Execution / control layers (frozen, non-sovereign)
from utils.accessibility import AccessibilityBackend
from audit.journal import ActionJournal
from policy.engine import PolicyEngine


# ----------------------------
# GLOBAL SINGLETONS (ALLOWED)
# ----------------------------

config = Config()
operating_system = OperatingSystem()

accessibility_backend = AccessibilityBackend()
journal = ActionJournal()
policy_engine = PolicyEngine()

# Lifecycle flag (formalized later)
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
            print(
                "Voice mode requires 'whisper_mic'. "
                "Install via 'pip install -r requirements-audio.txt'"
            )
            sys.exit(1)

    if not terminal_prompt:
        message_dialog(
            title="Self-Operating Computer",
            text="An experimental framework to enable multimodal models to operate computers",
            style=style,
        ).run()
    else:
        print("Running direct prompt...")

    # Clear console
    if platform.system() == "Windows":
        os.system("cls")
    else:
        print("\033c", end="")

    if terminal_prompt:
        objective = terminal_prompt
    elif voice_mode:
        print(f"{ANSI_GREEN}[SOC]{ANSI_RESET} Listening...")
        try:
            objective = mic.listen()
        except Exception as e:
            print(f"{ANSI_RED}Voice input error: {e}{ANSI_RESET}")
            return
    else:
        print(
            f"[{ANSI_GREEN}SOC{ANSI_RESET}|{ANSI_BRIGHT_MAGENTA} {model}{ANSI_RESET}]\n"
            f"{USER_QUESTION}"
        )
        print(f"{ANSI_YELLOW}[User]{ANSI_RESET}")
        objective = prompt(style=style)

    system_prompt = get_system_prompt(model, objective)
    messages = [{"role": "system", "content": system_prompt}]

    loop_count = 0
    session_id = None

    try:
        while True:
            operations, session_id = asyncio.run(
                get_next_action(model, messages, objective, session_id)
            )

            stop = operate(operations, model)
            if stop:
                break

            loop_count += 1
            if loop_count > 10:
                break

    except ModelNotRecognizedException as e:
        print(f"{ANSI_RED}[Error] {e}{ANSI_RESET}")
    except Exception as e:
        print(f"{ANSI_RED}[Error] {e}{ANSI_RESET}")
    finally:
        # Ledger must always seal
        journal.seal(reason="OBJECTIVE_COMPLETE")


# ----------------------------
# EXECUTION LOOP
# ----------------------------

def operate(operations, model):
    """
    Core SOC execution loop.

    - Legacy execution preserved
    - Node-based execution available
    - Policy + Audit enforced ONLY via AccessibilityBackend
    """

    # Freeze UI context ONCE per step batch (future use)
    frozen_nodes = accessibility_backend.get_nodes()

    for operation in operations:
        time.sleep(1)

        op_type = operation.get("operation", "").lower()
        thought = operation.get("thought")
        detail = ""

        # ----------------------------
        # LEGACY PATHS (UNCHANGED)
        # ----------------------------

        if op_type in ("press", "hotkey"):
            keys = operation.get("keys")
            detail = keys
            operating_system.press(keys)

        elif op_type == "write":
            content = operation.get("content")
            detail = content
            operating_system.write(content)

        elif op_type == "click":
            # Legacy coordinate click (still allowed)
            x = operation.get("x")
            y = operation.get("y")
            detail = {"x": x, "y": y}
            operating_system.mouse(detail)

        # ----------------------------
        # FUTURE NODE-BASED PATH
        # ----------------------------
        # elif op_type == "click_node":
        #     node_id = operation.get("node_id")
        #     node = frozen_nodes.get(node_id)
        #
        #     accessibility_backend.execute(
        #         mode=EXECUTION_MODE,
        #         policy_engine=policy_engine,
        #         audit_callback=journal.record,
        #         node=node,
        #         action_type="click"
        #     )

        elif op_type == "done":
            summary = operation.get("summary")
            print(
                f"[{ANSI_GREEN}SOC{ANSI_RESET}|{ANSI_BRIGHT_MAGENTA} {model}{ANSI_RESET}]"
            )
            print(f"{ANSI_BLUE}Objective Complete:{ANSI_RESET} {summary}\n")
            return True

        else:
            print(f"{ANSI_RED}[Error] Unknown operation{ANSI_RESET}")
            print(operation)
            return True

        # ----------------------------
        # DISPLAY
        # ----------------------------

        print(
            f"[{ANSI_GREEN}SOC{ANSI_RESET}|{ANSI_BRIGHT_MAGENTA} {model}{ANSI_RESET}]"
        )
        print(thought)
        print(f"{ANSI_BLUE}Action:{ANSI_RESET} {op_type} {detail}\n")

    return False
