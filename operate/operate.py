import sys
import os
import time
import asyncio
from prompt_toolkit.shortcuts import message_dialog
from prompt_toolkit import prompt
from operate.exceptions import ModelNotRecognizedException
import platform

from operate.models.prompts import (
    USER_QUESTION,
    get_system_prompt,
)
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
from operate.models.apis import get_next_action

# Execution instrument
from utils.accessibility import AccessibilityBackend

# Cryptographic execution ledger
from audit.journal import ActionJournal

# Load configuration
config = Config()
operating_system = OperatingSystem()

# Instantiate once (instruments, not controllers)
accessibility_backend = AccessibilityBackend()
journal = ActionJournal()

# Execution mode (will later be lifecycle-driven)
EXECUTION_MODE = "ACTIVE"  # or "OBSERVER"


def main(model, terminal_prompt, voice_mode=False, verbose_mode=False):
    """
    Main function for the Self-Operating Computer.
    """

    mic = None

    config.verbose = verbose_mode
    config.validation(model, voice_mode)

    if voice_mode:
        try:
            from whisper_mic import WhisperMic
            mic = WhisperMic()
        except ImportError:
            print(
                "Voice mode requires the 'whisper_mic' module. "
                "Please install it using 'pip install -r requirements-audio.txt'"
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
        print(
            f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_RESET} Listening for your command..."
        )
        try:
            objective = mic.listen()
        except Exception as e:
            print(f"{ANSI_RED}Voice input error: {e}{ANSI_RESET}")
            return
    else:
        print(
            f"[{ANSI_GREEN}Self-Operating Computer {ANSI_RESET}|"
            f"{ANSI_BRIGHT_MAGENTA} {model}{ANSI_RESET}]\n{USER_QUESTION}"
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
        print(
            f"{ANSI_GREEN}[Self-Operating Computer]"
            f"{ANSI_RED}[Error] -> {e}{ANSI_RESET}"
        )
    except Exception as e:
        print(
            f"{ANSI_GREEN}[Self-Operating Computer]"
            f"{ANSI_RED}[Error] -> {e}{ANSI_RESET}"
        )
    finally:
        # Seal audit ledger at task boundary
        journal.seal(reason="OBJECTIVE_COMPLETE")


def operate(operations, model):
    """
    Core SOC operation loop.
    Legacy execution preserved.
    Journal is wired but only used when AccessibilityBackend executes.
    """

    for operation in operations:
        time.sleep(1)

        operate_type = operation.get("operation", "").lower()
        operate_thought = operation.get("thought")
        operate_detail = ""

        if operate_type in ("press", "hotkey"):
            keys = operation.get("keys")
            operate_detail = keys
            operating_system.press(keys)

        elif operate_type == "write":
            content = operation.get("content")
            operate_detail = content
            operating_system.write(content)

        elif operate_type == "click":
            # Legacy coordinate-based click (unchanged)
            x = operation.get("x")
            y = operation.get("y")
            operate_detail = {"x": x, "y": y}
            operating_system.mouse(operate_detail)

            # NOTE:
            # Node-based execution will later replace this path:
            # accessibility_backend.execute(
            #     mode=EXECUTION_MODE,
            #     policy_engine=policy_engine,
            #     audit_callback=journal.record,
            #     node=node,
            #     action_type="click"
            # )

        elif operate_type == "done":
            summary = operation.get("summary")
            print(
                f"[{ANSI_GREEN}Self-Operating Computer {ANSI_RESET}|"
                f"{ANSI_BRIGHT_MAGENTA} {model}{ANSI_RESET}]"
            )
            print(f"{ANSI_BLUE}Objective Complete: {ANSI_RESET}{summary}\n")
            return True

        else:
            print(
                f"{ANSI_GREEN}[Self-Operating Computer]"
                f"{ANSI_RED}[Error] Unknown operation{ANSI_RESET}"
            )
            print(operation)
            return True

        print(
            f"[{ANSI_GREEN}Self-Operating Computer {ANSI_RESET}|"
            f"{ANSI_BRIGHT_MAGENTA} {model}{ANSI_RESET}]"
        )
        print(operate_thought)
        print(f"{ANSI_BLUE}Action: {ANSI_RESET}{operate_type} {operate_detail}\n")

    return False
