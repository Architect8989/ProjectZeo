import sys
import os
import json
import platform
from prompt_toolkit.shortcuts import message_dialog
from prompt_toolkit import prompt

from self_operating_computer.operating_system.main import OperatingSystem, StepResult
from self_operating_computer.models.interface import ModelProvider
from operate.utils.style import (
    ANSI_GREEN,
    ANSI_RESET,
    ANSI_RED,
    ANSI_BRIGHT_MAGENTA,
    style,
)

def main(model, terminal_prompt, voice_mode=False, verbose_mode=False):
    """
    Kernel-aligned orchestrator.
    Pure I/O pump. Zero execution authority.
    """

    # 1. Initialize Kernel (Authority Holder)
    try:
        os_controller = OperatingSystem(max_steps=50)
        model_worker = ModelProvider(provider_type="openai")
    except Exception as e:
        print(f"{ANSI_RED}[Initialization Error] {e}{ANSI_RESET}")
        sys.exit(1)

    # 2. Objective Acquisition
    if not terminal_prompt:
        message_dialog(
            title="ProjectZeo Enterprise",
            text="Deterministic OS Control Kernel via Semantic Accessibility",
            style=style,
        ).run()
        print(f"[{ANSI_GREEN}Self-Operating Computer{ANSI_RESET}]\nWhat is your objective?")
        objective = prompt(style=style)
    else:
        objective = terminal_prompt

    # Clear console (cosmetic only)
    if platform.system() == "Windows":
        os.system("cls")
    else:
        print("\033c", end="")

    print(f"{ANSI_BRIGHT_MAGENTA}Objective:{ANSI_RESET} {objective}")

    # 3. Kernel-Stepped Execution Loop
    while True:
        try:
            if verbose_mode:
                print(f"[Kernel] Step {os_controller.step_count} - capturing state")

            # STEP A: Kernel-owned state snapshot
            ui_summary = os_controller.get_state_summary()

            # STEP B: Model proposes next action (NO authority)
            action_data = model_worker.generate_action(ui_summary, objective)

            # STEP C: Kernel step (single source of truth)
            signal = os_controller.step(json.dumps(action_data))

            if signal == StepResult.FAIL:
                print(f"{ANSI_RED}Kernel violation. Execution terminated.{ANSI_RESET}")
                os_controller.shutdown("FAIL")
                sys.exit(1)

            # CONTINUE means mutation verified
            if verbose_mode:
                print(
                    f"{ANSI_BRIGHT_MAGENTA}"
                    f"Step {os_controller.step_count} verified and journaled."
                    f"{ANSI_RESET}"
                )

        except KeyboardInterrupt:
            print(f"{ANSI_BRIGHT_MAGENTA}Manual interruption detected. Sealing journal.{ANSI_RESET}")
            os_controller.shutdown("INTERRUPTED")
            break

        except Exception as e:
            if hasattr(os_controller, "journal"):
                os_controller.journal.record({
                    "outcome": "FATAL_EXCEPTION",
                    "error": str(e),
                    "trace": "operate_loop_panic",
                })
                os_controller.shutdown("FATAL_EXCEPTION")
            print(f"{ANSI_RED}[Fatal Kernel Error] -> {e}{ANSI_RESET}")
            sys.exit(1)
