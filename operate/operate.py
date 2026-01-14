import sys
import os
import json
import platform
from prompt_toolkit.shortcuts import message_dialog
from prompt_toolkit import prompt

# Enterprise Kernel Imports
from self_operating_computer.operating_system.main import OperatingSystem
from self_operating_computer.models.interface import ModelProvider
from operate.utils.style import (
    ANSI_GREEN,
    ANSI_RESET,
    ANSI_YELLOW,
    ANSI_RED,
    ANSI_BRIGHT_MAGENTA,
    style,
)

def main(model, terminal_prompt, voice_mode=False, verbose_mode=False):
    """
    Main entry point for the Deterministic Control Kernel.
    Removes vision-based loops and enforces semantic OS control.
    """
    # 1. Initialize Authority Layers
    # Initializes: AT-SPI Backend, Policy Engine v1, and Action Journal
    try:
        os_controller = OperatingSystem()
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

    # Clear Console for clean audit monitoring
    if platform.system() == "Windows":
        os.system("cls")
    else:
        print("\033c", end="")

    print(f"{ANSI_BRIGHT_MAGENTA}Objective:{ANSI_RESET} {objective}")

    # 3. Execution Loop (The Control Kernel)
    loop_count = 0
    while True:
        try:
            if verbose_mode:
                print(f"[Kernel] Iteration {loop_count} - Stabilizing UI...")

            # STEP A: Capture Semantic State
            # wait_for_ui_stabilization() is called internally
            ui_summary = os_controller.get_state_summary()

            # STEP B: Request Action from Reasoning Engine
            # Model receives JSON tree, NOT a screenshot.
            action_data = model_worker.generate_action(ui_summary, objective)

            # Check for termination signal from model
            if action_data.get("action") == "done":
                print(f"{ANSI_GREEN}Objective Complete:{ANSI_RESET} {action_data.get('summary')}")
                os_controller.shutdown()
                break

            # STEP C: Execute via Deterministic Kernel
            # Performs: Poison Scan -> Policy Check -> Action -> State-Diff Verification
            # This method triggers sys.exit(1) on any failure (Policy/Causality)
            os_controller.execute_deterministic_action(json.dumps(action_data))

            loop_count += 1
            
            # Final Safety: Max iterations to prevent logic drift in production
            if loop_count > 50:
                print(f"{ANSI_RED}[Guardrail] Max loop iterations reached.{ANSI_RESET}")
                os_controller.journal.record({"outcome": "LIMIT_EXCEEDED", "reason": "Max iterations"})
                os_controller.shutdown()
                break

        except Exception as e:
            # Fatal Kernel Panic: Record and Halt
            if hasattr(os_controller, 'journal'):
                os_controller.journal.record({
                    "outcome": "FATAL_EXCEPTION",
                    "error": str(e)
                })
                os_controller.shutdown()
            print(f"{ANSI_RED}[Fatal Exception] -> {e}{ANSI_RESET}")
            sys.exit(1)

# Deprecated: The old 'operate' function is replaced by the Kernel executor
# to ensure all actions are validated against the Policy Engine.
