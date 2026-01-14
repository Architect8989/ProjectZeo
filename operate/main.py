"""
Self-Operating Computer - Deterministic Control Kernel
Entry Point: Enterprise-Grade Edition (Kernel-Aligned)
"""

import argparse
import sys
import json
from operate.utils.style import ANSI_BRIGHT_MAGENTA
from self_operating_computer.operating_system.main import OperatingSystem, StepResult
from self_operating_computer.models.gpt_4_vision import format_prompt


# Production model interface must return RAW JSON STRING
def call_llm(prompt: str, model_name: str) -> str:
    """
    This function must synchronously return a JSON string:
    {"node_id": "...", "action": "...", "text": "..."}
    """
    raise NotImplementedError("Model provider not wired")


def main_entry():
    parser = argparse.ArgumentParser(
        description="Run the deterministic self-operating-computer kernel."
    )
    parser.add_argument(
        "-m",
        "--model",
        default="gpt-4-vision-preview",
        help="Model name (must support strict JSON output)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Objective prompt (required)",
    )

    args = parser.parse_args()

    # 1. Initialize Kernel (Authority Holder)
    try:
        os_controller = OperatingSystem(max_steps=50)
    except Exception as e:
        print(f"Kernel initialization failure: {e}")
        sys.exit(1)

    objective = args.prompt
    print(f"{ANSI_BRIGHT_MAGENTA}Objective:{ANSI_BRIGHT_MAGENTA} {objective}")

    try:
        while True:
            # 2. Kernel-Owned State Snapshot
            ui_summary = os_controller.get_state_summary()

            # 3. Model Proposal (NO authority)
            prompt = format_prompt(ui_summary, objective)
            raw_json = call_llm(prompt, args.model)

            # 4. Single Kernel Step
            signal = os_controller.step(raw_json)

            if signal == StepResult.FAIL:
                print(f"{ANSI_BRIGHT_MAGENTA}Kernel violation detected. Execution terminated.")
                os_controller.shutdown("FAIL")
                sys.exit(1)

            # CONTINUE means verified mutation, bounded by kernel
            print(f"{ANSI_BRIGHT_MAGENTA}Step {os_controller.step_count} verified.")

    except KeyboardInterrupt:
        os_controller.shutdown("INTERRUPTED")
        print(f"\n{ANSI_BRIGHT_MAGENTA}Manual interruption. Journal sealed.")
    except Exception as e:
        if hasattr(os_controller, "journal"):
            os_controller.journal.record({
                "outcome": "CRITICAL_FATAL",
                "error": str(e),
                "trace": "main_entry_panic"
            })
            os_controller.shutdown("CRITICAL_FATAL")
        raise


if __name__ == "__main__":
    main_entry()
