"""
Self-Operating Computer - Deterministic Control Kernel
Entry Point: Enterprise-Grade Edition
"""
import argparse
import sys
import json
from operate.utils.style import ANSI_BRIGHT_MAGENTA
from self_operating_computer.operating_system.main import OperatingSystem
from self_operating_computer.models.gpt_4_vision import format_prompt, parse_model_response

# Mock for model interface - replace with actual API client in production
def call_llm(prompt, model_name):
    # This must return a raw JSON string from the model provider
    pass

def main_entry():
    parser = argparse.ArgumentParser(
        description="Run the deterministic self-operating-computer."
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Specify the model (must support JSON output)",
        required=False,
        default="gpt-4-vision-preview",
    )
    parser.add_argument(
        "--prompt",
        help="Directly input the objective prompt",
        type=str,
        required=False,
    )

    args = parser.parse_args()
    
    # 1. Initialize Control Kernel
    # Automatically initializes: Backend, Policy Engine, and Action Journal
    try:
        os_controller = OperatingSystem()
    except Exception as e:
        print(f"Failed to initialize Control Kernel: {e}")
        sys.exit(1)

    if not args.prompt:
        print(f"{ANSI_BRIGHT_MAGENTA}Error: Objective prompt required for deterministic execution.")
        sys.exit(1)

    objective = args.prompt
    print(f"{ANSI_BRIGHT_MAGENTA}Objective: {objective}")

    try:
        while True:
            # 2. Stabilized UI Snapshot
            # backend.get_nodes() is called within get_state_summary() 
            # and enforces stabilization via wait_for_ui_stabilization()
            ui_summary = os_controller.get_state_summary() 
            
            # 3. Model Reasoning
            prompt = format_prompt(ui_summary, objective)
            
            # Simulated model call - In production, this receives the model's JSON
            raw_response = call_llm(prompt, args.model)
            
            # 4. Deterministic Execution
            # This method now performs:
            # - Poison check (coordinate scanning)
            # - Policy v1 validation (ACL/Regex/Role checking)
            # - Action journaling (Pre-action)
            # - Semantic OS interaction (No synthetic fallbacks)
            # - State-Diff Enforcement (Termination if pre_sig == post_sig)
            os_controller.execute_deterministic_action(raw_response)
            
            print(f"{ANSI_BRIGHT_MAGENTA}Step successful. State mutation verified and journaled.")
            
    except KeyboardInterrupt:
        os_controller.shutdown()
        print(f"\n{ANSI_BRIGHT_MAGENTA}Manual Interruption. Journal Sealed. Exiting...")
    except SystemExit:
        # sys.exit(1) is called by OperatingSystem for Policy/State/Poison violations
        # Journal is already sealed at the point of violation.
        pass
    except Exception as e:
        # Final safety net for unexpected kernel panic
        if hasattr(os_controller, 'journal'):
            os_controller.journal.record({
                "outcome": "CRITICAL_FATAL", 
                "error": str(e),
                "trace": "main_loop_panic"
            })
            os_controller.shutdown()
        print(f"Fatal Kernel Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main_entry()
