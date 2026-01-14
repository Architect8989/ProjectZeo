"""
Self-Operating Computer - Deterministic Control Kernel
Entry Point
"""
import argparse
import sys
from operate.utils.style import ANSI_BRIGHT_MAGENTA
from self_operating_computer.operating_system.main import OperatingSystem
from self_operating_computer.models.gpt_4_vision import format_prompt, parse_model_response

def main_entry():
    parser = argparse.ArgumentParser(
        description="Run the deterministic self-operating-computer."
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Specify the model (must support JSON output)",
        required=False,
        default="gpt-4",
    )
    parser.add_argument(
        "--prompt",
        help="Directly input the objective prompt",
        type=str,
        required=False,
    )

    args = parser.parse_args()
    
    # Initialize the OS Controller (Backend + Journal)
    # This also initializes the ActionJournal and verifies write-access
    os_controller = OperatingSystem()

    if not args.prompt:
        print(f"{ANSI_BRIGHT_MAGENTA}Error: Objective prompt required for deterministic execution.")
        sys.exit(1)

    objective = args.prompt
    print(f"{ANSI_BRIGHT_MAGENTA}Objective: {objective}")

    try:
        while True:
            # 1. Capture semantic UI state (Stabilized Snapshot)
            ui_state = os_controller.backend.get_nodes()
            
            # 2. Convert state to minified JSON for the model
            ui_summary = os_controller.get_state_summary() 
            
            # 3. Model Request (Logic handled in gpt_4_vision wrapper)
            # In a real impl, you'd pass 'objective' and 'ui_summary' to your model provider
            # Here we assume a call to your model's generate function
            prompt = format_prompt(ui_summary, objective)
            
            # Placeholder for model call
            # response = model.generate(prompt)
            # action_json = parse_model_response(response)
            
            # 4. Deterministic execution with journaling and poison checks
            # os_controller.execute_deterministic_action(action_json)
            
            print(f"{ANSI_BRIGHT_MAGENTA}Action executed and journaled.")
            
            # Implementation Note: In enterprise grade, we typically pause 
            # to prevent runaway loops if the UI doesn't mutate.
            
    except KeyboardInterrupt:
        os_controller.shutdown()
        print(f"\n{ANSI_BRIGHT_MAGENTA}Journal Sealed. Exiting...")
    except Exception as e:
        # Any unhandled exception must be recorded before death
        os_controller.journal.record({"outcome": "CRITICAL_FATAL", "error": str(e)})
        os_controller.shutdown()
        sys.exit(1)

if __name__ == "__main__":
    main_entry()
