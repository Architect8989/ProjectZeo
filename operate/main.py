"""
Self-Operating Computer – Execution Wrapper
"""

import argparse
import sys

from operate.utils.style import ANSI_BRIGHT_MAGENTA
from operate.operate import operate_main
from core.telemetry.logger import log_info, log_error

from operate.utils.operating_system import OperatingSystem
from utils.accessibility import AccessibilityBackend
from audit.journal import ActionJournal
from authority.input_arbitrator import InputArbitrator


def main_entry(
    *,
    model: str = None,
    terminal_prompt: str = None,
    voice_mode: bool = False,
    verbose_mode: bool = False,
    observer=None,
):
    """
    Thin execution wrapper.

    Responsibilities:
    - Parse CLI or accept programmatic input
    - Construct execution dependencies
    - Call operate_main()

    Does NOT:
    - manage lifecycle
    - manage snapshots
    - manage authority state
    """

    try:
        # --------------------------------------------------
        # CONFIG RESOLUTION
        # --------------------------------------------------
        if terminal_prompt is None:
            parser = argparse.ArgumentParser(
                description="Run the self-operating computer."
            )

            parser.add_argument(
                "-m", "--model",
                default=None,
            )

            parser.add_argument(
                "--prompt",
                type=str,
                required=True,
            )

            args = parser.parse_args()

            model = args.model
            objective = args.prompt
        else:
            objective = terminal_prompt

        if observer is None:
            raise RuntimeError(
                "observer is required for execution"
            )

        # --------------------------------------------------
        # DEPENDENCY CONSTRUCTION (EXPLICIT)
        # --------------------------------------------------
        os_backend = OperatingSystem()

        accessibility_backend = AccessibilityBackend()
        accessibility_backend.wire(observer=observer)

        journal = ActionJournal()
        input_arbitrator = InputArbitrator()

        log_info("[SYSTEM] Starting SOC execution")

        # --------------------------------------------------
        # EXECUTION
        # --------------------------------------------------
        operate_main(
            model=model,
            terminal_prompt=objective,
            execution_plan=None,  # expected to be provided by caller in SOC flow
            observer=observer,
        )

    except KeyboardInterrupt:
        log_info("[SYSTEM] Keyboard interrupt received")
        print(f"\n{ANSI_BRIGHT_MAGENTA}Exiting...")
        sys.exit(0)

    except Exception as e:
        log_error(f"[SYSTEM] Fatal startup error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main_entry()
