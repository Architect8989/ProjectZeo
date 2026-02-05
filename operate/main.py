"""
Self-Operating Computer – Execution Wrapper
"""

import argparse
import sys

from operate.utils.style import ANSI_BRIGHT_MAGENTA
from operate.operate import execute_soc
from core.telemetry.logger import log_info, log_error

from operate.utils.operating_system import OperatingSystem
from utils.accessibility import AccessibilityBackend
from audit.journal import AuditJournal
from authority.input_arbitrator import InputArbitrator


def main_entry(
    *,
    model: str = None,
    terminal_prompt: str = None,
    voice_mode: bool = False,
    verbose_mode: bool = False,
    observer=None,
    screenpipe=None,
):
    """
    Thin execution wrapper.

    Responsibilities:
    - Parse CLI or accept programmatic input
    - Construct execution dependencies
    - Call execute_soc()

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
                default="gpt-4-with-ocr",
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
            model = model or "gpt-4-with-ocr"

        if observer is None or screenpipe is None:
            raise RuntimeError(
                "observer and screenpipe are required for execution"
            )

        # --------------------------------------------------
        # DEPENDENCY CONSTRUCTION (EXPLICIT)
        # --------------------------------------------------
        os_backend = OperatingSystem()

        accessibility_backend = AccessibilityBackend(
            observer=observer,
            screenpipe=screenpipe,
        )

        journal = AuditJournal()
        input_arbitrator = InputArbitrator()

        log_info("[SYSTEM] Starting SOC execution")

        # --------------------------------------------------
        # EXECUTION
        # --------------------------------------------------
        execute_soc(
            model=model,
            objective=objective,
            observer=observer,
            screenpipe=screenpipe,
            os_backend=os_backend,
            accessibility_backend=accessibility_backend,
            journal=journal,
            input_arbitrator=input_arbitrator,
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
