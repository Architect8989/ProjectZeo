"""
Self-Operating Computer
"""

import argparse
import sys

from operate.utils.style import ANSI_BRIGHT_MAGENTA
from operate.operate import main as operate_main

from core.control.kernel_controller import KernelController
from core.telemetry.logger import log_info, log_error


def main_entry(
    *,
    model: str = None,
    terminal_prompt: str = None,
    voice_mode: bool = False,
    verbose_mode: bool = False,
):
    """
    Entry point for both:
    - CLI execution (argparse)
    - Programmatic invocation (kernel / main.py)

    If terminal_prompt is provided, argparse is skipped.
    """

    try:
        # ----------------------------
        # PROGRAMMATIC INVOCATION PATH
        # ----------------------------
        if terminal_prompt is not None:
            config = {
                "model": model or "gpt-4-with-ocr",
                "terminal_prompt": terminal_prompt,
                "voice_mode": bool(voice_mode),
                "verbose_mode": bool(verbose_mode),
            }

        # ----------------------------
        # CLI INVOCATION PATH
        # ----------------------------
        else:
            parser = argparse.ArgumentParser(
                description="Run the self-operating-computer with a specified model."
            )

            parser.add_argument(
                "-m",
                "--model",
                help="Specify the model to use",
                required=False,
                default="gpt-4-with-ocr",
            )

            parser.add_argument(
                "--voice",
                help="Use voice input mode",
                action="store_true",
            )

            parser.add_argument(
                "--verbose",
                help="Run operate in verbose mode",
                action="store_true",
            )

            parser.add_argument(
                "--prompt",
                help="Directly input the objective prompt",
                type=str,
                required=False,
            )

            args = parser.parse_args()

            config = {
                "model": args.model,
                "terminal_prompt": args.prompt,
                "voice_mode": args.voice,
                "verbose_mode": args.verbose,
            }

        # ----------------------------
        # BOOT KERNEL
        # ----------------------------

        log_info("[SYSTEM] Booting kernel")

        kernel = KernelController(
            config=config,
            operate_entry=operate_main,
        )

        kernel.start()

    except KeyboardInterrupt:
        log_info("[SYSTEM] Keyboard interrupt received")
        print(f"\n{ANSI_BRIGHT_MAGENTA}Exiting...")
        sys.exit(0)

    except Exception as e:
        log_error(f"[SYSTEM] Fatal startup error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main_entry()
