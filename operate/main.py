"""
Self-Operating Computer – Execution Wrapper
"""

import argparse
import sys

from operate.utils.style import ANSI_BRIGHT_MAGENTA
from operate.operate import operate_main
from core.telemetry.logger import log_info, log_error

from operate.utils.operating_system import OperatingSystem
from core.planner.execution_planner import ExecutionPlanner
from core.environment_fingerprint import collect_environment_fingerprint
from core.vision.world_graph import WorldGraph


def main_entry(
    *,
    model: str = None,
    terminal_prompt: str = None,
    observer=None,
):
    """
    Thin execution wrapper.

    Responsibilities:
    - Parse CLI or accept programmatic input
    - Construct minimal execution dependencies
    - Build ExecutionPlan
    - Call operate_main()
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
                required=True,
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
            if not model:
                raise RuntimeError("model must be provided")
            objective = terminal_prompt

        if observer is None:
            raise RuntimeError("observer is required for execution")

        # --------------------------------------------------
        # DEPENDENCY CONSTRUCTION
        # --------------------------------------------------
        os_backend = OperatingSystem()
        world_graph = WorldGraph()
        env_fingerprint = collect_environment_fingerprint()

        planner = ExecutionPlanner(
            llm_call=None,  # expected injected via higher-level orchestrator
            environment_fingerprint=env_fingerprint,
            world_graph=world_graph,
        )

        if not callable(planner._llm_call):
            raise RuntimeError("ExecutionPlanner LLM callable not configured")

        log_info("[SYSTEM] Starting SOC execution")

        # --------------------------------------------------
        # PLAN GENERATION
        # --------------------------------------------------
        execution_plan = planner.create_plan(
            objective=objective,
            requirements={
                "environment": env_fingerprint,
                "tools": env_fingerprint.get("tools", []),
            },
            high_level_steps=[{"goal": objective}],
        )

        # --------------------------------------------------
        # EXECUTION
        # --------------------------------------------------
        operate_main(
            terminal_prompt=objective,
            execution_plan=execution_plan,
            planner=planner,
            observer=observer,
            world_graph=world_graph,
            os_backend=os_backend,
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
