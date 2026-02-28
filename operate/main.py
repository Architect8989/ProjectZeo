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

        # FIX: llm_call=None immediately raises PlanningError in ExecutionPlanner.
        # This standalone entry builds a real adapter from LLM_MODEL env var.
        # In production the full kernel (run.py → main.py) wires this properly.
        import os as _os
        _model_name = model or _os.environ.get("LLM_MODEL", "").strip()
        if not _model_name:
            raise RuntimeError(
                "model must be provided via argument or LLM_MODEL env var"
            )
        from adapters.factory import build_llm as _build_llm
        import asyncio as _asyncio

        _adapter = _build_llm(_model_name)

        def _llm_callable(messages, objective=None, session_id=None):
            async def _invoke():
                return await _adapter.get_next_action(
                    messages=messages, objective=objective, session_id=session_id
                )
            ops, err = _asyncio.run(_invoke())
            if err:
                raise RuntimeError(f"LLM error: {err}")
            return ops or []

        planner = ExecutionPlanner(
            llm_call=_llm_callable,
            environment_fingerprint=env_fingerprint,
            world_graph=world_graph,
        )

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
