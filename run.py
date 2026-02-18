import os
import sys
import asyncio
import threading
from typing import Any

from adapters.factory import build_llm
from main import main


LLM_THREAD_TIMEOUT_SECONDS = 120.0


def resolve_model_name() -> str:
    if len(sys.argv) > 1:
        model = sys.argv[1].strip()
        if model:
            return model

    model = os.getenv("LLM_MODEL")
    if model and model.strip():
        return model.strip()

    raise RuntimeError(
        "No model specified. "
        "Pass model as CLI argument or set LLM_MODEL environment variable."
    )


def _run_coroutine_threadsafe(coro) -> Any:
    """
    Execute coroutine in a fresh event loop inside a dedicated thread.
    Enforces hard timeout to prevent permanent kernel freeze.
    """
    result_container: dict = {}
    error_container: dict = {}

    def _thread_target():
        try:
            result_container["result"] = asyncio.run(coro)
        except Exception as e:
            error_container["error"] = e

    t = threading.Thread(target=_thread_target, daemon=True)
    t.start()
    t.join(timeout=LLM_THREAD_TIMEOUT_SECONDS)

    if t.is_alive():
        raise RuntimeError(
            f"LLM thread timed out after {LLM_THREAD_TIMEOUT_SECONDS} seconds"
        )

    if "error" in error_container:
        raise error_container["error"]

    return result_container.get("result")


def _make_llm_callable(adapter):
    """
    Wrap async adapter into a safe synchronous callable
    compatible with ExecutionPlanner.
    """

    if not hasattr(adapter, "get_next_action"):
        raise RuntimeError("Adapter missing get_next_action()")

    def _call(messages, objective=None, session_id=None):

        async def _invoke():
            return await adapter.get_next_action(
                messages=messages,
                objective=objective,
                session_id=session_id,
            )

        try:
            try:
                asyncio.get_running_loop()
                result = _run_coroutine_threadsafe(_invoke())
            except RuntimeError:
                result = asyncio.run(_invoke())
        except Exception as e:
            raise RuntimeError(f"LLM adapter invocation failed: {e}") from e

        if isinstance(result, tuple) and len(result) == 2:
            ops, err = result
        else:
            ops = result
            err = None

        if err:
            raise RuntimeError(f"LLM adapter error: {err}")

        if not isinstance(ops, list):
            raise RuntimeError("LLM adapter returned invalid operation list")

        return ops

    return _call


if __name__ == "__main__":
    model_name = resolve_model_name()
    adapter = build_llm(model_name)
    llm_callable = _make_llm_callable(adapter)
    main(llm_callable, model_name=model_name)
