import os
import sys
import asyncio

from adapters.factory import build_llm
from main import main


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
                # If already inside event loop (rare but fatal case)
                loop = asyncio.get_running_loop()
                # Run coroutine in a fresh loop inside thread
                return asyncio.run(_invoke())
            except RuntimeError:
                # No running loop → safe to run directly
                ops, err = asyncio.run(_invoke())
        except Exception as e:
            raise RuntimeError(f"LLM adapter invocation failed: {e}") from e

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
