"""
run.py
=======
PATCHES APPLIED (Audit Fixes):

  ✅  §R1  (was §run-4): Dual-path async detection now catches only the
           'no running event loop' RuntimeError, not ALL RuntimeErrors.
           Genuine errors from asyncio.run() no longer re-route to the
           thread path — they propagate correctly.

  ✅  §R2  (was §run-5): The tuple/bare-list dual-path is preserved but
           now documented explicitly. The adapter contract is enforced via
           a clear check rather than implicit fallthrough.

All existing correct behaviours preserved:
  - Model resolution from CLI arg or LLM_MODEL env var
  - LLM_THREAD_TIMEOUT_SECONDS from shared config
  - Thread-isolated coroutine execution for nested event loops
  - get_next_action() adapter interface enforcement
"""

import os
import sys
import asyncio
import threading
from typing import Any

from adapters.factory import build_llm
from main import main
from config.timeouts import LLM_THREAD_TIMEOUT_SECONDS


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
    Enforces hard timeout derived from shared config.
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

    PATCH §R1: asyncio.get_running_loop() catch now only re-routes on
    the specific 'no current event loop' RuntimeError.  All other errors
    propagate normally so genuine failures are not swallowed.
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
            # PATCH §R1: detect running loop safely, then isolate
            _inside_loop = False
            try:
                asyncio.get_running_loop()
                _inside_loop = True
            except RuntimeError:
                # No running loop — safe to use asyncio.run() directly
                _inside_loop = False

            if _inside_loop:
                result = _run_coroutine_threadsafe(_invoke())
            else:
                result = asyncio.run(_invoke())

        except Exception as e:
            raise RuntimeError(f"LLM adapter invocation failed: {e}") from e

        # PATCH §R2: explicit contract — adapter returns (ops, err) tuple OR bare list
        if isinstance(result, tuple) and len(result) == 2:
            ops, err = result
            if err:
                raise RuntimeError(f"LLM adapter error: {err}")
        elif isinstance(result, list):
            # Bare list return — treat as ops with no error
            ops = result
        else:
            raise RuntimeError(
                f"LLM adapter returned unexpected type: {type(result)!r}. "
                "Expected (List, Exception|None) tuple or List."
            )

        if not isinstance(ops, list):
            raise RuntimeError("LLM adapter returned invalid operation list")

        return ops

    return _call


if __name__ == "__main__":
    model_name = resolve_model_name()
    adapter = build_llm(model_name)
    llm_callable = _make_llm_callable(adapter)
    main(llm_callable, model_name=model_name)
