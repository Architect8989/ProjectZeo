import os
import sys
import asyncio
import threading
from typing import Any

from adapters.factory import build_llm
from main import main
from config.timeouts import LLM_THREAD_TIMEOUT_SECONDS


# ---------------------------------------------------------------------------
# CLI ARGUMENT PARSING
# ---------------------------------------------------------------------------

def _parse_args():
    """
    Minimal argument parser. Does not use argparse to avoid extra deps.

    Recognised flags:
        --allow-cloud   Permit cloud model names to be routed through
                        PureLLMWrapper. Without this flag, only models
                        registered in adapters/factory._LOCAL_REGISTRY
                        are accepted. Default: DENIED.

    Positional argument (required):
        model_name      First non-flag argument is treated as the model name.
                        Can also be supplied via LLM_MODEL env var.

    Returns (model_name: str, allow_cloud: bool)
    """
    args = sys.argv[1:]
    allow_cloud = "--allow-cloud" in args
    positional = [a for a in args if not a.startswith("--")]

    model: str | None = None
    if positional:
        model = positional[0].strip() or None

    if not model:
        model = os.getenv("LLM_MODEL", "").strip() or None

    if not model:
        raise RuntimeError(
            "No model specified. "
            "Pass model as CLI argument or set LLM_MODEL environment variable.\n"
            "Example: python run.py qwen2.5-vl:7b-instruct\n"
            "         LLM_MODEL=qwen2.5-vl:7b-instruct python run.py"
        )

    return model, allow_cloud


# ---------------------------------------------------------------------------
# THREAD-SAFE COROUTINE EXECUTOR
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# LLM CALLABLE FACTORY
# ---------------------------------------------------------------------------

def _make_llm_callable(adapter):
    """
    Wrap async adapter into a safe synchronous callable
    compatible with ExecutionPlanner.

    PATCH §R1: asyncio.get_running_loop() catch only re-routes on
    the specific RuntimeError from no running loop. Genuine errors propagate.
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
            _inside_loop = False
            try:
                asyncio.get_running_loop()
                _inside_loop = True
            except RuntimeError:
                _inside_loop = False

            if _inside_loop:
                result = _run_coroutine_threadsafe(_invoke())
            else:
                result = asyncio.run(_invoke())

        except Exception as e:
            raise RuntimeError(f"LLM adapter invocation failed: {e}") from e

        # PATCH §R2: explicit contract enforcement
        if isinstance(result, tuple) and len(result) == 2:
            ops, err = result
            if err:
                raise RuntimeError(f"LLM adapter error: {err}")
        elif isinstance(result, list):
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


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model_name, allow_cloud = _parse_args()

    # FIX F-02: Persist the resolved model name into the environment so that
    # all downstream text-only Ollama calls (ExecutionPlanner._call_llm_text,
    # _decompose_if_complex) see the operator-specified model instead of the
    # hardcoded fallback "qwen2.5-vl:7b-instruct".
    os.environ["LLM_MODEL"] = model_name

    # FIX H-01: Enforce OLLAMA_ONLY by default unless --allow-cloud was given.
    # The factory checks this env var before routing to PureLLMWrapper.
    if not allow_cloud:
        os.environ.setdefault("OLLAMA_ONLY", "1")
    else:
        # Explicit opt-in: unset the flag so the factory allows cloud routing.
        os.environ.pop("OLLAMA_ONLY", None)
        print(
            "[run.py] WARNING: --allow-cloud is set. Cloud API routing is ENABLED. "
            "Ensure API keys are intentionally configured.",
            file=sys.stderr,
        )

    adapter = build_llm(model_name)
    llm_callable = _make_llm_callable(adapter)
    main(llm_callable, model_name=model_name)
