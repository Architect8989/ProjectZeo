import os
import sys

def _early_parse_args() -> bool:
    """
    Parse --allow-cloud from sys.argv BEFORE any imports execute.

    Returns True if --allow-cloud is present, False otherwise.

    This is intentionally minimal — full argument validation happens in
    _parse_args() after all imports are complete.
    """
    return "--allow-cloud" in sys.argv


# Resolve the operator's cloud intent at the earliest possible moment.
_allow_cloud_early = _early_parse_args()

if not _allow_cloud_early:
    # Default: Ollama-only. Set explicitly (don't rely on default in factory.py)
    # so the freeze always sees "1" when --allow-cloud is absent.
    os.environ["OLLAMA_ONLY"] = "1"
else:
    # Explicit --allow-cloud: set OLLAMA_ONLY="0" so factory.py freeze
    # sees a falsy value and permits cloud routing.
    #
    # RB-NEW-02 FIX: The previous code used os.environ.pop("OLLAMA_ONLY", None).
    # factory.py reads: os.environ.get("OLLAMA_ONLY", "1").strip().lower()
    # When the key is absent, get() returns the default "1" (cloud-blocked),
    # making --allow-cloud permanently non-functional (cloud always blocked).
    #
    # Fix: set "0" explicitly so the freeze captures the correct value and
    # _OLLAMA_ONLY_ENFORCEMENT_FROZEN = True (cloud permitted).
    os.environ["OLLAMA_ONLY"] = "0"
    # Warn early so operators see this even if something crashes during import.
    print(
        "[run.py] WARNING: --allow-cloud is set. Cloud API routing is ENABLED. "
        "Ensure API keys are intentionally configured.",
        file=sys.stderr,
    )

# ===========================================================================
# Imports — factory.py freeze now captures the correct OLLAMA_ONLY value.
# ===========================================================================

import asyncio
import threading
from typing import Any

from adapters.factory import build_llm
from main import main
from config.timeouts import LLM_THREAD_TIMEOUT_SECONDS


# ---------------------------------------------------------------------------
# CLI ARGUMENT PARSING (full)
# ---------------------------------------------------------------------------

def _parse_args():
    """
    Full argument parser. Called after imports.

    Recognised flags:
        --allow-cloud   Permit cloud model routing (processed pre-import above).

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

    # FIX F-02: Persist the resolved model name so all downstream text-only
    # Ollama calls see the operator-specified model.
    os.environ["LLM_MODEL"] = model_name

    # OLLAMA_ONLY was already set/cleared above before factory import.
    # The factory freeze has already captured the correct value.
    # Do NOT mutate os.environ["OLLAMA_ONLY"] again here — factory.py's
    # module-level freeze is immutable; any mutation after import is silently
    # ignored by the enforcement boundary.

    adapter = build_llm(model_name)
    llm_callable = _make_llm_callable(adapter)
    main(llm_callable, model_name=model_name)
