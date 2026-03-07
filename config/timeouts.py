from __future__ import annotations

import os as _os


# ---------------------------------------------------------------------------
# Deployment mode detection
# ---------------------------------------------------------------------------

def _is_gpu_mode() -> bool:
    """Return True when SGLang GPU inference is configured and enabled."""
    return _os.environ.get("PROJECTZEO_USE_SGLANG", "0").strip() in ("1", "true", "yes")



LLM_CALL_TIMEOUT_SECONDS_CPU: float = 150.0


LLM_THREAD_TIMEOUT_SECONDS_CPU: float = 600.0



LLM_CALL_TIMEOUT_SECONDS_GPU: float = float(
    _os.environ.get("PROJECTZEO_GPU_CALL_TIMEOUT", "30.0")
)

# §GPU: Thread timeout — 3 retries x 30s = 90s worst-case + 30s safety margin.
LLM_THREAD_TIMEOUT_SECONDS_GPU: float = float(
    _os.environ.get("PROJECTZEO_GPU_THREAD_TIMEOUT", "120.0")
)


LLM_CONSEQUENCE_TIMEOUT_SECONDS_GPU: float = float(
    _os.environ.get("PROJECTZEO_GPU_CONSEQUENCE_TIMEOUT", "90.0")
)


# ---------------------------------------------------------------------------
# Effective (deployment-aware) accessors
# ---------------------------------------------------------------------------

def effective_llm_call_timeout() -> float:
    """Return the correct LLM call timeout for the current deployment mode."""
    return LLM_CALL_TIMEOUT_SECONDS_GPU if _is_gpu_mode() else LLM_CALL_TIMEOUT_SECONDS_CPU


def effective_llm_thread_timeout() -> float:
    """Return the correct thread-wrapper timeout for the current deployment mode."""
    return LLM_THREAD_TIMEOUT_SECONDS_GPU if _is_gpu_mode() else LLM_THREAD_TIMEOUT_SECONDS_CPU


def effective_consequence_timeout() -> float:
    
    return LLM_CONSEQUENCE_TIMEOUT_SECONDS_GPU if _is_gpu_mode() else 180.0



LLM_CALL_TIMEOUT_SECONDS: float = effective_llm_call_timeout()
LLM_THREAD_TIMEOUT_SECONDS: float = effective_llm_thread_timeout()



INSTALL_COMMAND_TIMEOUT_SECONDS: float = 300.0



MAX_STAGNANT_ITERS_UI: int = 12

# For command_execution and tool_installation steps.
# 120 iterations x 0.25s heartbeat = 30s minimum guard.
MAX_STAGNANT_ITERS_COMMAND: int = 120


# ---------------------------------------------------------------------------
# Startup / warmup grace period
# ---------------------------------------------------------------------------

# GPU warmup is much faster than CPU — 60s vs 300s default.
STARTUP_GRACE_SECONDS: float = float(
    _os.environ.get(
        "PROJECTZEO_STARTUP_GRACE_SECONDS",
        "60.0" if _is_gpu_mode() else "300.0",
    )
)


# ---------------------------------------------------------------------------
# Introspection helper (used by startup banners)
# ---------------------------------------------------------------------------

def describe_timeout_mode() -> str:
    """Return a one-line description of the active timeout regime."""
    if _is_gpu_mode():
        return (
            f"GPU mode — call={LLM_CALL_TIMEOUT_SECONDS_GPU}s "
            f"thread={LLM_THREAD_TIMEOUT_SECONDS_GPU}s "
            f"consequence={LLM_CONSEQUENCE_TIMEOUT_SECONDS_GPU}s"
        )
    return (
        f"CPU mode — call={LLM_CALL_TIMEOUT_SECONDS_CPU}s "
        f"thread={LLM_THREAD_TIMEOUT_SECONDS_CPU}s "
        f"consequence=180.0s"
    )
