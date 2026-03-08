"""
config/timeouts.py — Unified timeout and stagnation configuration.

FIXES (v2 — March 2026 production hardening):
  CRITICAL-2: MAX_STAGNANT_ITERS_COMMAND raised from 120 → 480.
    At 0.25 s heartbeat = 120 seconds before REPLAN fires.
    Old 30-second limit interrupted renders, downloads, compiles.

  LOW-1: LLM_THREAD_TIMEOUT_SECONDS_CPU reduced 600 s → 240 s.
    Old value caused up to 10-minute SIGINT unresponsiveness.

  HIGH-3 FIX: CONSEQUENCE_EVAL_STAGNATION_GRACE added.
    Stagnation counting suspended while consequence reasoning runs.

  NEW: WAIT_OPERATION_* constants for explicit hold-and-recheck.
  NEW: Per-step stagnation override hint constants.
"""
from __future__ import annotations

import os as _os


# ---------------------------------------------------------------------------
# Deployment mode detection
# ---------------------------------------------------------------------------

def _is_gpu_mode() -> bool:
    """Return True when SGLang GPU inference is configured and enabled."""
    return _os.environ.get("PROJECTZEO_USE_SGLANG", "0").strip() in (
        "1", "true", "yes"
    )


# ---------------------------------------------------------------------------
# LLM call timeouts  (per single inference call)
# ---------------------------------------------------------------------------

LLM_CALL_TIMEOUT_SECONDS_CPU: float = float(
    _os.environ.get("PROJECTZEO_CPU_CALL_TIMEOUT", "150.0")
)

LLM_CALL_TIMEOUT_SECONDS_GPU: float = float(
    _os.environ.get("PROJECTZEO_GPU_CALL_TIMEOUT", "30.0")
)

# ---------------------------------------------------------------------------
# LLM thread-wrapper timeouts
# FIX LOW-1: CPU thread timeout reduced from 600 s -> 240 s.
# 240 s = LLM_CALL_TIMEOUT (150) + 90 s grace.
# ---------------------------------------------------------------------------

LLM_THREAD_TIMEOUT_SECONDS_CPU: float = float(
    _os.environ.get(
        "PROJECTZEO_CPU_THREAD_TIMEOUT",
        str(LLM_CALL_TIMEOUT_SECONDS_CPU + 90.0),
    )
)

LLM_THREAD_TIMEOUT_SECONDS_GPU: float = float(
    _os.environ.get("PROJECTZEO_GPU_THREAD_TIMEOUT", "120.0")
)

# ---------------------------------------------------------------------------
# Consequence reasoning timeouts
# ---------------------------------------------------------------------------

LLM_CONSEQUENCE_TIMEOUT_SECONDS_GPU: float = float(
    _os.environ.get("PROJECTZEO_GPU_CONSEQUENCE_TIMEOUT", "90.0")
)

LLM_CONSEQUENCE_TIMEOUT_SECONDS_CPU: float = float(
    _os.environ.get("PROJECTZEO_CPU_CONSEQUENCE_TIMEOUT", "180.0")
)

# ---------------------------------------------------------------------------
# Effective (deployment-aware) accessors
# ---------------------------------------------------------------------------

def effective_llm_call_timeout() -> float:
    return LLM_CALL_TIMEOUT_SECONDS_GPU if _is_gpu_mode() else LLM_CALL_TIMEOUT_SECONDS_CPU


def effective_llm_thread_timeout() -> float:
    return (
        LLM_THREAD_TIMEOUT_SECONDS_GPU if _is_gpu_mode()
        else LLM_THREAD_TIMEOUT_SECONDS_CPU
    )


def effective_consequence_timeout() -> float:
    return (
        LLM_CONSEQUENCE_TIMEOUT_SECONDS_GPU if _is_gpu_mode()
        else LLM_CONSEQUENCE_TIMEOUT_SECONDS_CPU
    )


LLM_CALL_TIMEOUT_SECONDS: float = effective_llm_call_timeout()
LLM_THREAD_TIMEOUT_SECONDS: float = effective_llm_thread_timeout()

# ---------------------------------------------------------------------------
# Installation command timeout
# ---------------------------------------------------------------------------

INSTALL_COMMAND_TIMEOUT_SECONDS: float = float(
    _os.environ.get("PROJECTZEO_INSTALL_TIMEOUT", "600.0")
)

# ---------------------------------------------------------------------------
# Stagnation limits  (CRITICAL-2 FIX)
#
# At 0.25 s heartbeat:
#   UI  :  12 iters =   3 s  (click/type — fast feedback expected)
#   CMD : 480 iters = 120 s  (commands/installs — slow ops expected)
#
# Old MAX_STAGNANT_ITERS_COMMAND = 120 (30 s) interrupted any long-running
# operation. Fixed to 480 (2 minutes) with per-step overrides available.
# ---------------------------------------------------------------------------

MAX_STAGNANT_ITERS_UI: int = int(
    _os.environ.get("PROJECTZEO_STAGNANT_UI", "12")
)

MAX_STAGNANT_ITERS_COMMAND: int = int(
    _os.environ.get("PROJECTZEO_STAGNANT_COMMAND", "480")
)

MAX_STAGNANT_ITERS_VERIFICATION: int = int(
    _os.environ.get("PROJECTZEO_STAGNANT_VERIFY", "60")
)

# ---------------------------------------------------------------------------
# Per-step stagnation override hints
# Used by ExecutionPlanner to populate ExecutionStep.stagnant_limit_override.
# Values are in ITERATIONS (multiply by 0.25 to get seconds).
# ---------------------------------------------------------------------------

STAGNANT_OVERRIDE_INSTALL: int = int(
    _os.environ.get("PROJECTZEO_STAGNANT_INSTALL", "2400")   # 10 min
)

STAGNANT_OVERRIDE_DOWNLOAD: int = int(
    _os.environ.get("PROJECTZEO_STAGNANT_DOWNLOAD", "4800")  # 20 min
)

STAGNANT_OVERRIDE_RENDER: int = int(
    _os.environ.get("PROJECTZEO_STAGNANT_RENDER", "14400")   # 60 min
)

STAGNANT_OVERRIDE_TEST: int = int(
    _os.environ.get("PROJECTZEO_STAGNANT_TEST", "4800")      # 20 min
)

STAGNANT_OVERRIDE_WAIT: int = int(
    _os.environ.get("PROJECTZEO_STAGNANT_WAIT", "7200")      # 30 min
)

# ---------------------------------------------------------------------------
# Wait operation constants  (NEW)
# Used by the "wait" operation handler in operate.py.
# When PerStepReasoner emits {"operation": "wait", "seconds": N},
# the loop holds for N seconds then resumes — stagnation NOT incremented.
# ---------------------------------------------------------------------------

WAIT_OPERATION_MIN_SECONDS: float = float(
    _os.environ.get("PROJECTZEO_WAIT_MIN_SECONDS", "1.0")
)

WAIT_OPERATION_MAX_SECONDS: float = float(
    _os.environ.get("PROJECTZEO_WAIT_MAX_SECONDS", "1800.0")   # 30 min max
)

WAIT_OPERATION_POLL_SECONDS: float = float(
    _os.environ.get("PROJECTZEO_WAIT_POLL_SECONDS", "5.0")
)

# ---------------------------------------------------------------------------
# Consequence reasoning — stagnation grace  (HIGH-3 FIX)
# While consequence evaluation is running the stagnation counter MUST NOT
# increment.  Grace = how many heartbeat ticks safety eval may consume.
# CPU: 180 s / 0.25 s = 720 ticks.  GPU: 90 s / 0.25 s = 360 ticks.
# ---------------------------------------------------------------------------

CONSEQUENCE_EVAL_STAGNATION_GRACE: int = int(
    _os.environ.get(
        "PROJECTZEO_CONSEQUENCE_STAGNATION_GRACE",
        "720" if not _is_gpu_mode() else "360",
    )
)

# ---------------------------------------------------------------------------
# Startup / warmup grace period
# ---------------------------------------------------------------------------

STARTUP_GRACE_SECONDS: float = float(
    _os.environ.get(
        "PROJECTZEO_STARTUP_GRACE_SECONDS",
        "60.0" if _is_gpu_mode() else "300.0",
    )
)

# ---------------------------------------------------------------------------
# Introspection helper
# ---------------------------------------------------------------------------

def describe_timeout_mode() -> str:
    """Return a one-line description of the active timeout regime."""
    if _is_gpu_mode():
        return (
            f"GPU mode — "
            f"call={LLM_CALL_TIMEOUT_SECONDS_GPU}s "
            f"thread={LLM_THREAD_TIMEOUT_SECONDS_GPU}s "
            f"consequence={LLM_CONSEQUENCE_TIMEOUT_SECONDS_GPU}s "
            f"stagnant_cmd={MAX_STAGNANT_ITERS_COMMAND}iters"
        )
    return (
        f"CPU mode — "
        f"call={LLM_CALL_TIMEOUT_SECONDS_CPU}s "
        f"thread={LLM_THREAD_TIMEOUT_SECONDS_CPU}s "
        f"consequence={LLM_CONSEQUENCE_TIMEOUT_SECONDS_CPU}s "
        f"stagnant_cmd={MAX_STAGNANT_ITERS_COMMAND}iters "
        f"(={MAX_STAGNANT_ITERS_COMMAND * 0.25:.0f}s)"
    )
