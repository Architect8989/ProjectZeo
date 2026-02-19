# config/timeouts.py
"""
Centralised timeout configuration for all LLM-related execution paths.

PATCHES APPLIED (Audit Fixes):

  ✅  §R4: LLM_CALL_TIMEOUT_SECONDS raised to 150s to accommodate CPU
           inference on Qwen2.5-VL 7B (40–90s) plus network overhead.
           The old 30s value caused the planner to abort mid-generation
           on CPU-only machines.

  ✅  §R4: LLM_THREAD_TIMEOUT_SECONDS raised proportionally to maintain
           the required safety margin above LLM_CALL_TIMEOUT_SECONDS.

Design rules:
  - Planner-level timeout must fire before thread-level timeout.
  - Thread timeout must exceed planner timeout by a small safety margin.
  - All LLM timeouts must be defined here to avoid drift.
"""

# Core LLM call timeout (used by ExecutionPlanner and AutonomousInstaller)
# PATCH §R4: raised from 30s to 150s for CPU-inference compatibility.
# CPU inference for Qwen2.5-VL 7B: 40–90s. 150s gives a comfortable margin.
LLM_CALL_TIMEOUT_SECONDS: float = 150.0

# Thread wrapper timeout (used in run.py)
# Must be strictly greater than LLM_CALL_TIMEOUT_SECONDS.
LLM_THREAD_TIMEOUT_SECONDS: float = LLM_CALL_TIMEOUT_SECONDS + 15.0

# Installation command timeout (used by AutonomousInstaller._try_terminal_install)
# Large downloads (e.g. apt-get install build-essential) can take 5+ minutes.
INSTALL_COMMAND_TIMEOUT_SECONDS: float = 300.0

# Step execution stagnation limits (used by operate.py)
# Default for UI interaction steps — expect fast feedback
MAX_STAGNANT_ITERS_UI: int = 12

# For command_execution and tool_installation steps.
# A slow download (e.g. nodejs via apt on a 10Mbps connection) can take 3+ minutes
# with no progress event. 120 iterations × 0.25s heartbeat = 30s minimum.
# Actual command blocking means real wall-clock time is the actual timeout
# from the subprocess, so this guard is for the outer loop health check.
MAX_STAGNANT_ITERS_COMMAND: int = 120
