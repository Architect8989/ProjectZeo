# config/timeouts.py
"""
Centralized timeout configuration for all LLM-related execution paths.

Design rules:
- Planner-level timeout must fire before thread-level timeout.
- Thread timeout must exceed planner timeout by a small safety margin.
- All LLM timeouts must be defined here to avoid drift.
"""

# Core LLM call timeout (used by ExecutionPlanner)
LLM_CALL_TIMEOUT_SECONDS: float = 30.0

# Thread wrapper timeout (used in run.py)
# Must be strictly greater than LLM_CALL_TIMEOUT_SECONDS.
LLM_THREAD_TIMEOUT_SECONDS: float = LLM_CALL_TIMEOUT_SECONDS + 10.0
