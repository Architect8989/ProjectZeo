"""
agent_orchestrator.py — Wires specialist agents and LangGraph into the execution loop.

AUDIT FIX: The LangGraph StateGraph and AgentPipeline were fully implemented but
never connected to operate.py's _execute_autonomous_loop(). This module provides
a clean, production-grade integration layer that:

  1. Builds and compiles the LangGraph pipeline from live agent instances.
  2. Provides a unified `run_task()` entry point compatible with both
     GIIGoalDirectedLoop and the LangGraph execution path.
  3. Auto-selects the best available backend:
       - PROJECTZEO_USE_LANGGRAPH=1  → LangGraph StateGraph (preferred)
       - AgentPipeline fallback        → Direct specialist-agent loop
       - GIIGoalDirectedLoop fallback  → Original loop (always available)
  4. Exposes health-check, stats, and graceful shutdown.

Usage in operate.py:
    from core.agents.agent_orchestrator import AgentOrchestrator
    orchestrator = AgentOrchestrator.create(
        llm_callable=llm, objective=objective, ...)
    result = orchestrator.run()
"""
from __future__ import annotations

import logging
import os
import time
import threading
from typing import Any, Callable, Dict, List, Optional

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AgentOrchestrator
# ---------------------------------------------------------------------------

class AgentOrchestrator:
    """
    Production-grade orchestrator that selects the best available execution
    backend and exposes a unified run() interface.

    Backend priority:
      1. LangGraph StateGraph (when PROJECTZEO_USE_LANGGRAPH=1 and langgraph installed)
      2. AgentPipeline (always available, specialist-agent loop)
      3. GIIGoalDirectedLoop (always available, original loop)
    """

    def __init__(
        self,
        *,
        # Core dependencies
        llm_callable: Callable,
        objective: str,
        os_backend,
        world_graph,
        policy_engine,
        journal,
        # Optional: pre-built components
        gii_controller=None,
        consequence_reasoner=None,
        vision_runtime=None,
        atspi_bridge=None,
        semantic_memory=None,
        application_memory=None,
        # Execution parameters
        max_wallclock_seconds: int = 3600,
        max_iterations: int = 500,
        watchdog=None,
        execute_decision_fn: Optional[Callable] = None,
        on_action_executed: Optional[Callable] = None,
    ) -> None:
        self._llm = llm_callable
        self._objective = objective
        self._os_backend = os_backend
        self._world_graph = world_graph
        self._policy = policy_engine
        self._journal = journal
        self._gii_controller = gii_controller
        self._consequence_reasoner = consequence_reasoner
        self._vision_runtime = vision_runtime
        self._atspi_bridge = atspi_bridge
        self._semantic_memory = semantic_memory
        self._application_memory = application_memory
        self._max_wallclock = max_wallclock_seconds
        self._max_iterations = max_iterations
        self._watchdog = watchdog
        self._execute_decision_fn = execute_decision_fn
        self._on_action_executed = on_action_executed

        # Selected backend
        self._backend_name: str = "unknown"
        self._pipeline = None       # LangGraph compiled graph or AgentPipeline
        self._gii_loop = None       # GIIGoalDirectedLoop (fallback)

        self._start_ts: float = 0.0
        self._result: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()

        self._build_pipeline()

    # =========================================================================
    # Factory
    # =========================================================================

    @classmethod
    def create(
        cls,
        *,
        llm_callable: Callable,
        objective: str,
        os_backend,
        world_graph,
        policy_engine,
        journal,
        **kwargs,
    ) -> "AgentOrchestrator":
        """
        Convenience factory. Accepts all constructor kwargs plus required core args.
        Any unrecognised kwargs are silently ignored to allow forward-compatibility.
        """
        known_keys = {
            "gii_controller", "consequence_reasoner", "vision_runtime",
            "atspi_bridge", "semantic_memory", "application_memory",
            "max_wallclock_seconds", "max_iterations", "watchdog",
            "execute_decision_fn", "on_action_executed",
        }
        filtered = {k: v for k, v in kwargs.items() if k in known_keys}
        return cls(
            llm_callable=llm_callable,
            objective=objective,
            os_backend=os_backend,
            world_graph=world_graph,
            policy_engine=policy_engine,
            journal=journal,
            **filtered,
        )

    # =========================================================================
    # Pipeline construction
    # =========================================================================

    def _build_pipeline(self) -> None:
        """Build the best available execution pipeline."""
        # Try LangGraph first
        if self._try_build_langgraph():
            return
        # Try AgentPipeline
        if self._try_build_agent_pipeline():
            return
        # Fallback: GIIGoalDirectedLoop
        self._build_gii_loop()

    def _build_specialist_agents(self):
        """Build specialist agent instances from available dependencies."""
        try:
            from core.agents.specialist_agents import (
                PerceiverAgent, ReasonerAgent, SafetyAgent,
                ExecutorAgent, MemoryAgent,
            )

            perceiver = PerceiverAgent(
                world_graph=self._world_graph,
                vision_runtime=self._vision_runtime,
                atspi_bridge=self._atspi_bridge,
            )
            # ReasonerAgent delegates to GIIController (which wraps PerStepReasoner)
            reasoner = ReasonerAgent(
                gii_controller=self._gii_controller,
            )
            safety = SafetyAgent(
                policy_engine=self._policy,
                consequence_reasoner=self._consequence_reasoner,
            )
            executor = ExecutorAgent(
                os_backend=self._os_backend,
                execute_decision_fn=self._execute_decision_fn,
                journal=self._journal,
            )
            memory = MemoryAgent(
                gii_controller=self._gii_controller,
                semantic_memory=self._semantic_memory,
                application_memory=self._application_memory,
                objective=self._objective,
            )
            return perceiver, reasoner, safety, executor, memory
        except Exception as exc:
            _logger.warning("[AgentOrchestrator] Specialist agent build failed: %s", exc)
            return None

    def _try_build_langgraph(self) -> bool:
        """Attempt to build and compile a LangGraph pipeline."""
        if not os.environ.get("PROJECTZEO_USE_LANGGRAPH", "0").strip() in ("1", "true", "yes"):
            return False
        try:
            agents = self._build_specialist_agents()
            if agents is None:
                return False
            perceiver, reasoner, safety, executor, memory = agents

            from core.agents.langgraph_pipeline import build_langgraph_pipeline
            graph = build_langgraph_pipeline(
                perceiver_agent=perceiver,
                reasoner_agent=reasoner,
                safety_agent=safety,
                executor_agent=executor,
                memory_agent=memory,
            )
            if graph is None:
                return False

            self._pipeline = graph
            self._backend_name = "langgraph"
            _logger.info(
                "[AgentOrchestrator] LangGraph StateGraph ready. "
                "Backend: langgraph"
            )
            return True
        except Exception as exc:
            _logger.warning("[AgentOrchestrator] LangGraph build failed: %s", exc)
            return False

    def _try_build_agent_pipeline(self) -> bool:
        """Build the AgentPipeline (specialist-agent direct loop)."""
        try:
            agents = self._build_specialist_agents()
            if agents is None:
                return False
            perceiver, reasoner, safety, executor, memory = agents

            from core.agents.specialist_agents import AgentPipeline
            pipeline = AgentPipeline(
                perceiver=perceiver,
                reasoner=reasoner,
                safety=safety,
                executor=executor,
                memory=memory,
            )  # keyword form supported after specialist_agents.py patch
            self._pipeline = pipeline
            self._backend_name = "agent_pipeline"
            _logger.info(
                "[AgentOrchestrator] AgentPipeline ready. Backend: agent_pipeline"
            )
            return True
        except Exception as exc:
            _logger.warning("[AgentOrchestrator] AgentPipeline build failed: %s", exc)
            return False

    def _build_gii_loop(self) -> None:
        """Build the GIIGoalDirectedLoop as the final fallback."""
        try:
            from core.gii.gii_loop import GIIGoalDirectedLoop
            self._gii_loop = GIIGoalDirectedLoop(
                gii_controller=self._gii_controller,
                os_backend=self._os_backend,
                world_graph=self._world_graph,
                policy_engine=self._policy,
                journal=self._journal,
                objective=self._objective,
                max_wallclock_seconds=self._max_wallclock,
                max_iterations=self._max_iterations,
                watchdog=self._watchdog,
                execute_decision_fn=self._execute_decision_fn,
                on_action_executed=self._on_action_executed,
            )
            self._backend_name = "gii_loop"
            _logger.info(
                "[AgentOrchestrator] GIIGoalDirectedLoop ready (fallback). "
                "Backend: gii_loop"
            )
        except Exception as exc:
            _logger.error(
                "[AgentOrchestrator] CRITICAL: GIIGoalDirectedLoop build failed: %s. "
                "No execution backend available.", exc
            )

    # =========================================================================
    # Execution
    # =========================================================================

    def run(self) -> Dict[str, Any]:
        """
        Execute the task using the best available backend.

        Returns:
            {
                "success": bool,
                "reason":  str,
                "backend": str,       # "langgraph" | "agent_pipeline" | "gii_loop"
                "iterations": int,
                "elapsed_seconds": float,
                "stats": dict,
            }
        """
        self._start_ts = time.time()

        _logger.info(
            "[AgentOrchestrator] Starting execution. objective=%r backend=%s",
            self._objective[:80], self._backend_name,
        )

        self._journal.record({
            "event":     "orchestrator_start",
            "objective": self._objective[:200],
            "backend":   self._backend_name,
        })

        try:
            if self._backend_name == "langgraph":
                raw_result = self._run_langgraph()
            elif self._backend_name == "agent_pipeline":
                raw_result = self._run_agent_pipeline()
            elif self._backend_name == "gii_loop":
                raw_result = self._run_gii_loop()
            else:
                raw_result = {
                    "success": False,
                    "reason":  "No execution backend available",
                    "iterations": 0,
                }
        except Exception as exc:
            _logger.error("[AgentOrchestrator] Execution error: %s", exc)
            raw_result = {
                "success": False,
                "reason":  f"Orchestration error: {exc}",
                "iterations": 0,
            }

        elapsed = time.time() - self._start_ts
        result = {
            **raw_result,
            "backend":          self._backend_name,
            "elapsed_seconds":  round(elapsed, 2),
            "stats":            self._collect_stats(),
        }

        self._journal.record({
            "event":   "orchestrator_complete",
            "success": result.get("success"),
            "reason":  result.get("reason", ""),
            "backend": self._backend_name,
            "elapsed": elapsed,
        })

        with self._lock:
            self._result = result

        return result

    def _run_langgraph(self) -> Dict[str, Any]:
        from core.agents.specialist_agents import WorldState
        from core.agents.langgraph_pipeline import run_langgraph_pipeline

        initial_ws = WorldState(
            objective=self._objective,
            start_ts=self._start_ts,
        )
        final_ws = run_langgraph_pipeline(
            compiled_graph=self._pipeline,
            initial_world_state=initial_ws,
            max_iterations=self._max_iterations,
            max_wallclock_seconds=self._max_wallclock,
        )
        return {
            "success":    final_ws.goal_complete,
            "reason":     final_ws.error or (
                "Goal complete" if final_ws.goal_complete else "Loop terminated"
            ),
            "iterations": final_ws.iteration,
        }

    def _run_agent_pipeline(self) -> Dict[str, Any]:
        from core.agents.specialist_agents import WorldState

        initial_ws = WorldState(
            objective=self._objective,
            start_ts=self._start_ts,
        )
        final_ws = self._pipeline.run_until_done(
            initial_state=initial_ws,
            max_iterations=self._max_iterations,
            max_wallclock_seconds=self._max_wallclock,
        )
        return {
            "success":    final_ws.goal_complete,
            "reason":     final_ws.error or (
                "Goal complete" if final_ws.goal_complete else "Loop terminated"
            ),
            "iterations": final_ws.iteration,
        }

    def _run_gii_loop(self) -> Dict[str, Any]:
        if self._gii_loop is None:
            return {"success": False, "reason": "GIIGoalDirectedLoop not initialised", "iterations": 0}
        return self._gii_loop.run(start_ts=self._start_ts)

    # =========================================================================
    # Helpers
    # =========================================================================

    def _collect_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {"backend": self._backend_name}
        if self._gii_controller:
            try:
                stats["gii"] = self._gii_controller.get_stats()
            except Exception:
                pass
        if self._consequence_reasoner:
            try:
                stats["consequence_reasoner"] = self._consequence_reasoner.get_stats()
            except Exception:
                pass
        return stats

    def get_backend_name(self) -> str:
        return self._backend_name

    def get_last_result(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._result
