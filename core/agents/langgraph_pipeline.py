from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

_logger = logging.getLogger(__name__)

# Lazy import
_langgraph_available: Optional[bool] = None


def _check_langgraph() -> bool:
    global _langgraph_available
    if _langgraph_available is not None:
        return _langgraph_available
    try:
        import langgraph  # noqa: PLC0415
        _langgraph_available = True
    except ImportError:
        _langgraph_available = False
        _logger.info(
            "[LangGraphPipeline] langgraph not installed. Using AgentPipeline fallback. "
            "Install: pip install langgraph"
        )
    return _langgraph_available


def _use_langgraph() -> bool:
    """Return True when LangGraph is available and enabled."""
    return (
        os.environ.get("PROJECTZEO_USE_LANGGRAPH", "0").strip() in ("1", "true", "yes")
        and _check_langgraph()
    )


# ---------------------------------------------------------------------------
# LangGraph State TypedDict
# ---------------------------------------------------------------------------

def _make_langgraph_state_type():
    """Create the LangGraph StateGraph state TypedDict at runtime."""
    from typing import TypedDict  # noqa: PLC0415
    from core.agents.specialist_agents import WorldState  # noqa: PLC0415

    class GraphState(TypedDict, total=False):
        """Type-safe state container for the LangGraph agent graph."""
        objective: str
        iteration: int
        goal_complete: bool
        error: Optional[str]

        # Perception
        screen_snapshot: Optional[Dict[str, Any]]
        entities: List[Dict[str, Any]]
        focused_app: str
        entity_count: int
        last_delta: Optional[Dict[str, Any]]

        # Reasoning
        proposed_action: Optional[Dict[str, Any]]
        reasoning_rationale: str

        # Safety
        safety_decision: str
        safety_reason: str

        # Execution
        last_action: Optional[Dict[str, Any]]
        last_action_success: bool
        last_action_output: str
        consecutive_failures: int

        # Memory
        memory_context: str

    return GraphState


def _world_state_to_graph_state(ws) -> Dict[str, Any]:
    """Convert a WorldState dataclass to a LangGraph-compatible dict."""
    return {
        "objective":           ws.objective,
        "iteration":           ws.iteration,
        "goal_complete":       ws.goal_complete,
        "error":               ws.error,
        "screen_snapshot":     ws.screen_snapshot,
        "entities":            ws.entities,
        "focused_app":         ws.focused_app,
        "entity_count":        ws.entity_count,
        "last_delta":          ws.last_delta,
        "proposed_action":     ws.proposed_action,
        "reasoning_rationale": ws.reasoning_rationale,
        "safety_decision":     ws.safety_decision,
        "safety_reason":       ws.safety_reason,
        "last_action":         ws.last_action,
        "last_action_success": ws.last_action_success,
        "last_action_output":  ws.last_action_output,
        "consecutive_failures": ws.consecutive_failures,
        "memory_context":      ws.memory_context,
    }


def _graph_state_to_world_state(state: Dict[str, Any]):
    """Convert a LangGraph graph state dict back to WorldState."""
    from core.agents.specialist_agents import WorldState  # noqa: PLC0415
    ws = WorldState()
    for key, val in state.items():
        if hasattr(ws, key):
            setattr(ws, key, val)
    return ws


# ---------------------------------------------------------------------------
# LangGraph Pipeline Builder
# ---------------------------------------------------------------------------

def build_langgraph_pipeline(
    perceiver_agent,
    reasoner_agent,
    safety_agent,
    executor_agent,
    memory_agent,
):
    
    if not _use_langgraph():
        return None

    try:
        from langgraph.graph import StateGraph, END  # noqa: PLC0415

        GraphState = _make_langgraph_state_type()

        # ----------------------------------------------------------------
        # Node functions — each wraps a specialist agent
        # ----------------------------------------------------------------

        def perceive_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Run PerceiverAgent to update world state from VisionRuntime."""
            ws = _graph_state_to_world_state(state)
            ws = perceiver_agent.run(ws)
            return _world_state_to_graph_state(ws)

        def reason_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Run ReasonerAgent to propose next action."""
            ws = _graph_state_to_world_state(state)
            ws = reasoner_agent.run(ws)
            return _world_state_to_graph_state(ws)

        def safety_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Run SafetyAgent (PolicyEngine + ConsequenceReasoner + LlamaGuard)."""
            ws = _graph_state_to_world_state(state)
            ws = safety_agent.run(ws)
            return _world_state_to_graph_state(ws)

        def confirm_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """
            GAP-3 FIX: Human confirmation wait loop for REQUIRE_HUMAN_CONFIRMATION.
            Writes a signal file and polls for .APPROVE file creation.
            If approved: sets safety_decision='ALLOW' and returns to execute.
            If timed out: sets safety_decision='DENY' and returns to reason.
            """
            import json as _json, secrets as _secrets, tempfile as _tempfile, time as _time
            APPROVAL_TIMEOUT = 300.0
            POLL_INTERVAL = 2.0

            proposed = state.get("proposed_action") or {}
            reason = state.get("safety_reason", "Requires human confirmation")
            objective = state.get("objective", "")

            signal_dir = _tempfile.gettempdir()
            token = _secrets.token_hex(8)
            signal_path = os.path.join(signal_dir, f"lgraph_approve_{token}.signal")
            approve_path = signal_path + ".APPROVE"

            try:
                with open(signal_path, "w", encoding="utf-8") as sf:
                    _json.dump({
                        "operation": proposed.get("operation", "?"),
                        "reason": reason[:300],
                        "objective": objective[:200],
                        "approve_by_creating": approve_path,
                    }, sf, indent=2)

                import sys
                print(
                    f"\n[LangGraph] ⚠  HUMAN APPROVAL REQUIRED\n"
                    f"  Operation: {proposed.get('operation', '?')}\n"
                    f"  Reason: {reason[:120]}\n"
                    f"  Approve: CREATE {approve_path}\n",
                    file=sys.stderr,
                )

                deadline = _time.time() + APPROVAL_TIMEOUT
                approved = False
                while _time.time() < deadline:
                    if os.path.exists(approve_path):
                        approved = True
                        break
                    _time.sleep(POLL_INTERVAL)
            except Exception as e:
                _logger.warning("[LangGraph] Approval signal error: %s", e)
                approved = False
            finally:
                for p in (signal_path, approve_path):
                    try: os.unlink(p)
                    except OSError: pass

            new_state = dict(state)
            if approved:
                new_state["safety_decision"] = "ALLOW"
                _logger.info("[LangGraph] Human confirmation approved.")
            else:
                new_state["safety_decision"] = "DENY"
                new_state["consecutive_failures"] = state.get("consecutive_failures", 0) + 1
                _logger.warning("[LangGraph] Human confirmation timed out — DENY.")
            return new_state

        def execute_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Run ExecutorAgent to dispatch the approved action."""
            ws = _graph_state_to_world_state(state)
            ws = executor_agent.run(ws)
            return _world_state_to_graph_state(ws)

        def memory_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Run MemoryAgent to update cross-session memory."""
            ws = _graph_state_to_world_state(state)
            ws = memory_agent.run(ws)
            return _world_state_to_graph_state(ws)

        # ----------------------------------------------------------------
        # Conditional edge functions
        # ----------------------------------------------------------------

        def should_execute(state: Dict[str, Any]) -> str:
            """After safety check: EXECUTE if ALLOW, REASON if DENY (retry).

            GAP-3 FIX: REQUIRE_HUMAN_CONFIRMATION now routes to 'confirm'
            node (wait loop) instead of silently falling through to 'execute'.
            Previously this caused every REQUIRE_HUMAN_CONFIRMATION decision to
            execute without approval when PROJECTZEO_USE_AGENT_ORCHESTRATOR=1.
            """
            if state.get("goal_complete") or state.get("error"):
                return "done"
            decision = state.get("safety_decision", "ALLOW")
            if decision == "ALLOW":
                return "execute"
            if decision == "REQUIRE_HUMAN_CONFIRMATION":
                # Route to confirmation wait node — DO NOT execute without approval
                return "confirm"
            # DENY → skip execution, loop back to reasoning
            return "reason"

        def should_continue(state: Dict[str, Any]) -> str:
            """After memory update: continue loop or terminate."""
            if state.get("goal_complete"):
                return "done"
            if state.get("error"):
                return "done"
            failures = state.get("consecutive_failures", 0)
            if failures >= 10:
                return "done"
            return "perceive"

        # ----------------------------------------------------------------
        # Build graph
        # ----------------------------------------------------------------

        graph_builder = StateGraph(GraphState)

        graph_builder.add_node("perceive",  perceive_node)
        graph_builder.add_node("reason",    reason_node)
        graph_builder.add_node("safety",    safety_node)
        graph_builder.add_node("confirm",   confirm_node)   # GAP-3 FIX
        graph_builder.add_node("execute",   execute_node)
        graph_builder.add_node("memory",    memory_node)

        # Entry point
        graph_builder.set_entry_point("perceive")

        # Linear edges
        graph_builder.add_edge("perceive", "reason")
        graph_builder.add_edge("reason", "safety")

        # Conditional: after safety (GAP-3: now routes REQUIRE_HUMAN to confirm)
        graph_builder.add_conditional_edges(
            "safety",
            should_execute,
            {
                "execute": "execute",
                "confirm": "confirm",  # GAP-3 FIX: wait for human approval
                "reason":  "reason",   # retry on DENY
                "done":    END,
            },
        )

        # After confirmation: execute if approved, reason if denied
        graph_builder.add_conditional_edges(
            "confirm",
            lambda s: "execute" if s.get("safety_decision") == "ALLOW" else "reason",
            {"execute": "execute", "reason": "reason"},
        )

        graph_builder.add_edge("execute", "memory")

        # Conditional: after memory
        graph_builder.add_conditional_edges(
            "memory",
            should_continue,
            {
                "perceive": "perceive",
                "done":     END,
            },
        )

        compiled = graph_builder.compile()
        _logger.info("[LangGraphPipeline] StateGraph compiled successfully.")
        return compiled

    except Exception as exc:
        _logger.warning(
            "[LangGraphPipeline] Graph compilation failed: %s. Using AgentPipeline fallback.",
            exc,
        )
        return None


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_langgraph_pipeline(
    compiled_graph,
    initial_world_state,
    *,
    max_iterations: int = 500,
    max_wallclock_seconds: int = 3600,
):
    
    if compiled_graph is None:
        # Fall back to linear AgentPipeline
        _logger.warning("[LangGraphPipeline] No compiled graph — cannot run.")
        return initial_world_state

    state = _world_state_to_graph_state(initial_world_state)
    start_ts = time.time()

    for _ in range(max_iterations):
        if time.time() - start_ts > max_wallclock_seconds:
            state["error"] = "Wall-clock timeout"
            break

        try:
            # LangGraph invoke runs the full graph from current state
            result = compiled_graph.invoke(state)
            state = result

            if state.get("goal_complete") or state.get("error"):
                break

        except Exception as exc:
            _logger.error("[LangGraphPipeline] Graph invocation error: %s", exc)
            state["error"] = f"LangGraph error: {exc}"
            break

    return _graph_state_to_world_state(state)
