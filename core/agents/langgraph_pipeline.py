from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

_logger = logging.getLogger(__name__)

_langgraph_available: Optional[bool] = None

def _check_langgraph() -> bool:
    global _langgraph_available
    if _langgraph_available is not None:
        return _langgraph_available
    try:
        import langgraph
        _langgraph_available = True
    except ImportError:
        _langgraph_available = False
        _logger.info(
            "[LangGraphPipeline] langgraph not installed. Using AgentPipeline fallback. "
            "Install: pip install langgraph"
        )
    return _langgraph_available

def _use_langgraph() -> bool:
    return (
        os.environ.get("PROJECTZEO_USE_LANGGRAPH", "1").strip() not in ("0", "false", "no")
        and _check_langgraph()
    )

def _make_langgraph_state_type():
    from typing import TypedDict
    from core.agents.specialist_agents import WorldState

    class GraphState(TypedDict, total=False):
        objective: str
        iteration: int
        goal_complete: bool
        error: Optional[str]

        screen_snapshot: Optional[Dict[str, Any]]
        entities: List[Dict[str, Any]]
        focused_app: str
        entity_count: int
        last_delta: Optional[Dict[str, Any]]

        proposed_action: Optional[Dict[str, Any]]
        reasoning_rationale: str

        safety_decision: str
        safety_reason: str

        last_action: Optional[Dict[str, Any]]
        last_action_success: bool
        last_action_output: str
        consecutive_failures: int

        memory_context: str

    return GraphState

def _world_state_to_graph_state(ws) -> Dict[str, Any]:
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
    from core.agents.specialist_agents import WorldState
    ws = WorldState()
    for key, val in state.items():
        if hasattr(ws, key):
            setattr(ws, key, val)
    return ws

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
        from langgraph.graph import StateGraph, END

        GraphState = _make_langgraph_state_type()

        def perceive_node(state: Dict[str, Any]) -> Dict[str, Any]:
            ws = _graph_state_to_world_state(state)
            ws = perceiver_agent.run(ws)
            return _world_state_to_graph_state(ws)

        def reason_node(state: Dict[str, Any]) -> Dict[str, Any]:
            ws = _graph_state_to_world_state(state)
            ws = reasoner_agent.run(ws)
            return _world_state_to_graph_state(ws)

        def safety_node(state: Dict[str, Any]) -> Dict[str, Any]:
            ws = _graph_state_to_world_state(state)
            ws = safety_agent.run(ws)
            return _world_state_to_graph_state(ws)

        def confirm_node(state: Dict[str, Any]) -> Dict[str, Any]:
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
            ws = _graph_state_to_world_state(state)
            ws = executor_agent.run(ws)
            return _world_state_to_graph_state(ws)

        def memory_node(state: Dict[str, Any]) -> Dict[str, Any]:
            ws = _graph_state_to_world_state(state)
            ws = memory_agent.run(ws)
            return _world_state_to_graph_state(ws)

        def should_execute(state: Dict[str, Any]) -> str:
            if state.get("goal_complete") or state.get("error"):
                return "done"
            decision = state.get("safety_decision", "ALLOW")
            if decision == "ALLOW":
                return "execute"
            if decision == "REQUIRE_HUMAN_CONFIRMATION":
                return "confirm"
            return "reason"

        def should_continue(state: Dict[str, Any]) -> str:
            if state.get("goal_complete"):
                return "done"
            if state.get("error"):
                return "done"
            failures = state.get("consecutive_failures", 0)
            if failures >= 10:
                return "done"
            return "perceive"

        graph_builder = StateGraph(GraphState)

        graph_builder.add_node("perceive",  perceive_node)
        graph_builder.add_node("reason",    reason_node)
        graph_builder.add_node("safety",    safety_node)
        graph_builder.add_node("confirm",   confirm_node)
        graph_builder.add_node("execute",   execute_node)
        graph_builder.add_node("memory",    memory_node)

        graph_builder.set_entry_point("perceive")

        graph_builder.add_edge("perceive", "reason")
        graph_builder.add_edge("reason", "safety")

        graph_builder.add_conditional_edges(
            "safety",
            should_execute,
            {
                "execute": "execute",
                "confirm": "confirm",
                "reason":  "reason",
                "done":    END,
            },
        )

        graph_builder.add_conditional_edges(
            "confirm",
            lambda s: "execute" if s.get("safety_decision") == "ALLOW" else "reason",
            {"execute": "execute", "reason": "reason"},
        )

        graph_builder.add_edge("execute", "memory")

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

def run_langgraph_pipeline(
    compiled_graph,
    initial_world_state,
    *,
    max_iterations: int = 500,
    max_wallclock_seconds: int = 3600,
):
    
    if compiled_graph is None:
        _logger.warning("[LangGraphPipeline] No compiled graph — cannot run.")
        return initial_world_state

    state = _world_state_to_graph_state(initial_world_state)
    start_ts = time.time()

    for _ in range(max_iterations):
        if time.time() - start_ts > max_wallclock_seconds:
            state["error"] = "Wall-clock timeout"
            break

        try:
            result = compiled_graph.invoke(state)
            state = result

            if state.get("goal_complete") or state.get("error"):
                break

        except Exception as exc:
            _logger.error("[LangGraphPipeline] Graph invocation error: %s", exc)
            state["error"] = f"LangGraph error: {exc}"
            break

    return _graph_state_to_world_state(state)
