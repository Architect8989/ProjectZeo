from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WorldState — shared state container passed between agents
# ---------------------------------------------------------------------------

@dataclass
class WorldState:
    
    # Perception
    screen_snapshot: Optional[Dict[str, Any]] = None
    entities: List[Dict[str, Any]] = field(default_factory=list)
    focused_app: str = "__unknown_app__"
    entity_count: int = 0
    last_delta: Optional[Dict[str, Any]] = None

    # Goal
    objective: str = ""
    goal_complete: bool = False

    # Reasoning
    proposed_action: Optional[Dict[str, Any]] = None
    reasoning_rationale: str = ""
    scaffold_context: str = ""

    # Safety
    safety_decision: str = "ALLOW"  # ALLOW | DENY | REQUIRE_HUMAN_CONFIRMATION
    safety_reason: str = ""

    # Execution
    last_action: Optional[Dict[str, Any]] = None
    last_action_success: bool = True
    last_action_output: str = ""
    consecutive_failures: int = 0

    # Memory context
    memory_context: str = ""
    semantic_facts: List[Dict[str, Any]] = field(default_factory=list)

    # Loop control
    iteration: int = 0
    start_ts: float = field(default_factory=time.time)
    error: Optional[str] = None

    def elapsed_seconds(self) -> float:
        return time.time() - self.start_ts


# ---------------------------------------------------------------------------
# Base Agent Protocol
# ---------------------------------------------------------------------------

class BaseAgent:
    """Base class for all specialist agents."""

    def __init__(self, name: str) -> None:
        self.name = name

    def run(self, state: WorldState) -> WorldState:
        
        raise NotImplementedError

    def _log(self, msg: str, *args) -> None:
        _logger.info(f"[{self.name}] {msg}", *args)

    def _warn(self, msg: str, *args) -> None:
        _logger.warning(f"[{self.name}] {msg}", *args)


# ---------------------------------------------------------------------------
# PerceiverAgent — Observation layer
# ---------------------------------------------------------------------------

class PerceiverAgent(BaseAgent):
    

    def __init__(
        self,
        *,
        world_graph,
        vision_runtime=None,
        atspi_bridge=None,
    ) -> None:
        super().__init__("PerceiverAgent")
        self._world_graph = world_graph
        self._vision = vision_runtime
        self._atspi = atspi_bridge

    def run(self, state: WorldState) -> WorldState:
        try:
            snapshot = self._world_graph.snapshot() if self._world_graph else {}
            if isinstance(snapshot, dict):
                entities = snapshot.get("entities", [])
                focused_app = str(snapshot.get("focused_app") or "__unknown_app__")
                delta = snapshot.get("delta")
                state.screen_snapshot = snapshot
                state.entities = entities[:50]  # cap for context window
                state.focused_app = focused_app
                state.entity_count = len(entities)
                state.last_delta = delta
                self._log(
                    "Perceived %d entities. focused_app=%r delta=%s",
                    len(entities), focused_app, "yes" if delta else "no",
                )
        except Exception as exc:
            self._warn("Perception failed: %s", exc)
            state.error = f"Perception error: {exc}"
        return state


# ---------------------------------------------------------------------------
# ReasonerAgent — Goal-directed action selection
# ---------------------------------------------------------------------------

class ReasonerAgent(BaseAgent):
    

    def __init__(
        self,
        *,
        gii_controller,
    ) -> None:
        super().__init__("ReasonerAgent")
        self._gii = gii_controller

    def run(self, state: WorldState) -> WorldState:
        if state.goal_complete or state.error:
            return state

        world_state_dict = {
            "entities": state.entities,
            "focused_app": state.focused_app,
            "entity_count": state.entity_count,
            "delta": state.last_delta,
            "_gii_loop_note": state.memory_context,
        }

        # GII-FIX: Inject DICP policy addendum + ToM user intent into world_state
        # so the GIIController's decide_next_action sees accumulated constraints
        # and the user's emotional/intentional state before selecting an action.
        try:
            _dicp = getattr(self._gii, "_dicp_engine", None)
            if _dicp is not None:
                _addendum = _dicp.get_policy_addendum(context={"world_state": world_state_dict})
                if _addendum:
                    world_state_dict["_dicp_policy_addendum"] = _addendum
        except Exception:
            pass
        try:
            _tom = getattr(self._gii, "_tom_agent", None)
            if _tom is not None:
                _tom_ctx = _tom.get_context_for_psr()
                if _tom_ctx:
                    world_state_dict["_tom_user_intent"] = _tom_ctx
        except Exception:
            pass

        try:
            action, reason = self._gii.decide_next_action(world_state_dict)
            state.proposed_action = action
            state.reasoning_rationale = reason

            if action is not None:
                op = action.get("operation", "unknown")
                if op == "done":
                    state.goal_complete = True
                    self._log("GOAL COMPLETE: %s", action.get("summary", ""))
                else:
                    self._log("Proposed: op=%s reason=%s", op, reason[:80])
            else:
                self._warn("No action proposed. reason=%s", reason[:100])

        except Exception as exc:
            self._warn("Reasoning failed: %s", exc)
            state.proposed_action = None
            state.reasoning_rationale = str(exc)

        return state


# ---------------------------------------------------------------------------
# SafetyAgent — Consequence evaluation
# ---------------------------------------------------------------------------

class SafetyAgent(BaseAgent):
    

    def __init__(
        self,
        *,
        consequence_reasoner,
        policy_engine,
    ) -> None:
        super().__init__("SafetyAgent")
        self._cr = consequence_reasoner
        self._policy = policy_engine

    def run(self, state: WorldState) -> WorldState:
        if state.proposed_action is None or state.goal_complete:
            state.safety_decision = "ALLOW"
            return state

        action = state.proposed_action
        op = action.get("operation", "")

        # Step 1: Policy engine check
        try:
            policy_decision, policy_reason = self._policy.validate_action_dict(
                action, focused_app=state.focused_app
            )
            if policy_decision == "DENY":
                state.safety_decision = "DENY"
                state.safety_reason = f"PolicyEngine: {policy_reason}"
                self._warn("Policy DENY: %s", policy_reason[:100])
                return state
        except Exception as policy_exc:
            self._warn("PolicyEngine error: %s", policy_exc)

        # Step 2: Consequence reasoner (for non-trivial ops)
        if self._cr is not None and op in ("command", "file_create", "install", "write", "type"):
            try:
                cr_result = self._cr.evaluate(
                    action=action,
                    objective=state.objective,
                    step_description=str(action.get("thought", "")),
                )
                from core.safety.consequence_reasoner import SafetyDecision
                if cr_result.decision == SafetyDecision.DENY:
                    state.safety_decision = "DENY"
                    state.safety_reason = f"ConsequenceReasoner T{cr_result.tier_reached}: {cr_result.reason}"
                    self._warn("CR DENY (Tier%d): %s", cr_result.tier_reached, cr_result.reason[:100])
                    return state
                elif cr_result.decision == SafetyDecision.REQUIRE_HUMAN_CONFIRMATION:
                    state.safety_decision = "REQUIRE_HUMAN_CONFIRMATION"
                    state.safety_reason = f"ConsequenceReasoner T{cr_result.tier_reached}: {cr_result.reason}"
                    self._warn(
                        "CR REQUIRE_CONFIRMATION (Tier%d): %s",
                        cr_result.tier_reached, cr_result.reason[:100],
                    )
                    return state
            except Exception as cr_exc:
                self._warn("ConsequenceReasoner error: %s", cr_exc)

        state.safety_decision = "ALLOW"
        state.safety_reason = ""
        return state


# ---------------------------------------------------------------------------
# ExecutorAgent — OS action dispatch
# ---------------------------------------------------------------------------

class ExecutorAgent(BaseAgent):
    

    def __init__(
        self,
        *,
        os_backend,
        execute_decision_fn: Optional[Callable] = None,
        execute_fn: Optional[Callable] = None,   # alias for execute_decision_fn
        journal=None,
    ) -> None:
        super().__init__("ExecutorAgent")
        self._os = os_backend
        self._execute_fn = execute_decision_fn or execute_fn
        self._journal = journal

    def run(self, state: WorldState) -> WorldState:
        if state.proposed_action is None or state.goal_complete:
            return state

        if state.safety_decision != "ALLOW":
            self._warn(
                "Skipping execution — safety decision: %s. %s",
                state.safety_decision, state.safety_reason[:80],
            )
            state.last_action_success = False
            state.last_action_output = f"Blocked: {state.safety_reason}"
            state.consecutive_failures += 1
            return state

        action = state.proposed_action
        op = action.get("operation", "unknown")
        success = False
        output = ""

        try:
            if self._execute_fn is not None:
                result = self._execute_fn(action, state.screen_snapshot or {})
                if isinstance(result, dict):
                    success = result.get("success", True)
                    output = str(result.get("output", ""))
                else:
                    success = bool(result)
            else:
                success = self._minimal_dispatch(op, action)

            state.last_action = action
            state.last_action_success = success
            state.last_action_output = output[:500]

            if success:
                state.consecutive_failures = 0
                self._log("Executed: op=%s success=True", op)
            else:
                state.consecutive_failures += 1
                self._warn("Executed: op=%s success=False output=%r", op, output[:80])

        except Exception as exc:
            state.last_action = action
            state.last_action_success = False
            state.last_action_output = str(exc)
            state.consecutive_failures += 1
            self._warn("Execution exception: %s", exc)

        return state

    def _minimal_dispatch(self, op: str, action: Dict[str, Any]) -> bool:
        if self._os is None:
            return False
        try:
            if op == "click":
                x, y = float(action.get("x", 0.5)), float(action.get("y", 0.5))
                self._os.click(x, y)
            elif op in ("write", "type"):
                self._os.write(str(action.get("content", "")))
            elif op == "press":
                self._os.press(action.get("keys", []))
            elif op == "command":
                result = self._os.exec(str(action.get("command", "")))
                return getattr(result, "returncode", 0) == 0
            elif op == "file_create":
                path = str(action.get("path", ""))
                content = str(action.get("content", ""))
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
            return True
        except Exception as exc:
            self._warn("Minimal dispatch error: %s", exc)
            return False


# ---------------------------------------------------------------------------
# MemoryAgent — Post-action memory update
# ---------------------------------------------------------------------------

class MemoryAgent(BaseAgent):
    

    def __init__(
        self,
        *,
        gii_controller=None,
        semantic_memory=None,
        application_memory=None,
        episodic_synthesizer=None,
        objective: str = "",
    ) -> None:
        super().__init__("MemoryAgent")
        self._gii = gii_controller
        self._objective = objective
        self._semantic = semantic_memory
        self._app_memory = application_memory
        self._synthesizer = episodic_synthesizer

    def run(self, state: WorldState) -> WorldState:
        # Record outcome in GII controller
        if state.last_action is not None:
            try:
                self._gii.record_outcome(
                    state.last_action,
                    success=state.last_action_success,
                    output=state.last_action_output,
                )
            except Exception as exc:
                self._warn("GII record_outcome failed: %s", exc)

        # Update memory context for next iteration
        try:
            context = self._gii.get_planning_context(focused_app=state.focused_app)
            state.memory_context = context[:2000]
        except Exception:
            pass

        return state


# ---------------------------------------------------------------------------
# Agent Pipeline — Sequential orchestration
# ---------------------------------------------------------------------------

class AgentPipeline:

    def __init__(
        self,
        agents: Optional[List[BaseAgent]] = None,
        *,
        perceiver=None,
        reasoner=None,
        safety=None,
        executor=None,
        memory=None,
        max_consecutive_failures: int = 10,
    ) -> None:
        # Accept either positional list or keyword agents
        if agents is not None:
            self._agents = list(agents)
        else:
            self._agents = [
                a for a in [perceiver, reasoner, safety, executor, memory]
                if a is not None
            ]
        self._max_failures = max_consecutive_failures

    def step(self, state: WorldState) -> WorldState:
        """Run all agents in sequence for one loop iteration."""
        for agent in self._agents:
            try:
                state = agent.run(state)
                if state.goal_complete or state.error:
                    break
                if state.consecutive_failures >= self._max_failures:
                    state.error = f"Too many consecutive failures ({self._max_failures})"
                    break
            except Exception as exc:
                _logger.error("[AgentPipeline] Agent %s crashed: %s", agent.name, exc)
                state.error = f"Agent {agent.name} crashed: {exc}"
                break
        state.iteration += 1
        return state

    def run_until_done(
        self,
        initial_state: WorldState,
        *,
        max_iterations: int = 500,
        max_wallclock_seconds: int = 3600,
    ) -> WorldState:
        """Run the pipeline loop until goal complete, timeout, or error."""
        state = initial_state
        start_ts = time.time()

        while state.iteration < max_iterations:
            if time.time() - start_ts > max_wallclock_seconds:
                state.error = "Wall-clock timeout"
                break
            state = self.step(state)
            if state.goal_complete or state.error:
                break

        return state


# ─────────────────────────────────────────────────────────────────────────────
# GII-FIX: build_pipeline() — unified factory for both LangGraph and fallback
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline(
    gii_controller,
    *,
    world_graph=None,
    vision_runtime=None,
    atspi_bridge=None,
    policy_engine=None,
    os_backend=None,
    execute_decision_fn=None,
    journal=None,
    objective: str = "",
    memory_dir: Optional[str] = None,
):
    """
    Build the complete GII specialist-agent pipeline.

    Tries LangGraph StateGraph first (if installed + PROJECTZEO_USE_LANGGRAPH=1).
    Falls back to AgentPipeline (sequential, no graph state).

    Returns:
        compiled_graph  — LangGraph compiled StateGraph, or None
        pipeline        — AgentPipeline fallback (always constructed)
    """
    # Wire individual agents
    perceiver = PerceiverAgent(
        world_graph=world_graph,
        vision_runtime=vision_runtime,
        atspi_bridge=atspi_bridge or getattr(gii_controller, "_atspi_bridge", None),
    )
    reasoner = ReasonerAgent(gii_controller=gii_controller)
    safety = SafetyAgent(
        consequence_reasoner=getattr(gii_controller, "consequence_reasoner", None),
        policy_engine=policy_engine,
    )
    executor = ExecutorAgent(
        os_backend=os_backend,
        execute_decision_fn=execute_decision_fn,
        journal=journal,
    )
    memory_agent = MemoryAgent(
        gii_controller=gii_controller,
        semantic_memory=getattr(gii_controller, "_semantic_memory", None),
        application_memory=getattr(gii_controller, "_application_memory", None),
        episodic_synthesizer=getattr(gii_controller, "_episodic_synthesizer", None),
        objective=objective,
    )

    # Try LangGraph
    compiled_graph = None
    try:
        from core.agents.langgraph_pipeline import build_langgraph_pipeline
        compiled_graph = build_langgraph_pipeline(
            perceiver, reasoner, safety, executor, memory_agent,
        )
    except Exception as _lg_exc:
        _logger.debug("[build_pipeline] LangGraph build failed (fallback active): %s", _lg_exc)

    # Always build AgentPipeline as fallback
    pipeline = AgentPipeline(
        perceiver=perceiver,
        reasoner=reasoner,
        safety=safety,
        executor=executor,
        memory=memory_agent,
    )

    _logger.info(
        "[build_pipeline] Pipeline ready. langgraph=%s fallback=AgentPipeline",
        compiled_graph is not None,
    )
    return compiled_graph, pipeline

