from __future__ import annotations

import json
import logging
import os
import queue
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

_logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

_FLYWHEEL_ENABLED        = os.environ.get("PROJECTZEO_FLYWHEEL_ENABLED", "1") == "1"
_ANALYSIS_MIN_INTERVAL   = float(os.environ.get("PROJECTZEO_FLYWHEEL_INTERVAL", "30"))
_QUEUE_MAX_SIZE          = int(os.environ.get("PROJECTZEO_FLYWHEEL_QUEUE", "100"))
_ANALYSIS_TIMEOUT        = float(os.environ.get("PROJECTZEO_FLYWHEEL_TIMEOUT", "60"))
_MAX_RETRY_BBON_N        = int(os.environ.get("PROJECTZEO_FLYWHEEL_BBON_N", "3"))
_PATTERN_WINDOW_SIZE     = int(os.environ.get("PROJECTZEO_FLYWHEEL_WINDOW", "20"))
_MIN_PATTERN_FAILURES    = int(os.environ.get("PROJECTZEO_FLYWHEEL_MIN_FAIL", "3"))
_TRAJECTORY_DATA_DIR     = os.path.expanduser(
    os.environ.get("PROJECTZEO_TRAJECTORY_DIR", "~/.projectzeo/trajectories")
)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryOutcome(str, Enum):
    SUCCESS        = "success"
    FAILURE        = "failure"
    PARTIAL        = "partial"
    TIMEOUT        = "timeout"
    HUMAN_REQUIRED = "human_required"


@dataclass
class TrajectoryEvent:
    """A completed trajectory submitted to the flywheel for processing."""
    trajectory_id:    str
    objective:        str
    app_context:      str
    outcome:          TrajectoryOutcome
    reward:           float                    # 0.0-1.0
    steps:            List[Dict[str, Any]]     # Action/observation pairs
    duration_sec:     float
    goal_progress:    float = 0.0             # Final goal progress (0.0-1.0)
    failure_reason:   str = ""
    iteration_count:  int = 0
    metadata:         Dict[str, Any] = field(default_factory=dict)
    submitted_at:     float = field(default_factory=time.time)


@dataclass
class FlywheelInsight:
    """An insight extracted from trajectory analysis."""
    insight_id:    str
    objective:     str
    app_context:   str
    insight_type:  str                          # "failure_pattern", "success_pattern", "lesson"
    content:       str
    confidence:    float = 0.7
    created_at:    float = field(default_factory=time.time)


@dataclass
class FailurePattern:
    """A recurring failure pattern detected across multiple trajectories."""
    pattern_id:   str
    description:  str
    task_type:    str
    app_context:  str
    occurrences:  int = 0
    last_seen:    float = field(default_factory=time.time)
    mitigation:   str = ""
    resolved:     bool = False


# ─────────────────────────────────────────────────────────────────────────────
# LLM prompts
# ─────────────────────────────────────────────────────────────────────────────

_FAILURE_ANALYSIS_SYSTEM = """\
You are a Trajectory Analyst for a GUI automation agent.

Given a failed task trajectory, analyse what went wrong and generate:
1. Root cause of the failure
2. Up to 3 alternative strategies the agent should try
3. A lesson to remember for future similar tasks
4. Whether a systemic pattern is likely (same type of failure)

OUTPUT FORMAT (JSON, no markdown):
{
  "root_cause": "<concise root cause>",
  "alternatives": [
    {
      "strategy": "<description>",
      "first_action": {"operation": "...", "target": "...", "text": "..."},
      "reasoning": "<why this would work>"
    }
  ],
  "lesson": "<generalised lesson for future tasks>",
  "is_systemic": <true|false>,
  "systemic_pattern": "<if systemic: description of the recurring pattern>"
}
"""

_SUCCESS_ANALYSIS_SYSTEM = """\
You are a Trajectory Analyst for a GUI automation agent.

Given a successful task trajectory, extract:
1. The key insight that made this approach work
2. Any UI patterns discovered that should be remembered
3. A procedural rule for this task type

OUTPUT FORMAT (JSON, no markdown):
{
  "key_insight": "<what made this work>",
  "ui_patterns": ["<pattern 1>", "<pattern 2>"],
  "procedural_rule": "<IF [condition] THEN [action sequence]>",
  "generalises_to": "<other task types this applies to>"
}
"""

_RETRY_STRATEGY_SYSTEM = """\
You are a Retry Strategist for a GUI automation agent.

Given a failed task and a proposed alternative strategy, generate the
first 3 concrete actions the agent should take to try this strategy.

OUTPUT FORMAT (JSON array, no markdown):
[
  {"operation": "<op>", "target": "<target>", "text": "<text>", "keys": "<keys>"},
  ...
]
"""


# ─────────────────────────────────────────────────────────────────────────────
# TrajectoryFlywheel
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryFlywheel:
    """
    Background self-improvement loop for ProjectZeo GII.

    Submit trajectories via submit(). The flywheel processes them
    asynchronously in a daemon thread, updating memory and learning systems.
    """

    def __init__(
        self,
        llm_call: Callable,
        *,
        openmemory: Optional[Any] = None,
        soar_chunker: Optional[Any] = None,
        arpo_trainer: Optional[Any] = None,
    ) -> None:
        self._llm        = llm_call
        self._openmemory = openmemory
        self._chunker    = soar_chunker
        self._arpo       = arpo_trainer

        # Trajectory queue (bounded, non-blocking submission)
        self._queue: queue.Queue = queue.Queue(maxsize=_QUEUE_MAX_SIZE)

        # Failure pattern detection
        self._failure_history: List[Dict[str, Any]] = []
        self._patterns: Dict[str, FailurePattern] = {}
        self._insights: List[FlywheelInsight] = []

        # Statistics
        self._lock              = threading.RLock()
        self._processed_count   = 0
        self._success_count     = 0
        self._failure_count     = 0
        self._insight_count     = 0
        self._last_analysis_ts  = 0.0
        self._running           = False

        # Trajectory persistence
        os.makedirs(_TRAJECTORY_DATA_DIR, exist_ok=True)

        # Start background thread
        if _FLYWHEEL_ENABLED:
            self._start()

        _logger.info(
            "[Flywheel] Initialised. enabled=%s openmemory=%s chunker=%s arpo=%s",
            _FLYWHEEL_ENABLED,
            openmemory is not None,
            soar_chunker is not None,
            arpo_trainer is not None,
        )

    # =========================================================================
    # Public interface
    # =========================================================================

    def submit(self, event: TrajectoryEvent) -> bool:
        """
        Submit a completed trajectory for asynchronous processing.
        Non-blocking: returns False if queue is full.
        """
        if not _FLYWHEEL_ENABLED:
            return False
        try:
            self._queue.put_nowait(event)
            _logger.debug(
                "[Flywheel] Queued trajectory %s outcome=%s reward=%.2f",
                event.trajectory_id, event.outcome.value, event.reward
            )
            return True
        except queue.Full:
            _logger.warning("[Flywheel] Queue full — dropping trajectory %s", event.trajectory_id)
            return False

    def submit_from_arpo(
        self,
        objective: str,
        app_context: str,
        steps: List[Dict[str, Any]],
        success: bool,
        reward: float,
        goal_progress: float = 0.0,
        failure_reason: str = "",
    ) -> bool:
        """Convenience wrapper for ARPO TrajectoryRecord → FlywheelEvent."""
        outcome = (
            TrajectoryOutcome.SUCCESS if success
            else (
                TrajectoryOutcome.PARTIAL if goal_progress > 0.3
                else TrajectoryOutcome.FAILURE
            )
        )
        event = TrajectoryEvent(
            trajectory_id = f"traj_{uuid.uuid4().hex[:12]}",
            objective     = objective,
            app_context   = app_context,
            outcome       = outcome,
            reward        = reward,
            steps         = steps,
            duration_sec  = 0.0,
            goal_progress = goal_progress,
            failure_reason = failure_reason,
        )
        return self.submit(event)

    def get_insights(
        self,
        objective: str = "",
        app_context: str = "",
        limit: int = 5,
    ) -> List[FlywheelInsight]:
        """Return recent insights, optionally filtered by objective/app."""
        with self._lock:
            insights = list(self._insights)

        if objective or app_context:
            filtered = []
            for ins in insights:
                if objective and objective.lower() not in ins.objective.lower():
                    if app_context and app_context.lower() not in ins.app_context.lower():
                        continue
                filtered.append(ins)
            insights = filtered

        return sorted(insights, key=lambda i: i.created_at, reverse=True)[:limit]

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled":           _FLYWHEEL_ENABLED,
                "running":           self._running,
                "queue_size":        self._queue.qsize(),
                "processed":         self._processed_count,
                "successes":         self._success_count,
                "failures":          self._failure_count,
                "insights":          self._insight_count,
                "active_patterns":   len([p for p in self._patterns.values() if not p.resolved]),
                "last_analysis_ts":  self._last_analysis_ts,
            }

    def shutdown(self) -> None:
        """Gracefully stop the flywheel background thread."""
        self._running = False
        # Unblock the queue.get()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        _logger.info("[Flywheel] Shutdown requested")

    # =========================================================================
    # Background loop
    # =========================================================================

    def _start(self) -> None:
        """Start the background processing thread."""
        t = threading.Thread(
            target=self._run_loop,
            name="flywheel_worker",
            daemon=True,
        )
        self._running = True
        t.start()
        _logger.info("[Flywheel] Background thread started")

    def _run_loop(self) -> None:
        """Main flywheel processing loop (runs in background thread)."""
        while self._running:
            try:
                # Block until event arrives (timeout allows shutdown check)
                event = self._queue.get(timeout=5.0)
                if event is None:
                    break  # Shutdown signal

                self._process_event(event)

            except queue.Empty:
                # Periodic: check for systemic failure patterns
                self._detect_systemic_patterns()
                continue
            except Exception as exc:
                _logger.error("[Flywheel] Loop error: %s", exc, exc_info=True)
                time.sleep(1.0)

        _logger.info("[Flywheel] Background thread stopped")

    def _process_event(self, event: TrajectoryEvent) -> None:
        """Process a single trajectory event."""
        t0 = time.perf_counter()

        # Rate limiting: don't overload LLM
        now = time.time()
        elapsed_since_last = now - self._last_analysis_ts
        if elapsed_since_last < _ANALYSIS_MIN_INTERVAL:
            wait = _ANALYSIS_MIN_INTERVAL - elapsed_since_last
            _logger.debug("[Flywheel] Rate limiting: waiting %.1fs", wait)
            time.sleep(wait)

        try:
            if event.outcome == TrajectoryOutcome.SUCCESS:
                self._process_success(event)
            elif event.outcome in (TrajectoryOutcome.FAILURE, TrajectoryOutcome.TIMEOUT):
                self._process_failure(event)
            elif event.outcome == TrajectoryOutcome.PARTIAL:
                self._process_partial(event)

            # Always persist the trajectory
            self._persist_trajectory(event)

            with self._lock:
                self._processed_count += 1
                self._last_analysis_ts = time.time()

        except Exception as exc:
            _logger.error(
                "[Flywheel] Error processing trajectory %s: %s",
                event.trajectory_id, exc, exc_info=True
            )

        latency = (time.perf_counter() - t0) * 1000
        _logger.info(
            "[Flywheel] Processed %s outcome=%s reward=%.2f in %.0fms",
            event.trajectory_id, event.outcome.value, event.reward, latency
        )

    # =========================================================================
    # Success processing
    # =========================================================================

    def _process_success(self, event: TrajectoryEvent) -> None:
        """Extract insights and procedural knowledge from a successful run."""
        with self._lock:
            self._success_count += 1

        # 1. SOAR Chunking: store operator sequence as procedural memory
        if self._chunker is not None and event.steps:
            try:
                action_seq = [
                    s["action"] for s in event.steps
                    if isinstance(s.get("action"), dict)
                ]
                self._chunker.chunk(
                    operator_sequence = action_seq,
                    goal_description  = event.objective,
                    app_context       = event.app_context,
                    success_reward    = event.reward,
                )
            except Exception as exc:
                _logger.debug("[Flywheel] SOAR chunking error: %s", exc)

        # 2. LLM success analysis
        insight_content = self._analyse_success(event)
        if insight_content:
            insight = FlywheelInsight(
                insight_id   = f"ins_{uuid.uuid4().hex[:8]}",
                objective    = event.objective,
                app_context  = event.app_context,
                insight_type = "success_pattern",
                content      = insight_content,
                confidence   = event.reward,
            )
            with self._lock:
                self._insights.append(insight)
                if len(self._insights) > 500:
                    self._insights = self._insights[-400:]
                self._insight_count += 1

            # Store in OpenMemory
            self._store_insight(insight)

        _logger.info(
            "[Flywheel] Success processed: obj=%r app=%r reward=%.2f steps=%d",
            event.objective[:40], event.app_context, event.reward, len(event.steps)
        )

    def _analyse_success(self, event: TrajectoryEvent) -> Optional[str]:
        """LLM analysis of a successful trajectory."""
        if not event.steps:
            return None

        steps_summary = self._summarize_steps(event.steps, max_steps=10)
        messages = [
            {"role": "system", "content": _SUCCESS_ANALYSIS_SYSTEM},
            {"role": "user", "content": (
                f"OBJECTIVE: {event.objective}\n"
                f"APPLICATION: {event.app_context}\n"
                f"REWARD: {event.reward:.2f}\n"
                f"STEPS ({len(event.steps)} total):\n{steps_summary}\n\n"
                "Extract insights from this successful run."
            )},
        ]

        result: Dict[str, Any] = {}

        def _call():
            try:
                raw = self._llm(messages, objective="flywheel_success_analysis")
                cleaned = re.sub(r"```(?:json)?", "", raw or "").strip()
                m = re.search(r"\{.*\}", cleaned, re.DOTALL)
                if m:
                    result.update(json.loads(m.group()))
            except Exception as exc:
                _logger.debug("[Flywheel] Success analysis LLM error: %s", exc)

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        t.join(timeout=_ANALYSIS_TIMEOUT)

        if not result:
            return None

        parts = []
        if result.get("key_insight"):
            parts.append(f"Insight: {result['key_insight']}")
        if result.get("procedural_rule"):
            parts.append(f"Rule: {result['procedural_rule']}")
        if result.get("ui_patterns"):
            parts.append(f"UI patterns: {'; '.join(result['ui_patterns'][:3])}")
        if result.get("generalises_to"):
            parts.append(f"Generalises to: {result['generalises_to']}")

        return "\n".join(parts) if parts else None

    # =========================================================================
    # Failure processing
    # =========================================================================

    def _process_failure(self, event: TrajectoryEvent) -> None:
        """Analyse failures and generate retry strategies."""
        with self._lock:
            self._failure_count += 1
            self._failure_history.append({
                "objective":    event.objective,
                "app_context":  event.app_context,
                "failure_reason": event.failure_reason,
                "ts":           event.submitted_at,
            })
            # Keep bounded
            if len(self._failure_history) > _PATTERN_WINDOW_SIZE:
                self._failure_history.pop(0)

        # LLM failure analysis
        analysis = self._analyse_failure(event)
        if not analysis:
            return

        # Store lesson in OpenMemory
        lesson = analysis.get("lesson", "")
        if lesson and self._openmemory:
            try:
                self._openmemory.store_reflective(
                    content   = f"Task failure lesson: {lesson}",
                    subject   = event.app_context,
                    importance = 0.7,
                )
            except Exception as _mem_err:
                _logger.debug("[Flywheel] Memory store error (non-fatal): %s", _mem_err)

        # Store failure insight
        root_cause = analysis.get("root_cause", "")
        if root_cause:
            insight = FlywheelInsight(
                insight_id   = f"ins_{uuid.uuid4().hex[:8]}",
                objective    = event.objective,
                app_context  = event.app_context,
                insight_type = "failure_pattern",
                content      = f"Root cause: {root_cause}\nLesson: {lesson}",
                confidence   = 0.6,
            )
            with self._lock:
                self._insights.append(insight)
                self._insight_count += 1
            self._store_insight(insight)

        # Check for systemic pattern
        if analysis.get("is_systemic") and analysis.get("systemic_pattern"):
            self._record_systemic_pattern(event, analysis["systemic_pattern"])

        # Generate and record retry strategies (for future reference — not executed here)
        alternatives = analysis.get("alternatives", [])
        if alternatives and self._openmemory:
            for alt in alternatives[:3]:
                try:
                    strategy_text = (
                        f"Alternative for '{event.objective[:80]}': "
                        f"{alt.get('strategy', '')} — "
                        f"Reasoning: {alt.get('reasoning', '')}"
                    )
                    self._openmemory.store_semantic(
                        content   = strategy_text,
                        subject   = event.app_context,
                        importance = 0.65,
                    )
                except Exception as _mem_err:
                    _logger.debug("[Flywheel] Memory store error (non-fatal): %s", _mem_err)

        _logger.info(
            "[Flywheel] Failure processed: obj=%r root_cause=%r systemic=%s",
            event.objective[:40], root_cause[:60] if root_cause else "",
            analysis.get("is_systemic", False)
        )

    def _analyse_failure(self, event: TrajectoryEvent) -> Optional[Dict[str, Any]]:
        """LLM analysis of a failed trajectory."""
        if not event.steps:
            return {"root_cause": event.failure_reason or "unknown", "alternatives": [], "lesson": ""}

        steps_summary = self._summarize_steps(event.steps, max_steps=15)
        messages = [
            {"role": "system", "content": _FAILURE_ANALYSIS_SYSTEM},
            {"role": "user", "content": (
                f"OBJECTIVE: {event.objective}\n"
                f"APPLICATION: {event.app_context}\n"
                f"FAILURE REASON: {event.failure_reason or 'unknown'}\n"
                f"GOAL PROGRESS REACHED: {event.goal_progress:.0%}\n"
                f"STEPS ATTEMPTED ({len(event.steps)} total):\n{steps_summary}\n\n"
                "Analyse this failure and provide alternatives."
            )},
        ]

        result: Dict[str, Any] = {}

        def _call():
            try:
                raw = self._llm(messages, objective="flywheel_failure_analysis")
                cleaned = re.sub(r"```(?:json)?", "", raw or "").strip()
                m = re.search(r"\{.*\}", cleaned, re.DOTALL)
                if m:
                    result.update(json.loads(m.group()))
            except Exception as exc:
                _logger.debug("[Flywheel] Failure analysis LLM error: %s", exc)

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        t.join(timeout=_ANALYSIS_TIMEOUT)

        return result if result else None

    # =========================================================================
    # Partial processing
    # =========================================================================

    def _process_partial(self, event: TrajectoryEvent) -> None:
        """Partial success: extract what worked, note what failed."""
        # Treat partial as a mini-success (extract the successful portion)
        # and a mini-failure (note where it stalled)

        if event.goal_progress >= 0.5:
            # Mostly successful: treat as success for chunking purposes
            self._process_success(event)
        else:
            # Mostly failed: extract the early steps as a partial procedure
            with self._lock:
                self._failure_count += 1

            lesson = (
                f"Partial completion ({event.goal_progress:.0%}) for "
                f"'{event.objective[:80]}': "
                f"Achieved {event.goal_progress:.0%} progress before stalling. "
                f"App: {event.app_context}. Reason: {event.failure_reason or 'stall'}"
            )
            if self._openmemory:
                try:
                    self._openmemory.store_reflective(
                        content=lesson,
                        subject=event.app_context,
                        importance=0.5,
                    )
                except Exception as _mem_err:
                    _logger.debug("[Flywheel] Memory store error (non-fatal): %s", _mem_err)

    # =========================================================================
    # Pattern detection
    # =========================================================================

    def _detect_systemic_patterns(self) -> None:
        """Periodically scan failure history for recurring patterns."""
        with self._lock:
            if len(self._failure_history) < _MIN_PATTERN_FAILURES:
                return
            history = list(self._failure_history)

        # Group by (app_context, failure type)
        groups: Dict[str, List[Dict]] = {}
        for event in history:
            key = f"{event.get('app_context', '')}:{event.get('failure_reason', '')[:50]}"
            groups.setdefault(key, []).append(event)

        for key, events in groups.items():
            if len(events) >= _MIN_PATTERN_FAILURES:
                pattern_id = f"pat_{hash(key) % 99999:05d}"
                with self._lock:
                    if pattern_id not in self._patterns:
                        parts = key.split(":", 1)
                        pattern = FailurePattern(
                            pattern_id  = pattern_id,
                            description = f"Recurring failure: {parts[-1][:100]}",
                            task_type   = events[0].get("objective", "")[:60],
                            app_context = parts[0],
                            occurrences = len(events),
                        )
                        self._patterns[pattern_id] = pattern
                        _logger.warning(
                            "[Flywheel] Systemic pattern detected: %s (occurrences=%d)",
                            pattern.description, pattern.occurrences
                        )
                        # Store as high-importance reflective memory
                        if self._openmemory:
                            try:
                                self._openmemory.store_reflective(
                                    content=(
                                        f"SYSTEMIC FAILURE PATTERN: {pattern.description}\n"
                                        f"App: {pattern.app_context}, "
                                        f"Occurrences: {pattern.occurrences}"
                                    ),
                                    subject=pattern.app_context,
                                    importance=0.9,
                                )
                            except Exception as _mem_err:
                                _logger.debug("[Flywheel] Pattern store error (non-fatal): %s", _mem_err)
                    else:
                        self._patterns[pattern_id].occurrences = len(events)
                        self._patterns[pattern_id].last_seen = time.time()

    def _record_systemic_pattern(self, event: TrajectoryEvent, pattern_desc: str) -> None:
        """Record a systemic pattern identified by LLM analysis."""
        pattern_id = f"pat_llm_{uuid.uuid4().hex[:8]}"
        pattern = FailurePattern(
            pattern_id  = pattern_id,
            description = pattern_desc[:200],
            task_type   = event.objective[:60],
            app_context = event.app_context,
            occurrences = 1,
        )
        with self._lock:
            self._patterns[pattern_id] = pattern

    # =========================================================================
    # Memory updates
    # =========================================================================

    def _store_insight(self, insight: FlywheelInsight) -> None:
        """Persist an insight to OpenMemory."""
        if self._openmemory is None:
            return
        try:
            sector = (
                "reflective" if insight.insight_type in ("failure_pattern", "lesson")
                else "semantic"
            )
            store_method = getattr(self._openmemory, f"store_{sector}", None)
            if store_method:
                store_method(
                    content=f"[{insight.insight_type}] {insight.content}",
                    subject=insight.app_context or insight.objective[:40],
                    importance=0.70 + 0.15 * insight.confidence,
                )
        except Exception as exc:
            _logger.debug("[Flywheel] Insight store error: %s", exc)

    # =========================================================================
    # Persistence
    # =========================================================================

    def _persist_trajectory(self, event: TrajectoryEvent) -> None:
        """Save trajectory to disk for offline RSSM training (Phase 2)."""
        try:
            filename = f"{event.trajectory_id}_{int(event.submitted_at)}.json"
            path = os.path.join(_TRAJECTORY_DATA_DIR, filename)
            data = {
                "trajectory_id": event.trajectory_id,
                "objective":     event.objective,
                "app_context":   event.app_context,
                "outcome":       event.outcome.value,
                "reward":        event.reward,
                "goal_progress": event.goal_progress,
                "duration_sec":  event.duration_sec,
                "step_count":    len(event.steps),
                "failure_reason": event.failure_reason,
                "steps":         event.steps[:50],  # Cap for storage
                "submitted_at":  event.submitted_at,
            }
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp, path)
            _logger.debug("[Flywheel] Persisted trajectory to %s", path)
        except Exception as exc:
            _logger.debug("[Flywheel] Trajectory persist error: %s", exc)

    # =========================================================================
    # Helpers
    # =========================================================================

    def _summarize_steps(self, steps: List[Dict[str, Any]], max_steps: int = 10) -> str:
        """Build a readable step summary for LLM analysis."""
        lines = []
        for i, step in enumerate(steps[:max_steps]):
            action = step.get("action", {})
            op  = action.get("operation", "?")
            tgt = action.get("target", "") or action.get("text", "") or ""
            ok  = "✓" if step.get("outcome", True) else "✗"
            lines.append(f"  {i+1}. [{ok}] {op} {tgt!r:.50}")
        if len(steps) > max_steps:
            lines.append(f"  ... ({len(steps) - max_steps} more steps)")
        return "\n".join(lines) if lines else "(no steps)"
