"""
core/cognition/self_model.py — NEW MODULE
==========================================
Agent self-model: persistent record of what the agent knows it can/cannot do.

This module implements one of the core GII gaps identified in the audit:
  "No self-modeling — agent has no explicit representation of its own
   capabilities, limitations, or history of success/failure per app/domain."

A SelfModel enables:
  1. Calibrated confidence: PSR knows its own error rate → adjusts approach
  2. Capability awareness: "I have successfully installed packages 8/10 times"
  3. Failure pattern recognition: "I always fail on modal dialogs in GIMP"
  4. Adaptive strategy selection: switch to conservative actions when error
     rate is high for current task domain
  5. Self-knowledge persistence: survives across sessions via JSON on disk

Architecture:
  - CapabilityRecord: tracks success/failure stats for a (domain, operation) pair
  - DomainStats: aggregates stats for an app/domain
  - SelfModel: the full model, persisted to ~/.projectzeo/self_model.json
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

_DEFAULT_SELF_MODEL_PATH = os.path.join(
    os.path.expanduser("~"), ".projectzeo", "self_model.json"
)


@dataclass
class CapabilityRecord:
    """Success/failure stats for a (domain, operation) pair."""
    domain:     str
    operation:  str
    successes:  int = 0
    failures:   int = 0
    last_ts:    float = field(default_factory=time.time)

    @property
    def total(self) -> int:
        return self.successes + self.failures

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.5  # unknown → neutral prior
        return self.successes / self.total

    @property
    def confidence_level(self) -> str:
        """Return a human-readable confidence label."""
        sr = self.success_rate
        n  = self.total
        if n < 3:
            return "untested"
        if sr >= 0.85:
            return "high"
        if sr >= 0.60:
            return "medium"
        if sr >= 0.35:
            return "low"
        return "unreliable"

    def record(self, success: bool) -> None:
        if success:
            self.successes += 1
        else:
            self.failures += 1
        self.last_ts = time.time()


@dataclass
class TaskOutcome:
    """Record of a completed task."""
    objective: str
    app:       str
    success:   bool
    ts:        float = field(default_factory=time.time)
    duration:  float = 0.0


class SelfModel:
    """
    Persistent agent self-model.

    Stores capability records, task outcomes, and learned limitations.
    Provides formatted context strings for injection into PSR prompts.
    """

    # Maximum records to keep in memory (older ones are pruned)
    _MAX_CAPABILITY_RECORDS = 200
    _MAX_TASK_OUTCOMES = 100

    def __init__(
        self,
        *,
        agent_id: str = "default",
        memory_dir: Optional[str] = None,
    ) -> None:
        self._agent_id = agent_id
        self._lock = threading.RLock()

        # Map of (domain, operation) -> CapabilityRecord
        self._capabilities: Dict[Tuple[str, str], CapabilityRecord] = {}

        # Recent task outcomes
        self._task_outcomes: List[TaskOutcome] = []

        # Current session stats
        self._session_start    = time.time()
        self._session_actions  = 0
        self._session_successes = 0
        self._session_failures  = 0

        # Known limitations (explicitly recorded)
        self._known_limitations: List[str] = []

        # Persistence path
        if memory_dir:
            self._path = os.path.join(memory_dir, "self_model.json")
        else:
            self._path = _DEFAULT_SELF_MODEL_PATH

        # Load from disk
        self._load()

    # =========================================================================
    # Recording API
    # =========================================================================

    def record_action_result(
        self,
        *,
        action: Dict[str, Any],
        success: bool,
        output: str = "",
        domain: Optional[str] = None,
    ) -> None:
        """Record the outcome of a single action dispatch."""
        op = str(action.get("operation", "unknown")).lower()
        d  = (domain or "unknown").lower()

        with self._lock:
            key = (d, op)
            if key not in self._capabilities:
                self._capabilities[key] = CapabilityRecord(domain=d, operation=op)
            self._capabilities[key].record(success)
            self._session_actions += 1
            if success:
                self._session_successes += 1
            else:
                self._session_failures += 1

            # Auto-detect limitations from failure outputs
            if not success and output:
                self._maybe_record_limitation(op, output, d)

    def record_task_outcome(
        self,
        *,
        objective: str,
        success: bool,
        focused_app: Optional[str] = None,
        duration: float = 0.0,
    ) -> None:
        """Record the outcome of a complete task."""
        with self._lock:
            outcome = TaskOutcome(
                objective=objective[:300],
                app=focused_app or "unknown",
                success=success,
                duration=duration,
            )
            self._task_outcomes.append(outcome)
            if len(self._task_outcomes) > self._MAX_TASK_OUTCOMES:
                self._task_outcomes = self._task_outcomes[-self._MAX_TASK_OUTCOMES:]
        self._save()

    def add_limitation(self, description: str) -> None:
        """Explicitly record a known limitation."""
        with self._lock:
            if description not in self._known_limitations:
                self._known_limitations.append(description[:300])

    # =========================================================================
    # Query API
    # =========================================================================

    def get_capability(
        self, domain: str, operation: str
    ) -> Optional[CapabilityRecord]:
        key = (domain.lower(), operation.lower())
        with self._lock:
            return self._capabilities.get(key)

    def get_domain_stats(self, domain: str) -> Dict[str, Any]:
        """Return aggregated stats for a domain (app)."""
        d = domain.lower()
        records = []
        with self._lock:
            for (dom, op), rec in self._capabilities.items():
                if dom == d:
                    records.append(rec)
        if not records:
            return {"domain": d, "total_actions": 0, "success_rate": None}

        total = sum(r.total for r in records)
        successes = sum(r.successes for r in records)
        return {
            "domain":       d,
            "total_actions": total,
            "successes":    successes,
            "failures":     total - successes,
            "success_rate": round(successes / max(total, 1), 3),
            "by_operation": {
                r.operation: {
                    "total": r.total,
                    "success_rate": round(r.success_rate, 3),
                    "confidence":   r.confidence_level,
                }
                for r in records
            },
        }

    def get_session_error_rate(self) -> float:
        with self._lock:
            total = max(self._session_actions, 1)
            return self._session_failures / total

    def format_context(self) -> str:
        """
        Return a compact self-model context for injection into PSR prompt.
        Includes: session stats, top weaknesses, known limitations.
        """
        with self._lock:
            session_total    = self._session_actions
            session_err_rate = self._session_failures / max(session_total, 1)

            # Find top 3 weakest capabilities this session
            weak = sorted(
                [r for r in self._capabilities.values() if r.total >= 3 and r.success_rate < 0.6],
                key=lambda r: r.success_rate,
            )[:3]

            # Find top 3 strongest
            strong = sorted(
                [r for r in self._capabilities.values() if r.total >= 3 and r.success_rate >= 0.85],
                key=lambda r: -r.success_rate,
            )[:3]

            limitations = list(self._known_limitations[-3:])

        lines = [
            f"Session: {session_total} actions, error_rate={session_err_rate:.1%}",
        ]
        if strong:
            sstr = ", ".join(f"{r.domain}/{r.operation}({r.success_rate:.0%})" for r in strong)
            lines.append(f"Strong at: {sstr}")
        if weak:
            wstr = ", ".join(f"{r.domain}/{r.operation}({r.success_rate:.0%})" for r in weak)
            lines.append(f"Weak at: {wstr}")
        if limitations:
            for lim in limitations:
                lines.append(f"Known limitation: {lim}")

        return "\n".join(lines)

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "agent_id":             self._agent_id,
                "capability_records":   len(self._capabilities),
                "task_outcomes":        len(self._task_outcomes),
                "session_actions":      self._session_actions,
                "session_error_rate":   round(
                    self._session_failures / max(self._session_actions, 1), 4
                ),
                "known_limitations":    len(self._known_limitations),
                "session_duration_s":   round(time.time() - self._session_start, 1),
            }

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _maybe_record_limitation(
        self, operation: str, output: str, domain: str
    ) -> None:
        """Heuristically detect limitation patterns from failure output."""
        output_lower = output.lower()
        patterns = [
            ("permission denied",  f"Lacks permission for {operation} in {domain}"),
            ("not found",          f"Cannot find target element for {operation} in {domain}"),
            ("timeout",            f"{operation} frequently times out in {domain}"),
            ("no such file",       f"File not found errors during {operation}"),
            ("command not found",  f"Required command unavailable for {operation}"),
        ]
        for pattern, limitation in patterns:
            if pattern in output_lower:
                self.add_limitation(limitation)
                break

    # =========================================================================
    # Persistence
    # =========================================================================

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            with self._lock:
                data = {
                    "agent_id": self._agent_id,
                    "saved_at": time.time(),
                    "capabilities": {
                        f"{k[0]}:{k[1]}": asdict(v)
                        for k, v in self._capabilities.items()
                    },
                    "task_outcomes": [asdict(o) for o in self._task_outcomes],
                    "known_limitations": self._known_limitations,
                }
            tmp = self._path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            os.replace(tmp, self._path)
        except Exception as exc:
            _logger.debug("[SelfModel] Save failed: %s", exc)

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                return

            with self._lock:
                for key_str, rec_dict in data.get("capabilities", {}).items():
                    parts = key_str.split(":", 1)
                    if len(parts) == 2 and isinstance(rec_dict, dict):
                        dom, op = parts
                        self._capabilities[(dom, op)] = CapabilityRecord(
                            domain=dom, operation=op,
                            successes=int(rec_dict.get("successes", 0)),
                            failures=int(rec_dict.get("failures", 0)),
                            last_ts=float(rec_dict.get("last_ts", time.time())),
                        )

                for o_dict in data.get("task_outcomes", []):
                    if isinstance(o_dict, dict):
                        self._task_outcomes.append(TaskOutcome(
                            objective=str(o_dict.get("objective", ""))[:300],
                            app=str(o_dict.get("app", "unknown")),
                            success=bool(o_dict.get("success", False)),
                            ts=float(o_dict.get("ts", time.time())),
                            duration=float(o_dict.get("duration", 0.0)),
                        ))

                raw_lims = data.get("known_limitations", [])
                if isinstance(raw_lims, list):
                    self._known_limitations = [str(l)[:300] for l in raw_lims[:50]]

            _logger.info(
                "[SelfModel] Loaded %d capability records, %d task outcomes from %s",
                len(self._capabilities), len(self._task_outcomes), self._path,
            )
        except Exception as exc:
            _logger.warning("[SelfModel] Load failed: %s", exc)
