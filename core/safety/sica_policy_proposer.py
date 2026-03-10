"""
core/safety/sica_policy_proposer.py — SICA: Self-Improving Consequence Analysis
==================================================================================
Blueprint §13 / §9.2 — Policy Self-Improvement Loop

WHAT THIS IS
------------
SICA watches consequence_reasoner.evaluate() results for UNCERTAIN verdicts and
uses an LLM to propose new policy.yaml rules that would resolve the ambiguity
in future runs. Proposed rules are accumulated in a pending queue and flushed
to policy.yaml (or a pending_rules.yaml sidecar) for human review.

HOW IT WORKS
------------
1. ConsequenceReasoner calls sica.observe(action, objective, verdict, reason)
   after each evaluate() call where verdict == UNCERTAIN.
2. SICA groups observations by action pattern. Once a pattern repeats >= MIN_REPEAT
   times (or immediately for HIGH_RISK ops), it calls the LLM to propose a rule.
3. Proposed rules are written to:
     ~/.projectzeo/sica_pending_rules.yaml   (awaiting human review)
   and optionally auto-applied to policy.yaml if
     PROJECTZEO_SICA_AUTO_APPLY=1 (DISABLED by default — safety critical)
4. SICAStats tracks how many patterns observed, rules proposed, rules applied.

INTEGRATION
-----------
ConsequenceReasoner._evaluate_with_sica() (added to consequence_reasoner.py)
calls this module. GIIController logs SICA stats on task completion.

REFERENCES
----------
Blueprint §13.3 "Policy Hot-Reload and Self-Improvement"
Blueprint §9.2 Algorithm conflict table (SICA compatible with all algorithms)
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

_SICA_ENABLED      = os.environ.get("PROJECTZEO_SICA_ENABLED", "1").strip() != "0"
_AUTO_APPLY        = os.environ.get("PROJECTZEO_SICA_AUTO_APPLY", "0").strip() == "1"
_MIN_REPEAT        = int(os.environ.get("PROJECTZEO_SICA_MIN_REPEAT", "2"))
_MAX_PENDING       = int(os.environ.get("PROJECTZEO_SICA_MAX_PENDING", "20"))

_PENDING_FILE      = os.path.expanduser(
    os.environ.get("PROJECTZEO_SICA_PENDING_FILE", "~/.projectzeo/sica_pending_rules.yaml")
)
_POLICY_FILE       = os.environ.get(
    "PROJECTZEO_POLICY_FILE",
    str(Path(__file__).resolve().parents[2] / "policy" / "policy.yaml"),
)

# Operations considered HIGH_RISK — propose rule immediately on first UNCERTAIN
_HIGH_RISK_OPS = frozenset({
    "command", "file_delete", "file_move", "install",
    "rm", "rmdir", "format", "wipe", "send", "post", "upload",
})

_PROPOSE_SYSTEM = """\
You are a policy rule generator for a safe autonomous desktop agent.
Given an action that caused an UNCERTAIN safety consequence verdict, propose
a precise policy.yaml rule that would resolve the ambiguity.

Policy rule format (YAML block):
  - operation: <operation_name>
    pattern: <regex or substring to match in command/content/path>
    rule: allow | deny | require_human_confirmation
    reason: <one sentence justification>
    confidence: 0.0-1.0

Rules:
  - prefer deny for irreversible file/system operations with unclear intent
  - prefer require_human_confirmation for network/send operations
  - prefer allow for well-understood, reversible UI interactions
  - set confidence < 0.7 if you are unsure — the human reviewer can adjust

Respond ONLY with the YAML block (no prose, no markdown fences).
"""

# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UncertainObservation:
    operation:   str
    action_hash: str
    objective:   str
    reason:      str
    count:       int = 1
    first_seen:  float = field(default_factory=time.time)
    last_seen:   float = field(default_factory=time.time)
    proposed:    bool = False


@dataclass
class ProposedRule:
    operation:   str
    pattern:     str
    rule:        str          # allow | deny | require_human_confirmation
    reason:      str
    confidence:  float
    proposed_at: float = field(default_factory=time.time)
    source_hash: str = ""
    applied:     bool = False


@dataclass
class SICAStats:
    total_observations:   int = 0
    patterns_accumulated: int = 0
    rules_proposed:       int = 0
    rules_applied:        int = 0
    llm_calls:            int = 0
    llm_failures:         int = 0


# ─────────────────────────────────────────────────────────────────────────────
# SICA Engine
# ─────────────────────────────────────────────────────────────────────────────

class SICAPolicyProposer:
    """
    Self-Improving Consequence Analysis — policy rule proposer.

    Observes UNCERTAIN consequence verdicts and uses an LLM to propose
    policy.yaml rules that would remove the ambiguity in future runs.
    """

    def __init__(
        self,
        llm_caller: Optional[Callable] = None,
        *,
        min_repeat: int = _MIN_REPEAT,
        max_pending: int = _MAX_PENDING,
        pending_file: str = _PENDING_FILE,
    ) -> None:
        self._llm        = llm_caller
        self._min_repeat = max(1, min_repeat)
        self._max_pending = max_pending
        self._pending_file = pending_file
        self._lock       = threading.Lock()

        self._observations: Dict[str, UncertainObservation] = {}
        self._proposed_rules: List[ProposedRule] = []
        self._stats = SICAStats()

        if _SICA_ENABLED:
            _logger.info(
                "[SICA] SICAPolicyProposer active. min_repeat=%d max_pending=%d "
                "auto_apply=%s pending_file=%s",
                self._min_repeat, self._max_pending, _AUTO_APPLY, self._pending_file,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def observe(
        self,
        action: Dict[str, Any],
        objective: str,
        verdict: str,           # "UNCERTAIN" | "COHERENT" | "INCOHERENT"
        reason: str = "",
    ) -> None:
        """
        Record a consequence verdict observation.

        Only UNCERTAIN verdicts accumulate toward rule proposals.
        COHERENT/INCOHERENT verdicts are recorded for stats only.
        """
        if not _SICA_ENABLED:
            return

        with self._lock:
            self._stats.total_observations += 1

        if str(verdict).upper() != "UNCERTAIN":
            return

        op      = str(action.get("operation", "unknown"))
        content = str(action.get("command") or action.get("content") or
                      action.get("path")    or action.get("text") or "")
        key     = self._action_key(op, content)

        with self._lock:
            if key in self._observations:
                obs = self._observations[key]
                obs.count   += 1
                obs.last_seen = time.time()
            else:
                obs = UncertainObservation(
                    operation   = op,
                    action_hash = key,
                    objective   = objective[:200],
                    reason      = reason[:200],
                )
                self._observations[key] = obs
                self._stats.patterns_accumulated += 1
                _logger.debug("[SICA] New UNCERTAIN pattern: op=%s key=%s", op, key[:8])

            # Propose rule if threshold reached or high-risk op
            should_propose = (
                not obs.proposed
                and (
                    obs.count >= self._min_repeat
                    or op in _HIGH_RISK_OPS
                )
            )

        if should_propose:
            # Propose asynchronously to avoid blocking consequence_reasoner
            threading.Thread(
                target=self._propose_rule,
                args=(action, objective, reason, obs),
                daemon=True,
                name=f"sica-propose-{key[:6]}",
            ).start()

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_observations":   self._stats.total_observations,
                "patterns_accumulated": self._stats.patterns_accumulated,
                "rules_proposed":       self._stats.rules_proposed,
                "rules_applied":        self._stats.rules_applied,
                "llm_calls":            self._stats.llm_calls,
                "llm_failures":         self._stats.llm_failures,
                "pending_rules":        len([r for r in self._proposed_rules if not r.applied]),
            }

    def flush_pending_to_file(self) -> int:
        """Write all un-applied proposed rules to the pending file for human review."""
        with self._lock:
            pending = [r for r in self._proposed_rules if not r.applied]
        if not pending:
            return 0

        try:
            os.makedirs(os.path.dirname(os.path.abspath(self._pending_file)), exist_ok=True)
            lines = [
                f"# SICA Pending Rules — generated {time.strftime('%Y-%m-%dT%H:%M:%S')}\n",
                "# Review and move desired entries to policy/policy.yaml\n\n",
                "pending_rules:\n",
            ]
            for r in pending:
                lines.append(f"  - operation: {r.operation}\n")
                lines.append(f"    pattern: {json.dumps(r.pattern)}\n")
                lines.append(f"    rule: {r.rule}\n")
                lines.append(f"    reason: {json.dumps(r.reason)}\n")
                lines.append(f"    confidence: {r.confidence:.2f}\n")
                lines.append(f"    proposed_at: {time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(r.proposed_at))}\n")
                lines.append("\n")

            with open(self._pending_file, "w", encoding="utf-8") as f:
                f.writelines(lines)

            _logger.info("[SICA] Wrote %d pending rules to %s", len(pending), self._pending_file)
            return len(pending)
        except Exception as exc:
            _logger.warning("[SICA] Failed to write pending rules: %s", exc)
            return 0

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _propose_rule(
        self,
        action: Dict[str, Any],
        objective: str,
        reason: str,
        obs: UncertainObservation,
    ) -> None:
        """LLM call to generate a policy rule for this UNCERTAIN pattern."""
        with self._lock:
            if obs.proposed:
                return  # Another thread already proposed for this pattern
            obs.proposed = True
            self._stats.llm_calls += 1

        if self._llm is None:
            _logger.debug("[SICA] No LLM available — rule proposal skipped for op=%s", obs.operation)
            with self._lock:
                self._stats.llm_failures += 1
            return

        try:
            action_json = json.dumps(
                {k: v for k, v in action.items() if k != "thought"},
                indent=2,
            )[:500]
            prompt = (
                f"ACTION (triggered UNCERTAIN consequence verdict):\n{action_json}\n\n"
                f"OBJECTIVE: {objective[:200]}\n\n"
                f"UNCERTAINTY REASON: {reason[:200]}\n\n"
                f"OBSERVATION COUNT: {obs.count} (pattern repeated {obs.count}x)\n\n"
                "Propose a policy.yaml rule to resolve this ambiguity:"
            )

            messages = [
                {"role": "system", "content": _PROPOSE_SYSTEM},
                {"role": "user",   "content": prompt},
            ]
            raw = self._llm(
                messages=messages,
                objective="policy_rule_proposal",
                session_id="sica_proposer",
            )

            raw_text = ""
            if isinstance(raw, list) and raw:
                raw_text = str(raw[0].get("content", "") if isinstance(raw[0], dict) else raw[0])
            elif isinstance(raw, str):
                raw_text = raw

            if not raw_text:
                with self._lock:
                    self._stats.llm_failures += 1
                return

            rule = self._parse_rule(raw_text, obs)
            if rule is None:
                with self._lock:
                    self._stats.llm_failures += 1
                return

            with self._lock:
                if len(self._proposed_rules) < self._max_pending:
                    self._proposed_rules.append(rule)
                    self._stats.rules_proposed += 1
                    _logger.info(
                        "[SICA] Rule proposed: op=%s rule=%s confidence=%.2f pattern=%r",
                        rule.operation, rule.rule, rule.confidence, rule.pattern[:60],
                    )

            if _AUTO_APPLY and rule.confidence >= 0.8:
                self._auto_apply_rule(rule)
            else:
                # Flush to file for human review
                self.flush_pending_to_file()

        except Exception as exc:
            _logger.warning("[SICA] Rule proposal failed: %s", exc)
            with self._lock:
                self._stats.llm_failures += 1

    def _parse_rule(
        self,
        raw: str,
        obs: UncertainObservation,
    ) -> Optional[ProposedRule]:
        """Parse LLM output into a ProposedRule. Falls back to defaults if parse fails."""
        try:
            # Strip markdown fences
            clean = re.sub(r"```(?:yaml|json)?", "", raw).strip()

            # Extract fields with simple regex
            op_match     = re.search(r"operation:\s*(.+)", clean)
            pat_match    = re.search(r"pattern:\s*(.+)", clean)
            rule_match   = re.search(r"\brule:\s*(allow|deny|require_human_confirmation)", clean, re.I)
            reason_match = re.search(r"reason:\s*(.+)", clean)
            conf_match   = re.search(r"confidence:\s*([0-9.]+)", clean)

            operation = op_match.group(1).strip().strip('"\'') if op_match else obs.operation
            pattern   = pat_match.group(1).strip().strip('"\'') if pat_match else ""
            rule_val  = rule_match.group(1).lower().strip() if rule_match else "require_human_confirmation"
            reason    = reason_match.group(1).strip().strip('"\'') if reason_match else obs.reason[:100]
            confidence = float(conf_match.group(1)) if conf_match else 0.5

            # Validate rule value
            if rule_val not in ("allow", "deny", "require_human_confirmation"):
                rule_val = "require_human_confirmation"

            return ProposedRule(
                operation   = operation[:40],
                pattern     = pattern[:200],
                rule        = rule_val,
                reason      = reason[:300],
                confidence  = max(0.0, min(1.0, confidence)),
                source_hash = obs.action_hash,
            )
        except Exception as exc:
            _logger.debug("[SICA] Rule parse failed: %s — using default", exc)
            # Return a conservative default rule
            return ProposedRule(
                operation   = obs.operation,
                pattern     = "",
                rule        = "require_human_confirmation",
                reason      = f"UNCERTAIN consequence pattern repeated {obs.count}x: {obs.reason[:100]}",
                confidence  = 0.4,
                source_hash = obs.action_hash,
            )

    def _auto_apply_rule(self, rule: ProposedRule) -> None:
        """
        AUTO_APPLY (DISABLED by default, PROJECTZEO_SICA_AUTO_APPLY=1 required).

        Appends the rule to policy.yaml under a `sica_auto_rules` section.
        This section is loaded by policy/engine.py when POLICY_HOT_RELOAD=1.

        WARNING: Only high-confidence rules (>= 0.8) are auto-applied.
        All auto-applied rules are still logged to the pending file for audit.
        """
        try:
            policy_path = _POLICY_FILE
            if not os.path.exists(policy_path):
                _logger.warning("[SICA] Policy file not found: %s — cannot auto-apply", policy_path)
                return

            append_block = (
                f"\n# SICA AUTO-APPLIED {time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
                f"# confidence={rule.confidence:.2f} source_hash={rule.source_hash[:8]}\n"
                f"sica_auto_rules:\n"
                f"  - operation: {rule.operation}\n"
                f"    pattern: {json.dumps(rule.pattern)}\n"
                f"    rule: {rule.rule}\n"
                f"    reason: {json.dumps(rule.reason)}\n"
            )

            with open(policy_path, "a", encoding="utf-8") as f:
                f.write(append_block)

            with self._lock:
                rule.applied = True
                self._stats.rules_applied += 1

            _logger.info(
                "[SICA] AUTO-APPLIED rule to %s: op=%s rule=%s (confidence=%.2f)",
                policy_path, rule.operation, rule.rule, rule.confidence,
            )
        except Exception as exc:
            _logger.warning("[SICA] Auto-apply failed: %s", exc)

    @staticmethod
    def _action_key(op: str, content: str) -> str:
        import hashlib
        raw = f"{op}:{content[:80]}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_instance: Optional[SICAPolicyProposer] = None
_instance_lock = threading.Lock()


def get_sica_proposer(
    llm_caller: Optional[Callable] = None,
) -> SICAPolicyProposer:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = SICAPolicyProposer(llm_caller=llm_caller)
    elif llm_caller is not None and _instance._llm is None:
        _instance._llm = llm_caller
    return _instance
