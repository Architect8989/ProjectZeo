"""
core/memory/memory_reconciler.py — Multi-Store Memory Reconciliation Protocol
==============================================================================
Blueprint §10 — Addressing the coherence problem in the 13-store memory stack.

PROBLEM
-------
ProjectZeo uses 13 memory subsystems:
  HippoRAG, Cognee+Qdrant, Mem0, A-MEM, OpenMemory, Graphiti, FAISS,
  SemanticMemory, EpisodicSynthesizer, KnowledgeVault, PlaybookStore,
  ApplicationMemory, MemoryManager

When multiple stores are queried for the same fact, they can return:
  - Contradictory answers (HippoRAG says X installed, Mem0 says X not installed)
  - Stale answers (SemanticMemory has outdated version number)
  - Duplicate answers (same fact from 3 stores, cluttering context)
  - Confidence-conflicting answers (same fact with different confidence scores)

Without reconciliation, the LLM receives a confused context window that
degrades decision quality and causes inconsistent behaviour.

SOLUTION: The Reconciliation Protocol
--------------------------------------
1. AGGREGATION: Gather candidate facts from all available stores.
2. GROUPING: Cluster candidates by semantic topic (subject + predicate hash).
3. CONFLICT DETECTION: Flag clusters where candidates disagree on the value.
4. RESOLUTION: Apply a weighted resolution strategy per cluster.
5. DEDUPLICATION: Return one canonical fact per cluster.

Resolution strategies (applied in order of cluster recency):
  - RECENCY_WINS: Most recently updated fact wins (safe default).
  - CONFIDENCE_WINS: Highest-confidence fact wins.
  - SOURCE_PRIORITY: Prefer stores in priority order.
  - LLM_ARBITRATION: When recency/confidence differ by <threshold, ask LLM.

SOURCE PRIORITY ORDER (trust ranking):
  1. Cognee + Qdrant (structured, LLM-extracted)
  2. Graphiti (bi-temporal: knows WHEN facts were true)
  3. HippoRAG (multi-hop graph retrieval)
  4. Mem0 (personal context)
  5. SemanticMemory (long-term structured)
  6. A-MEM (Zettelkasten associative)
  7. EpisodicSynthesizer (recent experience)
  8. OpenMemory (session-scoped)
  9. KnowledgeVault / PlaybookStore (static reference)
 10. ApplicationMemory (app-specific ephemeral)

INTEGRATION
-----------
  MemoryManager.query() → MemoryReconciler.reconcile(raw_results) → clean_facts
  GIIController._memory_manager → uses reconciler automatically
  PerStepReasoner → receives reconciled context string

REFERENCES
----------
  - Packer et al. "MemGPT" (2023) — tiered memory architecture
  - Park et al. "Generative Agents" (2023) — reflection-based memory synthesis
  - Blueprint §10 — ProjectZeo Long-Term Memory Architecture
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Minimum confidence gap to prefer one fact over another without LLM arbitration
_CONFIDENCE_GAP_THRESHOLD = float(
    os.environ.get("PROJECTZEO_RECONCILE_CONFIDENCE_GAP", "0.2")
)

# Age in seconds after which a fact is considered stale (override by recency)
_STALE_AGE_SECONDS = float(
    os.environ.get("PROJECTZEO_RECONCILE_STALE_AGE", "86400")  # 24h
)

# Max conflicts to resolve via LLM per query (prevent runaway API calls)
_MAX_LLM_ARBITRATIONS = int(
    os.environ.get("PROJECTZEO_RECONCILE_MAX_LLM_ARBITRATIONS", "3")
)

# Source trust ranking (lower index = higher trust)
_SOURCE_PRIORITY: List[str] = [
    "cognee", "graphiti", "hipporag", "mem0", "semantic",
    "amem", "episodic", "openmemory", "vault", "playbook",
    "application", "working_memory", "manual",
]

# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MemoryClaim:
    """
    A single factual claim from one memory store.
    Normalised representation of whatever the store returned.
    """
    source: str                        # Store name (e.g. "hipporag", "mem0")
    key: str                           # Subject:predicate or topic key
    value: str                         # The claimed value / content
    confidence: float = 0.5           # 0.0 – 1.0
    timestamp: float = field(default_factory=time.time)
    raw: Any = field(default=None, repr=False)  # Original dict from store

    def age_seconds(self) -> float:
        return time.time() - self.timestamp

    def is_stale(self) -> bool:
        return self.age_seconds() > _STALE_AGE_SECONDS

    def source_rank(self) -> int:
        """Lower = more trusted."""
        try:
            return _SOURCE_PRIORITY.index(self.source.lower())
        except ValueError:
            return len(_SOURCE_PRIORITY)


@dataclass
class ReconciliationResult:
    """
    Output of reconciling one cluster of conflicting claims.
    """
    winner: MemoryClaim                # The selected canonical fact
    conflicts: List[MemoryClaim]       # Other claims that were overridden
    resolution_method: str             # "recency" | "confidence" | "source_priority" | "llm" | "unanimous"
    conflict_detected: bool = False
    arbitration_reason: str = ""


@dataclass
class ReconciledMemory:
    """
    Output of a full reconciliation pass across all stores.
    """
    facts: List[MemoryClaim]                       # Deduplicated, reconciled facts
    conflicts_resolved: int = 0
    llm_arbitrations_used: int = 0
    sources_queried: List[str] = field(default_factory=list)
    query_time_ms: float = 0.0
    raw_claim_count: int = 0

    def to_context_string(self, max_chars: int = 3000) -> str:
        """Format for LLM context injection."""
        if not self.facts:
            return ""
        lines = ["[Memory Context — Reconciled]"]
        used = len(lines[0])
        for fact in self.facts:
            staleness = " [STALE]" if fact.is_stale() else ""
            confidence_tag = f" (conf={fact.confidence:.1f})" if fact.confidence < 0.7 else ""
            line = f"  [{fact.source}] {fact.key}: {fact.value[:200]}{confidence_tag}{staleness}"
            if used + len(line) > max_chars:
                break
            lines.append(line)
            used += len(line)

        if self.conflicts_resolved:
            lines.append(
                f"  [reconciler: {self.conflicts_resolved} conflict(s) resolved, "
                f"{self.llm_arbitrations_used} via LLM]"
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MemoryReconciler
# ─────────────────────────────────────────────────────────────────────────────

class MemoryReconciler:
    """
    Cross-store memory reconciliation engine.

    Usage:
        reconciler = MemoryReconciler(llm_caller=my_llm)
        claims = [
            MemoryClaim("hipporag", "chrome:version", "120.0", 0.8, ...),
            MemoryClaim("mem0",     "chrome:version", "119.0", 0.6, ...),
        ]
        result = reconciler.reconcile(claims)
        context = result.to_context_string()
    """

    _ARBITRATION_PROMPT = """\
You are a memory arbitration engine. Two sources disagree on a fact.
Select the more likely correct claim based on recency, confidence, and plausibility.

CLAIM A (source={source_a}, confidence={conf_a:.1f}, age={age_a:.0f}s):
  Key: {key}
  Value: {value_a}

CLAIM B (source={source_b}, confidence={conf_b:.1f}, age={age_b:.0f}s):
  Key: {key}
  Value: {value_b}

Context (what the agent is currently doing): {context}

Respond with ONLY a JSON object:
{{"winner": "A" or "B", "reason": "<one sentence>"}}
"""

    def __init__(
        self,
        llm_caller: Optional[Callable] = None,
        *,
        enable_llm_arbitration: bool = True,
        conflict_log_path: Optional[str] = None,
    ) -> None:
        self._llm = llm_caller
        self._enable_llm = enable_llm_arbitration and (llm_caller is not None)
        self._log_path = conflict_log_path

        # Conflict history for debugging
        self._conflict_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._total_conflicts = 0
        self._total_arbitrations = 0

        _logger.info(
            "[MemoryReconciler] Initialised. llm_arbitration=%s",
            self._enable_llm,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def reconcile(
        self,
        claims: List[MemoryClaim],
        *,
        context: str = "",
        max_results: int = 50,
    ) -> ReconciledMemory:
        """
        Reconcile a list of raw memory claims from multiple stores.

        Steps:
          1. Group claims by normalised key
          2. Within each group, detect conflicts (different values)
          3. Resolve conflicts using priority rules
          4. Return deduplicated, ordered facts

        Args:
            claims: Raw claims from all stores (mixed sources)
            context: Current task context (for LLM arbitration prompt)
            max_results: Maximum facts to return

        Returns:
            ReconciledMemory with deduplicated, conflict-resolved facts
        """
        t0 = time.time()
        raw_count = len(claims)
        llm_arb_used = 0
        conflicts_resolved = 0
        sources_seen: List[str] = []

        if not claims:
            return ReconciledMemory(facts=[], query_time_ms=0.0)

        # Step 1: Group by normalised key
        groups: Dict[str, List[MemoryClaim]] = {}
        for claim in claims:
            norm_key = self._normalise_key(claim.key)
            groups.setdefault(norm_key, []).append(claim)
            if claim.source not in sources_seen:
                sources_seen.append(claim.source)

        # Step 2 & 3: Resolve each group
        resolved_facts: List[MemoryClaim] = []
        for norm_key, cluster in groups.items():
            if len(cluster) == 1:
                resolved_facts.append(cluster[0])
                continue

            result = self._resolve_cluster(cluster, context=context, arb_budget=_MAX_LLM_ARBITRATIONS - llm_arb_used)
            resolved_facts.append(result.winner)

            if result.conflict_detected:
                conflicts_resolved += 1
                self._total_conflicts += 1
                with self._lock:
                    self._conflict_history.append({
                        "ts": time.time(),
                        "key": norm_key,
                        "winner_source": result.winner.source,
                        "loser_sources": [c.source for c in result.conflicts],
                        "method": result.resolution_method,
                    })

            if result.resolution_method == "llm":
                llm_arb_used += 1
                self._total_arbitrations += 1

        # Step 4: Sort by confidence desc, then source rank asc
        resolved_facts.sort(
            key=lambda f: (-f.confidence, f.source_rank())
        )

        return ReconciledMemory(
            facts=resolved_facts[:max_results],
            conflicts_resolved=conflicts_resolved,
            llm_arbitrations_used=llm_arb_used,
            sources_queried=sources_seen,
            query_time_ms=(time.time() - t0) * 1000,
            raw_claim_count=raw_count,
        )

    def ingest_store_results(
        self,
        store_name: str,
        raw_results: List[Any],
        *,
        default_confidence: float = 0.6,
        key_field: str = "key",
        value_field: str = "content",
        confidence_field: str = "confidence",
        timestamp_field: str = "timestamp",
    ) -> List[MemoryClaim]:
        """
        Convert raw results from a specific store into MemoryClaim objects.

        Handles the heterogeneous output formats of different stores:
          - HippoRAG returns: {"content": "...", "score": 0.8, "node_id": "..."}
          - Mem0 returns: {"memory": "...", "id": "...", "score": 0.7}
          - SemanticMemory returns: SemanticFact objects with .subject, .predicate, .object
          - OpenMemory returns: {"key": "...", "content": "..."}
        """
        claims: List[MemoryClaim] = []
        now = time.time()

        for item in raw_results:
            try:
                claim = self._item_to_claim(
                    store_name, item, now,
                    default_confidence=default_confidence,
                    key_field=key_field,
                    value_field=value_field,
                    confidence_field=confidence_field,
                    timestamp_field=timestamp_field,
                )
                if claim is not None:
                    claims.append(claim)
            except Exception as exc:
                _logger.debug(
                    "[MemoryReconciler] Failed to convert %s item: %s", store_name, exc
                )

        return claims

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_conflicts_detected": self._total_conflicts,
            "total_llm_arbitrations": self._total_arbitrations,
            "conflict_history_size": len(self._conflict_history),
            "llm_arbitration_enabled": self._enable_llm,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Cluster resolution
    # ─────────────────────────────────────────────────────────────────────────

    def _resolve_cluster(
        self,
        cluster: List[MemoryClaim],
        *,
        context: str = "",
        arb_budget: int = 3,
    ) -> ReconciliationResult:
        """
        Resolve a cluster of claims for the same key.
        Returns the winner and resolution metadata.
        """
        if not cluster:
            raise ValueError("Empty cluster")

        # Check if all claims agree (no conflict)
        values = set(self._normalise_value(c.value) for c in cluster)
        if len(values) == 1:
            # All sources agree — return highest-confidence claim
            winner = max(cluster, key=lambda c: c.confidence)
            return ReconciliationResult(
                winner=winner,
                conflicts=[c for c in cluster if c is not winner],
                resolution_method="unanimous",
                conflict_detected=False,
            )

        # Conflict detected — resolve
        _logger.debug(
            "[MemoryReconciler] Conflict in key=%r: %d distinct values from %s",
            cluster[0].key, len(values), [c.source for c in cluster],
        )

        # Strategy 1: Recency — most recently updated wins if not stale
        fresh_claims = [c for c in cluster if not c.is_stale()]
        if fresh_claims:
            most_recent = min(fresh_claims, key=lambda c: c.age_seconds())
            # If the most recent is significantly newer and confident enough
            others_max_conf = max(
                (c.confidence for c in cluster if c is not most_recent), default=0.0
            )
            if (
                most_recent.confidence >= 0.5
                and most_recent.age_seconds() < _STALE_AGE_SECONDS / 2
            ):
                return ReconciliationResult(
                    winner=most_recent,
                    conflicts=[c for c in cluster if c is not most_recent],
                    resolution_method="recency",
                    conflict_detected=True,
                    arbitration_reason=f"Most recent ({most_recent.age_seconds():.0f}s old) from {most_recent.source}",
                )

        # Strategy 2: Confidence gap — high-confidence source wins decisively
        most_confident = max(cluster, key=lambda c: c.confidence)
        second_confident = sorted(cluster, key=lambda c: c.confidence, reverse=True)
        if len(second_confident) > 1:
            gap = most_confident.confidence - second_confident[1].confidence
            if gap >= _CONFIDENCE_GAP_THRESHOLD:
                return ReconciliationResult(
                    winner=most_confident,
                    conflicts=[c for c in cluster if c is not most_confident],
                    resolution_method="confidence",
                    conflict_detected=True,
                    arbitration_reason=f"Confidence gap {gap:.2f} from {most_confident.source}",
                )

        # Strategy 3: Source priority ranking
        by_source_rank = sorted(cluster, key=lambda c: c.source_rank())
        highest_priority = by_source_rank[0]

        # If highest-priority source is notably different from lowest, use it
        if highest_priority.source_rank() < by_source_rank[-1].source_rank() - 2:
            return ReconciliationResult(
                winner=highest_priority,
                conflicts=[c for c in cluster if c is not highest_priority],
                resolution_method="source_priority",
                conflict_detected=True,
                arbitration_reason=f"Trusted source {highest_priority.source} (rank {highest_priority.source_rank()})",
            )

        # Strategy 4: LLM arbitration (last resort, budget-limited)
        if self._enable_llm and arb_budget > 0 and len(cluster) == 2:
            winner = self._llm_arbitrate(cluster[0], cluster[1], context=context)
            if winner is not None:
                loser = cluster[1] if winner is cluster[0] else cluster[0]
                return ReconciliationResult(
                    winner=winner,
                    conflicts=[loser],
                    resolution_method="llm",
                    conflict_detected=True,
                    arbitration_reason="LLM arbitration",
                )

        # Fallback: source priority (always produces a result)
        return ReconciliationResult(
            winner=highest_priority,
            conflicts=[c for c in cluster if c is not highest_priority],
            resolution_method="source_priority",
            conflict_detected=True,
            arbitration_reason=f"Fallback source_priority: {highest_priority.source}",
        )

    def _llm_arbitrate(
        self,
        claim_a: MemoryClaim,
        claim_b: MemoryClaim,
        *,
        context: str = "",
    ) -> Optional[MemoryClaim]:
        """Use LLM to arbitrate between two conflicting claims."""
        if self._llm is None:
            return None
        try:
            prompt = self._ARBITRATION_PROMPT.format(
                source_a=claim_a.source,
                conf_a=claim_a.confidence,
                age_a=claim_a.age_seconds(),
                source_b=claim_b.source,
                conf_b=claim_b.confidence,
                age_b=claim_b.age_seconds(),
                key=claim_a.key,
                value_a=claim_a.value[:300],
                value_b=claim_b.value[:300],
                context=context[:200] if context else "general task execution",
            )
            raw = self._llm(
                messages=[{"role": "user", "content": prompt}],
                objective="memory_reconciliation",
                session_id="reconciler",
            )
            if isinstance(raw, list):
                raw = "".join(
                    item.get("content", "") if isinstance(item, dict) else str(item)
                    for item in raw
                )
            raw = str(raw or "").strip()

            import re
            match = re.search(r"\{.*?\}", raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                winner_label = str(parsed.get("winner", "A")).strip().upper()
                if winner_label == "A":
                    return claim_a
                elif winner_label == "B":
                    return claim_b
        except Exception as exc:
            _logger.debug("[MemoryReconciler] LLM arbitration failed: %s", exc)
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Normalisation helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_key(key: str) -> str:
        """Normalise a fact key for grouping (lowercase, strip whitespace)."""
        return key.lower().strip().replace("  ", " ")

    @staticmethod
    def _normalise_value(value: str) -> str:
        """Normalise a fact value for equality checking."""
        return " ".join(value.lower().split())[:200]

    def _item_to_claim(
        self,
        store_name: str,
        item: Any,
        now: float,
        *,
        default_confidence: float,
        key_field: str,
        value_field: str,
        confidence_field: str,
        timestamp_field: str,
    ) -> Optional[MemoryClaim]:
        """Convert a raw store item to a MemoryClaim."""
        if item is None:
            return None

        # Handle SemanticFact objects (from SemanticMemory)
        if hasattr(item, "subject") and hasattr(item, "predicate") and hasattr(item, "object"):
            return MemoryClaim(
                source=store_name,
                key=f"{item.subject}:{item.predicate}",
                value=str(getattr(item, "object", "")),
                confidence=float(
                    item.current_confidence() if hasattr(item, "current_confidence") else default_confidence
                ),
                timestamp=float(getattr(item, "last_confirmed_at", now)),
                raw=item,
            )

        # Handle dict items
        if isinstance(item, dict):
            # Try common key patterns
            key = (
                item.get(key_field)
                or item.get("key")
                or item.get("subject", "") + ":" + item.get("predicate", "")
                or item.get("node_id", "")
                or item.get("id", "")
                or "unknown"
            )
            value = (
                item.get(value_field)
                or item.get("content")
                or item.get("memory")
                or item.get("text")
                or item.get("object")
                or str(item)[:300]
            )
            confidence = float(
                item.get(confidence_field)
                or item.get("confidence")
                or item.get("score")
                or default_confidence
            )
            ts = float(item.get(timestamp_field) or item.get("timestamp") or now)
            return MemoryClaim(
                source=store_name,
                key=str(key),
                value=str(value),
                confidence=min(1.0, max(0.0, confidence)),
                timestamp=ts,
                raw=item,
            )

        # Handle plain strings
        if isinstance(item, str):
            return MemoryClaim(
                source=store_name,
                key=f"{store_name}:{hashlib.md5(item.encode()).hexdigest()[:8]}",
                value=item[:500],
                confidence=default_confidence,
                timestamp=now,
                raw=item,
            )

        return None


# ─────────────────────────────────────────────────────────────────────────────
# Singleton accessor
# ─────────────────────────────────────────────────────────────────────────────

_instance: Optional[MemoryReconciler] = None
_instance_lock = threading.Lock()


def get_memory_reconciler(
    llm_caller: Optional[Callable] = None,
    *,
    enable_llm_arbitration: bool = True,
) -> MemoryReconciler:
    """Return the global MemoryReconciler instance."""
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = MemoryReconciler(
                llm_caller=llm_caller,
                enable_llm_arbitration=enable_llm_arbitration,
            )
        elif llm_caller is not None and _instance._llm is None:
            # Late-bind LLM caller if not set at creation
            _instance._llm = llm_caller
            _instance._enable_llm = enable_llm_arbitration
    return _instance


def reset_reconciler() -> None:
    """Reset singleton (mainly for testing)."""
    global _instance
    with _instance_lock:
        _instance = None
