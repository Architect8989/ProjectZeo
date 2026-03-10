"""
core/memory/memory_manager.py — MemGPT-style tiered memory manager.

GII-FIX: This file was absent from the codebase (Blueprint Phase 2 gap).
Implements the unified memory tier manager described in Blueprint §10.6.

Architecture:
  ┌─────────────────────────────────────────────────────┐
  │  Tier 0: Working Memory (in-context, ~4k tokens)    │  ← current task context
  │  Tier 1: Episodic Memory (SQLite, cross-session)    │  ← recent experiences
  │  Tier 2: Semantic Memory (JSON, ACT-R activation)   │  ← facts and skills
  │  Tier 3: Long-Term Archive (compressed, cold)       │  ← old episodes
  └─────────────────────────────────────────────────────┘

Responsibilities:
  - Working memory budget enforcement (prevent context overflow)
  - Automatic promotion of hot episodic items → semantic facts
  - Automatic demotion of cold semantic facts → archive
  - Query routing: routes queries to the appropriate tier(s)
  - Eviction: LRU + importance-weighted eviction when tiers overflow

Blueprint reference: §10 (MemGPT), §10.6 (multi-tier hierarchy)
Paper: Packer et al. "MemGPT: Towards LLMs as Operating Systems" (2023)
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

# GII-FIX: Memory Reconciler — deduplicates and resolves conflicts across the
# 13-store memory stack before results are returned to the LLM context window.
try:
    from core.memory.memory_reconciler import (
        MemoryReconciler, MemoryClaim, get_memory_reconciler,
    )
    _RECONCILER_AVAILABLE = True
except ImportError:
    _RECONCILER_AVAILABLE = False
    _logger.debug("[MemoryManager] MemoryReconciler not available (will skip reconciliation)")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_MEMORY_DIR = os.path.join(
    os.path.expanduser("~"), ".projectzeo", "memory_manager"
)

# Working memory token budget (approximate chars: 4 tokens ≈ 1 token limit)
_WORKING_MEMORY_CHAR_BUDGET = int(
    os.environ.get("PROJECTZEO_WM_CHAR_BUDGET", "4000")
)

# How many episodic items to keep hot in working memory
_WORKING_MEMORY_EPISODIC_SLOTS = int(
    os.environ.get("PROJECTZEO_WM_EPISODIC_SLOTS", "5")
)

# Promotion threshold: episodic item confirmed N times → promote to semantic
_PROMOTION_THRESHOLD = int(os.environ.get("PROJECTZEO_MEM_PROMOTE_THRESHOLD", "3"))

# Demotion threshold: semantic fact not accessed for N hours → demote to archive
_DEMOTION_HOURS = float(os.environ.get("PROJECTZEO_MEM_DEMOTE_HOURS", "72.0"))

# Max semantic facts before LRU eviction to archive
_MAX_SEMANTIC_FACTS = int(os.environ.get("PROJECTZEO_MEM_MAX_SEMANTIC", "5000"))


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WorkingMemorySlot:
    """A single slot in the working memory context window."""
    key: str
    content: str           # Formatted text for LLM context
    importance: float      # 0.0–1.0 (higher = keep longer)
    inserted_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    tier_source: str = "episodic"   # "episodic" | "semantic" | "manual"

    def age_seconds(self) -> float:
        return time.time() - self.inserted_at

    def staleness(self) -> float:
        """Higher staleness = more likely to be evicted."""
        age_hours = self.age_seconds() / 3600.0
        return age_hours / max(self.importance, 0.01)


@dataclass
class MemoryQueryResult:
    """Unified result from a cross-tier memory query."""
    working_memory: List[str]          # Working memory hits
    episodic_hits: List[Dict]          # Episodic memory hits
    semantic_hits: List[Dict]          # Semantic memory hits
    total_chars: int
    query_time_ms: float


# ─────────────────────────────────────────────────────────────────────────────
# MemoryManager
# ─────────────────────────────────────────────────────────────────────────────

class MemoryManager:
    """
    Unified tiered memory manager for ProjectZeo GII.

    Manages four memory tiers with automatic promotion/demotion,
    working memory budget enforcement, and unified query routing.

    Usage:
        mm = MemoryManager(memory_dir=~/.projectzeo/memory_manager)
        mm.insert("chrome installed", importance=0.8)
        result = mm.query("chrome", objective="open chrome browser")
        context_str = mm.get_context_string(max_chars=2000)
    """

    def __init__(
        self,
        memory_dir: Optional[str] = None,
        *,
        episodic_synthesizer=None,
        semantic_memory=None,
    ) -> None:
        self._dir = memory_dir or _DEFAULT_MEMORY_DIR
        os.makedirs(self._dir, exist_ok=True)

        # External memory stores (injected, optional)
        self._episodic_synth = episodic_synthesizer
        self._semantic_mem   = semantic_memory

        # Tier 0: Working memory (in-process dict)
        self._working_memory: Dict[str, WorkingMemorySlot] = {}
        self._wm_lock = threading.Lock()

        # Tier 3: Archive (SQLite)
        self._archive_db_path = os.path.join(self._dir, "archive.db")
        self._archive_conn: Optional[sqlite3.Connection] = None
        self._archive_lock = threading.Lock()
        self._init_archive_db()

        # GII-FIX: Reconciler for cross-store deduplication/conflict resolution
        self._reconciler: Optional[Any] = None
        if _RECONCILER_AVAILABLE:
            try:
                self._reconciler = MemoryReconciler(llm_caller=None)
                _logger.debug("[MemoryManager] MemoryReconciler active.")
            except Exception as _rec_exc:
                _logger.debug("[MemoryManager] MemoryReconciler init failed: %s", _rec_exc)

        # Stats
        self._query_count = 0
        self._promotion_count = 0
        self._eviction_count = 0
        self._demotion_count = 0

        _logger.info(
            "[MemoryManager] Initialised. dir=%r wm_budget=%d slots=%d",
            self._dir, _WORKING_MEMORY_CHAR_BUDGET, _WORKING_MEMORY_EPISODIC_SLOTS,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Tier 0: Working Memory
    # ──────────────────────────────────────────────────────────────────────────

    def insert(
        self,
        key: str,
        content: str,
        *,
        importance: float = 0.5,
        tier_source: str = "manual",
    ) -> None:
        """
        Insert or update a slot in working memory.
        Evicts least-important slots if budget is exceeded.
        """
        if not key or not content:
            return

        with self._wm_lock:
            self._working_memory[key] = WorkingMemorySlot(
                key=key,
                content=str(content)[:1000],
                importance=max(0.0, min(1.0, importance)),
                tier_source=tier_source,
            )
            self._enforce_wm_budget()

    def _enforce_wm_budget(self) -> None:
        """Evict slots until total chars ≤ budget. Must be called under _wm_lock."""
        total = sum(len(s.content) for s in self._working_memory.values())
        if total <= _WORKING_MEMORY_CHAR_BUDGET:
            return

        # Sort by staleness descending (stale + low importance evicted first)
        slots = sorted(self._working_memory.values(), key=lambda s: s.staleness(), reverse=True)
        for slot in slots:
            if total <= _WORKING_MEMORY_CHAR_BUDGET * 0.8:
                break
            # Demote to archive before evicting
            self._archive_slot(slot)
            del self._working_memory[slot.key]
            total -= len(slot.content)
            self._eviction_count += 1

    def _archive_slot(self, slot: WorkingMemorySlot) -> None:
        """Write a working memory slot to the archive tier."""
        try:
            with self._archive_lock:
                conn = self._get_archive_conn()
                conn.execute(
                    "INSERT OR REPLACE INTO archive "
                    "(key, content, importance, inserted_at, tier_source, archived_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        slot.key, slot.content, slot.importance,
                        slot.inserted_at, slot.tier_source, time.time(),
                    )
                )
                conn.commit()
        except Exception as exc:
            _logger.debug("[MemoryManager] Archive write failed: %s", exc)

    def get_working_memory(self, max_chars: int = _WORKING_MEMORY_CHAR_BUDGET) -> List[WorkingMemorySlot]:
        """Return working memory slots sorted by importance descending."""
        with self._wm_lock:
            slots = sorted(
                self._working_memory.values(),
                key=lambda s: s.importance,
                reverse=True,
            )
            result = []
            used = 0
            for slot in slots:
                if used + len(slot.content) > max_chars:
                    break
                result.append(slot)
                used += len(slot.content)
            # Update last_accessed
            for slot in result:
                slot.last_accessed = time.time()
            return result

    def get_context_string(
        self,
        objective: str = "",
        max_chars: int = _WORKING_MEMORY_CHAR_BUDGET,
    ) -> str:
        """
        Build a compact context string from working memory for LLM injection.
        Used by PerStepReasoner and GIIController as MEMORY CONTEXT section.
        """
        slots = self.get_working_memory(max_chars=max_chars)
        if not slots:
            return ""
        lines = ["[Working Memory]"]
        remaining = max_chars - len("[Working Memory]\n")
        for slot in slots:
            line = f"  [{slot.tier_source.upper()}] {slot.key}: {slot.content[:200]}"
            if remaining - len(line) < 0:
                break
            lines.append(line)
            remaining -= len(line)
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────────
    # Cross-tier Query
    # ──────────────────────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        *,
        objective: str = "",
        max_results: int = 10,
        include_archive: bool = False,
    ) -> MemoryQueryResult:
        """
        Query across all memory tiers and return a unified result.

        Query routing:
          1. Working memory: exact/fuzzy key match
          2. Semantic memory: ACT-R + 3-component retrieval score
          3. Episodic memory: recent episodes with keyword match
          4. Archive: only if include_archive=True (cold lookup)
        """
        t0 = time.time()
        self._query_count += 1

        query_lower = query_text.lower()
        working_hits: List[str] = []
        episodic_hits: List[Dict] = []
        semantic_hits: List[Dict] = []

        # Tier 0: Working memory
        with self._wm_lock:
            for key, slot in self._working_memory.items():
                if query_lower in key.lower() or query_lower in slot.content.lower():
                    working_hits.append(f"[WM:{slot.tier_source}] {slot.content[:200]}")
                    slot.last_accessed = time.time()
                    if len(working_hits) >= 3:
                        break

        # Tier 2: Semantic memory
        if self._semantic_mem is not None:
            try:
                sem_facts = self._semantic_mem.query(
                    query_text,
                    max_results=max_results,
                    goal_context=objective,
                )
                for fact in sem_facts:
                    semantic_hits.append({
                        "key": f"{fact.subject}:{fact.predicate}",
                        "content": fact.object,
                        "confidence": fact.current_confidence(),
                        "category": fact.category,
                    })
            except Exception as exc:
                _logger.debug("[MemoryManager] Semantic query failed: %s", exc)

        # Tier 1: Episodic memory (via synthesizer if available)
        if self._episodic_synth is not None:
            try:
                if hasattr(self._episodic_synth, "query"):
                    ep_results = self._episodic_synth.query(
                        query_text, max_results=max_results // 2
                    )
                    if isinstance(ep_results, list):
                        for r in ep_results:
                            episodic_hits.append(
                                r if isinstance(r, dict)
                                else {"content": str(r)}
                            )
            except Exception as exc:
                _logger.debug("[MemoryManager] Episodic query failed: %s", exc)

        # GII-FIX: Reconcile cross-store results to eliminate duplicates and
        # resolve conflicts before building the final MemoryQueryResult.
        if self._reconciler is not None and _RECONCILER_AVAILABLE:
            try:
                raw_claims: list = []
                for h in working_hits:
                    raw_claims.append(MemoryClaim(
                        source="working_memory", key=query_text[:50] + "_wm",
                        value=h, confidence=0.9,
                    ))
                for h in semantic_hits:
                    raw_claims.append(MemoryClaim(
                        source="semantic",
                        key=str(h.get("key", query_text[:50])),
                        value=str(h.get("content", "")),
                        confidence=float(h.get("confidence", 0.6)),
                    ))
                for h in episodic_hits:
                    raw_claims.append(MemoryClaim(
                        source="episodic",
                        key=str(h.get("key", query_text[:50] + "_ep")),
                        value=str(h.get("content", h)),
                        confidence=float(h.get("confidence", 0.5)),
                    ))
                if raw_claims:
                    reconciled = self._reconciler.reconcile(
                        raw_claims, context=objective, max_results=max_results,
                    )
                    # Rebuild typed results from reconciled facts
                    working_hits = [
                        f.value for f in reconciled.facts if f.source == "working_memory"
                    ]
                    semantic_hits = [
                        {"key": f.key, "content": f.value, "confidence": f.confidence,
                         "category": "reconciled"}
                        for f in reconciled.facts if f.source not in ("working_memory", "episodic")
                    ]
                    episodic_hits = [
                        {"key": f.key, "content": f.value}
                        for f in reconciled.facts if f.source == "episodic"
                    ]
                    if reconciled.conflicts_resolved:
                        _logger.debug(
                            "[MemoryManager] Reconciler resolved %d conflict(s) in query=%r",
                            reconciled.conflicts_resolved, query_text[:40],
                        )
            except Exception as _rec_err:
                _logger.debug("[MemoryManager] Reconciler skipped (non-fatal): %s", _rec_err)

        total_chars = (
            sum(len(h) for h in working_hits)
            + sum(len(str(h)) for h in semantic_hits)
            + sum(len(str(h)) for h in episodic_hits)
        )

        return MemoryQueryResult(
            working_memory=working_hits,
            episodic_hits=episodic_hits,
            semantic_hits=semantic_hits,
            total_chars=total_chars,
            query_time_ms=(time.time() - t0) * 1000,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Promotion / Demotion
    # ──────────────────────────────────────────────────────────────────────────

    def maybe_promote_to_semantic(
        self,
        subject: str,
        predicate: str,
        object_: str,
        *,
        confirmation_count: int = 0,
        confidence: float = 0.8,
        category: str = "general",
    ) -> bool:
        """
        GII-FIX: Promote a hot episodic fact to semantic memory when
        it has been confirmed enough times. Implements the Generative Agents
        reflection pattern: frequently-confirmed episodic observations
        crystallise into semantic knowledge (Blueprint §10.4).

        Returns True if promoted, False if threshold not met.
        """
        if confirmation_count < _PROMOTION_THRESHOLD:
            return False
        if self._semantic_mem is None:
            return False
        try:
            self._semantic_mem.store(
                subject=subject,
                predicate=predicate,
                object_=object_,
                category=category,
                confidence=confidence,
                source="promoted_from_episodic",
            )
            self._promotion_count += 1
            _logger.debug(
                "[MemoryManager] Promoted to semantic: %s→%s=%r",
                subject, predicate, object_[:50],
            )
            return True
        except Exception as exc:
            _logger.debug("[MemoryManager] Promotion failed: %s", exc)
            return False

    def run_maintenance(self) -> Dict[str, int]:
        """
        Run periodic maintenance:
          - Demote cold semantic facts to archive
          - Compact working memory
          - Log tier sizes

        Call this after task completion or on a background thread.
        Returns counts of actions taken.
        """
        promoted = 0
        demoted = 0
        evicted = 0

        # Compact working memory
        with self._wm_lock:
            before = len(self._working_memory)
            self._enforce_wm_budget()
            evicted = before - len(self._working_memory)

        # Demote cold semantic facts (if semantic memory supports it)
        if self._semantic_mem is not None:
            try:
                now = time.time()
                cutoff = now - (_DEMOTION_HOURS * 3600.0)
                all_facts = getattr(self._semantic_mem, "_facts", {})
                cold_facts = []
                for fid, fact in list(all_facts.items()):
                    last_access = max(
                        (max(fact.access_history) if fact.access_history else 0.0),
                        fact.last_confirmed_at,
                    )
                    if last_access < cutoff and fact.current_confidence() < 0.5:
                        cold_facts.append((fid, fact))

                # Archive cold facts (cap at 50 per maintenance run)
                for fid, fact in cold_facts[:50]:
                    try:
                        with self._archive_lock:
                            conn = self._get_archive_conn()
                            conn.execute(
                                "INSERT OR REPLACE INTO archive "
                                "(key, content, importance, inserted_at, tier_source, archived_at) "
                                "VALUES (?, ?, ?, ?, ?, ?)",
                                (
                                    f"semantic:{fid}",
                                    f"{fact.subject}→{fact.predicate}={fact.object}",
                                    fact.current_confidence(),
                                    fact.created_at,
                                    "semantic",
                                    now,
                                )
                            )
                            conn.commit()
                        # Remove from semantic memory if over limit
                        if len(all_facts) > _MAX_SEMANTIC_FACTS:
                            all_facts.pop(fid, None)
                            demoted += 1
                    except Exception:
                        pass
                self._demotion_count += demoted
            except Exception as exc:
                _logger.debug("[MemoryManager] Demotion pass failed: %s", exc)

        _logger.info(
            "[MemoryManager] Maintenance: promoted=%d demoted=%d evicted=%d "
            "wm_slots=%d",
            promoted, demoted, evicted, len(self._working_memory),
        )
        return {"promoted": promoted, "demoted": demoted, "evicted": evicted}

    # ──────────────────────────────────────────────────────────────────────────
    # Archive (Tier 3) — SQLite
    # ──────────────────────────────────────────────────────────────────────────

    def _init_archive_db(self) -> None:
        """Initialise the Tier 3 archive SQLite database."""
        try:
            conn = sqlite3.connect(self._archive_db_path, check_same_thread=False)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS archive (
                    key         TEXT PRIMARY KEY,
                    content     TEXT NOT NULL,
                    importance  REAL NOT NULL DEFAULT 0.5,
                    inserted_at REAL NOT NULL,
                    tier_source TEXT NOT NULL DEFAULT 'manual',
                    archived_at REAL NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_archive_importance ON archive(importance)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_archive_archived ON archive(archived_at)")
            conn.commit()
            self._archive_conn = conn
            _logger.debug("[MemoryManager] Archive DB initialised: %s", self._archive_db_path)
        except Exception as exc:
            _logger.warning("[MemoryManager] Archive DB init failed: %s", exc)
            self._archive_conn = None

    def _get_archive_conn(self) -> sqlite3.Connection:
        if self._archive_conn is None:
            self._archive_conn = sqlite3.connect(
                self._archive_db_path, check_same_thread=False
            )
        return self._archive_conn

    def search_archive(
        self,
        query_text: str,
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search the cold archive tier for a keyword."""
        try:
            with self._archive_lock:
                conn = self._get_archive_conn()
                rows = conn.execute(
                    "SELECT key, content, importance FROM archive "
                    "WHERE key LIKE ? OR content LIKE ? "
                    "ORDER BY importance DESC LIMIT ?",
                    (f"%{query_text}%", f"%{query_text}%", max_results),
                ).fetchall()
            return [{"key": r[0], "content": r[1], "importance": r[2]} for r in rows]
        except Exception as exc:
            _logger.debug("[MemoryManager] Archive search failed: %s", exc)
            return []

    # ──────────────────────────────────────────────────────────────────────────
    # Stats
    # ──────────────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        with self._wm_lock:
            wm_slots = len(self._working_memory)
            wm_chars = sum(len(s.content) for s in self._working_memory.values())

        archive_count = 0
        try:
            with self._archive_lock:
                conn = self._get_archive_conn()
                row = conn.execute("SELECT COUNT(*) FROM archive").fetchone()
                if row:
                    archive_count = row[0]
        except Exception:
            pass

        return {
            "working_memory_slots": wm_slots,
            "working_memory_chars": wm_chars,
            "working_memory_budget": _WORKING_MEMORY_CHAR_BUDGET,
            "archive_entries": archive_count,
            "query_count": self._query_count,
            "promotion_count": self._promotion_count,
            "eviction_count": self._eviction_count,
            "demotion_count": self._demotion_count,
            "promotion_threshold": _PROMOTION_THRESHOLD,
            "demotion_hours": _DEMOTION_HOURS,
        }

    def __repr__(self) -> str:
        s = self.get_stats()
        return (
            f"<MemoryManager wm={s['working_memory_slots']} slots "
            f"({s['working_memory_chars']} chars / {s['working_memory_budget']} budget) "
            f"archive={s['archive_entries']}>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Singleton accessor
# ─────────────────────────────────────────────────────────────────────────────

_instance: Optional[MemoryManager] = None
_instance_lock = threading.Lock()


def get_memory_manager(
    memory_dir: Optional[str] = None,
    *,
    episodic_synthesizer=None,
    semantic_memory=None,
) -> MemoryManager:
    """Return the global MemoryManager instance (creates if not exists)."""
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = MemoryManager(
                memory_dir=memory_dir,
                episodic_synthesizer=episodic_synthesizer,
                semantic_memory=semantic_memory,
            )
        return _instance
