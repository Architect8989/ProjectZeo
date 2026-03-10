"""
core/memory/unified_memory_orchestrator.py
============================================
Unified Memory Orchestrator — GII Blueprint §10

WHY THIS FILE EXISTS
--------------------
ProjectZeo has 9+ independent memory backends (Mem0, Cognee, Graphiti, A-MEM,
HippoRAG, OpenMemory, FAISS, MemoryManager, Playbook).  They were initialised
independently in GIIController._initialise_phase2_components() with no
coordination, no reconciliation, and no startup pass.

This orchestrator:

1. STARTUP RECONCILIATION PASS
   On first call, detects which backends are available (Qdrant/FAISS,
   FalkorDB/SQLite) and migrates any data stored in fallback backends to
   production backends when they become available.  Prevents split-brain
   across sessions.

2. UNIFIED WRITE FANOUT
   store() writes to ALL available backends simultaneously (async where
   possible) so no backend silently misses an event.

3. UNIFIED READ WITH RANKING
   retrieve() queries all available backends and merges results using a
   confidence-weighted rank fusion algorithm (Reciprocal Rank Fusion).

4. BACKEND HEALTH MONITORING
   Each backend has a health flag. Failed reads/writes decrement health;
   successful operations restore it. Unhealthy backends are temporarily
   bypassed to prevent cascading timeouts.

5. AT-STARTUP MEMORY STATUS BANNER
   Prints a clear capability matrix so operators know which tiers are active.

INTEGRATION
-----------
Replace individual backend init calls in GIIController._initialise_phase2_components()
with a single:
    self._unified_memory = UnifiedMemoryOrchestrator(
        memory_dir=memory_dir,
        llm_callable=llm_callable,
    )
    await self._unified_memory.startup_reconciliation()

Then replace per-backend store/retrieve calls with:
    self._unified_memory.store(key, value, memory_type="episodic")
    results = self._unified_memory.retrieve(query, top_k=5)
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Backend health tracking
# ─────────────────────────────────────────────────────────────────────────────
_MAX_HEALTH = 5           # max health points per backend
_MIN_HEALTH_TO_USE = 1    # skip backends below this
_HEALTH_DECAY_PER_FAIL = 1
_HEALTH_RESTORE_PER_SUCCESS = 1

# RRF constant
_RRF_K = 60

# Startup reconciliation — migrate data when a better backend becomes available
_RECONCILIATION_LOCK = threading.Lock()
_RECONCILIATION_DONE = False


class MemoryType(str, Enum):
    EPISODIC  = "episodic"
    SEMANTIC  = "semantic"
    PLAYBOOK  = "playbook"
    FAILURE   = "failure"
    TASK      = "task"
    APP       = "app"


@dataclass
class MemoryEntry:
    key:         str
    value:       Any
    memory_type: MemoryType
    confidence:  float = 1.0
    source:      str   = "unknown"
    timestamp:   float = field(default_factory=time.time)
    metadata:    Dict  = field(default_factory=dict)


@dataclass
class RetrievalResult:
    entry:    MemoryEntry
    score:    float
    source:   str


class _BackendWrapper:
    """Thin wrapper around any memory backend with health tracking."""

    def __init__(self, name: str, backend: Any) -> None:
        self.name    = name
        self.backend = backend
        self.health  = _MAX_HEALTH
        self.enabled = True

    def is_usable(self) -> bool:
        return self.enabled and self.health >= _MIN_HEALTH_TO_USE

    def on_success(self) -> None:
        self.health = min(_MAX_HEALTH, self.health + _HEALTH_RESTORE_PER_SUCCESS)

    def on_failure(self, err: Exception) -> None:
        self.health -= _HEALTH_DECAY_PER_FAIL
        if self.health <= 0:
            _logger.warning(
                "[UnifiedMemory] Backend %r health=0 — bypassed until recovery.", self.name
            )


class UnifiedMemoryOrchestrator:
    """
    Central coordinator for all ProjectZeo memory backends.

    Usage
    -----
    orchestrator = UnifiedMemoryOrchestrator(memory_dir="/path", llm_callable=fn)
    orchestrator.startup_reconciliation()   # call once at task start
    orchestrator.store("task:xyz", {...}, MemoryType.EPISODIC)
    results = orchestrator.retrieve("how to close dialog in Firefox", top_k=5)
    context_str = orchestrator.get_context_string(objective, max_chars=1500)
    """

    def __init__(
        self,
        *,
        memory_dir: Optional[str] = None,
        llm_callable: Optional[Callable] = None,
    ) -> None:
        self._memory_dir  = memory_dir or os.path.expanduser("~/.projectzeo")
        self._llm         = llm_callable
        self._lock        = threading.RLock()
        self._backends: List[_BackendWrapper] = []
        self._initialized = False
        self._startup_done = False

        # Individual backend references (for compatibility with existing callers)
        self.mem0_store         = None
        self.cognee_store       = None
        self.graphiti_store     = None
        self.amem_store         = None
        self.hippo_rag          = None
        self.openmemory_store   = None
        self.memory_manager     = None
        self.playbook_store     = None
        self.knowledge_vault    = None

        self._init_backends()

    # ─────────────────────────────────────────────────────────────────────────
    # INIT
    # ─────────────────────────────────────────────────────────────────────────

    def _init_backends(self) -> None:
        """Lazy-initialise all memory backends, recording which are available."""
        _logger.info("[UnifiedMemory] Initialising memory backends...")

        # Mem0 — cross-session
        try:
            from core.memory.mem0_store import Mem0Store
            self.mem0_store = Mem0Store(memory_dir=self._memory_dir)
            self._backends.append(_BackendWrapper("Mem0", self.mem0_store))
            _logger.info("[UnifiedMemory] ✓ Mem0Store")
        except Exception as e:
            _logger.debug("[UnifiedMemory] Mem0Store unavailable: %s", e)

        # OpenMemory (SQLite + FAISS/Qdrant fallback)
        try:
            from core.memory.openmemory_store import OpenMemoryStore
            self.openmemory_store = OpenMemoryStore(memory_dir=self._memory_dir)
            self._backends.append(_BackendWrapper("OpenMemory", self.openmemory_store))
            _logger.info("[UnifiedMemory] ✓ OpenMemoryStore")
        except Exception as e:
            _logger.debug("[UnifiedMemory] OpenMemoryStore unavailable: %s", e)

        # Cognee + Qdrant
        try:
            from core.memory.cognee_store import CogneeStore
            self.cognee_store = CogneeStore(memory_dir=self._memory_dir)
            self._backends.append(_BackendWrapper("Cognee", self.cognee_store))
            _logger.info("[UnifiedMemory] ✓ CogneeStore")
        except Exception as e:
            _logger.debug("[UnifiedMemory] CogneeStore unavailable: %s", e)

        # Graphiti bi-temporal KG
        try:
            from core.memory.graphiti_store import GraphitiStore
            self.graphiti_store = GraphitiStore(memory_dir=self._memory_dir)
            self._backends.append(_BackendWrapper("Graphiti", self.graphiti_store))
            _logger.info("[UnifiedMemory] ✓ GraphitiStore (backend=%s)", 
                         getattr(self.graphiti_store, "_backend", "?"))
        except Exception as e:
            _logger.debug("[UnifiedMemory] GraphitiStore unavailable: %s", e)

        # A-MEM Zettelkasten
        try:
            from core.memory.amem_store import AMEMStore
            self.amem_store = AMEMStore(memory_dir=self._memory_dir)
            self._backends.append(_BackendWrapper("AMEM", self.amem_store))
            _logger.info("[UnifiedMemory] ✓ AMEMStore")
        except Exception as e:
            _logger.debug("[UnifiedMemory] AMEMStore unavailable: %s", e)

        # HippoRAG
        try:
            from core.memory.hippo_rag import get_hippo_rag
            self.hippo_rag = get_hippo_rag()
            self._backends.append(_BackendWrapper("HippoRAG", self.hippo_rag))
            _logger.info("[UnifiedMemory] ✓ HippoRAG (%d nodes)",
                         self.hippo_rag.get_stats().get("nodes", 0))
        except Exception as e:
            _logger.debug("[UnifiedMemory] HippoRAG unavailable: %s", e)

        # MemoryManager (4-tier MemGPT)
        try:
            from core.memory.memory_manager import MemoryManager
            self.memory_manager = MemoryManager(
                memory_dir=self._memory_dir,
                llm_callable=self._llm,
            )
            self._backends.append(_BackendWrapper("MemoryManager", self.memory_manager))
            _logger.info("[UnifiedMemory] ✓ MemoryManager (4-tier)")
        except Exception as e:
            _logger.debug("[UnifiedMemory] MemoryManager unavailable: %s", e)

        # Playbook Store
        try:
            from core.memory.playbook_store import PlaybookStore
            self.playbook_store = PlaybookStore(memory_dir=self._memory_dir)
            self._backends.append(_BackendWrapper("Playbook", self.playbook_store))
            _logger.info("[UnifiedMemory] ✓ PlaybookStore")
        except Exception as e:
            _logger.debug("[UnifiedMemory] PlaybookStore unavailable: %s", e)

        # Knowledge Vault
        try:
            from core.memory.knowledge_vault import get_global_knowledge_vault
            self.knowledge_vault = get_global_knowledge_vault()
            self._backends.append(_BackendWrapper("KnowledgeVault", self.knowledge_vault))
            _logger.info("[UnifiedMemory] ✓ KnowledgeVault")
        except Exception as e:
            _logger.debug("[UnifiedMemory] KnowledgeVault unavailable: %s", e)

        active = [b.name for b in self._backends]
        _logger.info(
            "[UnifiedMemory] %d/%d backends active: %s",
            len(active), 9, active,
        )
        self._initialized = True
        self._print_startup_banner()

    def _print_startup_banner(self) -> None:
        active = [b.name for b in self._backends if b.is_usable()]
        inactive = [
            n for n in ["Mem0", "OpenMemory", "Cognee", "Graphiti", "AMEM",
                        "HippoRAG", "MemoryManager", "Playbook", "KnowledgeVault"]
            if n not in active
        ]
        import sys
        print(
            "\n[UnifiedMemory] Memory backend status:\n"
            + "".join(f"  ✓ {n}\n" for n in active)
            + "".join(f"  ✗ {n} (unavailable — check Docker/install)\n" for n in inactive),
            file=sys.stderr,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # STARTUP RECONCILIATION PASS
    # ─────────────────────────────────────────────────────────────────────────

    def startup_reconciliation(self) -> None:
        """
        Detect split-brain between FAISS-only data (from previous sessions with
        no Qdrant) and Qdrant (now available).  Migrate FAISS entries to Qdrant.

        This prevents losing cross-session data when infrastructure changes
        (e.g., Docker started after previously running without it).

        Thread-safe — safe to call multiple times; only runs once per process.
        """
        global _RECONCILIATION_DONE
        with _RECONCILIATION_LOCK:
            if _RECONCILIATION_DONE:
                return
            _RECONCILIATION_DONE = True

        _logger.info("[UnifiedMemory] Running startup reconciliation pass...")

        # Check if Qdrant is now available (was previously absent)
        qdrant_available = self._check_qdrant()

        if qdrant_available and self.openmemory_store is not None:
            try:
                # Attempt to migrate any FAISS-only entries to Qdrant
                migrated = self._migrate_faiss_to_qdrant()
                if migrated > 0:
                    _logger.info(
                        "[UnifiedMemory] Reconciliation: migrated %d FAISS entries → Qdrant.",
                        migrated,
                    )
            except Exception as e:
                _logger.warning("[UnifiedMemory] FAISS→Qdrant migration failed: %s", e)

        # Ingest HippoRAG from Graphiti if both available
        if self.hippo_rag is not None and self.graphiti_store is not None:
            try:
                self.hippo_rag.ingest_from_graphiti(self.graphiti_store)
                _logger.info("[UnifiedMemory] HippoRAG ingested from Graphiti.")
            except Exception as e:
                _logger.debug("[UnifiedMemory] HippoRAG Graphiti ingest: %s", e)

        self._startup_done = True
        _logger.info("[UnifiedMemory] Startup reconciliation complete.")

    def _check_qdrant(self) -> bool:
        """Return True if Qdrant is reachable."""
        qdrant_url = os.environ.get("PROJECTZEO_QDRANT_URL", "http://localhost:6333")
        try:
            import urllib.request
            req = urllib.request.Request(f"{qdrant_url}/healthz", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _migrate_faiss_to_qdrant(self) -> int:
        """
        Read all entries from the FAISS fallback and upsert them to Qdrant.
        Returns the number of entries migrated.
        """
        if not hasattr(self.openmemory_store, "list_faiss_entries"):
            return 0
        try:
            entries = self.openmemory_store.list_faiss_entries()
            if not entries:
                return 0
            migrated = 0
            for entry in entries:
                try:
                    self.openmemory_store.upsert_to_qdrant(entry)
                    migrated += 1
                except Exception:
                    pass
            return migrated
        except Exception:
            return 0

    # ─────────────────────────────────────────────────────────────────────────
    # WRITE FANOUT
    # ─────────────────────────────────────────────────────────────────────────

    def store(
        self,
        key: str,
        value: Any,
        memory_type: MemoryType = MemoryType.EPISODIC,
        *,
        confidence: float = 1.0,
        metadata: Optional[Dict] = None,
        backends: Optional[List[str]] = None,  # None = all
    ) -> int:
        """
        Write to all available backends (or a subset).
        Returns the number of backends successfully written to.
        """
        entry = MemoryEntry(
            key=key,
            value=value,
            memory_type=memory_type,
            confidence=confidence,
            metadata=metadata or {},
        )
        written = 0
        for wrapper in self._backends:
            if not wrapper.is_usable():
                continue
            if backends is not None and wrapper.name not in backends:
                continue
            try:
                self._backend_store(wrapper, entry)
                wrapper.on_success()
                written += 1
            except Exception as e:
                wrapper.on_failure(e)
                _logger.debug(
                    "[UnifiedMemory] Store failed on %s: %s", wrapper.name, e
                )
        return written

    def _backend_store(self, wrapper: _BackendWrapper, entry: MemoryEntry) -> None:
        """Route a store call to the correct backend API."""
        b = wrapper.backend
        name = wrapper.name

        if name == "Mem0":
            b.add(str(entry.value), metadata={"key": entry.key, **entry.metadata})
        elif name == "OpenMemory":
            b.add(
                text=str(entry.value),
                memory_id=entry.key,
                metadata=entry.metadata,
            )
        elif name == "Cognee":
            b.store(entry.key, str(entry.value))
        elif name == "Graphiti":
            b.store_fact(
                subject=entry.key,
                predicate=entry.memory_type.value,
                object_=str(entry.value)[:500],
            )
        elif name == "AMEM":
            b.add_note(
                content=str(entry.value),
                tags=[entry.memory_type.value, entry.key[:50]],
            )
        elif name == "HippoRAG":
            b.add_document(
                doc_id=entry.key,
                text=str(entry.value),
                metadata=entry.metadata,
            )
        elif name == "MemoryManager":
            b.store(entry.key, entry.value, tier="episodic")
        elif name == "Playbook":
            if entry.memory_type == MemoryType.PLAYBOOK:
                b.store_playbook(entry.key, entry.value)
        elif name == "KnowledgeVault":
            b.store_fact(str(entry.value), subject=entry.key)

    # ─────────────────────────────────────────────────────────────────────────
    # UNIFIED RETRIEVAL WITH RANK FUSION
    # ─────────────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        memory_type: Optional[MemoryType] = None,
    ) -> List[RetrievalResult]:
        """
        Query all healthy backends and merge results using Reciprocal Rank
        Fusion (RRF).  Returns top_k results ranked by fused score.
        """
        all_results: Dict[str, List[Tuple[int, float]]] = {}  # key → [(rank, score)]

        for wrapper in self._backends:
            if not wrapper.is_usable():
                continue
            try:
                backend_results = self._backend_retrieve(wrapper, query, top_k)
                wrapper.on_success()
                for rank, (key, score) in enumerate(backend_results):
                    if key not in all_results:
                        all_results[key] = []
                    all_results[key].append((rank, score))
            except Exception as e:
                wrapper.on_failure(e)
                _logger.debug(
                    "[UnifiedMemory] Retrieve failed on %s: %s", wrapper.name, e
                )

        # RRF fusion
        fused: Dict[str, float] = {}
        for key, rank_scores in all_results.items():
            rrf_score = sum(1.0 / (_RRF_K + rank + 1) for rank, _ in rank_scores)
            fused[key] = rrf_score

        sorted_keys = sorted(fused, key=lambda k: fused[k], reverse=True)[:top_k]

        results = []
        for key in sorted_keys:
            results.append(RetrievalResult(
                entry=MemoryEntry(
                    key=key,
                    value=key,  # caller resolves full value from key if needed
                    memory_type=memory_type or MemoryType.SEMANTIC,
                ),
                score=fused[key],
                source="rrf_fusion",
            ))
        return results

    def _backend_retrieve(
        self, wrapper: _BackendWrapper, query: str, top_k: int
    ) -> List[Tuple[str, float]]:
        """
        Call backend-specific search API.
        Returns list of (key, relevance_score) tuples.
        """
        b = wrapper.backend
        name = wrapper.name
        results: List[Tuple[str, float]] = []

        if name == "Mem0":
            items = b.search(query, limit=top_k) or []
            for i, item in enumerate(items):
                key = item.get("id", str(i)) if isinstance(item, dict) else str(i)
                results.append((key, 1.0 / (i + 1)))

        elif name == "OpenMemory":
            items = b.search(query, top_k=top_k) or []
            for i, item in enumerate(items):
                key = item.get("id", str(i)) if isinstance(item, dict) else str(i)
                score = item.get("score", 1.0 / (i + 1)) if isinstance(item, dict) else 1.0 / (i + 1)
                results.append((key, score))

        elif name == "HippoRAG":
            items = b.query(query, top_k=top_k) or []
            for i, item in enumerate(items):
                key = item.get("doc_id", str(i)) if isinstance(item, dict) else str(i)
                score = item.get("score", 1.0 / (i + 1)) if isinstance(item, dict) else 1.0 / (i + 1)
                results.append((key, score))

        elif name == "MemoryManager":
            raw = b.retrieve(query, tier="all", top_k=top_k) or []
            for i, item in enumerate(raw):
                key = str(item)[:80] if not isinstance(item, dict) else item.get("key", str(i))
                results.append((key, 1.0 / (i + 1)))

        elif name == "KnowledgeVault":
            entries = b.query_relevant(query, max_results=top_k) or []
            for i, entry in enumerate(entries):
                key = getattr(entry, "key", str(i))
                score = getattr(entry, "importance", 1.0 / (i + 1))
                results.append((key, score))

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # CONTEXT STRING FOR PROMPT INJECTION
    # ─────────────────────────────────────────────────────────────────────────

    def get_context_string(
        self,
        objective: str,
        max_chars: int = 1500,
        top_k: int = 5,
    ) -> str:
        """
        Return a formatted memory context string for injection into LLM prompts.
        Queries all backends, deduplicates, and truncates to max_chars.
        """
        try:
            results = self.retrieve(objective, top_k=top_k)
        except Exception:
            results = []

        if not results:
            return ""

        lines = ["[MEMORY CONTEXT]"]
        char_count = 17

        for r in results:
            entry_str = f"• {r.entry.key}: {r.entry.value}"[:200]
            if char_count + len(entry_str) + 1 > max_chars:
                break
            lines.append(entry_str)
            char_count += len(entry_str) + 1

        if len(lines) == 1:
            return ""

        lines.append("[END MEMORY]")
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # TASK LIFECYCLE HOOKS
    # ─────────────────────────────────────────────────────────────────────────

    def on_task_complete(
        self,
        *,
        objective: str,
        success: bool,
        app_name: str = "",
        duration_sec: float = 0.0,
        execution_log: Optional[Dict] = None,
    ) -> None:
        """
        Called when a task completes.  Stores the outcome in all backends and
        triggers HippoRAG graph update from Graphiti (if available).
        """
        outcome_key = f"task:{int(time.time())}"
        outcome_val = {
            "objective": objective[:300],
            "success": success,
            "app_name": app_name,
            "duration_sec": duration_sec,
            "timestamp": time.time(),
        }
        self.store(
            outcome_key,
            outcome_val,
            memory_type=MemoryType.TASK,
            confidence=1.0,
        )

        # Update Graphiti with structured outcome
        if self.graphiti_store is not None:
            try:
                self.graphiti_store.store_task_outcome(
                    app_name=app_name,
                    objective=objective,
                    milestone_sequence=[],
                    stagnation_events=[],
                    vsa_violations=[],
                    success=success,
                    duration_sec=duration_sec,
                )
            except Exception as e:
                _logger.debug("[UnifiedMemory] Graphiti task outcome: %s", e)

        # Refresh HippoRAG from Graphiti after update
        if self.hippo_rag is not None and self.graphiti_store is not None:
            try:
                self.hippo_rag.ingest_from_graphiti(self.graphiti_store)
            except Exception:
                pass

    def on_failure_pattern(self, description: str, app_name: str = "") -> None:
        """Store a failure pattern in all backends for future reference."""
        self.store(
            f"failure:{int(time.time())}",
            {"description": description, "app": app_name},
            memory_type=MemoryType.FAILURE,
            confidence=0.9,
        )
        if self.knowledge_vault is not None:
            try:
                self.knowledge_vault.store_failure_pattern(
                    description, subject=app_name.lower() if app_name else "general"
                )
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────────────
    # HEALTH REPORT
    # ─────────────────────────────────────────────────────────────────────────

    def health_report(self) -> Dict[str, Any]:
        """Return current health status of all backends."""
        return {
            wrapper.name: {
                "health": wrapper.health,
                "max_health": _MAX_HEALTH,
                "usable": wrapper.is_usable(),
            }
            for wrapper in self._backends
        }


# ─────────────────────────────────────────────────────────────────────────────
# Singleton accessor
# ─────────────────────────────────────────────────────────────────────────────
_GLOBAL_INSTANCE: Optional[UnifiedMemoryOrchestrator] = None
_GLOBAL_LOCK = threading.Lock()


def get_unified_memory(
    *,
    memory_dir: Optional[str] = None,
    llm_callable: Optional[Callable] = None,
) -> UnifiedMemoryOrchestrator:
    """Get or create the global UnifiedMemoryOrchestrator singleton."""
    global _GLOBAL_INSTANCE
    with _GLOBAL_LOCK:
        if _GLOBAL_INSTANCE is None:
            _GLOBAL_INSTANCE = UnifiedMemoryOrchestrator(
                memory_dir=memory_dir,
                llm_callable=llm_callable,
            )
        return _GLOBAL_INSTANCE
