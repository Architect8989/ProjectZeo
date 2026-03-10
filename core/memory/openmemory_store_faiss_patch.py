"""
openmemory_store_faiss_patch.py
────────────────────────────────────────────────────────────────────────────
PATCH INSTRUCTIONS: Apply to core/memory/openmemory_store.py

This patch adds FAISS vector search as an intermediate tier between
Qdrant (best) and SQLite keyword search (worst).

SEARCH PRIORITY after patch:
  1. Qdrant        — vector similarity (Docker required, highest quality)
  2. FAISS local   — vector similarity (in-process, medium quality)
  3. SQLite        — keyword search (always available, lowest quality)

═══════════════════════════════════════════════════════════════════════════
CHANGE 1: In OpenMemoryStore.__init__()
          Add after: self._init_qdrant()
═══════════════════════════════════════════════════════════════════════════
"""

# CHANGE 1 — Add in __init__() after self._init_qdrant():
FAISS_INIT_ADDITION = '''
        # ── FAISS local vector search (Tier 2 fallback when Qdrant unavailable)
        # Blueprint §10: Memory Backends — always-available semantic search
        self._faiss = None
        self._faiss_available = False
        if not self._qdrant_available:
            self._init_faiss()
'''

# ── CHANGE 2: Add new method _init_faiss() after _init_qdrant() ─────────────
FAISS_INIT_METHOD = '''
    def _init_faiss(self) -> None:
        """
        Initialise FAISS local vector store as Tier-2 fallback.
        Called when Qdrant is unavailable.
        """
        try:
            from core.memory.faiss_vector_store import get_faiss_store
            self._faiss = get_faiss_store()
            if self._faiss._available:
                self._faiss_available = True
                _logger.info(
                    "[OpenMemory] FAISS local vector store active "
                    "(Qdrant unavailable). Semantic search quality: MEDIUM. "
                    "Start Qdrant for best quality: docker-compose up -d"
                )
            else:
                _logger.info(
                    "[OpenMemory] FAISS not available — using SQLite keyword search only."
                )
        except Exception as exc:
            _logger.info("[OpenMemory] FAISS init failed: %s — SQLite only", exc)
            self._faiss_available = False
'''

# ── CHANGE 3: In store() method, add after self._store_qdrant(entry): ────────
FAISS_STORE_ADDITION = '''
        # Store in FAISS if Qdrant unavailable (Tier-2 vector search)
        elif self._faiss_available and self._faiss is not None:
            try:
                self._faiss.add(
                    memory_id=entry.memory_id,
                    content=entry.content,
                    sector=entry.sector,
                )
            except Exception as _faiss_store_exc:
                _logger.debug("[OpenMemory] FAISS store failed: %s", _faiss_store_exc)
'''

# ── CHANGE 4: In retrieve() method, replace the fallback section ─────────────
#  ORIGINAL (lines ~533-545):
#
#     # Try Qdrant vector search first (highest quality)
#     results: Optional[List[MemoryEntry]] = None
#     if self._qdrant_available:
#         try:
#             results = self._retrieve_qdrant(query, sector, top_k, at_time)
#         except Exception as exc:
#             _logger.debug("[OpenMemory] Qdrant retrieve failed: %s", exc)
#             results = None
#
#     # Fallback: SQLite keyword search
#     if not results:
#         results = self._sqlite.retrieve_by_text(query, sector, top_k, at_time)
#
#  REPLACE WITH:

FAISS_RETRIEVE_REPLACEMENT = '''
        # Tier 1: Qdrant vector search (highest quality, Docker required)
        results: Optional[List[MemoryEntry]] = None
        if self._qdrant_available:
            try:
                results = self._retrieve_qdrant(query, sector, top_k, at_time)
            except Exception as exc:
                _logger.debug("[OpenMemory] Qdrant retrieve failed: %s", exc)
                results = None

        # Tier 2: FAISS local vector search (in-process, medium quality)
        if not results and self._faiss_available and self._faiss is not None:
            try:
                faiss_hits = self._faiss.search(query, top_k=top_k, sector=sector)
                if faiss_hits:
                    # Resolve memory_ids to full MemoryEntry objects from SQLite
                    hit_ids = [mem_id for mem_id, _ in faiss_hits]
                    results = self._sqlite.retrieve_by_ids(hit_ids, at_time=at_time)
                    if results:
                        _logger.debug(
                            "[OpenMemory] FAISS retrieve: %d results for %r",
                            len(results), query[:40]
                        )
            except Exception as faiss_exc:
                _logger.debug("[OpenMemory] FAISS retrieve failed: %s", faiss_exc)
                results = None

        # Tier 3: SQLite keyword search (always available, lowest quality)
        if not results:
            results = self._sqlite.retrieve_by_text(query, sector, top_k, at_time)
'''

# ── CHANGE 5: Add retrieve_by_ids() to _SQLiteBackend ───────────────────────
SQLITE_RETRIEVE_BY_IDS = '''
    def retrieve_by_ids(
        self,
        memory_ids: List[str],
        at_time: Optional[float] = None,
    ) -> List[MemoryEntry]:
        """
        Retrieve MemoryEntry objects by their memory_id.
        Used by FAISS tier to hydrate search results.
        """
        if not memory_ids:
            return []
        now = at_time or time.time()
        placeholders = ",".join("?" * len(memory_ids))
        query = f"""
            SELECT memory_id, sector, content, subject, importance,
                   valid_from, valid_to, created_at, last_accessed, metadata
            FROM memories
            WHERE memory_id IN ({placeholders})
              AND (valid_to IS NULL OR valid_to >= ?)
              AND valid_from <= ?
        """
        params = memory_ids + [now, now]
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_entry(r) for r in rows]
'''

# ── CHANGE 6: In get_stats() add FAISS info ──────────────────────────────────
FAISS_STATS_ADDITION = '''
            "faiss_available": self._faiss_available,
            "faiss": (self._faiss.get_stats() if self._faiss_available and self._faiss else {}),
'''

if __name__ == "__main__":
    print("FAISS patch blocks defined.")
    print("Apply to core/memory/openmemory_store.py at the locations described above.")
    print("")
    print("Also place faiss_vector_store.py at: core/memory/faiss_vector_store.py")
    print("")
    print("After patching, the memory search priority becomes:")
    print("  1. Qdrant (Docker)  → best quality")
    print("  2. FAISS local      → semantic fallback (new)")
    print("  3. SQLite keyword   → always available")
