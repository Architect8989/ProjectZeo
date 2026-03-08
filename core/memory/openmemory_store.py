"""
core/memory/openmemory_store.py
================================
OpenMemory 5-Sector Memory System for ProjectZeo GII.

Blueprint Reference: §2.5.2 (github.com/CaviraOSS/OpenMemory)

Implements a five-sector memory system modelled on cognitive science:

  EPISODIC    — Events: what the agent did, when, and what happened
  SEMANTIC    — Facts: application knowledge, UI patterns, user preferences
  PROCEDURAL  — Skills: successful operator sequences (SOAR chunking target)
  EMOTIONAL   — Preferences: user comfort/discomfort patterns, trust signals
  REFLECTIVE  — Meta-insights: post-task lessons, failure patterns

Each sector has:
  - Independent salience/decay rate
  - Sector-specific retrieval strategy
  - valid_from/valid_to temporal indexing (point-in-time queries)
  - Waypoint trace: which memory nodes were traversed during retrieval (auditability)

Storage backends (tried in order):
  1. Qdrant vector store (production, persistent)
  2. Local SQLite + in-memory embeddings (fallback, no GPU)
  3. Pure in-memory dict (last resort, session-only)

Interface expected by GlobalWorkspace.MemoryModule:
  .retrieve(query, top_k, sector?) → List[MemoryEntry]  (entries have .content attr)

Interface expected by OperatorCycle:
  .store_procedural(content, subject, importance) → None
  .retrieve(query, top_k, sector="procedural") → List

Interface expected by GIIController:
  .store_episodic(...)
  .store_semantic(...)
  .store_reflective(...)
  .store_emotional(...)
  .query_at_time(sector, query, timestamp) → List   [temporal query]
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

_DATA_DIR          = os.path.expanduser(
    os.environ.get("PROJECTZEO_MEMORY_DIR", "~/.projectzeo/memory")
)
_DB_PATH           = os.path.join(_DATA_DIR, "openmemory.sqlite")
_QDRANT_URL        = os.environ.get("PROJECTZEO_QDRANT_URL", "").strip()
_QDRANT_COLLECTION = os.environ.get("PROJECTZEO_OPENMEM_COLLECTION", "projectzeo_openmemory")
_EMBED_MODEL       = os.environ.get("PROJECTZEO_EMBED_MODEL", "nomic-embed-text")
_OLLAMA_HOST       = os.environ.get("PROJECTZEO_OLLAMA_HOST", "localhost")
_OLLAMA_PORT       = int(os.environ.get("PROJECTZEO_OLLAMA_PORT", "11434"))

_MAX_MEMORY_PER_SECTOR = int(os.environ.get("PROJECTZEO_OPENMEM_MAX_PER_SECTOR", "5000"))
_RETRIEVAL_TIMEOUT     = float(os.environ.get("PROJECTZEO_OPENMEM_TIMEOUT", "5.0"))

# Sector-specific decay rates (fraction of salience lost per hour)
_DECAY_RATES: Dict[str, float] = {
    "episodic":   0.05,   # Moderate decay — events become less salient over time
    "semantic":   0.01,   # Slow decay — facts stay relevant
    "procedural": 0.005,  # Very slow — skills persist
    "emotional":  0.02,   # Moderate — preferences fade
    "reflective": 0.008,  # Slow — insights stay
}


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

class MemorySector(str, Enum):
    EPISODIC   = "episodic"
    SEMANTIC   = "semantic"
    PROCEDURAL = "procedural"
    EMOTIONAL  = "emotional"
    REFLECTIVE = "reflective"


@dataclass
class MemoryEntry:
    """A single memory unit stored in any sector."""
    memory_id:    str
    sector:       str
    content:      str                   # Free text or JSON string
    subject:      str = ""              # Primary entity/app this memory is about
    importance:   float = 0.5          # 0.0-1.0 salience score at creation
    current_salience: float = 0.5      # Decayed salience at retrieval time
    valid_from:   float = field(default_factory=time.time)
    valid_to:     Optional[float] = None   # None = currently valid
    created_at:   float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count:  int = 0
    metadata:      Dict[str, Any] = field(default_factory=dict)
    # Waypoint trace: IDs of related memory nodes traversed during retrieval
    waypoint_trace: List[str] = field(default_factory=list)

    def effective_salience(self, query_ts: Optional[float] = None) -> float:
        """
        ACT-R-style activation:
        S = base_level_learning * recency_boost * importance

        base_level_learning decays with time since last access.
        """
        now = query_ts or time.time()
        sector_decay = _DECAY_RATES.get(self.sector, 0.02)

        # Time since last access in hours
        hours_since_access = (now - self.last_accessed) / 3600.0

        # ACT-R base-level activation (simplified)
        decay_factor = max(0.01, 1.0 - sector_decay * hours_since_access)

        # Frequency boost: more accesses → more salient
        freq_boost = min(2.0, 1.0 + 0.1 * self.access_count)

        return min(1.0, self.importance * decay_factor * freq_boost)

    def is_valid_at(self, timestamp: float) -> bool:
        """True if this memory was valid at the given timestamp."""
        if timestamp < self.valid_from:
            return False
        if self.valid_to is not None and timestamp > self.valid_to:
            return False
        return True


@dataclass
class RetrievalResult:
    """Result of a memory retrieval operation."""
    entries:        List[MemoryEntry]
    query:          str
    sector:         Optional[str]
    latency_ms:     float
    waypoint_trace: List[str]   # Ordered list of memory IDs traversed


# ─────────────────────────────────────────────────────────────────────────────
# SQLite backend
# ─────────────────────────────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    memory_id      TEXT PRIMARY KEY,
    sector         TEXT NOT NULL,
    content        TEXT NOT NULL,
    subject        TEXT DEFAULT '',
    importance     REAL DEFAULT 0.5,
    valid_from     REAL NOT NULL,
    valid_to       REAL,
    created_at     REAL NOT NULL,
    last_accessed  REAL NOT NULL,
    access_count   INTEGER DEFAULT 0,
    metadata_json  TEXT DEFAULT '{}',
    embedding_blob BLOB
);

CREATE INDEX IF NOT EXISTS idx_memories_sector     ON memories(sector);
CREATE INDEX IF NOT EXISTS idx_memories_subject    ON memories(subject);
CREATE INDEX IF NOT EXISTS idx_memories_valid_from ON memories(valid_from);
CREATE INDEX IF NOT EXISTS idx_memories_valid_to   ON memories(valid_to);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);
"""


class _SQLiteBackend:
    """Persistent SQLite storage with BM25-style text search."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._lock    = threading.Lock()
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def store(self, entry: MemoryEntry) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO memories
                    (memory_id, sector, content, subject, importance,
                     valid_from, valid_to, created_at, last_accessed,
                     access_count, metadata_json)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        entry.memory_id, entry.sector, entry.content,
                        entry.subject, entry.importance,
                        entry.valid_from, entry.valid_to,
                        entry.created_at, entry.last_accessed,
                        entry.access_count,
                        json.dumps(entry.metadata),
                    )
                )

    def retrieve_by_text(
        self,
        query: str,
        sector: Optional[str] = None,
        top_k: int = 5,
        at_time: Optional[float] = None,
    ) -> List[MemoryEntry]:
        """Simple keyword-based retrieval with recency and importance ranking."""
        now = at_time or time.time()

        # Tokenize query
        query_words = set(
            w.lower() for w in query.split()
            if len(w) > 2 and w.lower() not in {
                "the", "and", "for", "are", "was", "has", "had",
                "with", "that", "this", "from", "not", "but"
            }
        )

        with self._lock:
            with self._connect() as conn:
                # Filter by sector and temporal validity
                if sector:
                    rows = conn.execute(
                        """
                        SELECT * FROM memories
                        WHERE sector = ?
                          AND valid_from <= ?
                          AND (valid_to IS NULL OR valid_to >= ?)
                        ORDER BY importance DESC, last_accessed DESC
                        LIMIT ?
                        """,
                        (sector, now, now, top_k * 10)
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT * FROM memories
                        WHERE valid_from <= ?
                          AND (valid_to IS NULL OR valid_to >= ?)
                        ORDER BY importance DESC, last_accessed DESC
                        LIMIT ?
                        """,
                        (now, now, top_k * 20)
                    ).fetchall()

        if not rows:
            return []

        # Score by keyword overlap + salience
        scored: List[Tuple[float, MemoryEntry]] = []
        for row in rows:
            entry = self._row_to_entry(row)
            salience = entry.effective_salience(now)

            if query_words:
                content_words = set(
                    w.lower() for w in entry.content.split() if len(w) > 2
                )
                overlap = len(query_words & content_words) / max(len(query_words), 1)
                score = 0.6 * overlap + 0.4 * salience
            else:
                score = salience

            scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [e for _, e in scored[:top_k]]

        # Update access timestamps
        ids = [e.memory_id for e in results]
        if ids:
            with self._lock:
                with self._connect() as conn:
                    conn.execute(
                        f"""
                        UPDATE memories
                        SET last_accessed = ?, access_count = access_count + 1
                        WHERE memory_id IN ({','.join('?'*len(ids))})
                        """,
                        [now] + ids
                    )

        return results

    def count_sector(self, sector: str) -> int:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) as n FROM memories WHERE sector=?", (sector,)
                ).fetchone()
                return row["n"] if row else 0

    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        try:
            meta = json.loads(row["metadata_json"] or "{}")
        except json.JSONDecodeError:
            meta = {}
        return MemoryEntry(
            memory_id     = row["memory_id"],
            sector        = row["sector"],
            content       = row["content"],
            subject       = row["subject"] or "",
            importance    = float(row["importance"]),
            valid_from    = float(row["valid_from"]),
            valid_to      = row["valid_to"],
            created_at    = float(row["created_at"]),
            last_accessed = float(row["last_accessed"]),
            access_count  = int(row["access_count"]),
            metadata      = meta,
        )

    def vacuum_sector(self, sector: str, keep_top_n: int = _MAX_MEMORY_PER_SECTOR) -> int:
        """Remove lowest-importance entries when sector exceeds capacity."""
        n = self.count_sector(sector)
        if n <= keep_top_n:
            return 0
        excess = n - keep_top_n
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    f"""
                    DELETE FROM memories WHERE memory_id IN (
                        SELECT memory_id FROM memories
                        WHERE sector = ?
                        ORDER BY importance ASC, last_accessed ASC
                        LIMIT ?
                    )
                    """,
                    (sector, excess)
                )
        _logger.debug("[OpenMemory] Vacuumed %d entries from sector '%s'", excess, sector)
        return excess


# ─────────────────────────────────────────────────────────────────────────────
# OpenMemoryStore — main class
# ─────────────────────────────────────────────────────────────────────────────

class OpenMemoryStore:
    """
    Five-sector temporal memory store for ProjectZeo GII.

    Provides sector-aware storage and retrieval with temporal validity,
    ACT-R-style salience decay, and waypoint tracing for auditability.
    """

    _instance: Optional["OpenMemoryStore"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        os.makedirs(_DATA_DIR, exist_ok=True)

        self._sqlite   = _SQLiteBackend(_DB_PATH)
        self._lock     = threading.Lock()
        self._qdrant   = None
        self._qdrant_available = False

        # Attempt Qdrant connection for vector search
        self._init_qdrant()

        # In-memory fast cache for recent retrievals
        self._cache: Dict[str, Tuple[List[MemoryEntry], float]] = {}
        self._cache_lock = threading.Lock()
        self._cache_ttl  = 10.0  # seconds

        _logger.info(
            "[OpenMemory] Initialised. SQLite=%s Qdrant=%s",
            _DB_PATH, self._qdrant_available
        )

    @classmethod
    def get_instance(cls) -> "OpenMemoryStore":
        if cls._instance is not None:
            return cls._instance
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def _init_qdrant(self) -> None:
        """Attempt to connect to Qdrant for vector similarity search."""
        if not _QDRANT_URL:
            return
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            client = QdrantClient(url=_QDRANT_URL, timeout=5.0)
            # Verify connection
            client.get_collections()

            # Ensure collection exists
            collections = [c.name for c in client.get_collections().collections]
            if _QDRANT_COLLECTION not in collections:
                client.create_collection(
                    collection_name=_QDRANT_COLLECTION,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
                )
                _logger.info("[OpenMemory] Created Qdrant collection: %s", _QDRANT_COLLECTION)

            self._qdrant = client
            self._qdrant_available = True
            _logger.info("[OpenMemory] Qdrant connected at %s", _QDRANT_URL)
        except Exception as exc:
            _logger.info("[OpenMemory] Qdrant not available: %s — using SQLite only", exc)
            self._qdrant_available = False

    # =========================================================================
    # Storage API
    # =========================================================================

    def store(
        self,
        sector: str,
        content: str,
        *,
        subject: str = "",
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        valid_from: Optional[float] = None,
        valid_to: Optional[float] = None,
    ) -> MemoryEntry:
        """
        Store a memory in the specified sector.

        Args:
            sector:     One of: episodic, semantic, procedural, emotional, reflective
            content:    Memory text (free form or JSON)
            subject:    Primary entity (app name, user, task type)
            importance: 0.0-1.0 salience at creation time
            metadata:   Additional structured data
            valid_from: Start of validity window (default: now)
            valid_to:   End of validity window (None = indefinitely valid)
        """
        sector = sector.lower().strip()
        if sector not in {s.value for s in MemorySector}:
            sector = MemorySector.EPISODIC.value

        now = time.time()
        entry = MemoryEntry(
            memory_id     = f"mem_{sector[:3]}_{uuid.uuid4().hex[:16]}",
            sector        = sector,
            content       = str(content)[:4000],
            subject       = str(subject)[:200],
            importance    = max(0.0, min(1.0, importance)),
            valid_from    = valid_from or now,
            valid_to      = valid_to,
            created_at    = now,
            last_accessed = now,
            metadata      = metadata or {},
        )

        self._sqlite.store(entry)

        # Optionally store embedding in Qdrant
        if self._qdrant_available:
            self._store_qdrant(entry)

        # Vacuum if over capacity (non-blocking, with error handling)
        def _vacuum():
            try:
                self._sqlite.vacuum_sector(sector)
            except Exception as vex:
                _logger.debug("[OpenMemory] Vacuum error (non-fatal): %s", vex)
        threading.Thread(target=_vacuum, daemon=True).start()

        _logger.debug(
            "[OpenMemory] Stored [%s] subject=%r importance=%.2f",
            sector, subject[:40], importance
        )
        return entry

    def store_episodic(
        self, content: str, *, subject: str = "", importance: float = 0.6
    ) -> MemoryEntry:
        return self.store(
            MemorySector.EPISODIC.value, content,
            subject=subject, importance=importance
        )

    def store_semantic(
        self, content: str, *, subject: str = "", importance: float = 0.7
    ) -> MemoryEntry:
        return self.store(
            MemorySector.SEMANTIC.value, content,
            subject=subject, importance=importance
        )

    def store_procedural(
        self, content: str, *, subject: str = "", importance: float = 0.8
    ) -> MemoryEntry:
        """Store a skill/procedure — high importance, very slow decay."""
        return self.store(
            MemorySector.PROCEDURAL.value, content,
            subject=subject, importance=importance
        )

    def store_emotional(
        self, content: str, *, subject: str = "", importance: float = 0.5
    ) -> MemoryEntry:
        return self.store(
            MemorySector.EMOTIONAL.value, content,
            subject=subject, importance=importance
        )

    def store_reflective(
        self, content: str, *, subject: str = "", importance: float = 0.75
    ) -> MemoryEntry:
        """Store a post-task insight — high importance, slow decay."""
        return self.store(
            MemorySector.REFLECTIVE.value, content,
            subject=subject, importance=importance
        )

    def expire(self, memory_id: str) -> None:
        """Mark a memory as expired (valid_to = now)."""
        now = time.time()
        with self._sqlite._lock:
            with self._sqlite._connect() as conn:
                conn.execute(
                    "UPDATE memories SET valid_to=? WHERE memory_id=?",
                    (now, memory_id)
                )

    # =========================================================================
    # Retrieval API
    # =========================================================================

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        sector: Optional[str] = None,
        at_time: Optional[float] = None,
        include_waypoints: bool = False,
    ) -> List[MemoryEntry]:
        """
        Retrieve relevant memories for a query.

        Args:
            query:             Natural language search query
            top_k:             Maximum number of entries to return
            sector:            Restrict to specific sector (None = all sectors)
            at_time:           Point-in-time query (None = current time)
            include_waypoints: If True, populate waypoint_trace on each result

        Returns:
            List of MemoryEntry sorted by relevance (most relevant first).
        """
        t0 = time.perf_counter()

        # Cache check
        cache_key = self._cache_key(query, sector, at_time)
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached:
                entries, ts = cached
                if time.time() - ts < self._cache_ttl:
                    return entries[:top_k]

        # Try Qdrant vector search first (highest quality)
        results: Optional[List[MemoryEntry]] = None
        if self._qdrant_available:
            try:
                results = self._retrieve_qdrant(query, sector, top_k, at_time)
            except Exception as exc:
                _logger.debug("[OpenMemory] Qdrant retrieve failed: %s", exc)
                results = None

        # Fallback: SQLite keyword search
        if not results:
            results = self._sqlite.retrieve_by_text(query, sector, top_k, at_time)

        # Build waypoint traces if requested
        if include_waypoints and results:
            for entry in results:
                entry.waypoint_trace = self._build_waypoint_trace(entry, query)

        latency = (time.perf_counter() - t0) * 1000
        _logger.debug(
            "[OpenMemory] retrieve(%r sector=%s) → %d results in %.0fms",
            query[:40], sector, len(results), latency
        )

        # Update cache
        with self._cache_lock:
            self._cache[cache_key] = (results, time.time())
            # Bound cache
            if len(self._cache) > 200:
                oldest = sorted(self._cache.items(), key=lambda kv: kv[1][1])[:50]
                for k, _ in oldest:
                    del self._cache[k]

        return results

    def query_at_time(
        self,
        sector: str,
        query: str,
        timestamp: float,
        top_k: int = 5,
    ) -> List[MemoryEntry]:
        """
        Temporal query: retrieve memories that were valid at a specific timestamp.
        Enables point-in-time reasoning about past UI states.
        """
        return self.retrieve(
            query=query,
            top_k=top_k,
            sector=sector,
            at_time=timestamp,
        )

    def get_sector_summary(self, sector: str) -> Dict[str, Any]:
        """Return statistics about a memory sector."""
        count = self._sqlite.count_sector(sector)
        decay = _DECAY_RATES.get(sector, 0.02)
        return {
            "sector":      sector,
            "count":       count,
            "decay_rate":  decay,
            "half_life_h": round(0.693 / decay if decay > 0 else float("inf"), 1),
        }

    # =========================================================================
    # Qdrant vector search
    # =========================================================================

    def _store_qdrant(self, entry: MemoryEntry) -> None:
        """Store embedding in Qdrant for vector similarity search."""
        try:
            embedding = self._embed(entry.content)
            if embedding is None:
                return

            from qdrant_client.models import PointStruct
            point = PointStruct(
                id      = abs(hash(entry.memory_id)) % (2**63),
                vector  = embedding,
                payload = {
                    "memory_id": entry.memory_id,
                    "sector":    entry.sector,
                    "subject":   entry.subject,
                    "content":   entry.content[:500],
                    "importance": entry.importance,
                    "valid_from": entry.valid_from,
                    "valid_to":   entry.valid_to,
                }
            )
            self._qdrant.upsert(
                collection_name=_QDRANT_COLLECTION,
                points=[point],
            )
        except Exception as exc:
            _logger.debug("[OpenMemory] Qdrant upsert error: %s", exc)

    def _retrieve_qdrant(
        self,
        query: str,
        sector: Optional[str],
        top_k: int,
        at_time: Optional[float],
    ) -> List[MemoryEntry]:
        """Vector similarity search via Qdrant."""
        embedding = self._embed(query)
        if embedding is None:
            raise RuntimeError("Embedding unavailable")

        from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
        filters = []

        if sector:
            filters.append(FieldCondition(key="sector", match=MatchValue(value=sector)))

        now = at_time or time.time()
        filters.append(FieldCondition(key="valid_from", range=Range(lte=now)))

        qdrant_filter = Filter(must=filters) if filters else None

        hits = self._qdrant.search(
            collection_name=_QDRANT_COLLECTION,
            query_vector=embedding,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        results: List[MemoryEntry] = []
        for hit in hits:
            payload = hit.payload or {}
            mid = payload.get("memory_id", "")
            if not mid:
                continue

            # Fetch full entry from SQLite (source of truth)
            try:
                entries = self._sqlite.retrieve_by_text(
                    payload.get("content", ""), sector, top_k=1, at_time=at_time
                )
                if entries:
                    e = entries[0]
                    e.current_salience = float(hit.score)
                    results.append(e)
                    continue
            except Exception:
                pass

            # Fallback: construct from Qdrant payload
            entry = MemoryEntry(
                memory_id   = mid,
                sector      = payload.get("sector", "episodic"),
                content     = payload.get("content", ""),
                subject     = payload.get("subject", ""),
                importance  = float(payload.get("importance", 0.5)),
                valid_from  = float(payload.get("valid_from", 0.0)),
                valid_to    = payload.get("valid_to"),
                current_salience = float(hit.score),
            )
            results.append(entry)

        return results

    # =========================================================================
    # Embeddings
    # =========================================================================

    def _embed(self, text: str) -> Optional[List[float]]:
        """Generate text embedding using Ollama or sentence-transformers."""
        # Try Ollama
        try:
            import httpx
            r = httpx.post(
                f"http://{_OLLAMA_HOST}:{_OLLAMA_PORT}/api/embeddings",
                json={"model": _EMBED_MODEL, "prompt": text[:2000]},
                timeout=10.0,
            )
            if r.status_code == 200:
                return r.json().get("embedding", [])
        except Exception:
            pass

        # Try sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            vec = model.encode(text[:2000]).tolist()
            return vec
        except Exception:
            pass

        return None

    # =========================================================================
    # Waypoint tracing
    # =========================================================================

    def _build_waypoint_trace(self, entry: MemoryEntry, query: str) -> List[str]:
        """
        Build a waypoint trace showing which memory IDs were considered
        during retrieval. This provides auditability for the agent's reasoning.
        """
        try:
            # Retrieve contextually related memories
            related = self._sqlite.retrieve_by_text(
                entry.subject or query, entry.sector, top_k=3
            )
            trace = [r.memory_id for r in related if r.memory_id != entry.memory_id]
            return [entry.memory_id] + trace[:3]
        except Exception:
            return [entry.memory_id]

    # =========================================================================
    # Helpers
    # =========================================================================

    def _cache_key(
        self,
        query: str,
        sector: Optional[str],
        at_time: Optional[float],
    ) -> str:
        raw = f"{query}::{sector}::{int(at_time or 0)}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def get_stats(self) -> Dict[str, Any]:
        """
        Return counts per sector and overall statistics.

        Returns a stable schema that includes:
          - top-level ``{sector}_count`` keys (backward-compatible)
          - ``sectors`` sub-dict with per-sector detail
          - ``total`` aggregate count
          - ``vacuum_needed`` flag when any sector is near capacity

        DEFECT FIX: Previously missing ``sectors`` sub-dict which the docstring
        promised and which any code expecting structured sector stats would fail
        to find. Now both flat and structured formats are provided.
        """
        sector_counts: Dict[str, int] = {}
        for sector in MemorySector:
            sector_counts[sector.value] = self._sqlite.count_sector(sector.value)

        total = sum(sector_counts.values())

        sectors_detail: Dict[str, Dict[str, Any]] = {}
        for sector in MemorySector:
            count = sector_counts[sector.value]
            decay = _DECAY_RATES.get(sector.value, 0.02)
            sectors_detail[sector.value] = {
                "count":      count,
                "capacity":   _MAX_MEMORY_PER_SECTOR,
                "pct_full":   round(100.0 * count / _MAX_MEMORY_PER_SECTOR, 1),
                "decay_rate": decay,
                "half_life_h": round(0.693 / decay if decay > 0 else float("inf"), 1),
            }

        vacuum_needed = any(
            info["count"] > 0.9 * _MAX_MEMORY_PER_SECTOR
            for info in sectors_detail.values()
        )

        stats: Dict[str, Any] = {
            "qdrant_active":  self._qdrant_available,
            "db_path":        _DB_PATH,
            "total":          total,
            "vacuum_needed":  vacuum_needed,
            "sectors":        sectors_detail,
        }
        # Backward-compatible flat keys
        for sector_name, count in sector_counts.items():
            stats[f"{sector_name}_count"] = count

        return stats

    def __repr__(self) -> str:
        stats = self.get_stats()
        total = sum(
            stats.get(f"{s.value}_count", 0) for s in MemorySector
        )
        return (
            f"<OpenMemoryStore total={total} "
            f"qdrant={self._qdrant_available} "
            f"db={_DB_PATH}>"
        )
