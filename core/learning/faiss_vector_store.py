"""
core/memory/faiss_vector_store.py — Local FAISS Vector Search
────────────────────────────────────────────────────────────────────────────
Blueprint §10: Memory Backends — Local Fallback

When Qdrant is not available (no Docker), this module provides in-process
FAISS-based vector similarity search.

Role: Intermediate tier between Qdrant (best) and SQLite keyword (worst).
  Qdrant (Docker)       ← Tier 1: semantic similarity, persisted, scalable
  FAISS local (this)    ← Tier 2: semantic similarity, in-process, RAM-based
  SQLite keyword search ← Tier 3: exact/prefix match, always available

Architecture:
  FAISSVectorStore
    ├── add(memory_id, content) → embeds + stores in index
    ├── search(query, top_k) → returns [(memory_id, score)]
    └── delete(memory_id) → marks as deleted (soft delete)

Embedding: Calls nomic-embed-text via Ollama (768-dim).
           Falls back to TF-IDF if Ollama unavailable.

Thread-safe: RLock protects all FAISS mutations.

Integration: OpenMemoryStore._init_faiss() called in __init__ when Qdrant
             is unavailable. Retrieval path tries FAISS before SQLite fallback.
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_logger = logging.getLogger(__name__)

_EMBED_MODEL = os.environ.get("PROJECTZEO_EMBED_MODEL", "nomic-embed-text")
_EMBED_DIM   = int(os.environ.get("PROJECTZEO_EMBED_DIM", "768"))
_MAX_INDEX   = int(os.environ.get("PROJECTZEO_FAISS_MAX_ENTRIES", "50000"))
_FAISS_PATH  = os.path.expanduser(
    os.environ.get("PROJECTZEO_FAISS_PATH", "~/.projectzeo/memory/faiss_index.bin")
)
_META_PATH   = _FAISS_PATH.replace(".bin", "_meta.json")


# ─────────────────────────────────────────────────────────────────────────────
# Embedding helpers
# ─────────────────────────────────────────────────────────────────────────────

class _OllamaEmbedder:
    """Call Ollama nomic-embed-text for 768-dim embeddings."""

    def __init__(self, model: str = _EMBED_MODEL, dim: int = _EMBED_DIM) -> None:
        self._model = model
        self._dim   = dim
        self._available = False
        self._check()

    def _check(self) -> None:
        try:
            import ollama
            ollama.Client().list()
            self._available = True
        except Exception:
            self._available = False

    def embed(self, text: str) -> Optional[np.ndarray]:
        if not self._available:
            return None
        try:
            import ollama
            resp = ollama.embeddings(model=self._model, prompt=text[:4000])
            vec = resp.get("embedding", [])
            if not vec:
                return None
            arr = np.array(vec, dtype=np.float32)
            # Normalize
            norm = np.linalg.norm(arr)
            if norm > 1e-8:
                arr = arr / norm
            return arr
        except Exception as exc:
            _logger.debug("[FAISSStore] Ollama embed failed: %s", exc)
            self._available = False
            return None


class _TFIDFEmbedder:
    """
    Pure-Python TF-IDF embedder fallback when Ollama unavailable.
    Returns 256-dim sparse vector (sufficient for keyword-style recall).
    """

    DIM = 256

    def embed(self, text: str) -> np.ndarray:
        tokens = text.lower().split()
        vec = np.zeros(self.DIM, dtype=np.float32)
        for tok in tokens:
            h = int(hashlib.sha1(tok.encode()).hexdigest(), 16) % self.DIM
            vec[h] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
        return vec


# ─────────────────────────────────────────────────────────────────────────────
# FAISSVectorStore
# ─────────────────────────────────────────────────────────────────────────────

class FAISSVectorStore:
    """
    In-process FAISS flat index for semantic memory search.

    Thread-safe. Persists index and metadata to disk.
    Falls back to TF-IDF embedding when Ollama unavailable.
    """

    def __init__(self, dim: int = _EMBED_DIM) -> None:
        self._dim   = dim
        self._lock  = threading.RLock()

        # Embedders (try Ollama first, fall back to TF-IDF)
        self._ollama = _OllamaEmbedder(dim=dim)
        self._tfidf  = _TFIDFEmbedder()

        if not self._ollama._available:
            _logger.info(
                "[FAISSStore] Ollama embedder unavailable — using TF-IDF "
                "(dim=%d). Semantic recall quality reduced.", _TFIDFEmbedder.DIM
            )
            self._dim = _TFIDFEmbedder.DIM

        # FAISS index + metadata
        self._index       = None    # faiss.IndexFlatIP
        self._meta:  List[Dict[str, Any]] = []   # parallel list to index
        self._deleted: set = set()  # soft-deleted memory_ids
        self._available   = False

        self._load_or_create()

    # ── Index lifecycle ───────────────────────────────────────────────────────

    def _load_or_create(self) -> None:
        try:
            import faiss
        except ImportError:
            _logger.warning(
                "[FAISSStore] faiss-cpu not installed — FAISS tier disabled. "
                "Install: pip install faiss-cpu"
            )
            return

        os.makedirs(os.path.dirname(_FAISS_PATH), exist_ok=True)

        if os.path.isfile(_FAISS_PATH) and os.path.isfile(_META_PATH):
            try:
                self._index = faiss.read_index(_FAISS_PATH)
                with open(_META_PATH, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    self._meta    = saved.get("meta", [])
                    self._deleted = set(saved.get("deleted", []))
                _logger.info(
                    "[FAISSStore] Loaded index: %d vectors (dim=%d) from %s",
                    self._index.ntotal, self._dim, _FAISS_PATH,
                )
                self._available = True
                return
            except Exception as exc:
                _logger.warning("[FAISSStore] Failed to load index: %s — creating fresh.", exc)

        # Create new flat IP (inner product = cosine with normalized vectors)
        self._index = faiss.IndexFlatIP(self._dim)
        self._meta  = []
        self._deleted = set()
        self._available = True
        _logger.info("[FAISSStore] Created new FAISS index (dim=%d).", self._dim)

    def _save(self) -> None:
        if not self._available or self._index is None:
            return
        try:
            import faiss
            faiss.write_index(self._index, _FAISS_PATH)
            with open(_META_PATH, "w", encoding="utf-8") as f:
                json.dump(
                    {"meta": self._meta, "deleted": list(self._deleted)},
                    f, indent=2, default=str,
                )
        except Exception as exc:
            _logger.debug("[FAISSStore] Save failed: %s", exc)

    # ── Core API ─────────────────────────────────────────────────────────────

    def add(self, memory_id: str, content: str, sector: str = "") -> bool:
        """
        Embed and add content to the index.
        Returns True if successfully added.
        """
        if not self._available:
            return False

        vec = self._embed(content)
        if vec is None:
            return False

        with self._lock:
            if self._index is None:
                return False
            # Bound index size
            if self._index.ntotal >= _MAX_INDEX:
                _logger.debug("[FAISSStore] Index at capacity (%d) — skipping add.", _MAX_INDEX)
                return False

            self._index.add(vec.reshape(1, -1))
            self._meta.append({
                "memory_id": memory_id,
                "sector":    sector,
                "added_at":  time.time(),
            })

            # Periodic save (every 50 additions)
            if len(self._meta) % 50 == 0:
                self._save()

        return True

    def search(
        self,
        query: str,
        top_k: int = 10,
        sector: Optional[str] = None,
        threshold: float = 0.3,
    ) -> List[Tuple[str, float]]:
        """
        Search for similar memories.

        Returns list of (memory_id, cosine_score) sorted by score descending.
        Only returns results with score > threshold.
        """
        if not self._available or self._index is None:
            return []

        vec = self._embed(query)
        if vec is None:
            return []

        with self._lock:
            if self._index.ntotal == 0:
                return []

            k = min(top_k * 3, self._index.ntotal)  # over-fetch for filtering
            try:
                scores, indices = self._index.search(vec.reshape(1, -1), k)
            except Exception as exc:
                _logger.debug("[FAISSStore] Search failed: %s", exc)
                return []

            results: List[Tuple[str, float]] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self._meta):
                    continue
                meta = self._meta[idx]
                mem_id = meta.get("memory_id", "")
                if mem_id in self._deleted:
                    continue
                if sector and meta.get("sector", "") != sector:
                    continue
                if float(score) < threshold:
                    continue
                results.append((mem_id, float(score)))
                if len(results) >= top_k:
                    break

        return results

    def delete(self, memory_id: str) -> None:
        """Soft-delete a memory (excluded from future searches)."""
        with self._lock:
            self._deleted.add(memory_id)

    def count(self) -> int:
        """Return total vectors in index (including soft-deleted)."""
        with self._lock:
            return self._index.ntotal if self._index is not None else 0

    def get_stats(self) -> Dict[str, Any]:
        return {
            "available":   self._available,
            "total":       self.count(),
            "deleted":     len(self._deleted),
            "dim":         self._dim,
            "embedder":    "ollama" if self._ollama._available else "tfidf",
            "index_path":  _FAISS_PATH,
        }

    # ── Internal ─────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> Optional[np.ndarray]:
        if self._ollama._available:
            vec = self._ollama.embed(text)
            if vec is not None:
                # Pad/truncate to match index dim
                if len(vec) != self._dim:
                    if len(vec) > self._dim:
                        vec = vec[:self._dim]
                    else:
                        vec = np.pad(vec, (0, self._dim - len(vec)))
                return vec

        # TF-IDF fallback
        if self._dim == _TFIDFEmbedder.DIM:
            return self._tfidf.embed(text)

        # Dimension mismatch: pad TF-IDF to match FAISS dim
        tfidf_vec = self._tfidf.embed(text)
        if len(tfidf_vec) >= self._dim:
            return tfidf_vec[:self._dim]
        return np.pad(tfidf_vec, (0, self._dim - len(tfidf_vec)))


# ─────────────────────────────────────────────────────────────────────────────
# Singleton accessor
# ─────────────────────────────────────────────────────────────────────────────

_global_faiss: Optional[FAISSVectorStore] = None
_faiss_lock = threading.Lock()


def get_faiss_store(dim: int = _EMBED_DIM) -> FAISSVectorStore:
    """Return (or create) the singleton FAISS vector store."""
    global _global_faiss
    with _faiss_lock:
        if _global_faiss is None:
            _global_faiss = FAISSVectorStore(dim=dim)
    return _global_faiss
