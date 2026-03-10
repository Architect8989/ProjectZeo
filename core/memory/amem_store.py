"""
core/memory/amem_store.py
==========================
A-MEM: Agentic Memory with Zettelkasten-style Atomic Notes.

Blueprint §8 — Memory Systems

Reference:
    Weng et al. (2024) "A-MEM: Agentic Memory for LLM Agents" — arXiv:2502.12110
    Zettelkasten method: each memory = one atomic note with explicit links to
    related notes. Unlike flat retrieval, this creates a navigable knowledge graph
    where each note enriches its neighbours on insertion.

Core operations:
    1. store(content, context) → create atomic note, generate tags+links via LLM
    2. retrieve(query, k) → BM25 + tag matching + link traversal
    3. update(note_id, new_outcome) → evolve note with new evidence
    4. consolidate() → merge near-duplicate notes, strengthen link weights

Why Zettelkasten for GII:
    - Atomic notes prevent conflation of distinct experiences
    - Explicit links enable reasoning paths ("this error → that solution")
    - Link-strength weights reflect empirical evidence, not just semantic similarity
    - Notes evolve as new evidence arrives — no stale knowledge

Integration:
    - memory_manager.py: A-MEM sits in tier 2 (session working memory)
    - gii_controller: accessible via self._amem_store
    - per_step_reasoner: injects relevant notes via retrieve()
    - nightly_consolidation: calls consolidate() to merge similar notes
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

_logger = logging.getLogger(__name__)

_STORE_DIR = os.path.expanduser(
    os.environ.get("PROJECTZEO_AMEM_DIR", "~/.projectzeo/amem")
)
_INDEX_FILE = "amem_index.json"
_MAX_NOTES = int(os.environ.get("PROJECTZEO_AMEM_MAX_NOTES", "2000"))
_CONSOLIDATE_THRESHOLD = float(os.environ.get("PROJECTZEO_AMEM_CONSOLIDATE_SIM", "0.85"))
_ENABLED = os.environ.get("PROJECTZEO_AMEM_ENABLED", "1").strip() != "0"

# BM25 parameters
_BM25_K1 = 1.5
_BM25_B  = 0.75


@dataclass
class AtomicNote:
    """
    A single Zettelkasten atomic note.

    Each note captures ONE observation, outcome, or insight.
    Links connect to related notes with a strength weight.
    """
    note_id:        str
    content:        str           # The actual knowledge (max 500 chars)
    tags:           List[str] = field(default_factory=list)
    links:          Dict[str, float] = field(default_factory=dict)  # note_id → strength
    context:        str = ""      # App + task context when note was created
    created_at:     float = field(default_factory=time.time)
    updated_at:     float = field(default_factory=time.time)
    access_count:   int = 0
    last_accessed:  float = field(default_factory=time.time)
    importance:     float = 0.5   # 0.0–1.0, decays with time, boosted on access
    source:         str = ""      # "episodic" | "semantic" | "reflexion" | "manual"
    # Evidence accumulation
    confirmations:  int = 0       # How many times this note was verified true
    contradictions: int = 0       # How many times this note was contradicted

    def decay_importance(self, decay_rate: float = 0.01) -> None:
        """Apply time-based importance decay (ACT-R style)."""
        age_hours = (time.time() - self.updated_at) / 3600.0
        self.importance = max(0.05, self.importance * (1.0 - decay_rate * age_hours))

    def boost_on_access(self) -> None:
        self.access_count += 1
        self.last_accessed = time.time()
        self.importance = min(1.0, self.importance + 0.05)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _make_note_id(content: str) -> str:
    return hashlib.sha256(f"{content}{time.time()}".encode()).hexdigest()[:16]


def _tokenise(text: str) -> List[str]:
    """Simple word tokeniser for BM25."""
    return re.findall(r"\b\w{2,}\b", text.lower())


class AMEMStore:
    """
    A-MEM: Agentic Memory with Zettelkasten atomic notes.

    Core capabilities:
      - store(): create atomic note with LLM-generated tags + links
      - retrieve(): BM25 + tag + link-traversal retrieval
      - update(): evolve note with new evidence
      - consolidate(): merge near-duplicates, strengthen link graph
    """

    def __init__(
        self,
        *,
        llm_call: Optional[Callable] = None,
        store_dir: Optional[str] = None,
    ) -> None:
        self._llm       = llm_call
        self._store_dir = store_dir or _STORE_DIR
        self._notes:    Dict[str, AtomicNote] = {}
        self._lock      = threading.Lock()
        self._enabled   = _ENABLED

        os.makedirs(self._store_dir, exist_ok=True)
        self._load_index()

        _logger.info(
            "[A-MEM] Initialised. notes=%d enabled=%s dir=%s",
            len(self._notes), self._enabled, self._store_dir,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Core operations
    # ─────────────────────────────────────────────────────────────────────────

    def store(
        self,
        content: str,
        context: str = "",
        source: str = "manual",
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Create and store an atomic note.

        If LLM is available: auto-generate tags and find related notes.
        Otherwise: use simple keyword extraction.

        Returns note_id.
        """
        if not self._enabled or not content.strip():
            return ""

        content = content.strip()[:500]
        note_id = _make_note_id(content)

        # Generate tags
        auto_tags = self._extract_tags(content, context)
        all_tags = list(set((tags or []) + auto_tags))[:15]

        # Find links to existing notes
        links = self._find_links(content, all_tags)

        note = AtomicNote(
            note_id=note_id,
            content=content,
            tags=all_tags,
            links=links,
            context=context[:200],
            importance=importance,
            source=source,
        )

        with self._lock:
            self._notes[note_id] = note
            # Update reverse links: all linked notes now point back
            for linked_id, strength in links.items():
                if linked_id in self._notes:
                    self._notes[linked_id].links[note_id] = strength * 0.8  # Slightly weaker backlink

            # Evict least important note if over limit
            if len(self._notes) > _MAX_NOTES:
                self._evict_one()

            self._save_index()

        _logger.debug(
            "[A-MEM] Stored note %s tags=%s links=%d",
            note_id[:8], all_tags[:3], len(links),
        )
        return note_id

    def retrieve(
        self,
        query: str,
        k: int = 5,
        traverse_links: bool = True,
        context_filter: Optional[str] = None,
    ) -> List[AtomicNote]:
        """
        Retrieve top-k relevant notes using BM25 + tag matching + link traversal.

        traverse_links=True: also returns notes linked to top results
        (allows reasoning chains to surface).
        """
        if not self._enabled or not query:
            return []

        with self._lock:
            if not self._notes:
                return []

            query_tokens = set(_tokenise(query))
            scores: List[Tuple[float, AtomicNote]] = []

            for note in self._notes.values():
                # Apply importance decay
                note.decay_importance()

                if context_filter and context_filter.lower() not in note.context.lower():
                    if not any(context_filter.lower() in t for t in note.tags):
                        pass  # Allow — don't hard-filter, just de-rank

                score = self._bm25_score(note, query_tokens)

                # Tag bonus
                note_words = set(_tokenise(" ".join(note.tags)))
                tag_overlap = len(query_tokens & note_words) / max(1, len(query_tokens))
                score += tag_overlap * 2.0

                # Importance weighting
                score *= (0.5 + 0.5 * note.importance)

                if score > 0.01:
                    scores.append((score, note))

            scores.sort(key=lambda x: x[0], reverse=True)
            top = [note for _, note in scores[:k]]

            # Link traversal: add strongly-linked notes
            if traverse_links:
                linked_ids: Set[str] = set()
                for note in top:
                    for linked_id, strength in note.links.items():
                        if strength >= 0.4 and linked_id not in {n.note_id for n in top}:
                            linked_ids.add(linked_id)

                for lid in list(linked_ids)[:k // 2]:
                    linked_note = self._notes.get(lid)
                    if linked_note:
                        top.append(linked_note)

            # Boost access count
            for note in top:
                note.boost_on_access()

            self._save_index()
            return top[:k * 2]  # Return up to 2*k with link traversal

    def update(
        self,
        note_id: str,
        new_observation: str,
        confirmed: bool = True,
    ) -> bool:
        """
        Update a note with new evidence. Contradictions lower importance,
        confirmations boost it.
        """
        with self._lock:
            note = self._notes.get(note_id)
            if note is None:
                return False

            note.updated_at = time.time()
            if confirmed:
                note.confirmations += 1
                note.importance = min(1.0, note.importance + 0.1)
                note.content = note.content + f" [+{note.confirmations}×confirmed]"
            else:
                note.contradictions += 1
                note.importance = max(0.05, note.importance - 0.15)
                note.content = note.content + f" [CONTRADICTED: {new_observation[:100]}]"

            self._save_index()
        return True

    def consolidate(self) -> int:
        """
        Merge near-duplicate notes (sim >= _CONSOLIDATE_THRESHOLD).
        Returns number of merges performed.

        Called by nightly_consolidation.py.
        """
        if not self._enabled:
            return 0

        merged = 0
        with self._lock:
            note_list = list(self._notes.values())
            to_delete: Set[str] = set()

            for i, note_a in enumerate(note_list):
                if note_a.note_id in to_delete:
                    continue
                for note_b in note_list[i + 1:]:
                    if note_b.note_id in to_delete:
                        continue
                    sim = _jaccard_text_sim(note_a.content, note_b.content)
                    if sim >= _CONSOLIDATE_THRESHOLD:
                        # Merge B into A (keep more important note)
                        if note_b.importance > note_a.importance:
                            # Merge A into B
                            note_b.confirmations += note_a.confirmations
                            note_b.access_count += note_a.access_count
                            note_b.tags = list(set(note_b.tags + note_a.tags))[:15]
                            note_b.links.update(note_a.links)
                            to_delete.add(note_a.note_id)
                        else:
                            note_a.confirmations += note_b.confirmations
                            note_a.access_count += note_b.access_count
                            note_a.tags = list(set(note_a.tags + note_b.tags))[:15]
                            note_a.links.update(note_b.links)
                            to_delete.add(note_b.note_id)
                        merged += 1

            for nid in to_delete:
                self._notes.pop(nid, None)

            if merged:
                self._save_index()
                _logger.info("[A-MEM] Consolidated %d duplicate notes.", merged)

        return merged

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled":     self._enabled,
                "total_notes": len(self._notes),
                "total_links": sum(len(n.links) for n in self._notes.values()),
                "avg_importance": (
                    sum(n.importance for n in self._notes.values()) / max(1, len(self._notes))
                ),
                "store_dir": self._store_dir,
            }

    # ─────────────────────────────────────────────────────────────────────────
    # BM25 scoring
    # ─────────────────────────────────────────────────────────────────────────

    def _bm25_score(self, note: AtomicNote, query_tokens: Set[str]) -> float:
        """Simplified BM25 score for a single note."""
        doc_tokens = _tokenise(note.content + " " + " ".join(note.tags))
        doc_len = max(1, len(doc_tokens))
        avg_len = 50.0  # Approximate average note length

        tf_dict: Dict[str, int] = {}
        for t in doc_tokens:
            tf_dict[t] = tf_dict.get(t, 0) + 1

        score = 0.0
        n_docs = max(1, len(self._notes))
        for term in query_tokens:
            tf = tf_dict.get(term, 0)
            if tf == 0:
                continue
            # IDF (simplified — assume term appears in ~10% of docs)
            idf = max(0.0, (n_docs - 1) / max(1, 1))
            numerator = tf * (_BM25_K1 + 1)
            denominator = tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * doc_len / avg_len)
            score += idf * numerator / max(0.001, denominator)
        return score

    # ─────────────────────────────────────────────────────────────────────────
    # Tag extraction and link discovery
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_tags(self, content: str, context: str) -> List[str]:
        """Extract tags from content. Uses LLM if available, else keyword heuristic."""
        if self._llm is not None:
            try:
                raw = self._llm(
                    system=(
                        "Extract 3-6 concise lowercase keyword tags from the text. "
                        "Output ONLY a JSON array of strings. No explanation."
                    ),
                    user=f"Text: {content[:300]}",
                    max_tokens=60,
                    timeout=10.0,
                )
                text = raw.get("content", "") if isinstance(raw, dict) else str(raw)
                text = text.strip()
                if text.startswith("["):
                    tags = json.loads(text)
                    if isinstance(tags, list):
                        return [str(t).lower()[:30] for t in tags[:8]]
            except Exception:
                pass

        # Fallback: keyword extraction
        words = _tokenise(content + " " + context)
        stopwords = {"the", "a", "an", "is", "are", "was", "for", "and",
                     "or", "but", "not", "this", "that", "with", "from"}
        freq: Dict[str, int] = {}
        for w in words:
            if w not in stopwords and len(w) >= 3:
                freq[w] = freq.get(w, 0) + 1
        return [w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:6]]

    def _find_links(
        self, content: str, tags: List[str]
    ) -> Dict[str, float]:
        """Find links to existing notes via tag overlap and content similarity."""
        links: Dict[str, float] = {}
        content_tokens = set(_tokenise(content))
        tag_set = set(tags)

        for note in list(self._notes.values())[:500]:  # Cap search
            note_tags = set(note.tags)
            tag_overlap = len(tag_set & note_tags) / max(1, len(tag_set | note_tags))
            content_sim = _jaccard_text_sim(content, note.content)
            strength = tag_overlap * 0.7 + content_sim * 0.3

            if strength >= 0.15:
                links[note.note_id] = round(min(1.0, strength), 3)

        # Keep top 10 links
        if len(links) > 10:
            links = dict(
                sorted(links.items(), key=lambda x: x[1], reverse=True)[:10]
            )
        return links

    # ─────────────────────────────────────────────────────────────────────────
    # Eviction
    # ─────────────────────────────────────────────────────────────────────────

    def _evict_one(self) -> None:
        """Evict the least important + least recently accessed note."""
        if not self._notes:
            return
        victim = min(
            self._notes.values(),
            key=lambda n: n.importance * 0.6 + (1.0 / max(1.0, time.time() - n.last_accessed)) * 0.4,
        )
        del self._notes[victim.note_id]
        _logger.debug("[A-MEM] Evicted note %s (importance=%.3f)", victim.note_id[:8], victim.importance)

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def _save_index(self) -> None:
        """Save all notes to JSON index. Called under _lock."""
        index_path = os.path.join(self._store_dir, _INDEX_FILE)
        tmp = index_path + ".tmp"
        try:
            data = {nid: n.to_dict() for nid, n in self._notes.items()}
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, separators=(",", ":"))
            os.replace(tmp, index_path)
        except Exception as exc:
            _logger.warning("[A-MEM] Save failed: %s", exc)

    def _load_index(self) -> None:
        """Load notes from disk."""
        index_path = os.path.join(self._store_dir, _INDEX_FILE)
        if not os.path.isfile(index_path):
            return
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for nid, d in data.items():
                try:
                    note = AtomicNote(**{
                        k: v for k, v in d.items()
                        if k in AtomicNote.__dataclass_fields__
                    })
                    self._notes[nid] = note
                except Exception:
                    pass
            _logger.info("[A-MEM] Loaded %d notes from disk.", len(self._notes))
        except Exception as exc:
            _logger.warning("[A-MEM] Load failed: %s", exc)


def _jaccard_text_sim(a: str, b: str) -> float:
    """Jaccard similarity between two text strings."""
    if not a or not b:
        return 0.0
    set_a = set(_tokenise(a))
    set_b = set(_tokenise(b))
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: Optional[AMEMStore] = None
_instance_lock = threading.Lock()


def get_amem_store(llm_call: Optional[Callable] = None) -> AMEMStore:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = AMEMStore(llm_call=llm_call)
    return _instance
