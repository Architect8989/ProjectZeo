"""
core/memory/knowledge_vault.py
================================
KnowledgeVault — MIRIX Memory Store 6: Structured Lesson Repository.

Blueprint §10.4 — Wang & Chen, 2025 (MIRIX Multi-Agent Memory)
Blueprint §8.4 — SAGE: Self-Evolving Agents structured knowledge base

MIRIX defines 6 specialized memory stores:
    1. Core           → BeliefState (working memory)
    2. Episodic       → EpisodicSynthesizer
    3. Semantic       → SemanticMemory
    4. Procedural     → SOAR Chunking
    5. Resource       → ApplicationMemory
    6. Knowledge Vault → THIS FILE ← Reflexion lessons + failure patterns + insights

A-MEM extension (Blueprint §10.5 — Xu et al., 2025):
    Each knowledge entry has:
    - LLM-generated keywords and tags
    - Contextual description
    - Dynamically constructed links to related entries
    - New experiences retroactively refine existing entry context

Cross-session persistence: ~/.projectzeo/knowledge_vault.db

Integration:
    - reflexion_engine.py → stores lessons via vault.store_lesson()
    - per_step_reasoner.py → retrieves vault.query_relevant() before step
    - gii_loop.py → loads vault on startup; vault.get_recent_insights()
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

_logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

_VAULT_DB_DIR  = os.path.join(os.path.expanduser("~"), ".projectzeo")
_VAULT_DB_FILE = os.path.join(_VAULT_DB_DIR, "knowledge_vault.db")
_MAX_LINKS_PER_ENTRY  = 5
_LINK_SIMILARITY_THRESHOLD = 0.3


# ─────────────────────────────────────────────────────────────────────────────
# Entry types
# ─────────────────────────────────────────────────────────────────────────────

class KVEntryType(str, Enum):
    LESSON           = "lesson"            # Reflexion verbal lesson
    FAILURE_PATTERN  = "failure_pattern"   # Anti-pattern to avoid
    INSIGHT          = "insight"           # Higher-level reflection
    WORKFLOW         = "workflow"          # Verified action sequence
    APP_QUIRK        = "app_quirk"         # Application-specific quirk
    SAFETY_NOTE      = "safety_note"       # Safety-relevant observation
    PERFORMANCE_TIP  = "performance_tip"   # Speed/efficiency insight


@dataclass
class KVEntry:
    """A single knowledge vault entry with A-MEM metadata."""
    entry_id:    str
    entry_type:  KVEntryType
    content:     str                    # The lesson/insight text
    subject:     str                    # Application or domain
    keywords:    List[str]              # Key terms for retrieval
    tags:        List[str]              # Categorical tags
    context_desc: str                   # Contextual description (A-MEM)
    importance:  float                  # 0.0-1.0
    linked_ids:  List[str]              # A-MEM: linked related entries
    source:      str                    # "reflexion" | "sage" | "manual"
    created_at:  float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id":    self.entry_id,
            "entry_type":  self.entry_type.value,
            "content":     self.content[:500],
            "subject":     self.subject[:100],
            "keywords":    self.keywords[:10],
            "tags":        self.tags[:10],
            "context_desc": self.context_desc[:300],
            "importance":  round(self.importance, 3),
            "linked_ids":  self.linked_ids[:_MAX_LINKS_PER_ENTRY],
            "source":      self.source,
            "created_at":  self.created_at,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
        }


# ─────────────────────────────────────────────────────────────────────────────
# KnowledgeVault
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeVault:
    """
    MIRIX Store 6: Structured lesson + insight repository.

    Features:
    - Cross-session SQLite persistence
    - A-MEM dynamic linking between related entries
    - Three-field retrieval: subject + keywords + tags
    - Retroactive context refinement on new entries
    - Importance-weighted retrieval for prompt injection

    Usage:
        vault = KnowledgeVault()

        # Store a lesson
        vault.store_lesson(
            content="In LibreOffice Calc, use Ctrl+S not File>Save to avoid dialog",
            subject="libreoffice_calc",
            entry_type=KVEntryType.LESSON,
            importance=0.8,
        )

        # Retrieve for prompt injection
        entries = vault.query_relevant("libreoffice saving", max_results=3)
        context = vault.format_for_prompt(entries)
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = db_path or _VAULT_DB_FILE
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._lock = threading.RLock()
        self._in_memory_cache: Dict[str, KVEntry] = {}
        self._cache_loaded = False
        self._init_db()
        _logger.info("[KnowledgeVault] Initialised. db=%r", self._db_path)

    # =========================================================================
    # Public API
    # =========================================================================

    def store_lesson(
        self,
        content: str,
        *,
        subject: str = "general",
        entry_type: KVEntryType = KVEntryType.LESSON,
        importance: float = 0.7,
        tags: Optional[List[str]] = None,
        source: str = "reflexion",
        context_desc: str = "",
    ) -> str:
        """
        Store a new lesson or insight.
        Automatically extracts keywords and creates A-MEM links to related entries.
        Returns the entry_id.
        """
        content = content.strip()[:1000]
        if not content:
            return ""

        keywords = _extract_keywords(content + " " + subject)
        entry_id = _make_entry_id(content, subject)

        # A-MEM: find related entries and create links
        related_ids = self._find_related(content, keywords, exclude_id=entry_id)

        entry = KVEntry(
            entry_id=entry_id,
            entry_type=entry_type,
            content=content,
            subject=subject.strip().lower()[:100],
            keywords=keywords[:15],
            tags=(tags or [])[:10],
            context_desc=context_desc[:300] or content[:100],
            importance=max(0.0, min(1.0, float(importance))),
            linked_ids=related_ids[:_MAX_LINKS_PER_ENTRY],
            source=source,
        )

        with self._lock:
            self._in_memory_cache[entry_id] = entry
            # A-MEM retroactive: add back-links to related entries
            for rid in related_ids:
                if rid in self._in_memory_cache:
                    re_entry = self._in_memory_cache[rid]
                    if entry_id not in re_entry.linked_ids:
                        re_entry.linked_ids.append(entry_id)
                        re_entry.linked_ids = re_entry.linked_ids[-_MAX_LINKS_PER_ENTRY:]

        self._persist_entry(entry)
        _logger.debug(
            "[KnowledgeVault] Stored: id=%s type=%s subject=%r links=%d",
            entry_id, entry_type.value, subject[:40], len(related_ids),
        )
        return entry_id

    def store_failure_pattern(
        self,
        pattern: str,
        *,
        subject: str = "general",
        importance: float = 0.8,
    ) -> str:
        """Convenience method: store a failure anti-pattern."""
        return self.store_lesson(
            content=pattern,
            subject=subject,
            entry_type=KVEntryType.FAILURE_PATTERN,
            importance=importance,
            source="reflexion",
        )

    def store_workflow(
        self,
        workflow: str,
        *,
        subject: str = "general",
        importance: float = 0.9,
    ) -> str:
        """Convenience method: store a verified workflow."""
        return self.store_lesson(
            content=workflow,
            subject=subject,
            entry_type=KVEntryType.WORKFLOW,
            importance=importance,
            source="sage",
        )

    def query_relevant(
        self,
        query: str,
        *,
        max_results: int = 5,
        subject_filter: Optional[str] = None,
        entry_types: Optional[List[KVEntryType]] = None,
        min_importance: float = 0.3,
    ) -> List[KVEntry]:
        """
        Retrieve relevant knowledge entries.

        Scoring:
            - Keyword overlap with query (primary)
            - Subject match bonus
            - Importance weight
            - Recency factor
        """
        self._ensure_cache_loaded()
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        now = time.time()
        scored: List[Tuple[float, KVEntry]] = []

        with self._lock:
            for entry in self._in_memory_cache.values():
                if entry.importance < min_importance:
                    continue
                if subject_filter and entry.subject != subject_filter.lower():
                    continue
                if entry_types and entry.entry_type not in entry_types:
                    continue

                # Keyword overlap
                entry_tokens = set(entry.keywords) | _tokenize(entry.content)
                overlap = len(query_tokens & entry_tokens)
                if overlap == 0:
                    continue

                relevance = overlap / max(len(query_tokens | entry_tokens), 1)

                # Subject bonus
                subj_bonus = 0.2 if subject_filter and entry.subject == subject_filter.lower() else 0.0

                # Recency (decay over days)
                days_old = (now - entry.created_at) / 86400.0
                recency = max(0.3, 1.0 - days_old * 0.01)

                score = (relevance + subj_bonus) * entry.importance * recency
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [e for _, e in scored[:max_results]]

        # Update access tracking
        if results:
            with self._lock:
                for e in results:
                    e.access_count += 1
                    e.last_accessed = now
            threading.Thread(
                target=self._batch_update_access,
                args=([e.entry_id for e in results], now),
                daemon=True,
            ).start()

        return results

    def get_recent_insights(self, n: int = 5) -> List[KVEntry]:
        """Return the N most recent high-importance entries for session startup."""
        self._ensure_cache_loaded()
        with self._lock:
            entries = sorted(
                [e for e in self._in_memory_cache.values() if e.importance >= 0.6],
                key=lambda e: e.created_at,
                reverse=True,
            )
        return entries[:n]

    def format_for_prompt(
        self,
        entries: List[KVEntry],
        *,
        max_chars: int = 800,
    ) -> str:
        """Format knowledge vault entries as a prompt-injectable string."""
        if not entries:
            return ""
        lines = ["=== Knowledge Vault (relevant lessons) ==="]
        total = 0
        for entry in entries:
            type_label = entry.entry_type.value.replace("_", " ").title()
            line = f"[{type_label}] ({entry.subject}): {entry.content[:200]}"
            if total + len(line) > max_chars:
                break
            lines.append(line)
            total += len(line)
        lines.append("=== End Knowledge Vault ===")
        return "\n".join(lines)

    def stats(self) -> Dict[str, Any]:
        self._ensure_cache_loaded()
        with self._lock:
            by_type: Dict[str, int] = {}
            for e in self._in_memory_cache.values():
                by_type[e.entry_type.value] = by_type.get(e.entry_type.value, 0) + 1
        return {
            "total_entries": len(self._in_memory_cache),
            "by_type": by_type,
            "db_path": self._db_path,
        }

    # =========================================================================
    # Private — A-MEM Linking
    # =========================================================================

    def _find_related(
        self,
        content: str,
        keywords: List[str],
        *,
        exclude_id: str,
    ) -> List[str]:
        """Find entries with high keyword overlap (A-MEM dynamic linking)."""
        query_tokens = set(keywords) | _tokenize(content)
        related: List[Tuple[float, str]] = []

        with self._lock:
            for eid, entry in self._in_memory_cache.items():
                if eid == exclude_id:
                    continue
                entry_tokens = set(entry.keywords) | _tokenize(entry.content[:100])
                if not entry_tokens:
                    continue
                overlap = len(query_tokens & entry_tokens) / max(len(query_tokens | entry_tokens), 1)
                if overlap >= _LINK_SIMILARITY_THRESHOLD:
                    related.append((overlap, eid))

        related.sort(key=lambda x: x[0], reverse=True)
        return [eid for _, eid in related[:_MAX_LINKS_PER_ENTRY]]

    # =========================================================================
    # Private — Persistence
    # =========================================================================

    def _init_db(self) -> None:
        try:
            conn = self._get_conn()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_vault (
                    entry_id TEXT PRIMARY KEY,
                    data_json TEXT NOT NULL,
                    entry_type TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    importance REAL NOT NULL,
                    created_at REAL NOT NULL,
                    access_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_kv_subject ON knowledge_vault(subject)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_kv_type ON knowledge_vault(entry_type)
            """)
            conn.commit()
        except Exception as exc:
            _logger.warning("[KnowledgeVault] DB init warning: %s", exc)

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _persist_entry(self, entry: KVEntry) -> None:
        try:
            d = entry.to_dict()
            conn = self._get_conn()
            conn.execute(
                "INSERT OR REPLACE INTO knowledge_vault(entry_id, data_json, entry_type, subject, importance, created_at, access_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (entry.entry_id, json.dumps(d), entry.entry_type.value,
                 entry.subject, entry.importance, entry.created_at, entry.access_count),
            )
            conn.commit()
        except Exception as exc:
            _logger.debug("[KnowledgeVault] Persist failed: %s", exc)

    def _ensure_cache_loaded(self) -> None:
        with self._lock:
            if self._cache_loaded:
                return
        self._load_all_from_db()

    def _load_all_from_db(self) -> None:
        try:
            conn = self._get_conn()
            rows = conn.execute("SELECT data_json FROM knowledge_vault").fetchall()
            loaded = 0
            with self._lock:
                for (data_json,) in rows:
                    try:
                        d = json.loads(data_json)
                        entry = KVEntry(
                            entry_id=d["entry_id"],
                            entry_type=KVEntryType(d["entry_type"]),
                            content=d["content"],
                            subject=d["subject"],
                            keywords=d.get("keywords", []),
                            tags=d.get("tags", []),
                            context_desc=d.get("context_desc", ""),
                            importance=d["importance"],
                            linked_ids=d.get("linked_ids", []),
                            source=d.get("source", "unknown"),
                            created_at=d.get("created_at", time.time()),
                            access_count=d.get("access_count", 0),
                            last_accessed=d.get("last_accessed", time.time()),
                        )
                        self._in_memory_cache[entry.entry_id] = entry
                        loaded += 1
                    except Exception:
                        pass
                self._cache_loaded = True
            _logger.info("[KnowledgeVault] Loaded %d entries from DB.", loaded)
        except Exception as exc:
            _logger.warning("[KnowledgeVault] DB load failed: %s", exc)
            with self._lock:
                self._cache_loaded = True

    def _batch_update_access(self, entry_ids: List[str], now: float) -> None:
        try:
            conn = self._get_conn()
            for eid in entry_ids:
                conn.execute(
                    "UPDATE knowledge_vault SET access_count=access_count+1 WHERE entry_id=?",
                    (eid,),
                )
            conn.commit()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> Set[str]:
    tokens = set(re.sub(r"[^\w]", " ", text.lower()).split())
    # Remove very short tokens and common stop words
    stop = {"the","a","an","is","in","on","at","to","of","and","or","for","with","from","by","this","that"}
    return {t for t in tokens if len(t) > 2 and t not in stop}


def _extract_keywords(text: str, max_kw: int = 10) -> List[str]:
    """Simple keyword extraction: remove stopwords, deduplicate, top by length."""
    tokens = _tokenize(text)
    # Sort by length (longer = more specific) then alphabetically
    return sorted(tokens, key=lambda t: (-len(t), t))[:max_kw]


def _make_entry_id(content: str, subject: str) -> str:
    raw = f"{content[:100]}{subject}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_global_vault: Optional[KnowledgeVault] = None
_global_lock = threading.Lock()


def get_global_knowledge_vault(db_path: Optional[str] = None) -> KnowledgeVault:
    """Return process-singleton KnowledgeVault."""
    global _global_vault
    with _global_lock:
        if _global_vault is None:
            _global_vault = KnowledgeVault(db_path=db_path)
    return _global_vault
