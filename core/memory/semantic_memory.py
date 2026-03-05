from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
_DEFAULT_MEMORY_DIR = os.path.join(
    os.path.expanduser("~"), ".projectzeo", "semantic_memory"
)
_MEMORY_FILE = "semantic_facts.json"
_MAX_FACTS = 10_000
_CONFIDENCE_DECAY_PER_DAY = 0.02   # 2% per day
_MIN_CONFIDENCE = 0.05              # facts below this are pruned
_RETRIEVAL_THRESHOLD = 0.20         # minimum confidence to return in query


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SemanticFact:
    
    fact_id: str
    subject: str          # Application name, error code, tool name, etc.
    predicate: str        # Relationship type
    object: str           # The fact value
    category: str         # One of: application_facts, ui_patterns, error_solutions,
                          #         install_outcomes, shortcut_map, general
    confidence: float     # 0.0–1.0
    source: str           # "observed" | "llm_extracted" | "operator_provided"
    created_at: float     # Unix timestamp
    last_confirmed_at: float
    confirmation_count: int = 0
    refutation_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def current_confidence(self) -> float:
        """Return confidence adjusted for age-based decay."""
        days_since_confirmed = (time.time() - self.last_confirmed_at) / 86400.0
        decayed = self.confidence - (days_since_confirmed * _CONFIDENCE_DECAY_PER_DAY)
        return max(_MIN_CONFIDENCE, min(1.0, decayed))

    def is_usable(self) -> bool:
        return self.current_confidence() >= _RETRIEVAL_THRESHOLD


VALID_CATEGORIES = frozenset({
    "application_facts",
    "ui_patterns",
    "error_solutions",
    "install_outcomes",
    "shortcut_map",
    "general",
})


def _fact_id(subject: str, predicate: str, object_: str) -> str:
    raw = f"{subject.strip().lower()}|{predicate.strip().lower()}|{object_.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# SemanticMemory
# ---------------------------------------------------------------------------

class SemanticMemory:
    

    def __init__(
        self,
        memory_dir: Optional[str] = None,
        *,
        max_facts: int = _MAX_FACTS,
        auto_save_interval: float = 30.0,
    ) -> None:
        self._memory_dir = memory_dir or _DEFAULT_MEMORY_DIR
        self._memory_path = os.path.join(self._memory_dir, _MEMORY_FILE)
        self._max_facts = max_facts
        self._auto_save_interval = auto_save_interval

        self._facts: Dict[str, SemanticFact] = {}
        self._lock = threading.RLock()
        self._last_save: float = 0.0
        self._dirty: bool = False

        os.makedirs(self._memory_dir, exist_ok=True)
        self._load()

        _logger.info(
            "[SemanticMemory] Initialised. dir=%r facts_loaded=%d",
            self._memory_dir, len(self._facts),
        )

    # =========================================================================
    # Public API
    # =========================================================================

    def store(
        self,
        subject: str,
        predicate: str,
        object_: str,
        *,
        category: str = "general",
        confidence: float = 0.8,
        source: str = "observed",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SemanticFact:
        
        subject = subject.strip().lower()[:200]
        predicate = predicate.strip().lower()[:200]
        object_ = object_.strip()[:1000]
        category = category if category in VALID_CATEGORIES else "general"
        confidence = max(0.0, min(1.0, float(confidence)))
        now = time.time()

        fid = _fact_id(subject, predicate, object_)

        with self._lock:
            existing = self._facts.get(fid)

            if existing:
                # Confirmation: boost confidence
                new_confidence = min(1.0, existing.confidence + 0.05)
                updated = SemanticFact(
                    fact_id=fid,
                    subject=existing.subject,
                    predicate=existing.predicate,
                    object=existing.object,
                    category=existing.category,
                    confidence=new_confidence,
                    source=existing.source,
                    created_at=existing.created_at,
                    last_confirmed_at=now,
                    confirmation_count=existing.confirmation_count + 1,
                    refutation_count=existing.refutation_count,
                    metadata={**existing.metadata, **(metadata or {})},
                )
                self._facts[fid] = updated
                self._dirty = True
                _logger.debug(
                    "[SemanticMemory] Confirmed: %s→%s=%r (confidence %.2f→%.2f)",
                    subject, predicate, object_[:60], existing.confidence, new_confidence,
                )
                return updated
            else:
                fact = SemanticFact(
                    fact_id=fid,
                    subject=subject,
                    predicate=predicate,
                    object=object_,
                    category=category,
                    confidence=confidence,
                    source=source,
                    created_at=now,
                    last_confirmed_at=now,
                    metadata=metadata or {},
                )
                self._facts[fid] = fact
                self._dirty = True
                _logger.debug(
                    "[SemanticMemory] Stored: %s→%s=%r (conf=%.2f)",
                    subject, predicate, object_[:60], confidence,
                )

                # Enforce max_facts limit (evict lowest-confidence facts)
                if len(self._facts) > self._max_facts:
                    self._evict_lowest_confidence_locked()

                self._maybe_auto_save()
                return fact

    def query(
        self,
        query_text: str,
        *,
        max_results: int = 10,
        min_confidence: float = _RETRIEVAL_THRESHOLD,
        category: Optional[str] = None,
    ) -> List[SemanticFact]:
        
        query_tokens = set(re.sub(r"[^\w\s]", "", query_text.lower()).split())
        if not query_tokens:
            return []

        with self._lock:
            candidates: List[Tuple[float, SemanticFact]] = []

            for fact in self._facts.values():
                conf = fact.current_confidence()
                if conf < min_confidence:
                    continue
                if category and fact.category != category:
                    continue

                # Token overlap scoring
                fact_tokens = set(
                    re.sub(r"[^\w\s]", "", f"{fact.subject} {fact.predicate}").split()
                )
                overlap = len(query_tokens & fact_tokens)
                if overlap == 0:
                    continue

                # Relevance = token overlap fraction × confidence
                relevance = (overlap / max(len(query_tokens), 1)) * conf
                candidates.append((relevance, fact))

            candidates.sort(key=lambda x: x[0], reverse=True)
            return [fact for _, fact in candidates[:max_results]]

    def query_by_subject(
        self,
        subject: str,
        *,
        min_confidence: float = _RETRIEVAL_THRESHOLD,
    ) -> List[SemanticFact]:
        """Return all facts about a specific subject."""
        subject_norm = subject.strip().lower()
        with self._lock:
            return [
                f for f in self._facts.values()
                if f.subject == subject_norm
                and f.current_confidence() >= min_confidence
            ]

    def refute(self, subject: str, predicate: str, object_: str) -> bool:
        """
        Mark a fact as refuted (decreases confidence, increases refutation count).
        Returns True if the fact was found and refuted.
        """
        fid = _fact_id(
            subject.strip().lower(),
            predicate.strip().lower(),
            object_.strip(),
        )
        with self._lock:
            if fid not in self._facts:
                return False
            fact = self._facts[fid]
            new_conf = max(0.0, fact.confidence - 0.3)
            self._facts[fid] = SemanticFact(
                fact_id=fact.fact_id,
                subject=fact.subject,
                predicate=fact.predicate,
                object=fact.object,
                category=fact.category,
                confidence=new_conf,
                source=fact.source,
                created_at=fact.created_at,
                last_confirmed_at=fact.last_confirmed_at,
                confirmation_count=fact.confirmation_count,
                refutation_count=fact.refutation_count + 1,
                metadata=fact.metadata,
            )
            self._dirty = True
            _logger.info(
                "[SemanticMemory] Refuted: %s→%s=%r (confidence %.2f→%.2f)",
                subject, predicate, object_[:60], fact.confidence, new_conf,
            )
            return True

    def format_for_prompt(self, facts: List[SemanticFact]) -> str:
        """
        Format a list of facts as a concise text block for inclusion in LLM prompts.

        Returns empty string if facts list is empty.
        """
        if not facts:
            return ""
        lines = ["Relevant knowledge from memory:"]
        for f in facts:
            conf_str = f"(confidence: {f.current_confidence():.0%})"
            lines.append(f"  [{f.category}] {f.subject} → {f.predicate}: {f.object} {conf_str}")
        return "\n".join(lines)

    def stats(self) -> dict:
        """Return memory statistics."""
        with self._lock:
            total = len(self._facts)
            by_cat: Dict[str, int] = {}
            usable = 0
            for f in self._facts.values():
                by_cat[f.category] = by_cat.get(f.category, 0) + 1
                if f.is_usable():
                    usable += 1
        return {
            "total_facts": total,
            "usable_facts": usable,
            "by_category": by_cat,
            "memory_path": self._memory_path,
        }

    def save(self) -> None:
        """Force-save memory to disk."""
        self._save_locked()

    # =========================================================================
    # Persistence
    # =========================================================================

    def _load(self) -> None:
        if not os.path.exists(self._memory_path):
            return
        try:
            with open(self._memory_path, "rb") as f:
                data = json.loads(f.read().decode("utf-8"))
            raw_facts = data.get("facts", [])
            loaded = 0
            for raw in raw_facts:
                try:
                    fact = SemanticFact(**raw)
                    self._facts[fact.fact_id] = fact
                    loaded += 1
                except Exception:
                    pass
            _logger.info("[SemanticMemory] Loaded %d facts from %r", loaded, self._memory_path)
        except Exception as exc:
            _logger.warning(
                "[SemanticMemory] Load failed (starting fresh): %s", exc
            )
            self._facts = {}

    def _save_locked(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            facts_data = []
            for fact in self._facts.values():
                try:
                    facts_data.append(asdict(fact))
                except Exception:
                    pass

            payload = json.dumps(
                {"facts": facts_data, "saved_at": time.time()},
                ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")

            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=self._memory_dir, delete=False
                ) as tmp:
                    tmp.write(payload)
                    tmp.flush()
                    os.fsync(tmp.fileno())
                    tmp_path = tmp.name
                os.replace(tmp_path, self._memory_path)
                self._dirty = False
                self._last_save = time.time()
                _logger.debug(
                    "[SemanticMemory] Saved %d facts to %r", len(facts_data), self._memory_path
                )
            except Exception as exc:
                _logger.error("[SemanticMemory] Save failed: %s", exc)
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    def _maybe_auto_save(self) -> None:
        if self._dirty and (time.time() - self._last_save) > self._auto_save_interval:
            self._save_locked()

    def _evict_lowest_confidence_locked(self) -> None:
        """Evict the lowest-confidence facts to stay under max_facts."""
        evict_count = len(self._facts) - self._max_facts + 100
        sorted_ids = sorted(
            self._facts.keys(),
            key=lambda fid: self._facts[fid].current_confidence(),
        )
        for fid in sorted_ids[:evict_count]:
            del self._facts[fid]
        _logger.debug("[SemanticMemory] Evicted %d low-confidence facts.", evict_count)

    def __del__(self):
        try:
            self._save_locked()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_global_semantic_memory: Optional[SemanticMemory] = None
_global_lock = threading.Lock()


def get_global_semantic_memory(memory_dir: Optional[str] = None) -> SemanticMemory:
    """Return the process-singleton SemanticMemory instance."""
    global _global_semantic_memory
    with _global_lock:
        if _global_semantic_memory is None:
            _global_semantic_memory = SemanticMemory(memory_dir=memory_dir)
    return _global_semantic_memory
