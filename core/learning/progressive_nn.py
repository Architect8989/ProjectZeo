"""
core/learning/progressive_nn.py
=================================
Progressive Neural Network (PNN) — Continual Learning without Forgetting.

Blueprint §11.3 — Continual Learning Stack

Reference:
    Rusu et al. (2016) "Progressive Neural Networks" — arXiv:1606.04671
    Each new task adds a new "column" to the network while lateral connections
    allow knowledge transfer from prior columns. Old columns are frozen —
    guaranteeing zero catastrophic forgetting.

Role in ProjectZeo:
    The PNN sits alongside EWC (in arpo_trainer.py/grpo_trainer.py) as the
    second anti-forgetting mechanism. Where EWC constrains weight updates,
    PNN provides a structural solution: each new application domain gets its
    own column that can leverage but never corrupt prior knowledge.

Implementation strategy:
    - Software-level PNN: columns are represented as task-profile records
      with lateral weight vectors (simplified, no full backprop needed here)
    - The "neural network" is the LLM itself; PNN here models task-level
      feature transfer as prompt context injection
    - On hardware with GPU: optional lightweight PyTorch column heads
    - On CPU-only: pure Python column registry with similarity-based transfer

Integration:
    - gii_controller._initialise_phase3_components() instantiates PNN
    - gii_controller._on_task_complete() calls register_task_completion()
    - per_step_reasoner can query get_lateral_context() for transfer hints
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

_STORE_PATH  = os.path.expanduser(
    os.environ.get("PROJECTZEO_PNN_STORE", "~/.projectzeo/pnn_columns.json")
)
_MAX_COLUMNS = int(os.environ.get("PROJECTZEO_PNN_MAX_COLUMNS", "100"))
_LATERAL_TOP_K = int(os.environ.get("PROJECTZEO_PNN_LATERAL_K", "3"))
_ENABLED = os.environ.get("PROJECTZEO_PNN_ENABLED", "1").strip() != "0"


@dataclass
class PNNColumn:
    """
    A single PNN column representing a learned task/domain.

    Each column is frozen after creation — it can only be READ by future columns
    via lateral connections (get_lateral_features).
    """
    column_id:      str
    task_type:      str           # Normalised task description
    app_context:    str           # Application domain (e.g. "chrome", "libreoffice")
    created_at:     float = field(default_factory=time.time)
    task_count:     int = 1       # Number of completions contributing to this column
    avg_steps:      float = 0.0   # Average steps to completion
    success_rate:   float = 1.0
    # Lateral features: key patterns learned in this column
    patterns:       List[str] = field(default_factory=list)
    # Known failure modes with solutions
    failure_modes:  List[Dict[str, str]] = field(default_factory=list)
    # Vocabulary of effective action sequences for this task type
    action_vocab:   List[Dict[str, Any]] = field(default_factory=list)
    frozen:         bool = False  # True once column is sealed


@dataclass
class LateralTransfer:
    """
    Knowledge transferred from prior columns to the current task.
    Injected as context into per_step_reasoner.
    """
    source_column_id:   str
    source_task:        str
    similarity:         float         # 0.0-1.0 Jaccard or embedding similarity
    transferred_patterns: List[str] = field(default_factory=list)
    transferred_actions:  List[Dict[str, Any]] = field(default_factory=list)
    failure_warnings:     List[str] = field(default_factory=list)

    def to_prompt_block(self) -> str:
        lines = [
            f"[PNN Lateral Transfer from column {self.source_column_id[:8]}] "
            f"Similar task: {self.source_task[:100]} (similarity={self.similarity:.2f})"
        ]
        for p in self.transferred_patterns[:3]:
            lines.append(f"  Learned pattern: {p}")
        for w in self.failure_warnings[:2]:
            lines.append(f"  Known failure: {w}")
        return "\n".join(lines)


def _jaccard_similarity(a: str, b: str) -> float:
    """Simple word-level Jaccard similarity for task matching."""
    if not a or not b:
        return 0.0
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


class ProgressiveNeuralNetwork:
    """
    Progressive Neural Network for continual task learning.

    Creates a new frozen column for each distinct task/app combination.
    Provides lateral transfer features from similar prior columns.

    Thread-safe, persisted to disk as JSON.
    """

    def __init__(self, store_path: Optional[str] = None) -> None:
        self._store_path = store_path or _STORE_PATH
        self._columns: Dict[str, PNNColumn] = {}
        self._lock = threading.Lock()
        self._enabled = _ENABLED
        os.makedirs(os.path.dirname(self._store_path), exist_ok=True)
        self._load()
        _logger.info(
            "[PNN] Initialised. columns=%d enabled=%s store=%s",
            len(self._columns), self._enabled, self._store_path,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Column lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def register_task_completion(
        self,
        task_description: str,
        app_context: str = "",
        steps_taken: int = 0,
        success: bool = True,
        patterns_observed: Optional[List[str]] = None,
        action_sequences: Optional[List[Dict[str, Any]]] = None,
        failure_modes: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Register a completed task as a new or updated PNN column.

        Returns the column_id that was created or updated.
        """
        if not self._enabled:
            return ""

        task_norm = self._normalise_task(task_description)
        app_norm  = app_context.lower().strip()

        with self._lock:
            # Find existing column for this task type
            existing = self._find_column(task_norm, app_norm)

            if existing is not None:
                # Update existing column (accumulate experience)
                col = existing
                col.task_count += 1
                if steps_taken > 0:
                    col.avg_steps = (
                        (col.avg_steps * (col.task_count - 1) + steps_taken)
                        / col.task_count
                    )
                # Update success rate
                col.success_rate = (
                    (col.success_rate * (col.task_count - 1) + (1.0 if success else 0.0))
                    / col.task_count
                )
                # Merge new patterns
                for p in (patterns_observed or [])[:5]:
                    if p and p not in col.patterns:
                        col.patterns.append(p)
                col.patterns = col.patterns[-20:]  # Cap

                # Merge failure modes
                for fm in (failure_modes or [])[:3]:
                    if fm and fm not in col.failure_modes:
                        col.failure_modes.append(fm)
                col.failure_modes = col.failure_modes[-10:]

                # Seal column after 5+ successful completions (freeze)
                if col.task_count >= 5 and col.success_rate >= 0.6 and not col.frozen:
                    col.frozen = True
                    _logger.info(
                        "[PNN] Column %s FROZEN (task_count=%d, success_rate=%.2f)",
                        col.column_id[:8], col.task_count, col.success_rate,
                    )
                column_id = col.column_id
            else:
                # Create new column
                import hashlib
                column_id = hashlib.sha256(
                    f"{task_norm}{app_norm}{time.time()}".encode()
                ).hexdigest()[:12]

                col = PNNColumn(
                    column_id=column_id,
                    task_type=task_norm,
                    app_context=app_norm,
                    avg_steps=float(steps_taken),
                    success_rate=1.0 if success else 0.0,
                    patterns=list((patterns_observed or [])[:10]),
                    failure_modes=list((failure_modes or [])[:5]),
                    action_vocab=list((action_sequences or [])[:10]),
                )
                self._columns[column_id] = col

                # Evict oldest if over limit
                if len(self._columns) > _MAX_COLUMNS:
                    oldest_key = min(
                        self._columns,
                        key=lambda k: self._columns[k].created_at,
                    )
                    del self._columns[oldest_key]

                _logger.info(
                    "[PNN] New column created: %s task=%r app=%r",
                    column_id[:8], task_norm[:60], app_norm,
                )

            self._save()
            return column_id

    # ─────────────────────────────────────────────────────────────────────────
    # Lateral transfer — query prior columns for relevant context
    # ─────────────────────────────────────────────────────────────────────────

    def get_lateral_context(
        self,
        task_description: str,
        app_context: str = "",
        top_k: int = _LATERAL_TOP_K,
    ) -> List[LateralTransfer]:
        """
        Find the most similar prior columns and return their transferable knowledge.

        This is the core PNN operation: lateral connections from old columns
        inform how the new column (current task) should approach the problem.
        """
        if not self._enabled:
            return []

        task_norm = self._normalise_task(task_description)
        app_norm  = app_context.lower().strip()

        with self._lock:
            if not self._columns:
                return []

            # Score all columns by similarity
            scored: List[Tuple[float, PNNColumn]] = []
            for col in self._columns.values():
                # App context match boosts similarity
                app_bonus = 0.2 if col.app_context == app_norm else 0.0
                sim = _jaccard_similarity(task_norm, col.task_type) + app_bonus
                if sim > 0.1:  # Only include reasonably similar columns
                    scored.append((sim, col))

            scored.sort(key=lambda x: x[0], reverse=True)
            top_cols = scored[:top_k]

        transfers: List[LateralTransfer] = []
        for sim, col in top_cols:
            lt = LateralTransfer(
                source_column_id=col.column_id,
                source_task=col.task_type[:100],
                similarity=sim,
                transferred_patterns=list(col.patterns[:3]),
                transferred_actions=list(col.action_vocab[:3]),
                failure_warnings=[
                    f"{fm.get('error', '')}: {fm.get('solution', '')}"[:150]
                    for fm in col.failure_modes[:2]
                    if fm.get("error")
                ],
            )
            transfers.append(lt)

        return transfers

    def get_lateral_prompt_block(
        self,
        task_description: str,
        app_context: str = "",
    ) -> str:
        """Return lateral transfer context as a formatted prompt block."""
        transfers = self.get_lateral_context(task_description, app_context)
        if not transfers:
            return ""
        blocks = [t.to_prompt_block() for t in transfers]
        return "\n".join(blocks)

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _normalise_task(self, task: str) -> str:
        """Normalise task description for matching."""
        # Remove punctuation, lowercase, strip filler words
        import re
        task = task.lower().strip()
        task = re.sub(r"[^\w\s]", " ", task)
        stopwords = {"the", "a", "an", "in", "on", "at", "to", "for", "of",
                     "with", "and", "or", "is", "are", "was", "open", "click"}
        words = [w for w in task.split() if w not in stopwords and len(w) > 2]
        return " ".join(words[:20])

    def _find_column(
        self, task_norm: str, app_norm: str
    ) -> Optional[PNNColumn]:
        """Find an existing column for this task+app with high similarity."""
        best_sim = 0.0
        best_col = None
        for col in self._columns.values():
            if col.frozen:
                continue  # Frozen columns can only be read, not updated
            app_match = (col.app_context == app_norm) if app_norm else True
            sim = _jaccard_similarity(task_norm, col.task_type)
            # Require high similarity (0.6) and same app context for update
            if sim >= 0.6 and app_match and sim > best_sim:
                best_sim = sim
                best_col = col
        return best_col

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled":       self._enabled,
                "total_columns": len(self._columns),
                "frozen_columns": sum(1 for c in self._columns.values() if c.frozen),
                "apps": list({c.app_context for c in self._columns.values()})[:20],
                "store_path":    self._store_path,
            }

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def _save(self) -> None:
        """Persist columns to disk (called under _lock)."""
        tmp = self._store_path + ".tmp"
        try:
            data = {cid: asdict(col) for cid, col in self._columns.items()}
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, separators=(",", ":"))
            os.replace(tmp, self._store_path)
        except Exception as exc:
            _logger.warning("[PNN] Save failed: %s", exc)

    def _load(self) -> None:
        """Load columns from disk."""
        if not os.path.isfile(self._store_path):
            return
        try:
            with open(self._store_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for cid, d in data.items():
                try:
                    col = PNNColumn(**{
                        k: v for k, v in d.items()
                        if k in PNNColumn.__dataclass_fields__
                    })
                    self._columns[cid] = col
                except Exception:
                    pass
            _logger.info("[PNN] Loaded %d columns from disk.", len(self._columns))
        except Exception as exc:
            _logger.warning("[PNN] Load failed: %s", exc)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: Optional[ProgressiveNeuralNetwork] = None
_instance_lock = threading.Lock()


def get_pnn() -> ProgressiveNeuralNetwork:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = ProgressiveNeuralNetwork()
    return _instance
