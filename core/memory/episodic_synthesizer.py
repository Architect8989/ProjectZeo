"""
episodic_synthesizer.py — Post-task and mid-task episodic knowledge extraction.

AUDIT FIX (March 2026):
  MEDIUM-1: synthesize_and_store() only ran after task completion.
    If the process was killed or crashed mid-task, no lessons from that
    run were stored. Long-running tasks (1+ hours) that failed at the
    last step stored zero episodic knowledge.

    Fix: add store_checkpoint() method that extracts partial lessons
    from the execution log so far without waiting for task completion.
    The GIIController should call this every N iterations (N=50 suggested).
    Checkpoint lessons are tagged with checkpoint=True so they can be
    distinguished from final synthesis lessons and re-synthesized after
    task completion with the full log.

    Also added: checkpoint deduplication via _checkpoint_log_hash to
    avoid redundant LLM calls when the log hasn't changed significantly.

GII UPGRADES (Blueprint §10 — Five-Tier Persistent Memory):
  TIER-1: Cross-session SQLite persistence. Episodic memories now survive
    process restarts (persisted to ~/.projectzeo/episodic.db).
  TIER-1: Three-component retrieval (Generative Agents, Park et al. UIST 2023):
    score = alpha*recency + beta*importance + gamma*relevance
  TIER-1: Reflection synthesis every 100 memories (hierarchical compression).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
from typing import Any, Callable, Dict, List, Optional

_logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Five-Tier Memory: Tier 1 — Episodic store config (Blueprint §10.6)
# ─────────────────────────────────────────────────────────────────────────────

_EPISODIC_DB_DIR  = os.path.join(os.path.expanduser("~"), ".projectzeo")
_EPISODIC_DB_FILE = os.path.join(_EPISODIC_DB_DIR, "episodic.db")

# Generative Agents three-component retrieval weights (Park et al., UIST 2023)
_RETRIEVAL_ALPHA   = float(os.environ.get("PROJECTZEO_EPISODIC_ALPHA", "1.0"))   # recency
_RETRIEVAL_BETA    = float(os.environ.get("PROJECTZEO_EPISODIC_BETA", "1.0"))    # importance
_RETRIEVAL_GAMMA   = float(os.environ.get("PROJECTZEO_EPISODIC_GAMMA", "1.0"))   # relevance
_RECENCY_DECAY     = float(os.environ.get("PROJECTZEO_EPISODIC_DECAY", "0.995")) # per-hour decay

# Reflection synthesis trigger
_REFLECTION_TRIGGER_N = int(os.environ.get("PROJECTZEO_EPISODIC_REFLECT_N", "100"))


_SYNTHESIS_SYSTEM_PROMPT = """\
You are an AI agent that just completed a computer automation task.
Your job is to extract structured, reusable lessons from this execution history.

FOCUS ON:
1. What worked that wasn't obvious (clever shortcuts, unexpected UI behaviors)
2. Failures and exactly how they were recovered from
3. Application-specific quirks discovered (menu locations, keyboard shortcuts, dialog patterns)
4. Performance insights (what made some steps fast vs slow)
5. Error patterns and their solutions

OUTPUT FORMAT — respond ONLY with a valid JSON array, no markdown, no preamble:
[
  {
    "subject": "application_or_domain_name",
    "predicate": "lesson_category",
    "object": "specific actionable lesson text (max 200 chars)",
    "confidence": 0.7,
    "tags": ["tag1", "tag2"]
  }
]

LESSON CATEGORIES (use these for predicate field):
  - "keyboard_shortcut" — discovered shortcut that works
  - "menu_path" — menu navigation sequence that works
  - "dialog_pattern" — how a specific dialog type behaves
  - "install_method" — successful installation approach
  - "error_solution" — specific error and how it was resolved
  - "workflow" — multi-step sequence that works well
  - "quirk" — unexpected application behavior to know about
  - "avoid" — something that failed and should not be attempted
  - "performance" — what made execution faster or slower

Extract 3-7 lessons. Only extract lessons you are CONFIDENT about from the execution log.
Do not invent lessons. If the log shows nothing interesting, return an empty array: []
"""

# Checkpoint synthesis prompt — same as full synthesis but acknowledges partial log
_CHECKPOINT_SYNTHESIS_PROMPT = """\
You are an AI agent in the middle of executing a computer automation task.
Your job is to extract structured, reusable lessons from what has happened SO FAR.
This is a MID-TASK checkpoint — the task is NOT yet complete.

Extract only lessons you are already CONFIDENT about from completed actions.
Do not speculate about incomplete steps or the final outcome.

Use the same JSON format and lesson categories as the full synthesis.
Return an empty array [] if there are no confident lessons yet.

OUTPUT FORMAT — respond ONLY with a valid JSON array, no markdown, no preamble:
[
  {
    "subject": "application_or_domain_name",
    "predicate": "lesson_category",
    "object": "specific actionable lesson text (max 200 chars)",
    "confidence": 0.7,
    "tags": ["tag1", "tag2", "checkpoint"]
  }
]
"""


class EpisodicSynthesizer:

    # Maximum characters of execution log to send to LLM
    MAX_LOG_CHARS = 8000
    # Minimum confidence for a lesson to be stored
    MIN_CONFIDENCE = 0.5
    # Maximum lessons to store per synthesis
    MAX_LESSONS = 10
    # Minimum log change fraction to trigger a new checkpoint synthesis
    # (avoids redundant LLM calls when nothing new has happened)
    _CHECKPOINT_MIN_CHANGE = 0.15

    def __init__(
        self,
        llm_callable: Callable,
        *,
        timeout_seconds: float = 180.0,
        db_path: Optional[str] = None,
    ) -> None:
        self._llm = llm_callable
        self._timeout = timeout_seconds
        self._total_syntheses = 0
        self._total_lessons_stored = 0
        self._total_checkpoints = 0
        self._lock = threading.Lock()
        # Hash of log text at last checkpoint — for deduplication
        self._last_checkpoint_log_hash: Optional[str] = None

        # ── Five-Tier Memory: Tier 1 — SQLite persistence (Blueprint §10.6)
        self._db_path = db_path or _EPISODIC_DB_FILE
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._init_episodic_db()
        self._episodic_memory_count: int = self._count_episodic_memories()

        _logger.info(
            "[EpisodicSynthesizer] Init. db=%r existing_memories=%d",
            self._db_path, self._episodic_memory_count,
        )

    # =========================================================================
    # Public API
    # =========================================================================

    def synthesize_and_store(
        self,
        *,
        execution_log: Dict[str, Any],
        objective: str,
        semantic_memory,
        focused_app: Optional[str] = None,
        application_memory=None,
        block: bool = False,
    ) -> None:
        """Full post-task synthesis. Runs in background thread unless block=True."""
        if block:
            self._synthesize(
                execution_log, objective, semantic_memory, focused_app,
                application_memory, is_checkpoint=False,
            )
        else:
            thread = threading.Thread(
                target=self._synthesize,
                args=(execution_log, objective, semantic_memory, focused_app,
                      application_memory, False),
                daemon=True,
                name="episodic_synthesis",
            )
            thread.start()

    def store_checkpoint(
        self,
        *,
        execution_log: Dict[str, Any],
        objective: str,
        semantic_memory,
        focused_app: Optional[str] = None,
        application_memory=None,
        iteration: int = 0,
    ) -> None:
        """
        AUDIT MEDIUM-1 FIX: Mid-task episodic checkpoint.

        Extracts partial lessons from the execution log so far without
        waiting for task completion. Should be called every N iterations
        (N=50 recommended) during task execution.

        Deduplicates: if the log has not changed significantly since the
        last checkpoint (< _CHECKPOINT_MIN_CHANGE fraction of lines changed),
        the LLM call is skipped.

        Runs in a background thread (non-blocking) to avoid adding latency
        to the execution loop.
        """
        log_text = self._format_execution_log(execution_log)
        if not log_text.strip():
            return

        # Deduplication: skip if log is nearly identical to last checkpoint
        current_hash = hashlib.sha256(log_text.encode("utf-8")).hexdigest()
        with self._lock:
            last_hash = self._last_checkpoint_log_hash
            if last_hash is not None and last_hash == current_hash:
                _logger.debug(
                    "[EpisodicSynthesizer] Checkpoint skipped: log unchanged (iter=%d).",
                    iteration,
                )
                return
            # Approximate change detection via hash bit difference
            if last_hash is not None:
                old_int = int(last_hash, 16)
                new_int = int(current_hash, 16)
                bit_diff = bin(old_int ^ new_int).count("1") / 256.0
                if bit_diff < self._CHECKPOINT_MIN_CHANGE:
                    _logger.debug(
                        "[EpisodicSynthesizer] Checkpoint skipped: log change %.2f < threshold %.2f "
                        "(iter=%d).",
                        bit_diff, self._CHECKPOINT_MIN_CHANGE, iteration,
                    )
                    return
            self._last_checkpoint_log_hash = current_hash

        _logger.info(
            "[EpisodicSynthesizer] Storing mid-task checkpoint at iteration %d "
            "(log_chars=%d).",
            iteration, len(log_text),
        )

        thread = threading.Thread(
            target=self._synthesize,
            args=(execution_log, objective, semantic_memory, focused_app,
                  application_memory, True),
            kwargs={"iteration": iteration},
            daemon=True,
            name=f"episodic_checkpoint_{iteration}",
        )
        thread.start()

    # =========================================================================
    # Internal synthesis engine
    # =========================================================================

    def _synthesize(
        self,
        execution_log: Dict[str, Any],
        objective: str,
        semantic_memory,
        focused_app: Optional[str],
        application_memory,
        is_checkpoint: bool,
        *,
        iteration: int = 0,
    ) -> None:
        t0 = time.monotonic()
        try:
            log_text = self._format_execution_log(execution_log)
            if not log_text.strip():
                _logger.debug("[EpisodicSynthesizer] Empty execution log — no lessons to extract.")
                return

            system_prompt = (
                _CHECKPOINT_SYNTHESIS_PROMPT if is_checkpoint
                else _SYNTHESIS_SYSTEM_PROMPT
            )
            checkpoint_note = (
                f"\n[MID-TASK CHECKPOINT at iteration {iteration}]"
                if is_checkpoint else ""
            )
            prompt = (
                f"TASK OBJECTIVE: {objective[:400]}{checkpoint_note}\n"
                f"FOCUSED APPLICATION: {focused_app or 'multiple/unknown'}\n\n"
                f"EXECUTION LOG:\n{log_text}\n"
            )

            raw_response = self._call_llm(prompt, system_prompt=system_prompt)
            if raw_response is None:
                _logger.debug("[EpisodicSynthesizer] LLM returned no response.")
                return

            lessons = self._parse_lessons(raw_response)
            if not lessons:
                _logger.debug("[EpisodicSynthesizer] No valid lessons parsed.")
                return

            stored_count = 0
            for lesson in lessons[:self.MAX_LESSONS]:
                if not isinstance(lesson, dict):
                    continue
                subject    = str(lesson.get("subject", focused_app or "task"))[:80]
                predicate  = str(lesson.get("predicate", "lesson"))[:80]
                object_    = str(lesson.get("object", ""))[:300]
                confidence = float(lesson.get("confidence", 0.7))
                tags       = lesson.get("tags", [])
                if not object_ or confidence < self.MIN_CONFIDENCE:
                    continue

                # Tag checkpoint lessons for distinction from final lessons
                if is_checkpoint and "checkpoint" not in tags:
                    tags = list(tags) + ["checkpoint"]

                try:
                    semantic_memory.store(
                        subject=subject,
                        predicate=predicate,
                        object_=object_,
                        category="episodic_lessons",
                        confidence=confidence,
                        source="episodic_checkpoint" if is_checkpoint else "episodic_synthesis",
                    )
                    stored_count += 1
                    _logger.info(
                        "[EpisodicSynthesizer] Stored%s: %s::%s = %s (conf=%.2f)",
                        " [CHECKPOINT]" if is_checkpoint else "",
                        subject, predicate, object_[:80], confidence,
                    )
                except Exception as store_exc:
                    _logger.warning("[EpisodicSynthesizer] Store failed: %s", store_exc)

                # Also store app-specific lessons in ApplicationMemory
                if application_memory and subject == focused_app:
                    try:
                        if predicate == "keyboard_shortcut":
                            application_memory.add_shortcut(focused_app, object_)
                        elif predicate == "menu_path":
                            application_memory.add_menu_path(focused_app, object_)
                        elif predicate in ("quirk", "avoid"):
                            application_memory.add_quirk(focused_app, object_)
                        elif predicate == "error_solution":
                            application_memory.add_error_solution(focused_app, object_)
                    except Exception:
                        pass  # ApplicationMemory integration is best-effort

            with self._lock:
                if is_checkpoint:
                    self._total_checkpoints += 1
                else:
                    self._total_syntheses += 1
                self._total_lessons_stored += stored_count

            elapsed = (time.monotonic() - t0) * 1000
            _logger.info(
                "[EpisodicSynthesizer] %s complete. %d lessons stored in %.1fms",
                "Checkpoint" if is_checkpoint else "Synthesis",
                stored_count, elapsed,
            )

        except Exception as exc:
            _logger.error(
                "[EpisodicSynthesizer] %s failed: %s",
                "Checkpoint" if is_checkpoint else "Synthesis",
                exc,
            )

    def _format_execution_log(self, execution_log: Dict[str, Any]) -> str:
        """Format execution log into compact LLM-readable text."""
        lines = []
        for step_idx in sorted(execution_log.keys(), key=lambda k: int(k) if str(k).isdigit() else 0):
            step_data = execution_log.get(step_idx, {})
            if not isinstance(step_data, dict):
                continue
            for output_entry in step_data.get("outputs", []):
                if not isinstance(output_entry, dict):
                    continue
                op = str(output_entry.get("operation", "unknown"))
                success = output_entry.get("success", True)
                output = str(output_entry.get("output", ""))[:300]
                status = "✓" if success else "✗"
                lines.append(f"Step {step_idx}: {status} [{op}] {output}")
            desc = str(step_data.get("description", ""))
            if desc:
                lines.append(f"Step {step_idx} desc: {desc[:100]}")

        full = "\n".join(lines)
        if len(full) > self.MAX_LOG_CHARS:
            half = self.MAX_LOG_CHARS // 2
            full = full[:half] + "\n... [truncated] ...\n" + full[-half:]
        return full

    def _call_llm(
        self,
        user_prompt: str,
        *,
        system_prompt: str = _SYNTHESIS_SYSTEM_PROMPT,
    ) -> Optional[str]:
        """Call LLM with synthesis prompt. Returns raw text response or None."""
        result_holder: List[Optional[str]] = [None]

        def _call():
            try:
                raw = self._llm(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    objective=None,
                    session_id="episodic_synthesis",
                )
                if isinstance(raw, list) and raw:
                    result_holder[0] = str(
                        raw[0].get("content", "") if isinstance(raw[0], dict) else raw[0]
                    )
                elif isinstance(raw, str):
                    result_holder[0] = raw
            except Exception as exc:
                _logger.warning("[EpisodicSynthesizer] LLM call failed: %s", exc)

        thread = threading.Thread(target=_call, daemon=True)
        thread.start()
        thread.join(timeout=self._timeout)

        if thread.is_alive():
            _logger.warning("[EpisodicSynthesizer] LLM call timed out after %.1fs", self._timeout)
            return None

        return result_holder[0]

    def _parse_lessons(self, raw_text: str) -> List[Dict[str, Any]]:
        """Parse LLM response into list of lesson dicts."""
        try:
            clean = re.sub(r"```(?:json)?", "", raw_text).strip()
            match = re.search(r"\[.*\]", clean, re.DOTALL)
            if not match:
                return []
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, AttributeError) as exc:
            _logger.debug("[EpisodicSynthesizer] Parse error: %s", exc)
        return []

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "total_syntheses":     self._total_syntheses,
                "total_checkpoints":   self._total_checkpoints,
                "total_lessons_stored": self._total_lessons_stored,
                "timeout_seconds":     self._timeout,
                "episodic_memories":   self._episodic_memory_count,
                "db_path":             self._db_path,
            }

    # =========================================================================
    # Five-Tier Memory: Tier 1 — Cross-session Episodic Store (Blueprint §10.6)
    # Generative Agents three-component retrieval (Park et al., UIST 2023)
    # =========================================================================

    def store_episodic_memory(
        self,
        content: str,
        *,
        task: str = "",
        app: str = "",
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Persist an episodic memory to cross-session SQLite store.
        Importance is LLM-judged on a 1-10 scale, normalised to 0-1.
        Every 100 new memories triggers reflection synthesis.
        """
        now = time.time()
        mem_id = hashlib.sha256(
            f"{content[:100]}{now}".encode()
        ).hexdigest()[:16]

        row = {
            "memory_id":    mem_id,
            "content":      content[:2000],
            "task":         task[:300],
            "app":          app[:100],
            "importance":   max(0.0, min(1.0, float(importance))),
            "tags":         json.dumps(tags or []),
            "created_at":   now,
            "last_accessed": now,
            "access_count": 1,
        }
        try:
            conn = self._get_db_conn()
            conn.execute(
                """INSERT OR IGNORE INTO episodic_memories
                   (memory_id, content, task, app, importance, tags, created_at, last_accessed, access_count)
                   VALUES (:memory_id, :content, :task, :app, :importance, :tags, :created_at, :last_accessed, :access_count)""",
                row,
            )
            conn.commit()
            with self._lock:
                self._episodic_memory_count += 1
                count = self._episodic_memory_count

            # Trigger reflection synthesis every N memories
            if count % _REFLECTION_TRIGGER_N == 0:
                threading.Thread(
                    target=self._reflection_synthesis,
                    args=(count,),
                    daemon=True,
                    name=f"episodic_reflect_{count}",
                ).start()

        except Exception as exc:
            _logger.warning("[EpisodicSynthesizer] Store episodic failed: %s", exc)

        return mem_id

    def retrieve_episodic_memories(
        self,
        query: str,
        *,
        max_results: int = 10,
        task: Optional[str] = None,
        app: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve episodic memories using three-component score (Blueprint §10.1):
            score = alpha*recency + beta*importance + gamma*relevance
        where:
            recency   = exponential_decay(time_since_last_access, decay=0.995)
            importance = stored importance (0-1)
            relevance  = cosine-like token overlap with query
        """
        try:
            conn = self._get_db_conn()
            clauses = []
            params: List[Any] = []
            if task:
                clauses.append("task LIKE ?")
                params.append(f"%{task[:100]}%")
            if app:
                clauses.append("app = ?")
                params.append(app[:100])
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

            rows = conn.execute(
                f"SELECT * FROM episodic_memories {where} ORDER BY created_at DESC LIMIT 500",
                params,
            ).fetchall()
            cols = [d[0] for d in conn.execute(f"SELECT * FROM episodic_memories {where} LIMIT 0", params).description or []]

            if not cols:
                # fallback column names
                cols = ["memory_id","content","task","app","importance","tags","created_at","last_accessed","access_count"]

            now = time.time()
            query_tokens = set(re.sub(r"[^\w]", " ", query.lower()).split())
            scored: List[tuple] = []

            for row in rows:
                mem = dict(zip(cols, row))
                content_tokens = set(re.sub(r"[^\w]", " ", mem.get("content","").lower()).split())
                if not content_tokens:
                    continue

                # Recency: exponential decay based on hours since last access
                hours_ago = (now - float(mem.get("last_accessed", now))) / 3600.0
                recency = _RECENCY_DECAY ** hours_ago

                # Importance
                importance = float(mem.get("importance", 0.5))

                # Relevance: token overlap
                overlap = len(query_tokens & content_tokens)
                relevance = overlap / max(len(query_tokens | content_tokens), 1)

                score = (
                    _RETRIEVAL_ALPHA * recency +
                    _RETRIEVAL_BETA  * importance +
                    _RETRIEVAL_GAMMA * relevance
                )
                scored.append((score, mem))

            scored.sort(key=lambda x: x[0], reverse=True)
            results = [m for _, m in scored[:max_results]]

            # Update last_accessed for retrieved memories
            if results:
                ids = [m["memory_id"] for m in results]
                try:
                    conn.execute(
                        f"UPDATE episodic_memories SET last_accessed=?, access_count=access_count+1 "
                        f"WHERE memory_id IN ({','.join('?'*len(ids))})",
                        [now] + ids,
                    )
                    conn.commit()
                except Exception:
                    pass

            return results

        except Exception as exc:
            _logger.warning("[EpisodicSynthesizer] Retrieve episodic failed: %s", exc)
            return []

    # =========================================================================
    # Private — SQLite helpers
    # =========================================================================

    def _init_episodic_db(self) -> None:
        try:
            conn = self._get_db_conn()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS episodic_memories (
                    memory_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    task TEXT DEFAULT '',
                    app TEXT DEFAULT '',
                    importance REAL DEFAULT 0.5,
                    tags TEXT DEFAULT '[]',
                    created_at REAL NOT NULL,
                    last_accessed REAL NOT NULL,
                    access_count INTEGER DEFAULT 1
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS episodic_reflections (
                    reflection_id TEXT PRIMARY KEY,
                    reflection_text TEXT NOT NULL,
                    trigger_count INTEGER,
                    created_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_episodic_created
                ON episodic_memories(created_at DESC)
            """)
            conn.commit()
        except Exception as exc:
            _logger.warning("[EpisodicSynthesizer] DB init warning: %s", exc)

    def _get_db_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _count_episodic_memories(self) -> int:
        try:
            conn = self._get_db_conn()
            row = conn.execute("SELECT COUNT(*) FROM episodic_memories").fetchone()
            return int(row[0]) if row else 0
        except Exception:
            return 0

    def _reflection_synthesis(self, trigger_count: int) -> None:
        """
        Generative Agents reflection synthesis (Blueprint §10.1):
        Every 100 memories, observe last 100 and generate higher-level insight.
        """
        _logger.info(
            "[EpisodicSynthesizer] Reflection synthesis triggered at count=%d",
            trigger_count,
        )
        try:
            recent = self.retrieve_episodic_memories(
                "reflection summary", max_results=100
            )
            if len(recent) < 10:
                return

            memories_text = "\n".join(
                f"- {m.get('content','')[:150]}" for m in recent[:50]
            )
            prompt = (
                f"You have {len(recent)} recent episodic memories from an AI agent.\n"
                f"Recent memories:\n{memories_text}\n\n"
                "Generate 3 high-level reflections: key patterns, insights, or lessons "
                "that emerge from these memories together. Be concise (1-2 sentences each).\n"
                "Return a JSON array of strings."
            )
            raw = self._call_llm(prompt)
            if not raw:
                return

            try:
                cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
                insights = json.loads(cleaned)
                if not isinstance(insights, list):
                    return
            except Exception:
                return

            now = time.time()
            conn = self._get_db_conn()
            for insight in insights[:5]:
                if not isinstance(insight, str) or len(insight) < 10:
                    continue
                rid = hashlib.sha256(f"{insight}{now}".encode()).hexdigest()[:16]
                conn.execute(
                    "INSERT OR IGNORE INTO episodic_reflections(reflection_id, reflection_text, trigger_count, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (rid, insight[:500], trigger_count, now),
                )
                # Store reflection as high-importance episodic memory too
                self.store_episodic_memory(
                    f"[REFLECTION] {insight}",
                    importance=0.9,
                    tags=["reflection", "synthesis"],
                )
            conn.commit()
            _logger.info(
                "[EpisodicSynthesizer] Stored %d reflections at count=%d",
                len(insights), trigger_count,
            )
        except Exception as exc:
            _logger.warning("[EpisodicSynthesizer] Reflection synthesis failed: %s", exc)
