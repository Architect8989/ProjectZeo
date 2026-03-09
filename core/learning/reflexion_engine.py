"""
core/learning/reflexion_engine.py
==================================
Reflexion: Verbal Reinforcement Learning via Per-Failure Reflection.

Blueprint §8.1 — Shinn, Cassano et al., NeurIPS 2023 (arXiv:2303.11366)

Architecture:
    After each milestone failure, the agent generates a verbal self-reflection
    stored in episodic memory. The next attempt begins with the last N reflections
    injected as context. Implements all three strategies (NONE, REFLEXION,
    LAST_ATTEMPT_AND_REFLEXION) with LAST_ATTEMPT_AND_REFLEXION as default.

Key results from paper:
    - 91% pass@1 on HumanEval coding (GPT-4: 80%)
    - 130/134 AlfWorld tasks (vs 109/134 without reflection)

Integration:
    - gii_loop.py → call reflect_on_failure() on milestone failure
    - per_step_reasoner.py → call inject_context() before each step
    - Chain of Hindsight: inject improving attempt sequence per milestone type

SAGE extension (§8.4):
    Structured ApplicationProfile updates alongside verbal reflection:
        lessons_learned, known_quirks, failed_approaches, verified_workflows
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
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Tunables
# ─────────────────────────────────────────────────────────────────────────────

_MAX_REFLECTIONS_PER_MILESTONE = int(
    os.environ.get("PROJECTZEO_REFLEXION_MAX_PER_MILESTONE", "5")
)
_MAX_CONTEXT_INJECT = int(
    os.environ.get("PROJECTZEO_REFLEXION_CONTEXT_SIZE", "3")
)
_REFLECTION_TIMEOUT = float(
    os.environ.get("PROJECTZEO_REFLEXION_TIMEOUT", "60.0")
)
_DB_DIR = os.path.join(os.path.expanduser("~"), ".projectzeo")
_DB_FILE = os.path.join(_DB_DIR, "reflexion.db")


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class ReflexionStrategy(str, Enum):
    NONE                        = "none"
    REFLEXION                   = "reflexion"
    LAST_ATTEMPT_AND_REFLEXION  = "last_attempt_and_reflexion"  # RECOMMENDED


class AttemptOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReflexionEntry:
    """A single verbal reflection stored after a failure."""
    entry_id:           str
    milestone_key:      str       # hash(milestone_description)
    milestone_desc:     str
    attempt_number:     int
    outcome:            AttemptOutcome
    trajectory_summary: str       # What was done
    failure_reason:     str       # Why it failed
    belief_state_summary: str     # World state at failure
    reflection_text:    str       # The verbal reflection
    lessons:            List[str] = field(default_factory=list)  # SAGE structured
    created_at:         float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id":           self.entry_id,
            "milestone_key":      self.milestone_key,
            "milestone_desc":     self.milestone_desc[:300],
            "attempt_number":     self.attempt_number,
            "outcome":            self.outcome.value,
            "trajectory_summary": self.trajectory_summary[:500],
            "failure_reason":     self.failure_reason[:300],
            "reflection_text":    self.reflection_text[:800],
            "lessons":            self.lessons[:10],
            "created_at":         self.created_at,
        }


@dataclass
class ApplicationProfile:
    """SAGE §8.4 — structured knowledge base per application."""
    app_name:          str
    lessons_learned:   List[str] = field(default_factory=list)   # Reflexion lessons
    known_quirks:      List[str] = field(default_factory=list)   # SAGE insights
    failed_approaches: List[str] = field(default_factory=list)   # CoH anti-patterns
    verified_workflows:List[str] = field(default_factory=list)   # Confirmed patterns
    last_updated:      float = field(default_factory=time.time)


# ─────────────────────────────────────────────────────────────────────────────
# ReflexionEngine
# ─────────────────────────────────────────────────────────────────────────────

class ReflexionEngine:
    """
    Verbal reinforcement learning engine.

    Usage:
        engine = ReflexionEngine(llm_caller=my_llm)

        # On milestone failure:
        await engine.reflect_on_failure(
            milestone_desc="Open spreadsheet app",
            trajectory_summary="Clicked Apps menu, searched 'spreadsheet'",
            failure_reason="Found LibreOffice Calc but double-click had no effect",
            belief_state_summary="Focus on desktop, LibreOffice icon visible",
        )

        # Before next reasoning step:
        context = engine.inject_context(milestone_desc="Open spreadsheet app")
        # → inject context str into per_step_reasoner prompt
    """

    def __init__(
        self,
        *,
        llm_caller: Optional[Callable] = None,
        strategy: ReflexionStrategy = ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION,
        db_path: Optional[str] = None,
        max_reflections_per_milestone: int = _MAX_REFLECTIONS_PER_MILESTONE,
        max_context_inject: int = _MAX_CONTEXT_INJECT,
    ) -> None:
        self._llm_caller = llm_caller
        self._strategy = strategy
        self._max_per_milestone = max_reflections_per_milestone
        self._max_context = max_context_inject
        self._lock = threading.RLock()

        # In-memory cache: milestone_key → list[ReflexionEntry]
        self._cache: Dict[str, List[ReflexionEntry]] = {}
        # ApplicationProfiles: app_name → ApplicationProfile
        self._app_profiles: Dict[str, ApplicationProfile] = {}
        # Attempt counters: milestone_key → int
        self._attempt_counters: Dict[str, int] = {}

        # Persistent SQLite store
        self._db_path = db_path or _DB_FILE
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._init_db()

        _logger.info(
            "[ReflexionEngine] Initialised. strategy=%s db=%r",
            strategy.value, self._db_path,
        )

    # =========================================================================
    # Public API
    # =========================================================================

    def reflect_on_failure(
        self,
        milestone_desc: str,
        trajectory_summary: str,
        failure_reason: str,
        belief_state_summary: str = "",
        app_name: Optional[str] = None,
    ) -> Optional[ReflexionEntry]:
        """
        Generate a verbal reflection after a milestone failure.
        Stores result persistently and updates ApplicationProfile.

        Returns the ReflexionEntry or None if LLM call fails.
        """
        if self._strategy == ReflexionStrategy.NONE:
            return None

        m_key = _milestone_key(milestone_desc)
        with self._lock:
            attempt_num = self._attempt_counters.get(m_key, 0) + 1
            self._attempt_counters[m_key] = attempt_num

        reflection_text = self._generate_reflection(
            milestone_desc=milestone_desc,
            trajectory_summary=trajectory_summary,
            failure_reason=failure_reason,
            belief_state_summary=belief_state_summary,
            attempt_num=attempt_num,
        )

        lessons = self._extract_lessons(reflection_text)

        entry = ReflexionEntry(
            entry_id=_entry_id(m_key, attempt_num),
            milestone_key=m_key,
            milestone_desc=milestone_desc[:300],
            attempt_number=attempt_num,
            outcome=AttemptOutcome.FAILURE,
            trajectory_summary=trajectory_summary[:500],
            failure_reason=failure_reason[:300],
            belief_state_summary=belief_state_summary[:300],
            reflection_text=reflection_text,
            lessons=lessons,
        )

        with self._lock:
            if m_key not in self._cache:
                self._cache[m_key] = []
            self._cache[m_key].append(entry)
            # Keep bounded
            if len(self._cache[m_key]) > self._max_per_milestone:
                self._cache[m_key] = self._cache[m_key][-self._max_per_milestone:]

        # Persist
        self._persist_entry(entry)

        # SAGE: update ApplicationProfile
        if app_name:
            self._update_app_profile(app_name, lessons, failure_reason)

        _logger.info(
            "[ReflexionEngine] Reflection stored: milestone_key=%r attempt=%d",
            m_key, attempt_num,
        )
        return entry

    def record_success(
        self,
        milestone_desc: str,
        trajectory_summary: str,
        app_name: Optional[str] = None,
    ) -> None:
        """Record a successful milestone for Chain of Hindsight context."""
        m_key = _milestone_key(milestone_desc)
        with self._lock:
            attempt_num = self._attempt_counters.get(m_key, 0) + 1
            self._attempt_counters[m_key] = attempt_num

        entry = ReflexionEntry(
            entry_id=_entry_id(m_key, attempt_num),
            milestone_key=m_key,
            milestone_desc=milestone_desc[:300],
            attempt_number=attempt_num,
            outcome=AttemptOutcome.SUCCESS,
            trajectory_summary=trajectory_summary[:500],
            failure_reason="",
            belief_state_summary="",
            reflection_text="[SUCCESS] No reflection needed.",
            lessons=[],
        )
        with self._lock:
            if m_key not in self._cache:
                self._cache[m_key] = []
            self._cache[m_key].append(entry)

        # SAGE: update verified workflow
        if app_name:
            self._update_verified_workflow(app_name, trajectory_summary)

        self._persist_entry(entry)

    def inject_context(
        self,
        milestone_desc: str,
        strategy: Optional[ReflexionStrategy] = None,
    ) -> str:
        """
        Build the reflection context string to inject into per_step_reasoner.

        Uses the configured strategy (default: LAST_ATTEMPT_AND_REFLEXION):
          - NONE: empty string
          - REFLEXION: last N reflections
          - LAST_ATTEMPT_AND_REFLEXION: last trajectory + reflections
        """
        strat = strategy or self._strategy
        if strat == ReflexionStrategy.NONE:
            return ""

        m_key = _milestone_key(milestone_desc)
        with self._lock:
            entries = list(self._cache.get(m_key, []))

        # Also load from DB if cache is empty (cross-session)
        if not entries:
            entries = self._load_entries_from_db(m_key)
            with self._lock:
                self._cache[m_key] = entries

        if not entries:
            return ""

        # Only use failure entries for reflection
        failures = [e for e in entries if e.outcome == AttemptOutcome.FAILURE]
        if not failures:
            return ""

        recent = failures[-self._max_context:]

        parts = [
            f"=== REFLEXION CONTEXT ({len(recent)} previous attempt(s)) ===",
        ]

        if strat == ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION:
            # Include full trajectory of last attempt
            last = recent[-1]
            parts.append(
                f"\nLast attempt #{last.attempt_number} trajectory:\n"
                f"{last.trajectory_summary}\n"
                f"Failed because: {last.failure_reason}"
            )

        parts.append("\nReflections from previous failures:")
        for r in recent:
            parts.append(
                f"  [Attempt {r.attempt_number}] {r.reflection_text}"
            )

        # Extract all lessons
        all_lessons = []
        for r in recent:
            all_lessons.extend(r.lessons)
        if all_lessons:
            parts.append("\nKey lessons:")
            for i, lesson in enumerate(all_lessons[-6:], 1):
                parts.append(f"  {i}. {lesson}")

        parts.append("=== END REFLEXION CONTEXT ===\n")
        return "\n".join(parts)

    def get_app_profile(self, app_name: str) -> ApplicationProfile:
        """Return the ApplicationProfile for an application."""
        with self._lock:
            if app_name not in self._app_profiles:
                profile = self._load_app_profile_from_db(app_name)
                self._app_profiles[app_name] = profile or ApplicationProfile(app_name=app_name)
            return self._app_profiles[app_name]

    def get_app_profile_for_prompt(self, app_name: str) -> str:
        """Format ApplicationProfile as a prompt-injectable string."""
        profile = self.get_app_profile(app_name)
        lines = []
        if profile.lessons_learned:
            lines.append(f"Lessons for {app_name}:")
            for l in profile.lessons_learned[-4:]:
                lines.append(f"  • {l}")
        if profile.known_quirks:
            lines.append(f"Known quirks:")
            for q in profile.known_quirks[-3:]:
                lines.append(f"  ! {q}")
        if profile.failed_approaches:
            lines.append(f"Do NOT try:")
            for fa in profile.failed_approaches[-3:]:
                lines.append(f"  ✗ {fa}")
        if profile.verified_workflows:
            lines.append(f"Verified approaches:")
            for vw in profile.verified_workflows[-3:]:
                lines.append(f"  ✓ {vw}")
        return "\n".join(lines) if lines else ""

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total_reflections = sum(len(v) for v in self._cache.values())
            failures = sum(
                1 for entries in self._cache.values()
                for e in entries if e.outcome == AttemptOutcome.FAILURE
            )
        return {
            "strategy": self._strategy.value,
            "tracked_milestones": len(self._cache),
            "total_reflections": total_reflections,
            "total_failures": failures,
            "app_profiles": list(self._app_profiles.keys()),
        }

    # =========================================================================
    # Private — Reflection Generation
    # =========================================================================

    def _generate_reflection(
        self,
        milestone_desc: str,
        trajectory_summary: str,
        failure_reason: str,
        belief_state_summary: str,
        attempt_num: int,
    ) -> str:
        """Call LLM to generate verbal reflection. Falls back to structured text."""
        if self._llm_caller is None:
            return self._structured_fallback_reflection(
                milestone_desc, trajectory_summary, failure_reason
            )

        prompt = _REFLECTION_PROMPT_TEMPLATE.format(
            milestone_desc=milestone_desc[:300],
            trajectory_summary=trajectory_summary[:500],
            failure_reason=failure_reason[:300],
            belief_state_summary=belief_state_summary[:300],
            attempt_num=attempt_num,
        )

        try:
            result = self._llm_caller(
                prompt=prompt,
                timeout=_REFLECTION_TIMEOUT,
                max_tokens=250,
            )
            reflection = result.get("text", "") if isinstance(result, dict) else str(result)
            reflection = reflection.strip()[:1000]
            if len(reflection) < 20:
                raise ValueError("Reflection too short")
            return reflection
        except Exception as exc:
            _logger.warning("[ReflexionEngine] LLM reflection failed: %s", exc)
            return self._structured_fallback_reflection(
                milestone_desc, trajectory_summary, failure_reason
            )

    def _structured_fallback_reflection(
        self,
        milestone_desc: str,
        trajectory_summary: str,
        failure_reason: str,
    ) -> str:
        """Generate structured reflection without LLM."""
        return (
            f"Attempted to: {milestone_desc}. "
            f"Approach taken: {trajectory_summary[:200]}. "
            f"Failure: {failure_reason[:200]}. "
            f"Next attempt should verify preconditions and consider alternative "
            f"UI elements or interaction methods."
        )

    def _extract_lessons(self, reflection_text: str) -> List[str]:
        """Extract bullet-point lessons from reflection text."""
        lessons = []
        lines = reflection_text.split(".")
        for line in lines:
            line = line.strip()
            if len(line) > 15 and any(
                kw in line.lower() for kw in
                ["should", "instead", "next time", "avoid", "try", "must", "never", "always"]
            ):
                lessons.append(line[:200])
        return lessons[:5]

    # =========================================================================
    # Private — ApplicationProfile (SAGE)
    # =========================================================================

    def _update_app_profile(
        self,
        app_name: str,
        lessons: List[str],
        failure_reason: str,
    ) -> None:
        profile = self.get_app_profile(app_name)
        with self._lock:
            for lesson in lessons:
                if lesson not in profile.lessons_learned:
                    profile.lessons_learned.append(lesson)
            # Keep bounded
            profile.lessons_learned = profile.lessons_learned[-20:]

            # Track failed approach
            short_failure = failure_reason[:150]
            if short_failure not in profile.failed_approaches:
                profile.failed_approaches.append(short_failure)
            profile.failed_approaches = profile.failed_approaches[-10:]
            profile.last_updated = time.time()

        self._persist_app_profile(profile)

    def _update_verified_workflow(self, app_name: str, trajectory: str) -> None:
        profile = self.get_app_profile(app_name)
        with self._lock:
            short_traj = trajectory[:200]
            if short_traj not in profile.verified_workflows:
                profile.verified_workflows.append(short_traj)
            profile.verified_workflows = profile.verified_workflows[-10:]
            profile.last_updated = time.time()
        self._persist_app_profile(profile)

    # =========================================================================
    # Private — Persistence
    # =========================================================================

    def _init_db(self) -> None:
        try:
            conn = self._get_conn()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reflections (
                    entry_id TEXT PRIMARY KEY,
                    milestone_key TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_reflections_milestone
                ON reflections(milestone_key, created_at)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS app_profiles (
                    app_name TEXT PRIMARY KEY,
                    data_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            conn.commit()
        except Exception as exc:
            _logger.warning("[ReflexionEngine] DB init warning: %s", exc)

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _persist_entry(self, entry: ReflexionEntry) -> None:
        try:
            conn = self._get_conn()
            conn.execute(
                "INSERT OR REPLACE INTO reflections(entry_id, milestone_key, data_json, created_at) "
                "VALUES (?, ?, ?, ?)",
                (entry.entry_id, entry.milestone_key,
                 json.dumps(entry.to_dict()), entry.created_at),
            )
            conn.commit()
        except Exception as exc:
            _logger.warning("[ReflexionEngine] Persist entry failed: %s", exc)

    def _load_entries_from_db(self, milestone_key: str) -> List[ReflexionEntry]:
        try:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT data_json FROM reflections WHERE milestone_key=? "
                "ORDER BY created_at DESC LIMIT ?",
                (milestone_key, self._max_per_milestone),
            ).fetchall()
            entries = []
            for (data_json,) in reversed(rows):
                try:
                    d = json.loads(data_json)
                    entries.append(ReflexionEntry(
                        entry_id=d["entry_id"],
                        milestone_key=d["milestone_key"],
                        milestone_desc=d["milestone_desc"],
                        attempt_number=d["attempt_number"],
                        outcome=AttemptOutcome(d["outcome"]),
                        trajectory_summary=d["trajectory_summary"],
                        failure_reason=d["failure_reason"],
                        belief_state_summary=d.get("belief_state_summary", ""),
                        reflection_text=d["reflection_text"],
                        lessons=d.get("lessons", []),
                        created_at=d["created_at"],
                    ))
                except Exception:
                    pass
            return entries
        except Exception as exc:
            _logger.warning("[ReflexionEngine] DB load failed: %s", exc)
            return []

    def _persist_app_profile(self, profile: ApplicationProfile) -> None:
        try:
            data = {
                "app_name":          profile.app_name,
                "lessons_learned":   profile.lessons_learned,
                "known_quirks":      profile.known_quirks,
                "failed_approaches": profile.failed_approaches,
                "verified_workflows":profile.verified_workflows,
                "last_updated":      profile.last_updated,
            }
            conn = self._get_conn()
            conn.execute(
                "INSERT OR REPLACE INTO app_profiles(app_name, data_json, updated_at) "
                "VALUES (?, ?, ?)",
                (profile.app_name, json.dumps(data), profile.last_updated),
            )
            conn.commit()
        except Exception as exc:
            _logger.warning("[ReflexionEngine] Persist app profile failed: %s", exc)

    def _load_app_profile_from_db(self, app_name: str) -> Optional[ApplicationProfile]:
        try:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT data_json FROM app_profiles WHERE app_name=?", (app_name,)
            ).fetchone()
            if row:
                d = json.loads(row[0])
                return ApplicationProfile(
                    app_name=d["app_name"],
                    lessons_learned=d.get("lessons_learned", []),
                    known_quirks=d.get("known_quirks", []),
                    failed_approaches=d.get("failed_approaches", []),
                    verified_workflows=d.get("verified_workflows", []),
                    last_updated=d.get("last_updated", time.time()),
                )
        except Exception:
            pass
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _milestone_key(desc: str) -> str:
    return hashlib.sha256(desc.strip().lower().encode()).hexdigest()[:16]


def _entry_id(milestone_key: str, attempt: int) -> str:
    return f"{milestone_key}-{attempt:04d}"


_REFLECTION_PROMPT_TEMPLATE = """\
You just attempted milestone (attempt #{attempt_num}): "{milestone_desc}"

Your actions this attempt:
{trajectory_summary}

The milestone failed because:
{failure_reason}

Current world state:
{belief_state_summary}

Write a concise reflection (2-3 sentences) covering:
1. What went wrong specifically
2. What assumption was incorrect
3. What you should try differently in the next attempt

Reflection:"""


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_global_reflexion_engine: Optional[ReflexionEngine] = None
_global_lock = threading.Lock()


def get_global_reflexion_engine(
    llm_caller: Optional[Callable] = None,
    strategy: ReflexionStrategy = ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION,
) -> ReflexionEngine:
    """Return the process-singleton ReflexionEngine."""
    global _global_reflexion_engine
    with _global_lock:
        if _global_reflexion_engine is None:
            _global_reflexion_engine = ReflexionEngine(
                llm_caller=llm_caller, strategy=strategy
            )
    return _global_reflexion_engine
