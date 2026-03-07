"""
letta_store.py — Letta (formerly MemGPT) memory store integration for ProjectZeo.

Letta provides persistent, self-editing agent memory with structured storage:
  - Human block:  facts about the user / operator
  - Persona block: agent self-description / context
  - Archival:     long-term vectorised episodic memory
  - In-context:   short sliding-window memory

This module wraps the Letta Python SDK with a ProjectZeo-compatible interface
that mirrors the Mem0Store and CogneeStore APIs, enabling drop-in substitution.

Optional dependency — falls back to JSON file store if `letta` is not installed:
    pip install letta

Environment variables:
    PROJECTZEO_LETTA_BASE_URL    — Letta server URL (default: http://localhost:8283)
    PROJECTZEO_LETTA_TOKEN       — Bearer token (optional for local servers)
    PROJECTZEO_LETTA_AGENT_ID    — Existing agent ID to reuse (optional)
    PROJECTZEO_LETTA_ENABLED     — Set to "1" to enable (default: 0)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

_logger = logging.getLogger(__name__)

# Lazy imports
_letta_client_cls = None
_LETTA_AVAILABLE: Optional[bool] = None
_LETTA_INIT_LOCK = threading.Lock()


def _check_letta() -> bool:
    global _letta_client_cls, _LETTA_AVAILABLE
    if _LETTA_AVAILABLE is not None:
        return _LETTA_AVAILABLE
    with _LETTA_INIT_LOCK:
        if _LETTA_AVAILABLE is not None:
            return _LETTA_AVAILABLE
        try:
            from letta import create_client  # noqa: PLC0415
            _letta_client_cls = create_client
            _LETTA_AVAILABLE = True
            _logger.info("[LettaStore] letta SDK available.")
        except ImportError:
            _LETTA_AVAILABLE = False
            _logger.info(
                "[LettaStore] letta not installed. Using JSON fallback. "
                "Install: pip install letta"
            )
    return _LETTA_AVAILABLE


# ---------------------------------------------------------------------------
# LettaStore
# ---------------------------------------------------------------------------

class LettaStore:
    """
    Persistent agent memory using Letta (formerly MemGPT).

    Supports:
      - store(fact: str, agent_id: str)  — add to archival memory
      - recall(query: str, n: int)        — semantic search in archival memory
      - update_human_block(text: str)     — update the Human memory block
      - update_persona_block(text: str)   — update the Persona memory block
      - get_context_summary()             — retrieve in-context memory
    """

    _instance: Optional["LettaStore"] = None
    _instance_lock = threading.Lock()

    _BASE_URL     = "http://localhost:8283"
    _FALLBACK_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "memory",
    )

    def __init__(self) -> None:
        self._enabled = (
            os.environ.get("PROJECTZEO_LETTA_ENABLED", "0").strip() in ("1", "true", "yes")
        )
        self._available = _check_letta() if self._enabled else False

        self._client = None
        self._agent_id: Optional[str] = os.environ.get("PROJECTZEO_LETTA_AGENT_ID", "").strip() or None
        self._base_url = os.environ.get("PROJECTZEO_LETTA_BASE_URL", self._BASE_URL).strip()
        self._token    = os.environ.get("PROJECTZEO_LETTA_TOKEN", "").strip() or None

        # Fallback JSON store
        self._fallback_path = os.path.join(self._FALLBACK_DIR, "letta_fallback.json")
        self._fallback_store: Dict[str, List[Dict]] = {}
        self._fallback_lock = threading.Lock()

        if self._available:
            self._init_client()
        else:
            self._load_fallback()

    @classmethod
    def get_instance(cls) -> "LettaStore":
        if cls._instance is not None:
            return cls._instance
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    # =========================================================================
    # Initialisation
    # =========================================================================

    def _init_client(self) -> None:
        try:
            kwargs: Dict[str, Any] = {"base_url": self._base_url}
            if self._token:
                kwargs["token"] = self._token

            self._client = _letta_client_cls(**kwargs)

            # Reuse existing agent or create a new one
            if self._agent_id:
                _logger.info("[LettaStore] Reusing agent: %s", self._agent_id)
            else:
                agent = self._client.create_agent(
                    name="projectzeo_memory_agent",
                    description=(
                        "ProjectZeo autonomous agent memory. Stores episodic lessons, "
                        "application facts, and execution history for cross-session learning."
                    ),
                )
                self._agent_id = agent.id
                _logger.info("[LettaStore] Created new Letta agent: %s", self._agent_id)

        except Exception as exc:
            _logger.warning(
                "[LettaStore] Client init failed: %s. Using JSON fallback.", exc
            )
            self._available = False
            self._load_fallback()

    def _load_fallback(self) -> None:
        try:
            os.makedirs(self._FALLBACK_DIR, exist_ok=True)
            if os.path.exists(self._fallback_path):
                with open(self._fallback_path, "r", encoding="utf-8") as f:
                    self._fallback_store = json.load(f)
        except Exception as exc:
            _logger.debug("[LettaStore] Fallback load error: %s", exc)
            self._fallback_store = {}

    def _save_fallback(self) -> None:
        try:
            os.makedirs(self._FALLBACK_DIR, exist_ok=True)
            tmp = self._fallback_path + ".tmp"
            with self._fallback_lock:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(self._fallback_store, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self._fallback_path)
        except Exception as exc:
            _logger.debug("[LettaStore] Fallback save error: %s", exc)

    # =========================================================================
    # Public API
    # =========================================================================

    def store(
        self,
        fact: str,
        *,
        agent_id: Optional[str] = None,
        category: str = "general",
        source: str = "observed",
    ) -> bool:
        """
        Store a fact in archival memory (Letta) or JSON fallback.

        Args:
            fact:     The text to store (will be vectorised by Letta).
            agent_id: Override the default agent ID.
            category: Tag for organisation (used in fallback).
            source:   Provenance label.

        Returns:
            True on success, False on failure.
        """
        if not fact or not fact.strip():
            return False

        _agent = agent_id or self._agent_id

        if self._available and self._client and _agent:
            try:
                self._client.insert_archival_memory(
                    agent_id=_agent,
                    memory=fact.strip(),
                )
                _logger.debug(
                    "[LettaStore] Stored archival memory (agent=%s): %s",
                    _agent, fact[:80],
                )
                return True
            except Exception as exc:
                _logger.warning("[LettaStore] store() failed: %s — writing to fallback.", exc)

        # Fallback
        entry = {
            "fact":     fact.strip(),
            "category": category,
            "source":   source,
            "ts":       time.time(),
            "id":       hashlib.sha1(fact.strip().encode()).hexdigest()[:12],
        }
        with self._fallback_lock:
            bucket = self._fallback_store.setdefault(category, [])
            bucket.append(entry)
        self._save_fallback()
        return True

    def recall(
        self,
        query: str,
        n: int = 10,
        *,
        agent_id: Optional[str] = None,
    ) -> List[str]:
        """
        Semantic search in archival memory.

        Args:
            query:    Natural language query string.
            n:        Maximum number of results.
            agent_id: Override the default agent ID.

        Returns:
            List of matching fact strings (most relevant first).
        """
        if not query or not query.strip():
            return []

        _agent = agent_id or self._agent_id

        if self._available and self._client and _agent:
            try:
                results = self._client.get_archival_memory(
                    agent_id=_agent,
                    query=query.strip(),
                    limit=n,
                )
                return [
                    r.text if hasattr(r, "text") else str(r)
                    for r in results
                ]
            except Exception as exc:
                _logger.warning("[LettaStore] recall() failed: %s — using fallback.", exc)

        # Fallback: simple keyword search
        query_lower = query.lower()
        matches: List[Dict] = []
        with self._fallback_lock:
            for bucket in self._fallback_store.values():
                for entry in bucket:
                    if any(w in entry["fact"].lower() for w in query_lower.split()):
                        matches.append(entry)

        matches.sort(key=lambda e: e.get("ts", 0), reverse=True)
        return [e["fact"] for e in matches[:n]]

    def update_human_block(
        self,
        text: str,
        *,
        agent_id: Optional[str] = None,
    ) -> bool:
        """Update the Letta Human memory block (operator context)."""
        _agent = agent_id or self._agent_id
        if not self._available or not self._client or not _agent:
            return False
        try:
            self._client.update_in_context_memory(
                agent_id=_agent,
                section="human",
                value=text[:2000],
            )
            return True
        except Exception as exc:
            _logger.warning("[LettaStore] update_human_block failed: %s", exc)
            return False

    def update_persona_block(
        self,
        text: str,
        *,
        agent_id: Optional[str] = None,
    ) -> bool:
        """Update the Letta Persona memory block (agent self-description)."""
        _agent = agent_id or self._agent_id
        if not self._available or not self._client or not _agent:
            return False
        try:
            self._client.update_in_context_memory(
                agent_id=_agent,
                section="persona",
                value=text[:2000],
            )
            return True
        except Exception as exc:
            _logger.warning("[LettaStore] update_persona_block failed: %s", exc)
            return False

    def get_context_summary(
        self,
        agent_id: Optional[str] = None,
    ) -> str:
        """Return the current in-context memory blocks as a formatted string."""
        _agent = agent_id or self._agent_id
        if not self._available or not self._client or not _agent:
            return ""
        try:
            memory = self._client.get_in_context_memory(agent_id=_agent)
            parts = []
            for block in getattr(memory, "memory", {}).values():
                label = getattr(block, "label", "?")
                value = getattr(block, "value", "")
                if value:
                    parts.append(f"[{label}]\n{value}")
            return "\n\n".join(parts)
        except Exception as exc:
            _logger.debug("[LettaStore] get_context_summary failed: %s", exc)
            return ""

    # =========================================================================
    # Diagnostics
    # =========================================================================

    def is_available(self) -> bool:
        return self._available

    def is_enabled(self) -> bool:
        return self._enabled

    def get_stats(self) -> Dict[str, Any]:
        fallback_count = sum(len(v) for v in self._fallback_store.values())
        return {
            "enabled":         self._enabled,
            "letta_available": self._available,
            "agent_id":        self._agent_id,
            "base_url":        self._base_url,
            "fallback_facts":  fallback_count,
        }


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def get_letta_store() -> LettaStore:
    """Return the process-global LettaStore singleton."""
    return LettaStore.get_instance()
