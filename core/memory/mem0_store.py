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
_mem0 = None
_MEM0_AVAILABLE: Optional[bool] = None
_MEM0_INIT_LOCK = threading.Lock()


def _check_mem0() -> bool:
    global _mem0, _MEM0_AVAILABLE
    if _MEM0_AVAILABLE is not None:
        return _MEM0_AVAILABLE
    with _MEM0_INIT_LOCK:
        if _MEM0_AVAILABLE is not None:
            return _MEM0_AVAILABLE
        try:
            from mem0 import Memory  # noqa: PLC0415
            _mem0 = Memory
            _MEM0_AVAILABLE = True
            _logger.info("[Mem0Store] mem0ai available.")
        except ImportError:
            _MEM0_AVAILABLE = False
            _logger.info(
                "[Mem0Store] mem0ai not installed. Using JSON file fallback. "
                "Install: pip install mem0ai"
            )
    return _MEM0_AVAILABLE


# ---------------------------------------------------------------------------
# Mem0Store
# ---------------------------------------------------------------------------

class Mem0Store:
    

    _instance: Optional["Mem0Store"] = None
    _instance_lock = threading.Lock()

    # Fallback JSON storage path
    _FALLBACK_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "memory", "mem0_fallback.json",
    )

    def __init__(self) -> None:
        self._available = _check_mem0()
        self._memory_client = None
        self._fallback_store: Dict[str, List[Dict]] = {}
        self._fallback_lock = threading.Lock()
        self._lock = threading.Lock()

        if self._available:
            self._init_mem0()
        else:
            self._load_fallback()

    @classmethod
    def get_instance(cls) -> "Mem0Store":
        if cls._instance is not None:
            return cls._instance
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def _init_mem0(self) -> None:
        """Configure Mem0 with Qdrant backend or in-memory."""
        try:
            qdrant_url = os.environ.get("PROJECTZEO_QDRANT_URL", "").strip()
            llm_model = os.environ.get("PROJECTZEO_LOCAL_MODEL", "qwen2.5-vl")
            ollama_host = os.environ.get("PROJECTZEO_OLLAMA_HOST", "localhost")
            ollama_port = int(os.environ.get("PROJECTZEO_OLLAMA_PORT", "11434"))

            config: Dict[str, Any] = {
                "llm": {
                    "provider": "ollama",
                    "config": {
                        "model": llm_model,
                        "ollama_base_url": f"http://{ollama_host}:{ollama_port}",
                    },
                },
                "embedder": {
                    "provider": "ollama",
                    "config": {
                        "model": "nomic-embed-text",
                        "ollama_base_url": f"http://{ollama_host}:{ollama_port}",
                    },
                },
            }

            if qdrant_url:
                config["vector_store"] = {
                    "provider": "qdrant",
                    "config": {
                        "url": qdrant_url,
                        "collection_name": os.environ.get(
                            "PROJECTZEO_MEM0_COLLECTION", "projectzeo_working_memory"
                        ),
                    },
                }
                _logger.info("[Mem0Store] Using Qdrant at %s for persistent storage.", qdrant_url)
            else:
                config["vector_store"] = {"provider": "memory"}
                _logger.info("[Mem0Store] Using in-memory storage (not persistent).")

            # Check for Mem0 cloud API key
            api_key = os.environ.get("PROJECTZEO_MEM0_API_KEY", "").strip()
            if api_key:
                from mem0 import MemoryClient  # noqa: PLC0415
                self._memory_client = MemoryClient(api_key=api_key)
                _logger.info("[Mem0Store] Using Mem0 Cloud API.")
                return

            self._memory_client = _mem0(config=config)
            _logger.info("[Mem0Store] Mem0 initialised with local backend.")

        except Exception as exc:
            _logger.warning("[Mem0Store] Mem0 init failed: %s. Using fallback.", exc)
            self._available = False
            self._load_fallback()

    def _load_fallback(self) -> None:
        """Load fallback JSON store from disk."""
        try:
            os.makedirs(os.path.dirname(self._FALLBACK_PATH), exist_ok=True)
            if os.path.exists(self._FALLBACK_PATH):
                with open(self._FALLBACK_PATH, "r", encoding="utf-8") as f:
                    self._fallback_store = json.load(f)
        except Exception:
            self._fallback_store = {}

    def _save_fallback(self) -> None:
        """Persist fallback JSON store to disk (best-effort)."""
        try:
            os.makedirs(os.path.dirname(self._FALLBACK_PATH), exist_ok=True)
            tmp = self._FALLBACK_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._fallback_store, f, indent=2)
            os.replace(tmp, self._FALLBACK_PATH)
        except Exception:
            pass

    # =========================================================================
    # Public API
    # =========================================================================

    def add_memory(
        self,
        messages: List[Dict[str, str]],
        agent_id: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Add memories extracted from a conversation turn.

        Args:
            messages:  List of {"role": ..., "content": ...} dicts
            agent_id:  Namespace identifier (use make_agent_id() for tasks)
            metadata:  Optional metadata dict to attach to memories
        """
        if not messages:
            return

        if self._available and self._memory_client is not None:
            try:
                self._memory_client.add(
                    messages,
                    agent_id=agent_id,
                    metadata=metadata or {},
                )
                return
            except Exception as exc:
                _logger.debug("[Mem0Store] add_memory failed: %s. Using fallback.", exc)

        # Fallback: store raw messages
        with self._fallback_lock:
            store = self._fallback_store.setdefault(agent_id, [])
            for msg in messages:
                entry = {
                    "content": str(msg.get("content", "")),
                    "role": str(msg.get("role", "user")),
                    "ts": time.time(),
                    "metadata": metadata or {},
                }
                store.append(entry)
            # Cap to 500 entries per agent
            if len(store) > 500:
                self._fallback_store[agent_id] = store[-500:]
            self._save_fallback()

    def search(
        self,
        query: str,
        agent_id: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search memories relevant to the query.

        Returns list of memory dicts: [{"memory": str, "score": float, ...}]
        """
        if not query:
            return []

        if self._available and self._memory_client is not None:
            try:
                results = self._memory_client.search(query, agent_id=agent_id, limit=limit)
                if isinstance(results, dict):
                    results = results.get("results", [])
                return results if isinstance(results, list) else []
            except Exception as exc:
                _logger.debug("[Mem0Store] search failed: %s. Using fallback.", exc)

        # Fallback: simple keyword search
        return self._fallback_search(query, agent_id, limit)

    def get_all(self, agent_id: str) -> List[Dict[str, Any]]:
        """Return all memories for an agent_id."""
        if self._available and self._memory_client is not None:
            try:
                results = self._memory_client.get_all(agent_id=agent_id)
                if isinstance(results, dict):
                    results = results.get("results", [])
                return results if isinstance(results, list) else []
            except Exception as exc:
                _logger.debug("[Mem0Store] get_all failed: %s.", exc)

        with self._fallback_lock:
            return list(self._fallback_store.get(agent_id, []))

    def format_context(self, memories: List[Dict[str, Any]], max_chars: int = 2000) -> str:
        
        if not memories:
            return ""

        lines = ["[Memory Context from previous sessions]"]
        total = len(lines[0])

        for mem in memories:
            if isinstance(mem, dict):
                text = str(mem.get("memory") or mem.get("content") or "")
            else:
                text = str(mem)

            if not text:
                continue

            line = f"• {text}"
            if total + len(line) > max_chars:
                break
            lines.append(line)
            total += len(line)

        return "\n".join(lines) if len(lines) > 1 else ""

    # =========================================================================
    # Utilities
    # =========================================================================

    @staticmethod
    def make_agent_id(objective: str) -> str:
        
        h = hashlib.sha256(objective.encode("utf-8", errors="replace")).hexdigest()[:8]
        slug = objective[:60].lower().replace(" ", "_").replace("/", "_")
        # Strip non-alphanumeric/underscore chars
        import re  # noqa: PLC0415
        slug = re.sub(r"[^\w_]", "", slug)[:40]
        return f"pz_{slug}_{h}"

    def _fallback_search(self, query: str, agent_id: str, limit: int) -> List[Dict]:
        """Simple keyword-based search over the fallback store."""
        with self._fallback_lock:
            entries = self._fallback_store.get(agent_id, [])

        if not entries:
            return []

        query_words = set(query.lower().split())
        scored: List[tuple] = []
        for entry in entries:
            content = str(entry.get("content", "")).lower()
            score = sum(1 for w in query_words if w in content)
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"memory": e.get("content", ""), "score": s / max(len(query_words), 1)}
            for s, e in scored[:limit]
        ]

    def is_available(self) -> bool:
        return self._available and self._memory_client is not None

    def get_stats(self) -> Dict[str, Any]:
        return {
            "available": self.is_available(),
            "fallback_agents": len(self._fallback_store),
            "fallback_total_entries": sum(
                len(v) for v in self._fallback_store.values()
            ),
        }
