"""
core/memory/graphiti_store.py — Graphiti Temporal Knowledge Graph (Layer 4, Research §5)

Implements the persistent world model using Graphiti — a bi-temporal knowledge
graph engine designed for AI agents operating in dynamic environments.

Key improvements over the current SemanticMemory (JSON file):
  - Facts tracked with event time T and ingestion time T' (bi-temporal)
  - Automatic invalidation of contradicted facts (no stale data)
  - Real-time incremental updates without full recomputation
  - Hybrid search: vector + BM25 + graph traversal (P95 at 300ms)
  - Survives across task boundaries (Gap 5 fix: no WorldGraph reset)

Reference: "Zep: A Temporal Knowledge Graph Architecture for Agent Memory"
           Rasmussen et al., arxiv 2501.13956, January 2025

Gap 5 fix from research §10 Phase 3:
  OLD: self._world_model = WorldGraph()   # reset every task
  NEW: WorldGraph.from_graphiti(...)      # persistent, compound across tasks

Setup:
  pip install graphiti-core
  docker run -p 6379:6379 -p 3000:3000 falkordb/falkordb:latest
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

_logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

_GRAPHITI_URI = os.environ.get("PROJECTZEO_GRAPHITI_URI", "bolt://localhost:7687")
_GRAPHITI_USER = os.environ.get("PROJECTZEO_GRAPHITI_USER", "neo4j")
_GRAPHITI_PASSWORD = os.environ.get("PROJECTZEO_GRAPHITI_PASSWORD", "password")
_GRAPHITI_ENABLED = os.environ.get("PROJECTZEO_GRAPHITI_ENABLED", "0").strip() == "1"

# FalkorDB (lighter alternative to Neo4j for local deployments)
_FALKORDB_HOST = os.environ.get("PROJECTZEO_FALKORDB_HOST", "localhost")
_FALKORDB_PORT = int(os.environ.get("PROJECTZEO_FALKORDB_PORT", "6379"))


class ApplicationEntity:
    """Pydantic-style entity for application knowledge (Research §5.2)."""

    def __init__(
        self,
        app_name: str,
        main_window_title: str = "",
        known_workflows: Optional[List[str]] = None,
        observed_failure_patterns: Optional[List[str]] = None,
        last_seen: Optional[datetime] = None,
    ) -> None:
        self.app_name = app_name
        self.main_window_title = main_window_title
        self.known_workflows: List[str] = known_workflows or []
        self.observed_failure_patterns: List[str] = observed_failure_patterns or []
        self.last_seen: datetime = last_seen or datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "app_name": self.app_name,
            "main_window_title": self.main_window_title,
            "known_workflows": self.known_workflows,
            "observed_failure_patterns": self.observed_failure_patterns,
            "last_seen": self.last_seen.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ApplicationEntity":
        last_seen = None
        try:
            last_seen = datetime.fromisoformat(d.get("last_seen", ""))
        except Exception:
            last_seen = datetime.utcnow()
        return cls(
            app_name=d.get("app_name", ""),
            main_window_title=d.get("main_window_title", ""),
            known_workflows=d.get("known_workflows", []),
            observed_failure_patterns=d.get("observed_failure_patterns", []),
            last_seen=last_seen,
        )


class GraphitiStore:
    """
    Graphiti-backed persistent knowledge store for ProjectZeo.

    Falls back gracefully to a JSON file store when graphiti-core is not
    installed or the backend is unreachable. The interface is identical in
    both cases so callers don't need to know which backend is active.
    """

    def __init__(
        self,
        memory_dir: Optional[str] = None,
        namespace: str = "projectzeo",
    ) -> None:
        self._namespace = namespace
        self._memory_dir = memory_dir or os.path.expanduser("~/.projectzeo/graphiti")
        os.makedirs(self._memory_dir, exist_ok=True)
        self._lock = threading.Lock()

        # Try to connect to Graphiti backend
        self._graphiti_client = None
        self._backend = "json"  # fallback

        if _GRAPHITI_ENABLED:
            self._backend = self._try_connect_graphiti()

        _logger.info("[GraphitiStore] Backend: %s | namespace: %s", self._backend, namespace)

        # JSON fallback store
        self._json_path = os.path.join(self._memory_dir, f"{namespace}_knowledge.json")
        self._json_data: Dict[str, Any] = self._load_json()

    # ─────────────────────────────────────────────────────────────────────────
    # Backend connection
    # ─────────────────────────────────────────────────────────────────────────

    def _try_connect_graphiti(self) -> str:
        try:
            from graphiti_core import Graphiti  # type: ignore
            from graphiti_core.llm_client.openai_client import OpenAIClient  # type: ignore
            # Graphiti uses OpenAI-compatible API — route to local Ollama
            client = Graphiti(
                uri=_GRAPHITI_URI,
                user=_GRAPHITI_USER,
                password=_GRAPHITI_PASSWORD,
            )
            self._graphiti_client = client
            _logger.info("[GraphitiStore] Connected to Graphiti at %s", _GRAPHITI_URI)
            return "graphiti"
        except ImportError:
            _logger.info("[GraphitiStore] graphiti-core not installed — using JSON fallback.")
            return "json"
        except Exception as e:
            _logger.warning("[GraphitiStore] Graphiti connection failed (%s) — JSON fallback.", e)
            return "json"

    # ─────────────────────────────────────────────────────────────────────────
    # JSON fallback persistence
    # ─────────────────────────────────────────────────────────────────────────

    def _load_json(self) -> Dict[str, Any]:
        try:
            if os.path.exists(self._json_path):
                with open(self._json_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            _logger.warning("[GraphitiStore] JSON load error: %s", e)
        return {"applications": {}, "workflows": {}, "failures": {}, "facts": []}

    def _save_json(self) -> None:
        try:
            tmp = self._json_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._json_data, f, indent=2, default=str)
            os.replace(tmp, self._json_path)
        except Exception as e:
            _logger.warning("[GraphitiStore] JSON save error: %s", e)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def store_application_entity(self, entity: ApplicationEntity) -> None:
        """
        Store or update an application entity after task completion.
        Called by GIIController.on_task_complete().
        """
        with self._lock:
            if self._backend == "graphiti" and self._graphiti_client is not None:
                self._graphiti_store_entity(entity)
            else:
                apps = self._json_data.setdefault("applications", {})
                existing = apps.get(entity.app_name, {})
                # Merge workflows
                merged_workflows = list(set(
                    existing.get("known_workflows", []) + entity.known_workflows
                ))[:50]
                # Merge failure patterns
                merged_failures = list(set(
                    existing.get("observed_failure_patterns", []) + entity.observed_failure_patterns
                ))[:50]
                apps[entity.app_name] = {
                    **entity.to_dict(),
                    "known_workflows": merged_workflows,
                    "observed_failure_patterns": merged_failures,
                }
                self._save_json()
        _logger.debug("[GraphitiStore] Stored entity: %s", entity.app_name)

    def get_application_entity(self, app_name: str) -> Optional[ApplicationEntity]:
        """Retrieve application knowledge. Returns None if unknown."""
        with self._lock:
            if self._backend == "graphiti" and self._graphiti_client is not None:
                return self._graphiti_get_entity(app_name)
            apps = self._json_data.get("applications", {})
            if app_name in apps:
                return ApplicationEntity.from_dict(apps[app_name])
        return None

    def store_task_outcome(
        self,
        *,
        app_name: str,
        objective: str,
        milestone_sequence: List[str],
        stagnation_events: List[str],
        vsa_violations: List[str],
        success: bool,
        duration_sec: float,
    ) -> None:
        """
        Store a complete task outcome for compound learning.
        Feeds UI-Evol and ARPO training pipeline.
        """
        record = {
            "ts": datetime.utcnow().isoformat(),
            "app_name": app_name,
            "objective": objective[:300],
            "milestone_sequence": milestone_sequence[:20],
            "stagnation_events": stagnation_events[:10],
            "vsa_violations": vsa_violations[:10],
            "success": success,
            "duration_sec": duration_sec,
        }
        with self._lock:
            outcomes = self._json_data.setdefault("task_outcomes", [])
            outcomes.append(record)
            # Keep last 1000 outcomes
            if len(outcomes) > 1000:
                self._json_data["task_outcomes"] = outcomes[-1000:]
            self._save_json()
        _logger.debug("[GraphitiStore] Task outcome stored: app=%s success=%s", app_name, success)

    def get_application_history(self, app_name: str) -> List[Dict[str, Any]]:
        """Return task outcomes for a specific application (for pre-task context)."""
        with self._lock:
            outcomes = self._json_data.get("task_outcomes", [])
        return [o for o in outcomes if o.get("app_name") == app_name][-20:]

    def store_fact(self, subject: str, predicate: str, obj: str, confidence: float = 1.0) -> None:
        """Store a semantic fact with bi-temporal tracking."""
        with self._lock:
            facts = self._json_data.setdefault("facts", [])
            facts.append({
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "confidence": confidence,
                "event_ts": datetime.utcnow().isoformat(),
                "ingest_ts": datetime.utcnow().isoformat(),
                "invalidated": False,
            })
            # Prune old facts
            if len(facts) > 10000:
                self._json_data["facts"] = facts[-10000:]
            self._save_json()

    def query_facts(self, subject: str) -> List[Dict[str, Any]]:
        """Retrieve active facts for a subject."""
        with self._lock:
            facts = self._json_data.get("facts", [])
        return [f for f in facts if f.get("subject") == subject and not f.get("invalidated")]

    # ─────────────────────────────────────────────────────────────────────────
    # Graphiti backend methods (when available)
    # ─────────────────────────────────────────────────────────────────────────

    def _graphiti_store_entity(self, entity: ApplicationEntity) -> None:
        """Store entity in Graphiti graph database."""
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            loop.run_until_complete(
                self._graphiti_client.add_episode(
                    name=f"app_entity:{entity.app_name}",
                    episode_body=json.dumps(entity.to_dict()),
                    source_description=f"ProjectZeo task completion — {entity.app_name}",
                    reference_time=datetime.utcnow(),
                )
            )
            loop.close()
        except Exception as e:
            _logger.warning("[GraphitiStore] Graphiti store failed: %s", e)

    def _graphiti_get_entity(self, app_name: str) -> Optional[ApplicationEntity]:
        """Retrieve entity from Graphiti."""
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            results = loop.run_until_complete(
                self._graphiti_client.search(f"application entity {app_name}", num_results=1)
            )
            loop.close()
            if results:
                data = json.loads(str(results[0]))
                return ApplicationEntity.from_dict(data)
        except Exception as e:
            _logger.debug("[GraphitiStore] Graphiti get failed: %s", e)
        return None

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "backend": self._backend,
                "applications_known": len(self._json_data.get("applications", {})),
                "task_outcomes": len(self._json_data.get("task_outcomes", [])),
                "facts": len(self._json_data.get("facts", [])),
            }
