from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

_logger = logging.getLogger(__name__)

# Lazy imports — cognee and qdrant are optional
_cognee = None
_qdrant_client = None
_COGNEE_AVAILABLE: Optional[bool] = None
_COGNEE_INIT_LOCK = threading.Lock()


def _check_cognee() -> bool:
    global _cognee, _COGNEE_AVAILABLE
    if _COGNEE_AVAILABLE is not None:
        return _COGNEE_AVAILABLE
    with _COGNEE_INIT_LOCK:
        if _COGNEE_AVAILABLE is not None:
            return _COGNEE_AVAILABLE
        try:
            import cognee as _cog  # noqa: PLC0415
            _cognee = _cog
            _COGNEE_AVAILABLE = True
            _logger.info("[CogneeStore] cognee library available.")
        except ImportError:
            _COGNEE_AVAILABLE = False
            _logger.info(
                "[CogneeStore] cognee not installed. Falling back to SemanticMemory. "
                "Install: pip install cognee"
            )
    return _COGNEE_AVAILABLE


# ---------------------------------------------------------------------------
# Helper: run async from sync context
# ---------------------------------------------------------------------------

def _run_async(coro) -> Any:
    """Run an async coroutine from a synchronous context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside an already-running event loop (e.g. Jupyter)
            import concurrent.futures  # noqa: PLC0415
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=120)
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# CogneeStore
# ---------------------------------------------------------------------------

class CogneeStore:
    

    _instance: Optional["CogneeStore"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._available = _check_cognee()
        self._qdrant_url = os.environ.get("PROJECTZEO_QDRANT_URL", "http://localhost:6333").strip()
        self._collection = os.environ.get("PROJECTZEO_COGNEE_COLLECTION", "projectzeo_memory")
        self._initialized = False
        self._init_lock = threading.Lock()

        # Fallback SemanticMemory for when cognee is not available
        self._semantic_fallback = None

    @classmethod
    def get_instance(cls) -> "CogneeStore":
        if cls._instance is not None:
            return cls._instance
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def _ensure_initialized(self) -> bool:
        """One-time Cognee + Qdrant configuration."""
        if self._initialized or not self._available:
            return self._initialized
        with self._init_lock:
            if self._initialized:
                return True
            try:
                _run_async(self._async_init())
                self._initialized = True
                _logger.info("[CogneeStore] Cognee + Qdrant initialised.")
            except Exception as exc:
                _logger.warning("[CogneeStore] Init failed: %s. Using SemanticMemory fallback.", exc)
                self._available = False
        return self._initialized

    async def _async_init(self) -> None:
        """Configure cognee to use local Qdrant as the vector store."""
        await _cognee.config.set_vector_db_url(self._qdrant_url)
        await _cognee.config.set_vector_db_collection(self._collection)
        # Use local LLM for cognee's internal graph operations if available
        local_model = os.environ.get("PROJECTZEO_LOCAL_MODEL", "qwen2.5-vl")
        try:
            await _cognee.config.set_llm_api_base(
                f"http://{os.environ.get('PROJECTZEO_OLLAMA_HOST', 'localhost')}:11434"
            )
            await _cognee.config.set_llm_model(local_model)
        except Exception:
            pass  # Not all cognee versions expose this config

    def ingest_execution_log(
        self,
        execution_log: Dict[str, Any],
        objective: str,
        focused_app: Optional[str],
    ) -> None:
        
        if not self._ensure_initialized():
            self._fallback_ingest(execution_log, objective, focused_app)
            return

        def _run():
            try:
                _run_async(
                    self._async_ingest(execution_log, objective, focused_app)
                )
            except Exception as exc:
                _logger.debug("[CogneeStore] Async ingest failed: %s", exc)
                self._fallback_ingest(execution_log, objective, focused_app)

        t = threading.Thread(target=_run, daemon=True, name="cognee-ingest")
        t.start()
        # Do not join — non-blocking post-task operation

    async def _async_ingest(
        self,
        execution_log: Dict[str, Any],
        objective: str,
        focused_app: Optional[str],
    ) -> None:
        """Build natural-language documents from execution log and add to Cognee."""
        docs = self._build_documents(execution_log, objective, focused_app)
        for doc in docs:
            await _cognee.add(doc)
        await _cognee.cognify()
        _logger.debug("[CogneeStore] Ingested %d documents for objective=%r", len(docs), objective[:80])

    def _build_documents(
        self,
        execution_log: Dict[str, Any],
        objective: str,
        focused_app: Optional[str],
    ) -> List[str]:
        """Convert execution log into natural-language sentences for the knowledge graph."""
        docs = []

        app_label = focused_app or "the system"
        docs.append(f"Task objective: {objective[:500]}")
        docs.append(f"Application context: {app_label}")

        for step_idx, step_data in execution_log.items():
            if not isinstance(step_data, dict):
                continue
            for output_entry in step_data.get("outputs", []):
                op = str(output_entry.get("operation", "unknown"))
                success = output_entry.get("success", True)
                output = str(output_entry.get("output", ""))[:500]
                status_word = "succeeded" if success else "failed"

                if op == "command":
                    cmd = str(output_entry.get("command", ""))[:200]
                    docs.append(
                        f"Step {step_idx}: Shell command '{cmd}' {status_word} "
                        f"in {app_label}. Output: {output}"
                    )
                elif op == "file_create":
                    path = str(output_entry.get("path", ""))
                    docs.append(
                        f"Step {step_idx}: Created file '{path}' {status_word} "
                        f"while working on {app_label}."
                    )
                elif op == "install":
                    docs.append(
                        f"Step {step_idx}: Installation {status_word} in {app_label}. "
                        f"Output: {output}"
                    )
                elif not success:
                    docs.append(
                        f"Step {step_idx}: Action '{op}' failed in {app_label}. "
                        f"Error: {output}"
                    )

        return [d for d in docs if len(d) > 20]  # Filter trivially short docs

    def query(self, objective: str, max_results: int = 5) -> List[str]:
        """
        Query the knowledge graph for context relevant to the objective.
        Returns a list of relevant fact strings.
        """
        if not self._ensure_initialized():
            return self._fallback_query(objective, max_results)

        try:
            results = _run_async(self._async_query(objective, max_results))
            return results if results else self._fallback_query(objective, max_results)
        except Exception as exc:
            _logger.debug("[CogneeStore] Query failed: %s", exc)
            return self._fallback_query(objective, max_results)

    async def _async_query(self, objective: str, max_results: int) -> List[str]:
        """Semantic search over the Cognee knowledge graph."""
        results = await _cognee.search(objective, limit=max_results)
        return [str(r) for r in results if r]

    # =========================================================================
    # Fallback: SemanticMemory (regex-based)
    # =========================================================================

    def _get_semantic_fallback(self):
        if self._semantic_fallback is None:
            try:
                from core.memory.semantic_memory import SemanticMemory  # noqa: PLC0415
                self._semantic_fallback = SemanticMemory()
            except Exception:
                pass
        return self._semantic_fallback

    def _fallback_ingest(
        self,
        execution_log: Dict[str, Any],
        objective: str,
        focused_app: Optional[str],
    ) -> None:
        """SemanticMemory regex extraction as fallback."""
        sm = self._get_semantic_fallback()
        if sm is None:
            return
        try:
            # Minimal regex extraction — same as GIIController._extract_semantic_facts_from_log
            import re  # noqa: PLC0415
            version_re = re.compile(r"(\w[\w\-]*)\s+(?:version\s+)?v?(\d+\.\d[\d.]*)", re.IGNORECASE)
            for step_idx, step_data in execution_log.items():
                if not isinstance(step_data, dict):
                    continue
                for output_entry in step_data.get("outputs", []):
                    output_text = str(output_entry.get("output", ""))[:1000]
                    for m in version_re.finditer(output_text):
                        sm.store(
                            subject=m.group(1).lower(), predicate="version",
                            object_=m.group(2), category="application_facts",
                            confidence=0.9, source="observed",
                        )
        except Exception:
            pass

    def _fallback_query(self, objective: str, max_results: int) -> List[str]:
        """SemanticMemory query as fallback."""
        sm = self._get_semantic_fallback()
        if sm is None:
            return []
        try:
            facts = sm.query(objective, max_results=max_results)
            return sm.format_for_prompt(facts).splitlines() if facts else []
        except Exception:
            return []
