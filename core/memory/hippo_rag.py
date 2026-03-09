"""
core/memory/hippo_rag.py

HippoRAG-style multi-hop knowledge graph retrieval for ProjectZeo.

Reference: Gutierrez et al., NeurIPS 2024 — "HippoRAG: Neurologically
Inspired Long-Term Memory for Large Language Models"

Design:
  - Entities and relations are extracted from task observations and stored
    as a lightweight in-process knowledge graph (node → edges dict)
  - Retrieval uses Personalized PageRank (PPR) seeded from query entities
    to find multi-hop relevant context — matching HippoRAG's hippocampal
    index structure
  - Graphiti is still used for persistent storage; HippoRAG adds the
    multi-hop PPR retrieval layer on top

Why not just Graphiti?
  Graphiti provides single-hop entity lookup. HippoRAG adds K-hop graph
  traversal so the agent can connect e.g. "VSCode → Python extension →
  pylint → config path" across multiple stored facts.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple

_logger = logging.getLogger(__name__)

_STORE_PATH   = os.path.expanduser(os.environ.get("PROJECTZEO_HIPPORAG_PATH", "~/.projectzeo/hipporag.json"))
_MAX_NODES    = int(os.environ.get("PROJECTZEO_HIPPORAG_MAX_NODES", "5000"))
_PPR_ALPHA    = float(os.environ.get("PROJECTZEO_HIPPORAG_PPR_ALPHA", "0.85"))
_PPR_ITERS    = int(os.environ.get("PROJECTZEO_HIPPORAG_PPR_ITERS", "20"))
_TOP_K        = int(os.environ.get("PROJECTZEO_HIPPORAG_TOP_K", "10"))


@dataclass
class KGNode:
    node_id:    str
    label:      str
    node_type:  str   # "entity" | "fact" | "app" | "task"
    weight:     float = 1.0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class KGEdge:
    src:       str
    dst:       str
    relation:  str
    weight:    float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HippoRAG:
    """
    In-process knowledge graph with Personalized PageRank retrieval.

    Nodes are entities/facts. Edges are semantic relations.
    PPR retrieval finds multi-hop relevant context given a query.
    """

    def __init__(self, store_path: Optional[str] = None) -> None:
        self._path  = store_path or _STORE_PATH
        self._lock  = threading.Lock()
        self._nodes: Dict[str, KGNode] = {}
        self._adj:   Dict[str, List[Tuple[str, float]]] = defaultdict(list)  # node_id → [(dst, weight)]
        self._rev:   Dict[str, List[Tuple[str, float]]] = defaultdict(list)  # reversed
        self._load()
        _logger.debug("[HippoRAG] Loaded %d nodes.", len(self._nodes))

    # -------------------------------------------------------------------------
    # Ingestion
    # -------------------------------------------------------------------------

    def add_node(self, node_id: str, label: str, node_type: str = "entity", weight: float = 1.0) -> None:
        with self._lock:
            if node_id not in self._nodes:
                self._nodes[node_id] = KGNode(node_id=node_id, label=label, node_type=node_type, weight=weight)
            if len(self._nodes) > _MAX_NODES:
                self._evict()

    def add_edge(self, src: str, dst: str, relation: str, weight: float = 1.0) -> None:
        with self._lock:
            self._adj[src].append((dst, weight))
            self._rev[dst].append((src, weight))

    def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        subject_type: str = "entity",
        object_type:  str = "entity",
        weight: float = 1.0,
    ) -> None:
        subj_id = f"e:{subject.lower()[:80]}"
        obj_id  = f"e:{obj.lower()[:80]}"
        self.add_node(subj_id, subject[:80], node_type=subject_type, weight=weight)
        self.add_node(obj_id, obj[:80], node_type=object_type, weight=weight)
        self.add_edge(subj_id, obj_id, predicate, weight=weight)

    def ingest_from_graphiti(self, graphiti_store) -> int:
        """Pull facts from GraphitiStore into the PPR graph."""
        if graphiti_store is None:
            return 0
        count = 0
        try:
            if hasattr(graphiti_store, "get_all_facts"):
                for fact in graphiti_store.get_all_facts():
                    self.add_fact(
                        subject=str(fact.get("subject", "")),
                        predicate=str(fact.get("predicate", "")),
                        obj=str(fact.get("object", "")),
                        weight=float(fact.get("confidence", 0.7)),
                    )
                    count += 1
        except Exception as e:
            _logger.debug("[HippoRAG] Graphiti ingest error: %s", e)
        return count

    # -------------------------------------------------------------------------
    # Retrieval — Personalized PageRank
    # -------------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = _TOP_K,
    ) -> List[Dict[str, Any]]:
        """
        Multi-hop retrieval via PPR seeded from query-matching nodes.

        1. Find seed nodes whose labels overlap with query tokens
        2. Run PPR on the graph from those seeds
        3. Return top-k nodes by PPR score
        """
        with self._lock:
            if not self._nodes:
                return []

            seed_ids = self._find_seeds(query)
            if not seed_ids:
                return self._fallback_bm25(query, top_k)

            scores = self._ppr(seed_ids)

            ranked = sorted(self._nodes.keys(), key=lambda n: -scores.get(n, 0.0))
            results = []
            for nid in ranked[:top_k]:
                node = self._nodes[nid]
                neighbors = [dst for dst, _ in self._adj.get(nid, [])]
                results.append({
                    "node_id":   nid,
                    "label":     node.label,
                    "type":      node.node_type,
                    "ppr_score": round(scores.get(nid, 0.0), 4),
                    "neighbors": neighbors[:5],
                })
            return results

    def _find_seeds(self, query: str) -> Set[str]:
        tokens = set(query.lower().split())
        seeds: Set[str] = set()
        for nid, node in self._nodes.items():
            label_tokens = set(node.label.lower().split())
            if tokens & label_tokens:
                seeds.add(nid)
            if len(seeds) >= 20:
                break
        return seeds

    def _ppr(self, seeds: Set[str]) -> Dict[str, float]:
        """
        Personalised PageRank with restart probability (1 - alpha).

        R_{t+1} = α * A^T R_t + (1-α) * s
        where s = uniform over seed nodes.
        """
        all_nodes = list(self._nodes.keys())
        n         = len(all_nodes)
        if n == 0:
            return {}

        idx = {nid: i for i, nid in enumerate(all_nodes)}

        # Personalisation vector
        s = {nid: 0.0 for nid in all_nodes}
        seed_w = 1.0 / max(len(seeds), 1)
        for seed in seeds:
            if seed in s:
                s[seed] = seed_w

        r = dict(s)  # initial distribution

        for _ in range(_PPR_ITERS):
            new_r: Dict[str, float] = {nid: 0.0 for nid in all_nodes}
            for src, edges in self._adj.items():
                if not edges:
                    continue
                total_w = sum(w for _, w in edges)
                for dst, w in edges:
                    if dst in new_r:
                        new_r[dst] += _PPR_ALPHA * (w / total_w) * r.get(src, 0.0)
            # Add restart
            for nid in all_nodes:
                new_r[nid] += (1.0 - _PPR_ALPHA) * s[nid]
            r = new_r

        return r

    def _fallback_bm25(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Simple BF text search when no seeds found."""
        tokens = query.lower().split()
        scored: List[Tuple[float, str]] = []
        for nid, node in self._nodes.items():
            score = sum(1 for t in tokens if t in node.label.lower())
            if score > 0:
                scored.append((score, nid))
        scored.sort(reverse=True)
        return [
            {"node_id": nid, "label": self._nodes[nid].label,
             "type": self._nodes[nid].node_type, "ppr_score": 0.0, "neighbors": []}
            for _, nid in scored[:top_k]
        ]

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def _evict(self) -> None:
        sorted_nodes = sorted(self._nodes.values(), key=lambda n: n.created_at)
        to_remove = sorted_nodes[:len(self._nodes) - _MAX_NODES + 500]
        for node in to_remove:
            nid = node.node_id
            del self._nodes[nid]
            self._adj.pop(nid, None)
            self._rev.pop(nid, None)

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            for nd in data.get("nodes", []):
                self._nodes[nd["node_id"]] = KGNode(**{
                    k: v for k, v in nd.items() if k in KGNode.__dataclass_fields__
                })
            for ed in data.get("edges", []):
                self._adj[ed["src"]].append((ed["dst"], float(ed.get("weight", 1.0))))
                self._rev[ed["dst"]].append((ed["src"], float(ed.get("weight", 1.0))))
        except Exception as e:
            _logger.warning("[HippoRAG] Load error: %s", e)

    def save(self) -> None:
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        tmp = self._path + ".tmp"
        try:
            with self._lock:
                edges: List[Dict] = []
                for src, dsts in self._adj.items():
                    for dst, w in dsts:
                        edges.append({"src": src, "dst": dst, "weight": w, "relation": ""})
                data = {
                    "nodes": [n.to_dict() for n in self._nodes.values()],
                    "edges": edges,
                }
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, separators=(",", ":"))
            os.replace(tmp, self._path)
        except Exception as e:
            _logger.debug("[HippoRAG] Save error: %s", e)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "nodes":      len(self._nodes),
                "edge_count": sum(len(v) for v in self._adj.values()),
                "store_path": self._path,
            }


_instance: Optional[HippoRAG] = None
_instance_lock = threading.Lock()


def get_hippo_rag() -> HippoRAG:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = HippoRAG()
    return _instance
