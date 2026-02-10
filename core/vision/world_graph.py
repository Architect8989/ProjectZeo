from __future__ import annotations

import time
import threading
import hashlib
from typing import Dict, Any, List, Optional
import copy


# -------------------------------------------------
# ERRORS
# -------------------------------------------------


class WorldGraphError(RuntimeError):
    pass


# -------------------------------------------------
# CONFIG
# -------------------------------------------------


MAX_ENTITIES = 5000
MAX_HISTORY = 2000
ENTITY_STALE_SECONDS = 30.0

# spatial quantization to reduce jitter but avoid collisions
COORD_QUANT = 0.001  # ~0.1% of screen


# -------------------------------------------------
# WORLD GRAPH
# -------------------------------------------------


class WorldGraph:
    """
    Incremental semantic world model.

    THINK OF THIS AS:
    - The system's subconscious
    - A continuously updated belief graph
    - Evidence ledger for planners and verifiers
    """

    def __init__(self):
        self._lock = threading.RLock()

        self._entities: Dict[str, Dict[str, Any]] = {}
        self._focused_app: Optional[str] = None
        self._last_frame_ts: Optional[float] = None

        self._history: List[Dict[str, Any]] = []

    # -------------------------------------------------
    # INGESTION (VISION ONLY)
    # -------------------------------------------------

    def ingest(self, perception: Dict[str, Any]) -> None:
        """
        Merge a new perception frame into the world graph.

        Only VisionRuntime / ObserverLoop may call this.
        """
        if not isinstance(perception, dict):
            return

        frame_ts = perception.get("frame_ts")
        if not isinstance(frame_ts, (int, float)):
            return

        elements = perception.get("elements")
        if not isinstance(elements, list):
            return

        now = time.monotonic()

        with self._lock:
            self._last_frame_ts = frame_ts
            self._focused_app = perception.get("focused_app")

            for el in elements:
                if not isinstance(el, dict):
                    continue

                entity_id = self._stable_entity_id(el)

                if entity_id not in self._entities:
                    self._entities[entity_id] = {
                        "id": entity_id,
                        "type": el.get("type"),
                        "text": el.get("text"),
                        "x": el.get("x"),
                        "y": el.get("y"),
                        "first_seen": now,
                        "last_seen": now,
                        "confidence": 0.5,
                    }
                else:
                    ent = self._entities[entity_id]
                    ent["last_seen"] = now
                    ent["x"] = el.get("x")
                    ent["y"] = el.get("y")
                    ent["confidence"] = min(
                        ent.get("confidence", 0.5) + 0.05, 1.0
                    )

            self._prune(now=now)
            self._record_history()

    # ---- compatibility alias (CRITICAL) ----
    def update(self, perception: Dict[str, Any]) -> None:
        """
        Compatibility alias.
        """
        self.ingest(perception)

    # -------------------------------------------------
    # SNAPSHOT API
    # -------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """
        Immutable snapshot for planner / verifier consumption.
        """
        with self._lock:
            return copy.deepcopy(
                {
                    "timestamp": self._last_frame_ts,
                    "focused_app": self._focused_app,
                    "entities": list(self._entities.values()),
                    "entity_count": len(self._entities),
                }
            )

    # -------------------------------------------------
    # QUERIES (READ-ONLY)
    # -------------------------------------------------

    def find_by_text(
        self, *, contains: Optional[str] = None, exact: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic lookup, not pixel search.
        """
        if not contains and not exact:
            return []

        contains_l = contains.lower() if contains else None
        exact_l = exact.lower() if exact else None

        with self._lock:
            results = []
            for ent in self._entities.values():
                text = (ent.get("text") or "").lower()

                if exact_l is not None and text != exact_l:
                    continue
                if contains_l is not None and contains_l not in text:
                    continue

                results.append(copy.deepcopy(ent))

            return results

    def focused_application(self) -> Optional[str]:
        with self._lock:
            return self._focused_app

    def entity_count(self) -> int:
        with self._lock:
            return len(self._entities)

    # -------------------------------------------------
    # HISTORY (FORENSICS)
    # -------------------------------------------------

    def history(self) -> List[Dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._history)

    # -------------------------------------------------
    # INTERNALS
    # -------------------------------------------------

    def _stable_entity_id(self, el: Dict[str, Any]) -> str:
        """
        Deterministic identity across frames.

        Uses quantized spatial buckets to avoid jitter
        without introducing collisions.
        """
        try:
            x = float(el.get("x", 0.0))
            y = float(el.get("y", 0.0))
        except Exception:
            x = 0.0
            y = 0.0

        qx = int(x / COORD_QUANT)
        qy = int(y / COORD_QUANT)

        raw = (
            f"{el.get('type')}|"
            f"{el.get('text')}|"
            f"{qx}|{qy}"
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    def _prune(self, *, now: float) -> None:
        """
        Remove stale or excessive entities.
        """
        stale_ids = [
            eid
            for eid, ent in self._entities.items()
            if now - ent.get("last_seen", now) > ENTITY_STALE_SECONDS
        ]

        for eid in stale_ids:
            del self._entities[eid]

        if len(self._entities) > MAX_ENTITIES:
            # drop lowest confidence first (deterministic)
            sorted_ids = sorted(
                self._entities.items(),
                key=lambda kv: kv[1].get("confidence", 0.0),
            )
            overflow = len(self._entities) - MAX_ENTITIES
            for eid, _ in sorted_ids[:overflow]:
                del self._entities[eid]

    def _record_history(self) -> None:
        snapshot = {
            "ts": self._last_frame_ts,
            "entity_count": len(self._entities),
            "focused_app": self._focused_app,
        }
        self._history.append(snapshot)

        if len(self._history) > MAX_HISTORY:
            self._history.pop(0)
