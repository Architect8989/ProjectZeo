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

# finer spatial quantization (~0.01% of screen)
COORD_QUANT = 0.0001


# -------------------------------------------------
# WORLD GRAPH
# -------------------------------------------------


class WorldGraph:
    """
    Incremental semantic world model.

    ARCHITECTURAL CONTRACT:
    - Observer NEVER mutates this directly
    - Planner ingests perception on-demand
    - All mutation occurs under lock
    """

    def __init__(self):
        self._lock = threading.RLock()

        self._entities: Dict[str, Dict[str, Any]] = {}
        self._focused_app: Optional[str] = None
        self._last_frame_ts: Optional[float] = None

        self._history: List[Dict[str, Any]] = []

    # -------------------------------------------------
    # INGESTION (PLANNER-DRIVEN)
    # -------------------------------------------------

    def ingest(self, perception: Dict[str, Any]) -> None:
        """
        Merge a new perception frame into the world graph.

        Must be called under lock or via snapshot_from_perception().
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

    # ---- compatibility alias ----
    def update(self, perception: Dict[str, Any]) -> None:
        self.ingest(perception)

    # -------------------------------------------------
    # SNAPSHOT API
    # -------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """
        Immutable snapshot for planner / verifier consumption.
        """
        return copy.deepcopy(
            {
                "timestamp": self._last_frame_ts,
                "focused_app": self._focused_app,
                "entities": list(self._entities.values()),
                "entity_count": len(self._entities),
            }
        )

    def snapshot_from_perception(
        self, perception: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Atomic ingest + snapshot.

        Planner calls this using latest perception from ObserverCore.
        Observer must NOT mutate world graph directly.
        """
        with self._lock:
            if isinstance(perception, dict):
                self.ingest(perception)
            return self.snapshot()

    # -------------------------------------------------
    # QUERIES (READ-ONLY)
    # -------------------------------------------------

    def find_by_text(
        self, *, contains: Optional[str] = None, exact: Optional[str] = None
    ) -> List[Dict[str, Any]]:

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

        Uses finer quantization to reduce collision risk.
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
        stale_ids = [
            eid
            for eid, ent in self._entities.items()
            if now - ent.get("last_seen", now) > ENTITY_STALE_SECONDS
        ]

        for eid in stale_ids:
            del self._entities[eid]

        if len(self._entities) > MAX_ENTITIES:
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
