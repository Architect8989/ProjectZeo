from __future__ import annotations

import time
import threading
import hashlib
from typing import Dict, Any, List, Optional
import copy


class WorldGraphError(RuntimeError):
    pass


MAX_ENTITIES = 5000
MAX_HISTORY = 2000
ENTITY_STALE_SECONDS = 30.0
COORD_QUANT = 0.0001


class WorldGraph:
    """
    Incremental semantic world model.

    CONTRACT:
    - All mutation under lock
    - Snapshot is immutable copy
    - Ingestion reflects authoritative latest frame
    - No unbounded growth
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._entities: Dict[str, Dict[str, Any]] = {}
        self._focused_app: Optional[str] = None
        self._last_frame_ts: Optional[float] = None
        self._history: List[Dict[str, Any]] = []

    # -------------------------------------------------
    # TASK ISOLATION
    # -------------------------------------------------

    def reset(self) -> None:
        with self._lock:
            self._entities.clear()
            self._focused_app = None
            self._last_frame_ts = None
            self._history.clear()

    # -------------------------------------------------
    # INGESTION
    # -------------------------------------------------

    def ingest(self, perception: Dict[str, Any]) -> None:
        if not isinstance(perception, dict):
            return

        frame_ts = perception.get("frame_ts")
        elements = perception.get("elements")

        if not isinstance(frame_ts, (int, float)):
            return
        if not isinstance(elements, list):
            return

        now = time.monotonic()

        with self._lock:

            # Reject stale frames
            if (
                self._last_frame_ts is not None
                and frame_ts <= self._last_frame_ts
            ):
                return

            self._last_frame_ts = frame_ts
            self._focused_app = perception.get("focused_app")

            new_entities: Dict[str, Dict[str, Any]] = {}

            for el in elements:
                if not isinstance(el, dict):
                    continue

                entity_id = self._stable_entity_id(el)

                # Preserve temporal continuity
                prev = self._entities.get(entity_id)
                first_seen = prev["first_seen"] if prev else now

                new_entities[entity_id] = {
                    "id": entity_id,
                    "type": el.get("type"),
                    "text": el.get("text"),
                    "x": self._safe_float(el.get("x")),
                    "y": self._safe_float(el.get("y")),
                    "interactable": el.get("interactable"),
                    "state": el.get("state"),
                    "first_seen": first_seen,
                    "last_seen": now,
                    "confidence": 1.0,
                }

            # Replace frame atomically
            self._entities = new_entities

            # Remove stale entities (temporal guard)
            cutoff = now - ENTITY_STALE_SECONDS
            self._entities = {
                eid: ent
                for eid, ent in self._entities.items()
                if ent["last_seen"] >= cutoff
            }

            # Hard cap enforcement
            if len(self._entities) > MAX_ENTITIES:
                self._entities = dict(
                    list(self._entities.items())[:MAX_ENTITIES]
                )

            self._record_history_locked()

    def update(self, perception: Dict[str, Any]) -> None:
        self.ingest(perception)

    # -------------------------------------------------
    # SNAPSHOT
    # -------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
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
        if isinstance(perception, dict):
            self.ingest(perception)
        return self.snapshot()

    # -------------------------------------------------
    # QUERIES
    # -------------------------------------------------

    def find_by_text(
        self,
        *,
        contains: Optional[str] = None,
        exact: Optional[str] = None,
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

    def find_by_type(self, entity_type: str) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                copy.deepcopy(ent)
                for ent in self._entities.values()
                if ent.get("type") == entity_type
            ]

    def focused_application(self) -> Optional[str]:
        with self._lock:
            return self._focused_app

    def entity_count(self) -> int:
        with self._lock:
            return len(self._entities)

    # -------------------------------------------------
    # HISTORY
    # -------------------------------------------------

    def history(self) -> List[Dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._history)

    # -------------------------------------------------
    # INTERNALS
    # -------------------------------------------------

    def _safe_float(self, value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return 0.0

    def _stable_entity_id(self, el: Dict[str, Any]) -> str:
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

    def _record_history_locked(self) -> None:
        snapshot = {
            "ts": self._last_frame_ts,
            "entity_count": len(self._entities),
            "focused_app": self._focused_app,
        }

        self._history.append(snapshot)

        if len(self._history) > MAX_HISTORY:
            self._history.pop(0)
