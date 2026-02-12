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
SPATIAL_MATCH_THRESHOLD = 0.01


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

            if (
                self._last_frame_ts is not None
                and frame_ts <= self._last_frame_ts
            ):
                return

            self._last_frame_ts = frame_ts
            self._focused_app = (
                perception.get("focused_app")
                if isinstance(perception.get("focused_app"), str)
                else None
            )

            updated_entities: Dict[str, Dict[str, Any]] = {}

            for el in elements:
                if not isinstance(el, dict):
                    continue

                x = self._safe_float(el.get("x"))
                y = self._safe_float(el.get("y"))

                entity_id = self._stable_entity_id(el)

                prev = self._entities.get(entity_id)
                if not prev:
                    prev = self._find_spatial_match(x, y)

                first_seen = prev["first_seen"] if prev else now

                updated_entities[entity_id] = {
                    "id": entity_id,
                    "type": el.get("type"),
                    "text": el.get("text"),
                    "x": x,
                    "y": y,
                    "interactable": bool(el.get("interactable")),
                    "state": el.get("state"),
                    "first_seen": first_seen,
                    "last_seen": now,
                    "confidence": 1.0,
                }

            self._entities = updated_entities

            cutoff = now - ENTITY_STALE_SECONDS
            self._entities = {
                eid: ent
                for eid, ent in self._entities.items()
                if ent["last_seen"] >= cutoff
            }

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

    # -------------------------------------------------
    # DELTA COMPUTATION (NEW)
    # -------------------------------------------------

    def compute_delta(
        self,
        previous_snapshot: Dict[str, Any]
    ) -> Dict[str, Any]:

        if not isinstance(previous_snapshot, dict):
            raise WorldGraphError("Invalid previous snapshot")

        with self._lock:
            current_snapshot = self.snapshot()

        prev_entities = {
            e["id"]: e
            for e in previous_snapshot.get("entities", [])
        }

        curr_entities = {
            e["id"]: e
            for e in current_snapshot.get("entities", [])
        }

        added_ids = set(curr_entities) - set(prev_entities)
        removed_ids = set(prev_entities) - set(curr_entities)
        common_ids = set(prev_entities) & set(curr_entities)

        added = [curr_entities[eid] for eid in added_ids]
        removed = [prev_entities[eid] for eid in removed_ids]

        modified = []

        for eid in common_ids:
            prev_e = prev_entities[eid]
            curr_e = curr_entities[eid]

            changes = {}

            if prev_e.get("text") != curr_e.get("text"):
                changes["text"] = {
                    "old": prev_e.get("text"),
                    "new": curr_e.get("text"),
                }

            if prev_e.get("state") != curr_e.get("state"):
                changes["state"] = {
                    "old": prev_e.get("state"),
                    "new": curr_e.get("state"),
                }

            if changes:
                modified.append({
                    "id": eid,
                    "changes": changes,
                    "entity": curr_e,
                })

        focus_changed = (
            previous_snapshot.get("focused_app")
            != current_snapshot.get("focused_app")
        )

        return {
            "entities_added": added,
            "entities_removed": removed,
            "entities_modified": modified,
            "focus_changed": focus_changed,
            "prev_focus": previous_snapshot.get("focused_app"),
            "curr_focus": current_snapshot.get("focused_app"),
            "entity_count_delta": len(added) - len(removed),
            "significant_change": (
                len(added) > 5
                or len(removed) > 5
                or focus_changed
            ),
        }

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

    def focused_application(self) -> Optional[str]:
        with self._lock:
            return self._focused_app

    def entity_count(self) -> int:
        with self._lock:
            return len(self._entities)

    # -------------------------------------------------
    # INTERNALS
    # -------------------------------------------------

    def _safe_float(self, value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return 0.0

    def _stable_entity_id(self, el: Dict[str, Any]) -> str:
        """
        Stable ID generation:
        - If text is meaningful → include it
        - If text short/empty → spatial only
        """

        try:
            x = float(el.get("x", 0.0))
            y = float(el.get("y", 0.0))
        except Exception:
            x = 0.0
            y = 0.0

        qx = int(x / COORD_QUANT)
        qy = int(y / COORD_QUANT)

        text = (el.get("text") or "").strip()
        el_type = el.get("type")

        if len(text) < 2:
            raw = f"{el_type}|{qx}|{qy}"
        else:
            raw = f"{el_type}|{text}|{qx}|{qy}"

        return hashlib.sha256(raw.encode()).hexdigest()

    def _find_spatial_match(
        self,
        x: float,
        y: float,
    ) -> Optional[Dict[str, Any]]:

        for ent in self._entities.values():
            if (
                abs(ent["x"] - x) <= SPATIAL_MATCH_THRESHOLD
                and abs(ent["y"] - y) <= SPATIAL_MATCH_THRESHOLD
            ):
                return ent

        return None

    def _record_history_locked(self) -> None:
        snapshot = {
            "ts": self._last_frame_ts,
            "entity_count": len(self._entities),
            "focused_app": self._focused_app,
        }

        self._history.append(snapshot)

        if len(self._history) > MAX_HISTORY:
            self._history.pop(0)
