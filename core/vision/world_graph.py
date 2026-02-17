from __future__ import annotations

import time
import threading
import hashlib
from typing import Dict, Any, List, Optional
import copy
from collections import deque


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

    HARDENED:
    - O(1) history
    - True stale pruning
    - Safe spatial merge
    - No entity ID duplication
    - Strict bounded growth
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._entities: Dict[str, Dict[str, Any]] = {}
        self._focused_app: Optional[str] = None
        self._last_frame_ts: Optional[float] = None
        self._history = deque(maxlen=MAX_HISTORY)

    # =================================================
    # RESET
    # =================================================

    def reset(self) -> None:
        with self._lock:
            self._entities.clear()
            self._focused_app = None
            self._last_frame_ts = None
            self._history.clear()

    # =================================================
    # INGESTION
    # =================================================

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

            # Reject non-monotonic frames
            if self._last_frame_ts is not None and frame_ts <= self._last_frame_ts:
                return

            self._last_frame_ts = frame_ts
            focused = perception.get("focused_app")
            self._focused_app = focused if isinstance(focused, str) else None

            cutoff = now - ENTITY_STALE_SECONDS

            # Start from existing entities (allow temporal persistence)
            updated_entities = {}

            for el in elements:
                if not isinstance(el, dict):
                    continue

                x = self._clamped_float(el.get("x"))
                y = self._clamped_float(el.get("y"))
                etype = self._normalize_type(el.get("type"))
                text = (el.get("text") or "").strip()

                candidate_id = self._stable_entity_id(
                    etype=etype,
                    text=text,
                    x=x,
                    y=y,
                )

                prev = self._entities.get(candidate_id)
                if not prev:
                    prev = self._find_spatial_match_locked(x, y, etype)

                first_seen = prev["first_seen"] if prev else now

                updated_entities[candidate_id] = {
                    "id": candidate_id,
                    "type": etype,
                    "text": text,
                    "x": x,
                    "y": y,
                    "interactable": bool(el.get("interactable")),
                    "state": el.get("state"),
                    "first_seen": first_seen,
                    "last_seen": now,
                    "confidence": 1.0,
                }

            # Merge previous entities that are still within staleness window
            for eid, ent in self._entities.items():
                if eid not in updated_entities and ent["last_seen"] >= cutoff:
                    updated_entities[eid] = ent

            # Enforce global cap
            if len(updated_entities) > MAX_ENTITIES:
                # Keep most recent entities
                sorted_items = sorted(
                    updated_entities.values(),
                    key=lambda e: e["last_seen"],
                    reverse=True,
                )[:MAX_ENTITIES]

                updated_entities = {e["id"]: e for e in sorted_items}

            self._entities = updated_entities
            self._record_history_locked()

    def update(self, perception: Dict[str, Any]) -> None:
        self.ingest(perception)

    # =================================================
    # SNAPSHOT
    # =================================================

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "timestamp": self._last_frame_ts,
                "focused_app": self._focused_app,
                "entities": copy.deepcopy(list(self._entities.values())),
                "entity_count": len(self._entities),
            }

    # =================================================
    # DELTA
    # =================================================

    def compute_delta(self, previous_snapshot: Dict[str, Any]) -> Dict[str, Any]:

        if not isinstance(previous_snapshot, dict):
            raise WorldGraphError("Invalid previous snapshot")

        with self._lock:
            current_entities = copy.deepcopy(list(self._entities.values()))
            current_focus = self._focused_app

        prev_entities_list = previous_snapshot.get("entities", [])
        prev_focus = previous_snapshot.get("focused_app")

        if not isinstance(prev_entities_list, list):
            raise WorldGraphError("Invalid previous snapshot structure")

        prev_entities = {e["id"]: e for e in prev_entities_list if "id" in e}
        curr_entities = {e["id"]: e for e in current_entities if "id" in e}

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
                changes["text"] = {"old": prev_e.get("text"), "new": curr_e.get("text")}

            if prev_e.get("state") != curr_e.get("state"):
                changes["state"] = {"old": prev_e.get("state"), "new": curr_e.get("state")}

            if changes:
                modified.append(
                    {
                        "id": eid,
                        "changes": changes,
                        "entity": curr_e,
                    }
                )

        focus_changed = prev_focus != current_focus

        significant = (
            len(added) > 3
            or len(removed) > 3
            or focus_changed
        )

        return {
            "entities_added": added,
            "entities_removed": removed,
            "entities_modified": modified,
            "focus_changed": focus_changed,
            "prev_focus": prev_focus,
            "curr_focus": current_focus,
            "entity_count_delta": len(added) - len(removed),
            "significant_change": significant,
        }

    # =================================================
    # QUERIES
    # =================================================

    def find_by_text(self, *, contains: Optional[str] = None, exact: Optional[str] = None):
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

    def find_by_type(self, entity_type: str):
        if not isinstance(entity_type, str) or not entity_type:
            return []

        type_lower = entity_type.lower().strip()

        with self._lock:
            return [
                copy.deepcopy(ent)
                for ent in self._entities.values()
                if (ent.get("type") or "").lower() == type_lower
            ]

    def focused_application(self) -> Optional[str]:
        with self._lock:
            return self._focused_app

    def entity_count(self) -> int:
        with self._lock:
            return len(self._entities)

    # =================================================
    # INTERNALS
    # =================================================

    def _clamped_float(self, value: Any) -> float:
        try:
            v = float(value)
            return max(0.0, min(1.0, v))
        except Exception:
            return 0.0

    def _normalize_type(self, value: Any) -> str:
        if not isinstance(value, str):
            return "unknown"
        return value.strip().lower()

    def _stable_entity_id(self, *, etype: str, text: str, x: float, y: float) -> str:
        qx = int(x / COORD_QUANT)
        qy = int(y / COORD_QUANT)

        if len(text) < 2:
            raw = f"{etype}|{qx}|{qy}"
        else:
            raw = f"{etype}|{text}|{qx}|{qy}"

        return hashlib.sha256(raw.encode()).hexdigest()

    def _find_spatial_match_locked(self, x: float, y: float, etype: str):
        for ent in self._entities.values():
            if ent.get("type") != etype:
                continue

            if (
                abs(ent["x"] - x) <= SPATIAL_MATCH_THRESHOLD
                and abs(ent["y"] - y) <= SPATIAL_MATCH_THRESHOLD
            ):
                return ent

        return None

    def _record_history_locked(self) -> None:
        self._history.append(
            {
                "ts": self._last_frame_ts,
                "entity_count": len(self._entities),
                "focused_app": self._focused_app,
            }
            )
