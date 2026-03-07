from __future__ import annotations

import time
import threading
import hashlib
from typing import Dict, Any, List, Optional
import copy
from collections import deque
import math


class WorldGraphError(RuntimeError):
    pass


MAX_ENTITIES = 5000
MAX_HISTORY = 2000
# AUDIT-MEDIUM-1 FIX: Reduce stale entity timeout from 30s to 5s.
#
# Root cause: with observer_loop.pause() (old behaviour), entities that
# disappeared from the screen (closed dialogs, dismissed popups, navigated
# pages) remained in the world graph for 30 seconds. This caused the planner
# to target non-existent UI elements and produced spurious click actions on
# ghost entities.
#
# Fix: default stale threshold reduced to 5s (configurable via env var for
# operators who need a longer window).  At 5 Hz full inference rate, an entity
# must be absent from 25 consecutive frames before being evicted — sufficient
# to handle brief occlusions without holding stale state for 30s.
#
# In lightweight mode (1 Hz screenshot-only, no entity extraction), the
# stale timer is artificially extended by the calling code to avoid mass
# eviction during execution phases where the VL model is not running.
ENTITY_STALE_SECONDS: float = float(
    __import__("os").environ.get("PROJECTZEO_ENTITY_STALE_SECONDS", "5.0")
)
ENTITY_STALE_SECONDS_LIGHTWEIGHT: float = float(
    __import__("os").environ.get("PROJECTZEO_ENTITY_STALE_SECONDS_LIGHTWEIGHT", "120.0")
)
COORD_QUANT = 0.0001
SPATIAL_MATCH_THRESHOLD = 0.01


class WorldGraph:

    def __init__(self):
        self._lock = threading.RLock()
        self._entities: Dict[str, Dict[str, Any]] = {}
        self._focused_app: Optional[str] = None
        self._last_frame_ts: Optional[float] = None
        self._history = deque(maxlen=MAX_HISTORY)
        # CRIT-5 FIX: Track whether current frame is from a browser context.
        self._browser_context: bool = False

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
        # CRIT-1 FIX: ObserverLoop._validate_perception_schema() normalises the
        # VisionRuntime output key "elements" → "entities" before handing the
        # dict to downstream consumers (WorldGraph, planner, belief state).
        # WorldGraph.ingest() was still reading "elements" — so entity_count
        # was permanently 0 regardless of what the vision model returned.
        # Fix: accept either key, preferring the canonical "entities".
        elements = perception.get("entities") or perception.get("elements")

        # BUG-07 FIX: The original code returned early when frame_ts was not
        # an int/float (e.g. None, which ObserverLoop._validate_perception_schema()
        # can produce when conversion fails).  This silently dropped all entity
        # ingestion, leaving the world graph permanently stale on CPU-only hardware
        # where VisionRuntime occasionally emits non-numeric timestamps.
        # Fix: fall back to wall-clock time rather than discarding the frame.
        if not isinstance(frame_ts, (int, float)):
            frame_ts = time.time()

        if not isinstance(elements, list):
            return

        now = time.monotonic()

        with self._lock:

            if self._last_frame_ts is not None and frame_ts <= self._last_frame_ts:
                return

            self._last_frame_ts = frame_ts

            focused = perception.get("focused_app")
            self._focused_app = focused if isinstance(focused, str) else None
            # CRIT-5 FIX: Capture the browser-context flag from vision_runtime.py.
            self._browser_context = bool(perception.get("_browser_context", False))

            # AUDIT-MEDIUM-1 FIX: Use extended stale window in lightweight mode
            # (observer running screenshot-only with no entity extraction).
            # Without this, all entities would be evicted within 5s when the
            # VL model is suspended during execution, losing all world model state.
            _is_lightweight = bool(perception.get("_lightweight", False))
            _stale_threshold = (
                ENTITY_STALE_SECONDS_LIGHTWEIGHT if _is_lightweight
                else ENTITY_STALE_SECONDS
            )
            cutoff = now - _stale_threshold
            updated_entities: Dict[str, Dict[str, Any]] = {}

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
                    # CRIT-5 FIX: Preserve _external_content_source flag from
                    # vision_runtime.py.  Once True it is never overridden to False
                    # so browser DOM entities remain flagged across world-state snapshots.
                    "_external_content_source": bool(el.get("_external_content_source", False)),
                }

            for eid, ent in self._entities.items():
                if eid not in updated_entities and ent["last_seen"] >= cutoff:
                    updated_entities[eid] = ent

            if len(updated_entities) > MAX_ENTITIES:
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
                # CRIT-5 FIX: Propagate browser-context flag to consumers
                # (operate.py, consequence_reasoner) so they know all entities
                # may originate from untrusted external web content.
                "_browser_context": self._browser_context,
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
            if math.isnan(v) or math.isinf(v):
                return 0.0
            return max(0.0, min(1.0, v))
        except Exception:
            return 0.0

    def _normalize_type(self, value: Any) -> str:
        if not isinstance(value, str):
            return "unknown"
        return value.strip().lower()

    def _quantize(self, value: float) -> int:
        return int(round(value / COORD_QUANT))

    def _stable_entity_id(self, *, etype: str, text: str, x: float, y: float) -> str:
        qx = self._quantize(x)
        qy = self._quantize(y)

        text_norm = text.strip().lower()

        if text_norm:
            text_hash = hashlib.sha256(text_norm.encode()).hexdigest()[:8]
        else:
            text_hash = "∅"

        raw = f"{etype}|{text_hash}|{qx}|{qy}"

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

    def proactive_cleanup(self, *, lightweight_mode: bool = False) -> int:
        """
        AUDIT-MEDIUM-1: Force-evict entities that have exceeded the stale
        threshold based on wall-clock time.  Call periodically (e.g. every
        10s) when VL inference is running at reduced rate or after task
        completion to ensure no ghost entities survive into the next task.

        AUDIT FIX: Accepts ``lightweight_mode`` parameter so callers in the
        ObserverLoop can pass the current mode flag.  When ``lightweight_mode``
        is True (1 Hz / inference-in-progress), the extended 120-second window
        (``ENTITY_STALE_SECONDS_LIGHTWEIGHT``) is used to prevent mass-eviction
        of legitimately present entities during long CPU inference cycles.

        Returns the number of entities evicted.
        """
        now = time.monotonic()
        stale_threshold = (
            ENTITY_STALE_SECONDS_LIGHTWEIGHT if lightweight_mode
            else ENTITY_STALE_SECONDS
        )
        cutoff = now - stale_threshold

        with self._lock:
            before = len(self._entities)
            self._entities = {
                eid: ent
                for eid, ent in self._entities.items()
                if ent["last_seen"] >= cutoff
            }
            evicted = before - len(self._entities)

        if evicted > 0:
            import logging as _log
            _log.getLogger(__name__).debug(
                "[WorldGraph] proactive_cleanup: evicted %d stale entities "
                "(threshold=%.1fs, lightweight=%s).",
                evicted, stale_threshold, lightweight_mode,
            )

        return evicted
