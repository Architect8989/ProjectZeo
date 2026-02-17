from typing import Dict, Any, List, Tuple
import math


class SemanticResolver:
    """
    Grounded semantic resolver.

    Converts natural-language target description
    into ranked world_graph entity candidates.

    Assumes world_graph.snapshot() returns:

    {
        "timestamp": ...,
        "focused_app": ...,
        "entities": [ {entity_dict}, ... ],
        "entity_count": ...
    }

    Entity schema (expected):
        {
            "id": str,
            "type": str,
            "text": str,
            "x": float (0-1),
            "y": float (0-1),
            "interactable": bool,
            "confidence": float,
            ...
        }
    """

    MIN_CONFIDENCE = 0.55

    def __init__(self, world_graph):
        self._world_graph = world_graph

    # ==================================================
    # PUBLIC API
    # ==================================================

    def resolve(self, description: str) -> Dict[str, Any]:

        if not isinstance(description, str) or not description.strip():
            return {"confidence": 0.0}

        snapshot = self._world_graph.snapshot() or {}
        entity_list = snapshot.get("entities", [])

        if not isinstance(entity_list, list) or not entity_list:
            return {"confidence": 0.0}

        candidates = self._extract_interactive_entities(entity_list)

        if not candidates:
            return {"confidence": 0.0}

        scored: List[Tuple[float, Dict[str, Any]]] = []

        for entity in candidates:
            score = self._score(description, entity)
            if score > 0.0:
                scored.append((score, entity))

        if not scored:
            return {"confidence": 0.0}

        scored.sort(key=lambda x: x[0], reverse=True)

        best_score, best_entity = scored[0]

        alternatives = [
            {"entity": e, "score": s}
            for s, e in scored[1:3]
        ]

        confidence = self._calibrate_confidence(best_score, scored)

        if confidence < self.MIN_CONFIDENCE:
            return {
                "confidence": confidence,
                "alternatives": alternatives,
            }

        return {
            "entity": best_entity,
            "confidence": confidence,
            "alternatives": alternatives,
        }

    # ==================================================
    # ENTITY FILTERING
    # ==================================================

    def _extract_interactive_entities(
        self,
        entity_list: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        interactive = []

        for entity in entity_list:

            if not isinstance(entity, dict):
                continue

            if not entity.get("interactable", True):
                continue

            entity_type = str(entity.get("type", "")).lower()

            if entity_type in {
                "button",
                "link",
                "input",
                "checkbox",
                "menuitem",
                "tab",
                "icon",
            }:
                interactive.append(entity)

        return interactive

    # ==================================================
    # SCORING
    # ==================================================

    def _score(self, description: str, entity: Dict[str, Any]) -> float:

        label = entity.get("text", "") or ""
        if not isinstance(label, str) or not label.strip():
            return 0.0

        desc_tokens = self._tokenize(description)
        label_tokens = self._tokenize(label)

        if not desc_tokens or not label_tokens:
            return 0.0

        # ---- Token Overlap ----
        overlap = len(set(desc_tokens) & set(label_tokens))
        token_score = overlap / max(len(desc_tokens), 1)

        # ---- Substring Match ----
        substring_score = (
            1.0 if description.lower() in label.lower() else 0.0
        )

        # ---- Spatial Prominence ----
        spatial_score = self._spatial_prominence(entity)

        # ---- Visual Confidence Boost ----
        visual_conf = float(entity.get("confidence", 0.0))
        visual_score = max(0.0, min(1.0, visual_conf))

        # Weighted combination (bounded 0–1)
        score = (
            0.4 * token_score +
            0.2 * substring_score +
            0.2 * spatial_score +
            0.2 * visual_score
        )

        return max(0.0, min(1.0, score))

    # ==================================================
    # TOKENIZATION
    # ==================================================

    def _tokenize(self, text: str) -> List[str]:
        return [
            t.strip().lower()
            for t in text.split()
            if t.strip()
        ]

    # ==================================================
    # SPATIAL PROMINENCE
    # ==================================================

    def _spatial_prominence(self, entity: Dict[str, Any]) -> float:

        x = entity.get("x")
        y = entity.get("y")

        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return 0.0

        # Assume normalized coordinates [0,1]
        dx = float(x) - 0.5
        dy = float(y) - 0.5

        distance = math.sqrt(dx * dx + dy * dy)

        # Normalize so center = 1.0, edges ≈ 0
        return max(0.0, 1.0 - min(distance, 1.0))

    # ==================================================
    # CONFIDENCE CALIBRATION
    # ==================================================

    def _calibrate_confidence(
        self,
        best_score: float,
        scored: List[Tuple[float, Dict[str, Any]]],
    ) -> float:

        if not scored:
            return 0.0

        if len(scored) == 1:
            return max(0.0, min(1.0, best_score))

        second_score = scored[1][0]

        margin = best_score - second_score

        # Stable margin-based confidence
        confidence = best_score * (1.0 + margin)

        return max(0.0, min(1.0, confidence))
