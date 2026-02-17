# core/vision/semantic_resolver.py

from typing import Dict, Any, List, Tuple
import math


class SemanticResolver:
    """
    Grounded semantic resolver.

    Converts natural-language target description
    into ranked world_graph entity candidates.

    Scoring factors:
        - Token overlap
        - Partial string similarity
        - Entity visibility
        - Spatial prominence
        - Recent change boost
    """

    MIN_CONFIDENCE = 0.55

    def __init__(self, world_graph):
        self._world_graph = world_graph

    # ==================================================
    # PUBLIC API
    # ==================================================

    def resolve(self, description: str) -> Dict[str, Any]:

        if not description or not isinstance(description, str):
            return {"confidence": 0.0}

        entities = self._world_graph.snapshot() or {}

        candidates = self._extract_interactive_entities(entities)

        if not candidates:
            return {"confidence": 0.0}

        scored = [
            (self._score(description, entity), entity)
            for entity in candidates
        ]

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
        entities: Dict[str, Any],
    ) -> List[Dict[str, Any]]:

        interactive = []

        for entity in entities.values():

            if not isinstance(entity, dict):
                continue

            if not entity.get("visible", True):
                continue

            entity_type = entity.get("type", "").lower()

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

        label = entity.get("label", "") or ""
        desc_tokens = self._tokenize(description)
        label_tokens = self._tokenize(label)

        if not label_tokens:
            return 0.0

        token_overlap = len(set(desc_tokens) & set(label_tokens))
        token_score = token_overlap / max(len(desc_tokens), 1)

        substring_score = 1.0 if description.lower() in label.lower() else 0.0

        spatial_score = self._spatial_prominence(entity)

        recency_score = 1.0 if entity.get("recently_changed") else 0.0

        return (
            0.4 * token_score +
            0.2 * substring_score +
            0.2 * spatial_score +
            0.2 * recency_score
        )

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

        coords = entity.get("coordinates")

        if not coords or not isinstance(coords, dict):
            return 0.0

        x = coords.get("x", 0.5)
        y = coords.get("y", 0.5)

        # assume normalized coordinates 0-1
        center_distance = math.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)

        return max(0.0, 1.0 - center_distance)

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
            return min(1.0, best_score + 0.1)

        second_score = scored[1][0]

        margin = best_score - second_score

        # strong margin increases confidence
        confidence = best_score * (1.0 + margin)

        return max(0.0, min(1.0, confidence))
