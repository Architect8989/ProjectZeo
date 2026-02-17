# core/vision/semantic_resolver.py

from typing import Dict, Any, List


class SemanticResolver:
    """
    Maps natural language action targets
    to world_graph entities.
    """

    def __init__(self, world_graph):
        self._world_graph = world_graph

    def resolve(self, description: str) -> Dict[str, Any]:
        entities = self._world_graph.snapshot()

        candidates = []

        for entity_id, entity in entities.items():
            label = entity.get("label", "")
            if description.lower() in label.lower():
                candidates.append(entity)

        if not candidates:
            return {"confidence": 0.0}

        # naive ranking for now
        best = candidates[0]

        return {
            "entity": best,
            "confidence": 0.6 if len(candidates) == 1 else 0.4,
      }
