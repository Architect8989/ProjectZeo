from typing import Dict, Any, List
import json
import logging


from core.security.injection_markers import INJECTION_MARKERS, normalize_for_injection_check

_logger = logging.getLogger(__name__)


class ReasoningEngine:
    """
    Multi-hypothesis LLM reasoning with bounded perception
    and injection-resistant prompt construction.
    """

    MAX_ENTITIES = 20
    MAX_TEXT_CHARS = 400
    MAX_JSON_CHARS = 8000

    
    _INJECTION_MARKERS = INJECTION_MARKERS

    def __init__(self, llm_callable, *, ollama_client=None):
        
        self._llm = llm_callable
        self._ollama_client = ollama_client

    # ==================================================
    # PUBLIC ACTION PROPOSAL
    # ==================================================

    def propose_actions(
        self,
        *,
        objective: str,
        belief_summary: Dict[str, Any],
        perception: Dict[str, Any],
        k: int = 3,
    ) -> List[Dict[str, Any]]:

        safe_perception = self._sanitize_perception(perception)
        safe_belief = self._safe_json(belief_summary)

        payload = {
            "objective": objective,
            "belief_summary": safe_belief,
            "perception": safe_perception,
            "instructions": {
                "propose_k_actions": k,
                "format": "JSON list of action objects only",
            },
        }

        serialized = json.dumps(payload, ensure_ascii=False)

        if len(serialized) > self.MAX_JSON_CHARS:
            raise RuntimeError("Perception payload exceeds size limit")

        prompt = {
            "role": "user",
            "content": serialized,
        }

        
        if self._ollama_client is not None:
            try:
                import os as _os_h5
                _model = _os_h5.environ.get("LLM_TEXT_MODEL", "") or _os_h5.environ.get("LLM_MODEL", "")
                if _model:
                    response = self._ollama_client.chat(
                        model=_model,
                        messages=[{"role": "user", "content": serialized}],
                        format="json",
                    )
                    raw_content = (
                        response.get("message", {}).get("content", "")
                        if isinstance(response, dict)
                        else getattr(
                            getattr(response, "message", None), "content", ""
                        ) or ""
                    )
                    try:
                        parsed = __import__("json").loads(raw_content)
                    except Exception:
                        parsed = []
                    return self._normalize_actions(parsed)
            except Exception as _text_err:
                _logger.warning(
                    "[ReasoningEngine] Text-only path failed (%s), "
                    "falling back to vision adapter: %s",
                    type(_text_err).__name__, _text_err,
                )
                # Fall through to vision adapter below

        result = self._llm(
            messages=[prompt],
            objective=objective,
            session_id="cognition",
        )

        return self._normalize_actions(result)

    # ==================================================
    # PERCEPTION SANITIZATION
    # ==================================================

    def _sanitize_perception(
        self, perception: Dict[str, Any]
    ) -> Dict[str, Any]:

        if not isinstance(perception, dict):
            return {}

        safe: Dict[str, Any] = {}

        for key in ["focused_app", "entity_count"]:
            if key in perception:
                safe[key] = perception.get(key)

        entities = perception.get("entities", [])
        if isinstance(entities, list):

            bounded_entities = []

            for ent in entities[: self.MAX_ENTITIES]:

                if not isinstance(ent, dict):
                    continue

                safe_ent = {}

                for k, v in ent.items():

                    if isinstance(v, str):
                        safe_ent[k] = self._truncate_text(v)

                    elif isinstance(v, (int, float, bool)):
                        safe_ent[k] = v

                    elif isinstance(v, dict):
                        nested = {}
                        for nk, nv in v.items():
                            if isinstance(nv, str):
                                nested[nk] = self._truncate_text(nv)
                            elif isinstance(nv, (int, float, bool)):
                                nested[nk] = nv
                        safe_ent[k] = nested

                bounded_entities.append(safe_ent)

            safe["entities"] = bounded_entities

        return safe

    # ==================================================
    # TEXT TRUNCATION + INJECTION DAMPENING  (HAR-5 FIX)
    # ==================================================

    def _truncate_text(self, text: str) -> str:
        
        if not isinstance(text, str):
            return ""

        
        lowered = normalize_for_injection_check(text)
        for marker in self._INJECTION_MARKERS:
            if marker in lowered:
                
                _logger.warning(
                    "[ReasoningEngine] Injection marker %r detected in entity text "
                    "(first 80 chars: %r) — text suppressed from LLM prompt.",
                    marker,
                    text[:80],
                )
                return ""

        return text[: self.MAX_TEXT_CHARS]

    # ==================================================
    # SAFE JSON SERIALIZATION
    # ==================================================

    def _safe_json(self, obj: Any) -> Any:
        try:
            return json.loads(json.dumps(obj, default=str))
        except Exception:
            return {}

    # ==================================================
    # OUTPUT NORMALIZATION
    # ==================================================

    def _normalize_actions(
        self, result: Any
    ) -> List[Dict[str, Any]]:

        if isinstance(result, list):
            return [
                a for a in result
                if isinstance(a, dict)
            ]

        if isinstance(result, dict):
            actions = result.get("actions")
            if isinstance(actions, list):
                return [
                    a for a in actions
                    if isinstance(a, dict)
                ]

        return []
