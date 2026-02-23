from typing import Dict, Any, List
import json


class ReasoningEngine:
    """
    Multi-hypothesis LLM reasoning with bounded perception
    and injection-resistant prompt construction.
    """

    MAX_ENTITIES = 20
    MAX_TEXT_CHARS = 400
    MAX_JSON_CHARS = 8000

    # HAR-5 (MATH-9): Comprehensive injection marker set.
    #
    # Each string is lowercased; _truncate_text() lowercases the input before
    # checking so the comparison is case-insensitive.
    #
    # Markers cover:
    #   - Classic direct override phrases
    #   - Common LLM template delimiters used in fine-tuned models
    #   - Structural injections (new system prompt, role override)
    #   - Common paraphrases found in adversarial prompt research
    #
    # Operators who deploy this system against richer adversarial environments
    # should extend this frozenset or replace _truncate_text() with a
    # dedicated classifier.
    _INJECTION_MARKERS = frozenset({
        # Classic override phrases
        "ignore previous instructions",
        "ignore all previous",
        "ignore prior instructions",
        "disregard previous",
        "disregard all previous",
        "disregard prior instructions",
        "disregard the above",
        "forget previous instructions",
        "forget all previous",
        "override previous instructions",
        "override the above",

        # "New instruction" injection patterns
        "new instruction:",
        "new instructions:",
        "updated instruction:",
        "revised instruction:",
        "important instruction:",

        # System prompt / role injection
        "system:",
        "system prompt:",
        "new system prompt:",
        "<|system|>",
        "[system]",
        "###system",
        "### system",

        # LLM template delimiters used in fine-tuned / instruction-tuned models
        "</s>",
        "[inst]",
        "</inst>",
        "[/inst]",
        "<s>[inst]",
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
        "assistant:",
        "human:",
        "user:",

        # Jailbreak prefix patterns
        "act as",
        "pretend you are",
        "pretend to be",
        "you are now",
        "from now on",
        "your new role",
        "your task is now",

        # Escaping / structural attacks
        "```system",
        "---system",
        "=== system",
    })

    def __init__(self, llm_callable):
        self._llm = llm_callable

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
        """
        HAR-5 (MATH-9): Truncate and sanitize entity text.

        Checks against a comprehensive set of injection markers
        (_INJECTION_MARKERS) instead of the original single-marker check
        for "ignore previous instructions".

        The original single-marker check was trivially bypassed by any
        alternative phrasing.  ExecutionPlanner already used 7 markers;
        this method now uses an aligned and extended set for consistency.

        Returns "" (empty string) if any injection marker is found,
        suppressing the hostile text from the LLM prompt entirely.
        Returns text[:MAX_TEXT_CHARS] otherwise.
        """
        if not isinstance(text, str):
            return ""

        lowered = text.lower()
        for marker in self._INJECTION_MARKERS:
            if marker in lowered:
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
