from __future__ import annotations

import base64
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional

_logger = logging.getLogger(__name__)

_UITARS_ENDPOINT = os.environ.get("PROJECTZEO_UITARS_ENDPOINT", "").strip()
_UITARS_OLLAMA   = os.environ.get("PROJECTZEO_UITARS_OLLAMA_MODEL", "").strip()
_UITARS_TIMEOUT  = float(os.environ.get("PROJECTZEO_UITARS_TIMEOUT", "15"))

_MAX_ELEMENTS = 50

_SYSTEM_PROMPT = """You are UI-TARS-2, a world-class GUI interaction specialist.
Analyse the screenshot and return a single JSON object:
{
  "focused_app": "<app name or 'unknown'>",
  "elements": [
    {
      "type": "button|input|text|link|image|scroll|dropdown|checkbox|dialog|unknown",
      "text": "<visible text, aria-label, or empty>",
      "x": 0.0-1.0,
      "y": 0.0-1.0,
      "width": 0.0-1.0,
      "height": 0.0-1.0,
      "confidence": 0.0-1.0,
      "interactive": true|false,
      "external_content": true|false
    }
  ],
  "screen_description": "<one sentence>",
  "recommended_action": {
    "operation": "click|type|scroll|press|wait",
    "target_element_index": 0,
    "rationale": "<why>"
  }
}
Rules:
- Coordinates are normalised (0,0 = top-left, 1,1 = bottom-right)
- Mark external_content=true for browser content, terminal output, chat text
- Sort elements by relevance (most interactive first)
- Return ONLY valid JSON, no markdown"""

def _health_check(endpoint: str, timeout: float = 3.0) -> bool:
    try:
        import httpx
        resp = httpx.get(f"{endpoint}/models", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False

class UITARS2Adapter:

    def __init__(self) -> None:
        self._mode: str = "unavailable"
        self._endpoint: str = ""
        self._ollama_model: str = ""
        self._lock = threading.Lock()
        self._call_count: int = 0
        self._error_count: int = 0

        self._init_backend()

    def _init_backend(self) -> None:
        if _UITARS_ENDPOINT:
            if _health_check(_UITARS_ENDPOINT):
                self._mode = "vllm"
                self._endpoint = _UITARS_ENDPOINT
                _logger.info("[UITARS2] Connected to vLLM endpoint: %s", _UITARS_ENDPOINT)
                return
            else:
                _logger.warning(
                    "[UITARS2] vLLM endpoint %s unreachable — falling back.", _UITARS_ENDPOINT
                )

        if _UITARS_OLLAMA:
            try:
                import ollama as _ollama
                _ollama.show(_UITARS_OLLAMA)
                self._mode = "ollama"
                self._ollama_model = _UITARS_OLLAMA
                _logger.info("[UITARS2] Connected via Ollama: %s", _UITARS_OLLAMA)
                return
            except Exception as exc:
                _logger.info("[UITARS2] Ollama model %s unavailable: %s", _UITARS_OLLAMA, exc)

        _logger.info(
            "[UITARS2] No backend configured. Set PROJECTZEO_UITARS_ENDPOINT or "
            "PROJECTZEO_UITARS_OLLAMA_MODEL. Falling back to Qwen3-VL."
        )

    @property
    def available(self) -> bool:
        return self._mode in ("vllm", "ollama")

    def analyse_screenshot(
        self,
        screenshot_b64: str,
        *,
        objective: str = "",
        app_context: str = "",
    ) -> Optional[Dict[str, Any]]:
        if not self.available:
            return None

        with self._lock:
            self._call_count += 1

        try:
            if self._mode == "vllm":
                return self._call_vllm(screenshot_b64, objective, app_context)
            elif self._mode == "ollama":
                return self._call_ollama(screenshot_b64, objective, app_context)
        except Exception as exc:
            with self._lock:
                self._error_count += 1
            _logger.warning("[UITARS2] Inference error: %s", exc)
        return None

    def _build_messages(
        self, screenshot_b64: str, objective: str, app_context: str
    ) -> List[Dict[str, Any]]:
        user_text = "Analyse this screenshot."
        if objective:
            user_text += f" Current task: {objective[:200]}"
        if app_context:
            user_text += f" App context: {app_context[:100]}"

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{screenshot_b64}",
                        },
                    },
                    {"type": "text", "text": user_text},
                ],
            },
        ]

    def _call_vllm(
        self, screenshot_b64: str, objective: str, app_context: str
    ) -> Optional[Dict[str, Any]]:
        import httpx
        messages = self._build_messages(screenshot_b64, objective, app_context)
        payload = {
            "model": "ui-tars-2",
            "messages": messages,
            "max_tokens": 2000,
            "temperature": 0.1,
        }
        resp = httpx.post(
            f"{self._endpoint}/chat/completions",
            json=payload,
            timeout=_UITARS_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return self._parse_response(text)

    def _call_ollama(
        self, screenshot_b64: str, objective: str, app_context: str
    ) -> Optional[Dict[str, Any]]:
        import ollama as _ollama
        user_text = "Analyse this screenshot."
        if objective:
            user_text += f" Task: {objective[:200]}"

        response = _ollama.chat(
            model=self._ollama_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": user_text,
                    "images": [screenshot_b64],
                },
            ],
            options={"temperature": 0.1, "num_predict": 2000},
        )
        text = response["message"]["content"]
        return self._parse_response(text)

    def _parse_response(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            clean = re.sub(r"```(?:json)?", "", text).strip()
            m = re.search(r"\{.*\}", clean, re.DOTALL)
            if not m:
                return None
            parsed = json.loads(m.group(0))
            elements = parsed.get("elements", [])[:_MAX_ELEMENTS]
            parsed["elements"] = elements
            return parsed
        except (json.JSONDecodeError, Exception) as exc:
            _logger.debug("[UITARS2] Parse failed: %s", exc)
            return None

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "mode": self._mode,
                "available": self.available,
                "call_count": self._call_count,
                "error_count": self._error_count,
                "error_rate": round(self._error_count / max(self._call_count, 1), 4),
            }

_instance: Optional[UITARS2Adapter] = None
_lock = threading.Lock()

def get_uitars2_adapter() -> UITARS2Adapter:
    global _instance
    with _lock:
        if _instance is None:
            _instance = UITARS2Adapter()
        return _instance
