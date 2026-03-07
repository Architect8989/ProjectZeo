"""
qwen_vl_runtime.py — Qwen3-VL-235B vision-language runtime for ProjectZeo.

Provides high-accuracy visual understanding using the Qwen3-VL-235B-Instruct
model served via the SGLang vision endpoint (port 30002 by default).

Capabilities:
  - screenshot_to_entities(): full UI entity extraction with bounding boxes
  - describe_screen():        natural language description of the screen
  - find_element():           semantic search for UI elements
  - ocr():                    optical character recognition from image regions

Fallback chain:
  Qwen3-VL-235B (GPU/SGLang) → UI-TARS-2 → VisionRuntime (Qwen2.5-VL Ollama)

Environment variables:
    PROJECTZEO_USE_SGLANG         — must be "1" to enable SGLang inference
    PROJECTZEO_VISION_PORT        — vLLM/SGLang port for vision model (default: 30002)
    PROJECTZEO_VISION_MODEL       — model ID (default: Qwen/Qwen3-VL-235B-Instruct)
    PROJECTZEO_QWEN_VL_ENABLED    — set to "1" to prefer this runtime (default: auto)
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

# Default model configuration
_DEFAULT_MODEL   = "Qwen/Qwen3-VL-235B-Instruct"
_DEFAULT_PORT    = 30002
_DEFAULT_TIMEOUT = 60.0   # Vision inference is faster on GPU (~5-30s per frame)

try:
    import httpx as _httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _httpx = None  # type: ignore[assignment]
    _HTTPX_AVAILABLE = False


# ---------------------------------------------------------------------------
# UI Entity schema (compatible with VisionRuntime output)
# ---------------------------------------------------------------------------

def _entity(
    label: str,
    x: float,
    y: float,
    w: float = 0.0,
    h: float = 0.0,
    etype: str = "unknown",
    confidence: float = 1.0,
) -> Dict[str, Any]:
    return {
        "label":      label,
        "text":       label,
        "x":          round(x, 4),
        "y":          round(y, 4),
        "width":      round(w, 4),
        "height":     round(h, 4),
        "type":       etype,
        "confidence": round(confidence, 3),
    }


# ---------------------------------------------------------------------------
# QwenVLRuntime
# ---------------------------------------------------------------------------

class QwenVLRuntime:
    """
    Vision-language runtime backed by Qwen3-VL-235B via SGLang.

    All methods accept a screenshot as bytes (PNG/JPEG) or a base64-encoded
    string. Results are returned in the VisionRuntime entity format so they
    can be consumed by the same downstream code without modification.
    """

    _ENTITY_EXTRACTION_PROMPT = """\
You are a UI entity extractor. Analyse the screenshot and return ALL visible
interactive UI elements: buttons, text fields, links, checkboxes, menus, icons,
labels, and any visible text content.

For each element provide:
  - label: visible text or icon description
  - type:  button | text_field | link | checkbox | menu | icon | text | image | unknown
  - x, y:  normalised centre coordinates (0.0–1.0 from top-left)
  - w, h:  normalised width and height (0.0–1.0)
  - confidence: 0.0–1.0

Respond ONLY with a JSON array. No prose. No markdown fences.
[{"label":"...", "type":"...", "x":0.5, "y":0.5, "w":0.1, "h":0.05, "confidence":0.95}, ...]
"""

    _DESCRIBE_PROMPT = """\
Describe this screenshot in 2–3 sentences. Include: what application is open,
what the current screen shows, and any notable UI state (errors, dialogs, etc.).
"""

    _FIND_ELEMENT_TEMPLATE = """\
Find the UI element matching this description: "{query}"

Return ONLY a JSON object with:
  {{"found": true/false, "label": "...", "x": 0.0, "y": 0.0, "confidence": 0.0}}

If not found, set found=false and use 0.0 for all numbers.
"""

    _OCR_PROMPT = """\
Extract all visible text from this screenshot. Return a JSON object:
  {"text": "full extracted text", "regions": [{"text": "...", "x": 0.0, "y": 0.0}]}
"""

    def __init__(
        self,
        *,
        model_id: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: float = _DEFAULT_TIMEOUT,
        max_tokens: int = 2048,
    ) -> None:
        if not _HTTPX_AVAILABLE:
            raise RuntimeError(
                "QwenVLRuntime requires httpx. Install: pip install httpx"
            )

        self._model_id = model_id or os.environ.get(
            "PROJECTZEO_VISION_MODEL", _DEFAULT_MODEL
        )
        port = int(os.environ.get("PROJECTZEO_VISION_PORT", str(_DEFAULT_PORT)))
        host = os.environ.get("PROJECTZEO_SGLANG_HOST", "localhost").strip()
        self._base_url = (base_url or f"http://{host}:{port}").rstrip("/")
        self._timeout  = timeout_seconds
        self._max_tokens = max_tokens

        self._client: Optional[Any] = None
        self._client_lock = threading.Lock()
        self._call_count  = 0
        self._error_count = 0

        _logger.info(
            "[QwenVLRuntime] Initialised. model=%s url=%s",
            self._model_id, self._base_url,
        )

    def _get_client(self):
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is None:
                self._client = _httpx.Client(
                    timeout=_httpx.Timeout(
                        connect=10.0, read=self._timeout,
                        write=10.0, pool=5.0,
                    )
                )
        return self._client

    def health_check(self) -> bool:
        """Return True if the SGLang vision endpoint is reachable."""
        try:
            resp = self._get_client().get(
                f"{self._base_url}/health", timeout=5.0
            )
            return resp.status_code == 200
        except Exception:
            return False

    # =========================================================================
    # Core API
    # =========================================================================

    def screenshot_to_entities(
        self,
        screenshot: bytes,
    ) -> List[Dict[str, Any]]:
        """
        Extract all UI entities from a screenshot.

        Args:
            screenshot: PNG/JPEG bytes of the screen.

        Returns:
            List of entity dicts compatible with VisionRuntime format.
        """
        raw = self._vision_call(
            image_bytes=screenshot,
            prompt=self._ENTITY_EXTRACTION_PROMPT,
        )
        return self._parse_entities(raw)

    def describe_screen(self, screenshot: bytes) -> str:
        """Return a natural language description of the screen state."""
        return self._vision_call(
            image_bytes=screenshot,
            prompt=self._DESCRIBE_PROMPT,
        )

    def find_element(
        self,
        screenshot: bytes,
        query: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Semantically locate a UI element by description.

        Returns:
            Entity dict if found, None otherwise.
        """
        prompt = self._FIND_ELEMENT_TEMPLATE.format(query=query[:200])
        raw = self._vision_call(image_bytes=screenshot, prompt=prompt)
        try:
            clean = re.sub(r"```(?:json)?", "", raw).strip()
            data  = json.loads(clean)
            if data.get("found"):
                return _entity(
                    label=str(data.get("label", query)),
                    x=float(data.get("x", 0.5)),
                    y=float(data.get("y", 0.5)),
                    confidence=float(data.get("confidence", 0.8)),
                )
        except Exception:
            pass
        return None

    def ocr(self, screenshot: bytes) -> Dict[str, Any]:
        """
        Perform OCR on the screenshot.

        Returns:
            {"text": str, "regions": list}
        """
        raw = self._vision_call(image_bytes=screenshot, prompt=self._OCR_PROMPT)
        try:
            clean = re.sub(r"```(?:json)?", "", raw).strip()
            return json.loads(clean)
        except Exception:
            return {"text": raw, "regions": []}

    # =========================================================================
    # Internal
    # =========================================================================

    def _vision_call(
        self,
        image_bytes: bytes,
        prompt: str,
        *,
        session_id: str = "qwen_vl",
    ) -> str:
        t0 = time.monotonic()
        b64 = base64.b64encode(image_bytes).decode("utf-8")

        payload = {
            "model":      self._model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens":  self._max_tokens,
            "temperature": 0.0,
        }

        try:
            client = self._get_client()
            resp   = client.post(
                f"{self._base_url}/v1/chat/completions",
                content=json.dumps(payload),
                headers={"Content-Type": "application/json"},
            )
            elapsed = time.monotonic() - t0

            if resp.status_code != 200:
                self._error_count += 1
                raise RuntimeError(
                    f"QwenVL endpoint returned HTTP {resp.status_code}"
                )

            data    = resp.json()
            content = data["choices"][0]["message"]["content"]
            self._call_count += 1
            _logger.debug(
                "[QwenVLRuntime] %s: %.2fs, %d chars",
                session_id, elapsed, len(content),
            )
            return content

        except Exception as exc:
            self._error_count += 1
            _logger.warning("[QwenVLRuntime] Vision call failed: %s", exc)
            return ""

    def _parse_entities(self, raw: str) -> List[Dict[str, Any]]:
        if not raw:
            return []
        try:
            clean = re.sub(r"```(?:json)?", "", raw).strip()
            arr   = json.loads(clean)
            if not isinstance(arr, list):
                return []
            result = []
            for item in arr:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label") or item.get("text") or "")
                if not label:
                    continue
                result.append(_entity(
                    label=label,
                    x=float(item.get("x", 0.5)),
                    y=float(item.get("y", 0.5)),
                    w=float(item.get("w", 0.0)),
                    h=float(item.get("h", 0.0)),
                    etype=str(item.get("type", "unknown")),
                    confidence=float(item.get("confidence", 0.9)),
                ))
            return result
        except Exception as exc:
            _logger.debug("[QwenVLRuntime] Entity parse error: %s", exc)
            return []

    def get_stats(self) -> Dict[str, Any]:
        return {
            "model_id":    self._model_id,
            "base_url":    self._base_url,
            "call_count":  self._call_count,
            "error_count": self._error_count,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_runtime_instance: Optional[QwenVLRuntime] = None
_runtime_lock = threading.Lock()


def get_qwen_vl_runtime(
    *,
    timeout_seconds: float = _DEFAULT_TIMEOUT,
) -> Optional[QwenVLRuntime]:
    """
    Return a singleton QwenVLRuntime if SGLang vision endpoint is available.
    Returns None if the endpoint is unreachable (graceful degradation).
    """
    global _runtime_instance

    if not os.environ.get("PROJECTZEO_USE_SGLANG", "0").strip() in ("1", "true", "yes"):
        return None

    if _runtime_instance is not None:
        return _runtime_instance

    with _runtime_lock:
        if _runtime_instance is not None:
            return _runtime_instance
        try:
            rt = QwenVLRuntime(timeout_seconds=timeout_seconds)
            if rt.health_check():
                _runtime_instance = rt
                _logger.info(
                    "[QwenVLRuntime] Vision endpoint healthy: %s", rt._base_url
                )
                return _runtime_instance
            else:
                _logger.info(
                    "[QwenVLRuntime] Vision endpoint unreachable at %s — "
                    "falling back to UI-TARS-2 / VisionRuntime.",
                    rt._base_url,
                )
                return None
        except Exception as exc:
            _logger.debug("[QwenVLRuntime] Init failed: %s", exc)
            return None
