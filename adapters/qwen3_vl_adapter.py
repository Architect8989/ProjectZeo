"""
adapters/qwen3_vl_adapter.py

Qwen3-VL family adapter for ProjectZeo.

Supports all Qwen3-VL variants (2B, 8B, 30B, 32B) via Ollama or a
vLLM/SGLang OpenAI-compatible endpoint. Implements the same
llm_callable(messages, objective, session_id) contract as QwenOllamaAdapter
so the factory can drop it in transparently.

Model selection (env vars):
    PROJECTZEO_QWEN3_BACKEND   ollama | vllm | sglang   (default: ollama)
    PROJECTZEO_QWEN3_HOST      host for vllm/sglang      (default: localhost)
    PROJECTZEO_QWEN3_PORT      port for vllm/sglang      (default: 8000)
    PROJECTZEO_QWEN3_MODEL     model tag                  (default: qwen3-vl:8b)
    PROJECTZEO_OLLAMA_HOST     Ollama host                (default: localhost)
    PROJECTZEO_OLLAMA_PORT     Ollama port                (default: 11434)
    PROJECTZEO_QWEN3_TIMEOUT   request timeout seconds    (default: 120)
    PROJECTZEO_QWEN3_MAX_TOKENS                           (default: 2048)
    PROJECTZEO_QWEN3_THINK     enable think-mode (1/0)    (default: 0)
"""
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

_BACKEND      = os.environ.get("PROJECTZEO_QWEN3_BACKEND", "ollama").lower().strip()
_OLLAMA_HOST  = os.environ.get("PROJECTZEO_OLLAMA_HOST", "localhost").strip()
_OLLAMA_PORT  = int(os.environ.get("PROJECTZEO_OLLAMA_PORT", "11434"))
_VLLM_HOST    = os.environ.get("PROJECTZEO_QWEN3_HOST", "localhost").strip()
_VLLM_PORT    = int(os.environ.get("PROJECTZEO_QWEN3_PORT", "8000"))
_MODEL_TAG    = os.environ.get("PROJECTZEO_QWEN3_MODEL", "qwen3-vl:8b").strip()
_TIMEOUT      = float(os.environ.get("PROJECTZEO_QWEN3_TIMEOUT", "120"))
_MAX_TOKENS   = int(os.environ.get("PROJECTZEO_QWEN3_MAX_TOKENS", "2048"))
_THINK_MODE   = os.environ.get("PROJECTZEO_QWEN3_THINK", "0").strip() == "1"

try:
    import httpx as _httpx
    _HTTPX_OK = True
except ImportError:
    _httpx = None  # type: ignore
    _HTTPX_OK = False


def _b64(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode("utf-8")


def _strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_text(response_json: Dict[str, Any]) -> str:
    try:
        content = response_json["choices"][0]["message"]["content"]
        if isinstance(content, list):
            parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
            content = " ".join(parts)
        raw = str(content)
        return _strip_think_tags(raw) if not _THINK_MODE else raw
    except (KeyError, IndexError, TypeError):
        return ""


class Qwen3VLAdapter:
    """
    Vision-language adapter for the Qwen3-VL model family.

    Provides the llm_callable interface expected by operate.py and GIIController.
    Supports image attachments in messages via bytes or base64 strings.
    """

    def __init__(self, model: str = _MODEL_TAG) -> None:
        self._model   = model
        self._lock    = threading.Lock()
        self._calls   = 0
        self._errors  = 0
        self._client: Optional[Any] = None

        if _BACKEND == "ollama":
            self._base_url = f"http://{_OLLAMA_HOST}:{_OLLAMA_PORT}"
            self._endpoint = f"{self._base_url}/api/chat"
            self._mode     = "ollama"
        else:
            self._base_url = f"http://{_VLLM_HOST}:{_VLLM_PORT}"
            self._endpoint = f"{self._base_url}/v1/chat/completions"
            self._mode     = "openai"

        if not _HTTPX_OK:
            _logger.warning("[Qwen3VL] httpx not installed — will use urllib fallback.")

        _logger.info(
            "[Qwen3VL] Adapter ready. model=%s backend=%s url=%s think=%s",
            self._model, self._mode, self._base_url, _THINK_MODE,
        )

    def _get_client(self):
        if not _HTTPX_OK:
            return None
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is None:
                self._client = _httpx.Client(
                    timeout=_httpx.Timeout(connect=10.0, read=_TIMEOUT, write=15.0, pool=5.0)
                )
        return self._client

    def _build_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        out = []
        for msg in messages:
            role    = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, str):
                out.append({"role": role, "content": content})
                continue

            if isinstance(content, list):
                parts = []
                for block in content:
                    if not isinstance(block, dict):
                        parts.append({"type": "text", "text": str(block)})
                        continue
                    btype = block.get("type", "text")
                    if btype == "text":
                        parts.append({"type": "text", "text": block.get("text", "")})
                    elif btype == "image_url":
                        parts.append({"type": "image_url", "image_url": block.get("image_url", {})})
                    elif btype == "image" and "data" in block:
                        media = block.get("media_type", "image/png")
                        data  = block["data"]
                        parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{media};base64,{data}"},
                        })
                    else:
                        parts.append({"type": "text", "text": json.dumps(block)})
                out.append({"role": role, "content": parts})
                continue

            if isinstance(content, bytes):
                b64 = _b64(content)
                out.append({
                    "role": role,
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                })
                continue

            out.append({"role": role, "content": str(content)})

        return out

    def _call_openai_compat(self, messages: List[Dict[str, Any]]) -> str:
        payload = {
            "model":       self._model,
            "messages":    messages,
            "max_tokens":  _MAX_TOKENS,
            "temperature": 0.0,
        }
        if _THINK_MODE:
            payload["chat_template_kwargs"] = {"enable_thinking": True}

        client = self._get_client()
        if client is None:
            return self._call_urllib(payload)

        resp = client.post(
            self._endpoint,
            content=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        return _extract_text(resp.json())

    def _call_ollama(self, messages: List[Dict[str, Any]]) -> str:
        payload = {
            "model":    self._model,
            "messages": messages,
            "stream":   False,
            "options":  {"num_predict": _MAX_TOKENS, "temperature": 0.0},
        }
        client = self._get_client()
        if client is None:
            return self._call_urllib(payload)

        resp = client.post(
            self._endpoint,
            content=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        raw  = data.get("message", {}).get("content", "")
        return _strip_think_tags(raw) if not _THINK_MODE else raw

    def _call_urllib(self, payload: Dict[str, Any]) -> str:
        import urllib.request
        body = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            self._endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read())
        if self._mode == "ollama":
            return data.get("message", {}).get("content", "")
        return _extract_text(data)

    def __call__(
        self,
        messages: List[Dict[str, Any]],
        objective: Optional[str] = None,
        session_id: str = "qwen3vl",
    ) -> List[Dict[str, Any]]:
        t0 = time.monotonic()
        built = self._build_messages(messages)
        try:
            if self._mode == "ollama":
                text = self._call_ollama(built)
            else:
                text = self._call_openai_compat(built)
            with self._lock:
                self._calls += 1
            elapsed = time.monotonic() - t0
            _logger.debug("[Qwen3VL] %s %.2fs %d chars", session_id, elapsed, len(text))
            return [{"role": "assistant", "content": text}]
        except Exception as exc:
            with self._lock:
                self._errors += 1
            _logger.warning("[Qwen3VL] Call failed (%s): %s", session_id, exc)
            return [{"role": "assistant", "content": ""}]

    def health_check(self) -> bool:
        try:
            if self._mode == "ollama":
                url = f"{self._base_url}/api/tags"
            else:
                url = f"{self._base_url}/health"
            client = self._get_client()
            if client:
                resp = client.get(url, timeout=5.0)
                return resp.status_code in (200, 404)
            import urllib.request
            urllib.request.urlopen(url, timeout=5.0)
            return True
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        return {
            "model":   self._model,
            "backend": self._mode,
            "calls":   self._calls,
            "errors":  self._errors,
        }
