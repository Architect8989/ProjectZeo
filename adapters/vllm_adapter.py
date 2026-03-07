"""
vllm_adapter.py — vLLM inference adapter for ProjectZeo.

Provides a drop-in replacement for SGLangAdapter when vLLM is preferred
as the GPU inference backend. Supports OpenAI-compatible /v1/chat/completions.

Usage:
    PROJECTZEO_USE_VLLM=1
    PROJECTZEO_VLLM_HOST=localhost
    PROJECTZEO_VLLM_PORT=8000
    PROJECTZEO_VLLM_MODEL=Qwen/Qwen3-32B

The adapter is selected by the factory.py when PROJECTZEO_USE_VLLM=1 and
PROJECTZEO_USE_SGLANG is not set.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

_logger = logging.getLogger(__name__)

try:
    import httpx as _httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _httpx = None  # type: ignore[assignment]
    _HTTPX_AVAILABLE = False


class VLLMConnectionError(RuntimeError):
    """Raised when vLLM server is unreachable."""
    pass


class VLLMAdapter:
    """
    vLLM inference adapter using the OpenAI-compatible REST API.

    vLLM exposes /v1/chat/completions identically to OpenAI, making this
    adapter structurally similar to SGLangAdapter. Key differences:
      - vLLM does not support a separate "thinking budget" parameter.
      - vLLM supports tensor-parallel multi-GPU inference out of the box.
      - vLLM streaming is supported but this adapter uses non-streaming for
        simplicity (consistent with the rest of ProjectZeo).

    Environment variables:
        PROJECTZEO_VLLM_HOST          — default: localhost
        PROJECTZEO_VLLM_PORT          — default: 8000
        PROJECTZEO_VLLM_MODEL         — model served by this vLLM instance
        PROJECTZEO_USE_VLLM           — set to "1" to activate
    """

    _DEFAULT_TIMEOUT = 180.0
    _HEALTH_CHECK_TIMEOUT = 5.0

    def __init__(
        self,
        *,
        model_id: str,
        base_url: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        timeout_seconds: float = _DEFAULT_TIMEOUT,
    ) -> None:
        if not _HTTPX_AVAILABLE:
            raise RuntimeError(
                "VLLMAdapter requires httpx. Install it: pip install httpx"
            )

        self.model_name = model_id   # satisfy llm_callable protocol
        self._model_id  = model_id
        self._base_url  = base_url.rstrip("/")
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout_seconds

        self._client: Optional[Any] = None
        self._client_lock = threading.Lock()

        self._total_calls: int = 0
        self._total_errors: int = 0
        self._total_time_seconds: float = 0.0

        _logger.info(
            "[VLLMAdapter] Initialised. model=%s base_url=%s",
            model_id, base_url,
        )

    def _get_client(self):
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is None:
                self._client = _httpx.Client(
                    timeout=_httpx.Timeout(
                        connect=10.0,
                        read=self._timeout,
                        write=10.0,
                        pool=5.0,
                    )
                )
        return self._client

    def health_check(self) -> bool:
        """Return True if the vLLM server is responsive."""
        try:
            client = self._get_client()
            resp = client.get(
                f"{self._base_url}/health",
                timeout=self._HEALTH_CHECK_TIMEOUT,
            )
            if resp.status_code == 200:
                return True
            # Some vLLM versions return 200 on /v1/models
            resp2 = client.get(
                f"{self._base_url}/v1/models",
                timeout=self._HEALTH_CHECK_TIMEOUT,
            )
            return resp2.status_code == 200
        except Exception as exc:
            _logger.debug("[VLLMAdapter] Health check failed: %s", exc)
            return False

    def __call__(
        self,
        messages: List[Dict[str, Any]],
        objective: Optional[str] = None,
        session_id: str = "vllm_call",
    ) -> List[Dict[str, Any]]:
        """
        Call the vLLM chat completions endpoint.

        Returns a list containing one dict with key "content" (matching the
        SGLangAdapter protocol used by operate.py / PerStepReasoner).
        """
        t0 = time.monotonic()
        client = self._get_client()

        payload: Dict[str, Any] = {
            "model":       self._model_id,
            "messages":    messages,
            "max_tokens":  self._max_tokens,
            "temperature": self._temperature,
            "stream":      False,
        }

        try:
            resp = client.post(
                f"{self._base_url}/v1/chat/completions",
                content=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=self._timeout,
            )
            elapsed = time.monotonic() - t0

            if resp.status_code != 200:
                self._total_errors += 1
                raise VLLMConnectionError(
                    f"vLLM returned HTTP {resp.status_code}: {resp.text[:200]}"
                )

            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                self._total_errors += 1
                raise VLLMConnectionError("vLLM returned empty choices")

            content = choices[0].get("message", {}).get("content", "")
            self._total_calls += 1
            self._total_time_seconds += elapsed

            _logger.debug(
                "[VLLMAdapter] Call complete. session=%s latency=%.2fs tokens≈%d",
                session_id, elapsed, len(content.split()),
            )
            return [{"role": "assistant", "content": content}]

        except VLLMConnectionError:
            raise
        except Exception as exc:
            self._total_errors += 1
            elapsed = time.monotonic() - t0
            _logger.warning(
                "[VLLMAdapter] Call failed. session=%s latency=%.2fs error=%s",
                session_id, elapsed, exc,
            )
            raise

    def get_stats(self) -> Dict[str, Any]:
        return {
            "model_id":              self._model_id,
            "base_url":              self._base_url,
            "total_calls":           self._total_calls,
            "total_errors":          self._total_errors,
            "avg_latency_seconds":   (
                self._total_time_seconds / self._total_calls
                if self._total_calls > 0 else 0.0
            ),
        }


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def _vllm_host() -> str:
    return os.environ.get("PROJECTZEO_VLLM_HOST", "localhost").strip()


def _vllm_port() -> int:
    try:
        return int(os.environ.get("PROJECTZEO_VLLM_PORT", "8000"))
    except (ValueError, TypeError):
        return 8000


def create_vllm_adapter(
    *,
    model_id: Optional[str] = None,
    port: Optional[int] = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    timeout_seconds: float = 180.0,
) -> VLLMAdapter:
    """
    Create a VLLMAdapter from environment variables or explicit arguments.

    Args:
        model_id:         Override PROJECTZEO_VLLM_MODEL
        port:             Override PROJECTZEO_VLLM_PORT
        max_tokens:       Max response tokens
        temperature:      Sampling temperature (0.0 = greedy)
        timeout_seconds:  Per-request timeout

    Returns:
        Configured VLLMAdapter instance.

    Raises:
        RuntimeError: if httpx is not available.
    """
    _model = model_id or os.environ.get(
        "PROJECTZEO_VLLM_MODEL", "Qwen/Qwen3-32B"
    )
    _port  = port or _vllm_port()
    _host  = _vllm_host()
    _url   = f"http://{_host}:{_port}"

    return VLLMAdapter(
        model_id=_model,
        base_url=_url,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
    )


def is_vllm_mode() -> bool:
    """Return True when vLLM is configured and SGLang is not."""
    use_vllm  = os.environ.get("PROJECTZEO_USE_VLLM", "0").strip() in ("1", "true", "yes")
    use_sglang = os.environ.get("PROJECTZEO_USE_SGLANG", "0").strip() in ("1", "true", "yes")
    return use_vllm and not use_sglang
