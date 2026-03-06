from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional

_logger = logging.getLogger(__name__)

# httpx is already a project dependency (via anthropic/openai)
try:
    import httpx as _httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _httpx = None  # type: ignore[assignment]
    _HTTPX_AVAILABLE = False


class SGLangConnectionError(RuntimeError):
    
    pass


class SGLangAdapter:
    

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
        thinking_mode: bool = False,
        thinking_budget_tokens: int = 4096,
    ) -> None:
        
        if not _HTTPX_AVAILABLE:
            raise RuntimeError(
                "SGLangAdapter requires httpx. Install it: pip install httpx"
            )

        self.model_name = model_id   # satisfy llm_callable protocol
        self._model_id = model_id
        self._base_url = base_url.rstrip("/")
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout_seconds
        self._thinking_mode = thinking_mode
        self._thinking_budget = thinking_budget_tokens

        self._client: Optional["_httpx.Client"] = None
        self._client_lock = threading.Lock()

        # Request statistics
        self._total_calls: int = 0
        self._total_errors: int = 0
        self._total_time_seconds: float = 0.0

        _logger.info(
            "[SGLangAdapter] Initialised. model=%s base_url=%s thinking=%s",
            model_id, base_url, thinking_mode,
        )

    def _get_client(self) -> "_httpx.Client":
        """Return (or create) a persistent httpx client."""
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
                    ),
                    headers={"Content-Type": "application/json"},
                )
        return self._client

    def health_check(self) -> bool:
        
        try:
            client = self._get_client()
            # SGLang exposes /health on the same port as /v1/chat/completions
            resp = client.get(
                f"{self._base_url}/health",
                timeout=self._HEALTH_CHECK_TIMEOUT,
            )
            return resp.status_code == 200
        except Exception as exc:
            _logger.debug("[SGLangAdapter] Health check failed: %s", exc)
            return False

    def __call__(
        self,
        messages: List[Dict[str, Any]],
        objective: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        
        if not messages:
            return [{"role": "assistant", "content": ""}]

        payload = self._build_payload(messages)
        start_ts = time.monotonic()
        self._total_calls += 1

        try:
            client = self._get_client()
            response = client.post(
                f"{self._base_url}/v1/chat/completions",
                content=json.dumps(payload),
                timeout=self._timeout,
            )

            elapsed = time.monotonic() - start_ts
            self._total_time_seconds += elapsed

            if response.status_code != 200:
                self._total_errors += 1
                error_body = response.text[:500]
                raise SGLangConnectionError(
                    f"SGLang returned HTTP {response.status_code}: {error_body}"
                )

            data = response.json()
            return self._extract_response(data)

        except SGLangConnectionError:
            raise
        except _httpx.TimeoutException as exc:
            self._total_errors += 1
            raise SGLangConnectionError(
                f"SGLang request timed out after {self._timeout}s: {exc}"
            ) from exc
        except _httpx.ConnectError as exc:
            self._total_errors += 1
            raise SGLangConnectionError(
                f"Cannot connect to SGLang at {self._base_url}: {exc}. "
                "Is the SGLang server running? "
                "Launch: python -m sglang.launch_server --model <model> --port 30000"
            ) from exc
        except Exception as exc:
            self._total_errors += 1
            raise SGLangConnectionError(
                f"SGLang request failed: {type(exc).__name__}: {exc}"
            ) from exc

    def _build_payload(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        
        payload: Dict[str, Any] = {
            "model": self._model_id,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "stream": False,
        }

        
        payload["thinking"] = {
            "type": "enabled",
            "budget_tokens": self._thinking_budget,
        } if self._thinking_mode else {"type": "disabled"}

        return payload

    def _extract_response(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        
        try:
            choices = data.get("choices", [])
            if not choices:
                _logger.warning("[SGLangAdapter] Response had no choices: %s", data)
                return [{"role": "assistant", "content": ""}]

            message = choices[0].get("message", {})
            content = str(message.get("content", ""))

            # Strip <think>…</think> for clean downstream processing
            if self._thinking_mode and "<think>" in content:
                think_end = content.find("</think>")
                if think_end != -1:
                    think_block = content[: think_end + len("</think>")]
                    content = content[think_end + len("</think>"):].strip()
                    _logger.debug(
                        "[SGLangAdapter] Thinking block (%d chars) stripped from response.",
                        len(think_block),
                    )

            return [{"role": "assistant", "content": content}]

        except Exception as exc:
            _logger.warning("[SGLangAdapter] Response parse error: %s. data=%r", exc, str(data)[:200])
            return [{"role": "assistant", "content": ""}]

    def get_llm_callable(self):
        """Satisfy the llm_callable protocol introspection used by ExecutionPlanner."""
        return self

    def get_stats(self) -> Dict[str, Any]:
        """Return call statistics for monitoring."""
        avg_time = (
            self._total_time_seconds / self._total_calls
            if self._total_calls > 0 else 0.0
        )
        return {
            "total_calls": self._total_calls,
            "total_errors": self._total_errors,
            "total_time_seconds": round(self._total_time_seconds, 2),
            "avg_time_seconds": round(avg_time, 3),
            "error_rate": round(
                self._total_errors / self._total_calls if self._total_calls > 0 else 0.0,
                3,
            ),
            "model_id": self._model_id,
            "base_url": self._base_url,
            "thinking_mode": self._thinking_mode,
        }

    def with_thinking(self, enabled: bool) -> "SGLangAdapter":
        
        return SGLangAdapter(
            model_id=self._model_id,
            base_url=self._base_url,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            timeout_seconds=self._timeout,
            thinking_mode=enabled,
            thinking_budget_tokens=self._thinking_budget,
        )

    def close(self) -> None:
        """Close the underlying httpx client. Call during shutdown."""
        with self._client_lock:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def create_sglang_adapter_from_tier(tier: str) -> SGLangAdapter:
    
    from config.model_config import get_endpoint, is_gpu_mode  # noqa: PLC0415

    if not is_gpu_mode():
        raise RuntimeError(
            "SGLang adapters require PROJECTZEO_USE_SGLANG=1. "
            "Set that environment variable before importing this module, "
            "then ensure the SGLang server is running on the configured port."
        )

    endpoint = get_endpoint(tier)

    return SGLangAdapter(
        model_id=endpoint.model_id,
        base_url=endpoint.base_url,
        max_tokens=endpoint.max_tokens,
        temperature=endpoint.temperature,
        timeout_seconds=endpoint.timeout_seconds,
        thinking_mode=endpoint.default_thinking,
        thinking_budget_tokens=8192 if endpoint.default_thinking else 4096,
    )
