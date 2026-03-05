"""
adapters/cloud_adapter.py

Production-grade cloud model adapter for Anthropic Claude and OpenAI GPT / O-series.
Implements the same llm_callable(messages, objective, session_id) contract as
QwenOllamaAdapter so it can be dropped in as a direct replacement.

Supported model prefixes:
  anthropic:<model>  e.g. anthropic:claude-sonnet-4-20250514
  openai:<model>     e.g. openai:gpt-4o, openai:o3

Environment variables (required):
  ANTHROPIC_API_KEY   — Anthropic API key (for anthropic:* models)
  OPENAI_API_KEY      — OpenAI API key   (for openai:* models)

Optional tuning:
  PROJECTZEO_CLOUD_MAX_TOKENS       default 2048
  PROJECTZEO_CLOUD_TIMEOUT_SECONDS  default 120
  PROJECTZEO_CLOUD_MAX_RETRIES      default 3
"""
from __future__ import annotations

import logging
import os
import time
import threading
from typing import Any, Dict, List, Optional

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_TOKENS: int = int(os.environ.get("PROJECTZEO_CLOUD_MAX_TOKENS", "2048"))
_TIMEOUT: float = float(os.environ.get("PROJECTZEO_CLOUD_TIMEOUT_SECONDS", "120"))
_MAX_RETRIES: int = int(os.environ.get("PROJECTZEO_CLOUD_MAX_RETRIES", "3"))

_ANTHROPIC_PREFIX = "anthropic:"
_OPENAI_PREFIX = "openai:"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_model(model_name: str) -> tuple[str, str]:
    """
    Return (provider, bare_model_name).
    Raises ValueError for unrecognised prefixes.
    """
    if model_name.startswith(_ANTHROPIC_PREFIX):
        return "anthropic", model_name[len(_ANTHROPIC_PREFIX):]
    if model_name.startswith(_OPENAI_PREFIX):
        return "openai", model_name[len(_OPENAI_PREFIX):]
    raise ValueError(
        f"Unrecognised cloud model prefix in {model_name!r}. "
        f"Use 'anthropic:<model>' or 'openai:<model>'."
    )


def _backoff(attempt: int, base: float = 1.0, cap: float = 30.0) -> float:
    return min(base * (2 ** attempt), cap)


def _scrub_key(key: str) -> str:
    """Return a safe display version of an API key for log lines."""
    if not key or len(key) < 8:
        return "<redacted>"
    return key[:4] + "..." + key[-4:]


# ---------------------------------------------------------------------------
# AnthropicCloudAdapter
# ---------------------------------------------------------------------------

class AnthropicCloudAdapter:
    """
    Wraps the Anthropic messages API.

    llm_callable contract:
        adapter(messages, objective=None, session_id=None) -> str | list
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._lock = threading.Lock()

        _api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not _api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Export it before starting ProjectZeo with an Anthropic model."
            )
        _logger.info(
            "[AnthropicCloudAdapter] Initialised. model=%r key_prefix=%s",
            model_name, _scrub_key(_api_key),
        )

        try:
            import anthropic as _anthropic
            self._client = _anthropic.Anthropic(api_key=_api_key)
        except ImportError:
            raise RuntimeError(
                "anthropic package not installed. "
                "Install with: pip install anthropic"
            )

    # ------------------------------------------------------------------
    # llm_callable implementation
    # ------------------------------------------------------------------

    def __call__(
        self,
        messages: List[Dict[str, Any]],
        objective: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Translate the internal messages list to Anthropic format and call the API.
        Returns the text of the first content block.
        """
        system_text: str = ""
        user_messages: List[Dict[str, Any]] = []

        for msg in messages:
            role = str(msg.get("role", "")).lower()
            content = msg.get("content", "")
            if role == "system":
                system_text = str(content)
            elif role in ("user", "assistant"):
                user_messages.append({"role": role, "content": self._coerce_content(content)})

        if not user_messages:
            raise ValueError("No user/assistant messages to send to Anthropic API.")

        last_exc: Exception = RuntimeError("unreachable")

        for attempt in range(_MAX_RETRIES):
            try:
                kwargs: Dict[str, Any] = {
                    "model": self.model_name,
                    "max_tokens": _MAX_TOKENS,
                    "messages": user_messages,
                }
                if system_text:
                    kwargs["system"] = system_text

                response = self._client.messages.create(**kwargs)
                return self._extract_text(response)

            except Exception as exc:
                last_exc = exc
                _exc_name = type(exc).__name__

                # Non-retryable: auth / validation errors
                if any(s in _exc_name for s in ("Auth", "Permission", "Invalid", "NotFound")):
                    raise

                if attempt < _MAX_RETRIES - 1:
                    wait = _backoff(attempt)
                    _logger.warning(
                        "[AnthropicCloudAdapter] Attempt %d/%d failed (%s: %s). "
                        "Retrying in %.1fs.",
                        attempt + 1, _MAX_RETRIES, _exc_name, exc, wait,
                    )
                    time.sleep(wait)

        raise RuntimeError(
            f"AnthropicCloudAdapter: all {_MAX_RETRIES} attempts failed. "
            f"Last error: {last_exc}"
        ) from last_exc

    @staticmethod
    def _coerce_content(content: Any) -> Any:
        """Normalise content to string or Anthropic content-block list."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Already a list of content blocks — pass through.
            return content
        return str(content)

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Extract the first text block from an Anthropic response object."""
        content = getattr(response, "content", None)
        if content is None:
            return ""
        if isinstance(content, list):
            for block in content:
                text = getattr(block, "text", None)
                if text is not None:
                    return str(text)
        if isinstance(content, str):
            return content
        return str(content)


# ---------------------------------------------------------------------------
# OpenAICloudAdapter
# ---------------------------------------------------------------------------

class OpenAICloudAdapter:
    """
    Wraps the OpenAI chat completions API (including o-series reasoning models).

    llm_callable contract:
        adapter(messages, objective=None, session_id=None) -> str
    """

    # O-series models use max_completion_tokens instead of max_tokens
    # and do not support system messages or temperature.
    _O_SERIES_PREFIXES = ("o1", "o3", "o4")

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._is_o_series = any(
            model_name.startswith(p) for p in self._O_SERIES_PREFIXES
        )
        self._lock = threading.Lock()

        _api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not _api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set. "
                "Export it before starting ProjectZeo with an OpenAI model."
            )
        _logger.info(
            "[OpenAICloudAdapter] Initialised. model=%r o_series=%s key_prefix=%s",
            model_name, self._is_o_series, _scrub_key(_api_key),
        )

        try:
            import openai as _openai
            self._client = _openai.OpenAI(api_key=_api_key)
        except ImportError:
            raise RuntimeError(
                "openai package not installed. "
                "Install with: pip install openai"
            )

    # ------------------------------------------------------------------
    # llm_callable implementation
    # ------------------------------------------------------------------

    def __call__(
        self,
        messages: List[Dict[str, Any]],
        objective: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        api_messages = self._coerce_messages(messages)
        last_exc: Exception = RuntimeError("unreachable")

        for attempt in range(_MAX_RETRIES):
            try:
                kwargs: Dict[str, Any] = {
                    "model": self.model_name,
                    "messages": api_messages,
                }
                if self._is_o_series:
                    kwargs["max_completion_tokens"] = _MAX_TOKENS
                else:
                    kwargs["max_tokens"] = _MAX_TOKENS

                response = self._client.chat.completions.create(**kwargs)
                return self._extract_text(response)

            except Exception as exc:
                last_exc = exc
                _exc_name = type(exc).__name__

                if any(s in _exc_name for s in ("Auth", "Permission", "Invalid", "NotFound")):
                    raise

                if attempt < _MAX_RETRIES - 1:
                    wait = _backoff(attempt)
                    _logger.warning(
                        "[OpenAICloudAdapter] Attempt %d/%d failed (%s: %s). "
                        "Retrying in %.1fs.",
                        attempt + 1, _MAX_RETRIES, _exc_name, exc, wait,
                    )
                    time.sleep(wait)

        raise RuntimeError(
            f"OpenAICloudAdapter: all {_MAX_RETRIES} attempts failed. "
            f"Last error: {last_exc}"
        ) from last_exc

    def _coerce_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Translate to OpenAI format.  O-series models do not support system
        messages — prepend as a user message instead.
        """
        result = []
        for msg in messages:
            role = str(msg.get("role", "")).lower()
            content = msg.get("content", "")
            if role == "system" and self._is_o_series:
                # O-series: inject system text as a leading user turn.
                result.append({"role": "user", "content": str(content)})
            else:
                result.append({"role": role, "content": self._coerce_content(content)})
        return result

    @staticmethod
    def _coerce_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(str(item.get("text", item)))
            return " ".join(parts)
        return str(content)

    @staticmethod
    def _extract_text(response: Any) -> str:
        try:
            return response.choices[0].message.content or ""
        except (AttributeError, IndexError):
            return str(response)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def create_cloud_adapter(model_name: str):
    """
    Return the appropriate adapter instance for the given model name.

    model_name must carry a provider prefix:
      anthropic:<bare-model>  → AnthropicCloudAdapter
      openai:<bare-model>     → OpenAICloudAdapter
    """
    provider, _ = _resolve_model(model_name)
    if provider == "anthropic":
        return AnthropicCloudAdapter(model_name[len(_ANTHROPIC_PREFIX):])
    if provider == "openai":
        return OpenAICloudAdapter(model_name[len(_OPENAI_PREFIX):])
    raise ValueError(f"Unsupported provider: {provider!r}")


def is_cloud_model(model_name: str) -> bool:
    """Return True if model_name carries a recognised cloud provider prefix."""
    return model_name.startswith(_ANTHROPIC_PREFIX) or model_name.startswith(_OPENAI_PREFIX)
