from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Runtime flags
# ---------------------------------------------------------------------------

def _use_sglang() -> bool:
    """Return True iff SGLang GPU inference is configured and enabled."""
    return os.environ.get("PROJECTZEO_USE_SGLANG", "0").strip() in ("1", "true", "yes")


def _sglang_host() -> str:
    return os.environ.get("PROJECTZEO_SGLANG_HOST", "localhost").strip()


def _ollama_host() -> str:
    return os.environ.get("PROJECTZEO_OLLAMA_HOST", "localhost").strip()


def _ollama_port() -> int:
    try:
        return int(os.environ.get("PROJECTZEO_OLLAMA_PORT", "11434"))
    except (ValueError, TypeError):
        return 11434


# ---------------------------------------------------------------------------
# ModelEndpoint dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelEndpoint:
    """Complete specification for one model inference endpoint."""

    # Human-readable tier label: "fast" | "deep" | "vision" | "coder" | "local"
    tier: str

    # Model identifier as known to the inference server
    model_id: str

    # Base URL for OpenAI-compatible /v1/chat/completions
    base_url: str

    # Max tokens for the response
    max_tokens: int = 4096

    # Whether this endpoint supports thinking mode (Qwen3 thinking=True/False)
    supports_thinking: bool = False

    # Default thinking mode for this endpoint
    #   True  → Qwen3 outputs <think>…</think> before answering (deep reasoning)
    #   False → Qwen3 instruct mode: fast, no chain-of-thought
    default_thinking: bool = False

    # Request timeout seconds (generous for consequence reasoning on CPU)
    timeout_seconds: float = 180.0

    # Whether this is an Ollama endpoint (different API format)
    is_ollama: bool = False

    # Temperature
    temperature: float = 0.0

    @property
    def chat_completions_url(self) -> str:
        """Full URL for chat completions requests."""
        if self.is_ollama:
            return f"{self.base_url}/api/chat"
        return f"{self.base_url}/v1/chat/completions"

    def with_thinking(self, enabled: bool) -> "ModelEndpoint":
        """Return a copy with thinking mode toggled (for dual-mode routing)."""
        import dataclasses
        return dataclasses.replace(self, default_thinking=enabled)


# ---------------------------------------------------------------------------
# Endpoint registry builder
# ---------------------------------------------------------------------------

def _build_sglang_url(port_env: str, default_port: int) -> str:
    port = int(os.environ.get(port_env, str(default_port)))
    return f"http://{_sglang_host()}:{port}"


def _build_ollama_url() -> str:
    return f"http://{_ollama_host()}:{_ollama_port()}"


def get_fast_endpoint() -> ModelEndpoint:
    
    if _use_sglang():
        return ModelEndpoint(
            tier="fast",
            model_id=os.environ.get("PROJECTZEO_FAST_MODEL", "Qwen/Qwen3-32B"),
            base_url=_build_sglang_url("PROJECTZEO_FAST_PORT", 30000),
            max_tokens=4096,
            supports_thinking=True,
            default_thinking=False,   # instruct mode for speed
            timeout_seconds=60.0,
            temperature=0.0,
        )
    # CPU fallback: Ollama
    return ModelEndpoint(
        tier="fast",
        model_id=os.environ.get("PROJECTZEO_LOCAL_MODEL", "qwen2.5-vl"),
        base_url=_build_ollama_url(),
        max_tokens=2048,
        supports_thinking=False,
        default_thinking=False,
        timeout_seconds=150.0,
        is_ollama=True,
        temperature=0.0,
    )


def get_deep_endpoint() -> ModelEndpoint:
    
    if _use_sglang():
        return ModelEndpoint(
            tier="deep",
            model_id=os.environ.get(
                "PROJECTZEO_DEEP_MODEL", "Qwen/Qwen3-235B-A22B-Thinking-2507"
            ),
            base_url=_build_sglang_url("PROJECTZEO_DEEP_PORT", 30001),
            max_tokens=8192,
            supports_thinking=True,
            default_thinking=True,    # thinking mode for IRREVERSIBLE decisions
            timeout_seconds=300.0,    # 5 minutes for complex reasoning
            temperature=0.0,
        )
    # CPU fallback: same Ollama model, generous timeout
    return ModelEndpoint(
        tier="deep",
        model_id=os.environ.get("PROJECTZEO_LOCAL_MODEL", "qwen2.5-vl"),
        base_url=_build_ollama_url(),
        max_tokens=2048,
        supports_thinking=False,
        default_thinking=False,
        timeout_seconds=180.0,
        is_ollama=True,
        temperature=0.0,
    )


def get_vision_endpoint() -> ModelEndpoint:
    
    if _use_sglang():
        return ModelEndpoint(
            tier="vision",
            model_id=os.environ.get(
                "PROJECTZEO_VISION_MODEL", "bytedance-research/UI-TARS-2-7B-Instruct"
            ),
            base_url=_build_sglang_url("PROJECTZEO_VISION_PORT", 30002),
            max_tokens=2048,
            supports_thinking=False,
            default_thinking=False,
            timeout_seconds=30.0,    # UI-TARS-2 is fast on GPU
            temperature=0.0,
        )
    # CPU fallback: generic VLM
    return ModelEndpoint(
        tier="vision",
        model_id=os.environ.get("PROJECTZEO_LOCAL_MODEL", "qwen2.5-vl"),
        base_url=_build_ollama_url(),
        max_tokens=1024,
        supports_thinking=False,
        default_thinking=False,
        timeout_seconds=90.0,
        is_ollama=True,
        temperature=0.0,
    )


def get_coder_endpoint() -> ModelEndpoint:
    
    if _use_sglang():
        return ModelEndpoint(
            tier="coder",
            model_id=os.environ.get(
                "PROJECTZEO_CODER_MODEL", "Qwen/Qwen3-Coder-480B-A35B-Instruct"
            ),
            base_url=_build_sglang_url("PROJECTZEO_CODER_PORT", 30003),
            max_tokens=8192,
            supports_thinking=True,
            default_thinking=False,
            timeout_seconds=120.0,
            temperature=0.0,
        )
    # CPU fallback: reuse fast endpoint
    return get_fast_endpoint()


def get_local_endpoint() -> ModelEndpoint:
    """
    Local Ollama endpoint — always available, no GPU required.
    Used as the universal fallback when SGLang is not configured.
    """
    return ModelEndpoint(
        tier="local",
        model_id=os.environ.get("PROJECTZEO_LOCAL_MODEL", "qwen2.5-vl"),
        base_url=_build_ollama_url(),
        max_tokens=2048,
        supports_thinking=False,
        default_thinking=False,
        timeout_seconds=150.0,
        is_ollama=True,
        temperature=0.0,
    )


# ---------------------------------------------------------------------------
# Convenience accessor
# ---------------------------------------------------------------------------

def get_endpoint(tier: str) -> ModelEndpoint:
    
    _MAP = {
        "fast":   get_fast_endpoint,
        "deep":   get_deep_endpoint,
        "vision": get_vision_endpoint,
        "coder":  get_coder_endpoint,
        "local":  get_local_endpoint,
    }
    builder = _MAP.get(tier)
    if builder is None:
        raise ValueError(
            f"Unknown tier {tier!r}. Valid tiers: {sorted(_MAP.keys())}"
        )
    return builder()


def is_gpu_mode() -> bool:
    """Return True when SGLang GPU inference is active."""
    return _use_sglang()


def describe_deployment() -> str:
    """Return a human-readable deployment summary for the startup banner."""
    if is_gpu_mode():
        return (
            f"GPU mode (SGLang @ {_sglang_host()}): "
            f"fast=:{os.environ.get('PROJECTZEO_FAST_PORT', '30000')} "
            f"deep=:{os.environ.get('PROJECTZEO_DEEP_PORT', '30001')} "
            f"vision=:{os.environ.get('PROJECTZEO_VISION_PORT', '30002')} "
            f"coder=:{os.environ.get('PROJECTZEO_CODER_PORT', '30003')}"
        )
    return (
        f"CPU mode (Ollama @ {_ollama_host()}:{_ollama_port()}): "
        f"model={os.environ.get('PROJECTZEO_LOCAL_MODEL', 'qwen2.5-vl')} "
        f"— set PROJECTZEO_USE_SGLANG=1 for GPU acceleration"
    )
