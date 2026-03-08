from __future__ import annotations

import importlib
import logging
import os
import threading
import re
from typing import Any, Dict, List, Optional, Type

from operate.exceptions import ModelNotRecognizedException
from adapters.apis_safety_layer import apply_patches

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Local model registry
# ---------------------------------------------------------------------------

_LOCAL_REGISTRY: Dict[str, str] = {
    # Qwen2.5-VL (vision-language) via Ollama — primary local model
    "qwen2.5-vl":   "adapters.qwen_ollama_adapter.QwenOllamaAdapter",
    "llava":         "adapters.llava_ollama_adapter.LLaVAOllamaAdapter",
    "llava-llama3":  "adapters.llava_ollama_adapter.LLaVAOllamaAdapter",
    "llava-phi3":    "adapters.llava_ollama_adapter.LLaVAOllamaAdapter",
    # Qwen3-VL (GII Blueprint Phase 0.1 — arXiv:2505.09388)
    # 256K context, built-in GUI agent training, thinking mode
    "qwen3-vl":      "adapters.qwen3_vl_adapter.Qwen3VLAdapter",
    "qwen3-vl:8b":   "adapters.qwen3_vl_adapter.Qwen3VLAdapter",
    "qwen3-vl:32b":  "adapters.qwen3_vl_adapter.Qwen3VLAdapter",
    "qwen3-vl:2b":   "adapters.qwen3_vl_adapter.Qwen3VLAdapter",
    "qwen3-vl:30b":  "adapters.qwen3_vl_adapter.Qwen3VLAdapter",
    # Qwen3 text-only variants
    "qwen3-32b":     "adapters.qwen_ollama_adapter.QwenOllamaAdapter",
    "qwen3-235b":    "adapters.qwen_ollama_adapter.QwenOllamaAdapter",
    # GUI-Actor (GII Blueprint Phase 0.2 — arXiv:2506.03143)
    # Coordinate-free grounding via attention peaks
    "gui-actor":     "adapters.gui_actor_adapter.GUIActorAdapter",
    "gui-actor-7b":  "adapters.gui_actor_adapter.GUIActorAdapter",
}

# SGLang tier aliases: "sglang/<tier>" maps to a tier name
_SGLANG_TIER_PREFIX = "sglang/"
_VALID_SGLANG_TIERS = frozenset({"fast", "deep", "vision", "coder", "local"})

_CLOUD_REGISTRY = frozenset({
    "gpt-4",
    "gpt-4o",
    "gpt-4-with-som",
    "gpt-4-with-ocr",
    "gpt-4o-with-ocr",
    "gpt-4.1-with-ocr",
    "gpt-4o-labeled",
    "o1-with-ocr",
    "claude-3",
    "claude-3-opus",
    "claude-3-sonnet",
    "gemini-pro-vision",
    "qwen-vl",
})

# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

_PATCHES_APPLIED = False
_PATCH_LOCK = threading.Lock()

# Allows: letters, digits, dots, hyphens, underscores, colons, forward-slash.
_MODEL_PATTERN = re.compile(r"^[a-zA-Z0-9.\-_:/]+$")

_ADAPTER_CACHE_MAX_SIZE: int = 20   # raised from 10 to accommodate sglang tiers
_BUILD_LOCKS_MAX_SIZE: int = _ADAPTER_CACHE_MAX_SIZE * 2

from collections import OrderedDict as _OrderedDict
_ADAPTER_CACHE: "_OrderedDict[str, Any]" = _OrderedDict()
_ADAPTER_CACHE_LOCK = threading.Lock()
_ADAPTER_BUILD_LOCKS: "_OrderedDict[str, threading.Lock]" = _OrderedDict()
_BUILD_LOCKS_LOCK = threading.Lock()


def _cache_put(model_name: str, instance: Any) -> None:
    _ADAPTER_CACHE[model_name] = instance
    _ADAPTER_CACHE.move_to_end(model_name)
    while len(_ADAPTER_CACHE) > _ADAPTER_CACHE_MAX_SIZE:
        _, evicted = _ADAPTER_CACHE.popitem(last=False)
        _evicted_executor = getattr(evicted, "_executor", None)
        if _evicted_executor is not None:
            try:
                _evicted_executor.shutdown(wait=False)
            except Exception:
                pass
        # Also close SGLang adapters cleanly
        if hasattr(evicted, "close"):
            try:
                evicted.close()
            except Exception:
                pass


def _cache_get(model_name: str) -> "Any | None":
    instance = _ADAPTER_CACHE.get(model_name)
    if instance is not None:
        _ADAPTER_CACHE.move_to_end(model_name)
    return instance


import os as _os_module

_raw_ollama_only: str = _os_module.environ.get("OLLAMA_ONLY", "1").strip().lower()
_CLOUD_ACCESS_PERMITTED: bool = _raw_ollama_only not in ("1", "true", "yes")

_ollama_only_set: bool = _raw_ollama_only in ("1", "true", "yes")
if _ollama_only_set and _CLOUD_ACCESS_PERMITTED:
    raise RuntimeError(
        "FACTORY_INIT_CONTRADICTION: OLLAMA_ONLY is set but "
        "_CLOUD_ACCESS_PERMITTED=True — cloud access is forbidden."
    )

_OLLAMA_ONLY_FROZEN: bool = _ollama_only_set
del _raw_ollama_only, _ollama_only_set


def _get_model_build_lock(model_name: str) -> threading.Lock:
    with _BUILD_LOCKS_LOCK:
        if model_name in _ADAPTER_BUILD_LOCKS:
            _ADAPTER_BUILD_LOCKS.move_to_end(model_name)
            return _ADAPTER_BUILD_LOCKS[model_name]
        lock = threading.Lock()
        _ADAPTER_BUILD_LOCKS[model_name] = lock
        _ADAPTER_BUILD_LOCKS.move_to_end(model_name)
        while len(_ADAPTER_BUILD_LOCKS) > _BUILD_LOCKS_MAX_SIZE:
            _ADAPTER_BUILD_LOCKS.popitem(last=False)
        return lock


def _ensure_patches() -> None:
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return
    with _PATCH_LOCK:
        if not _PATCHES_APPLIED:
            apply_patches()
            _PATCHES_APPLIED = True


def _import_class(path: str) -> Type:
    try:
        module_path, class_name = path.rsplit(".", 1)
    except ValueError:
        raise RuntimeError(f"Invalid adapter path format: {path!r}")
    module = importlib.import_module(module_path)
    try:
        return getattr(module, class_name)
    except AttributeError:
        raise RuntimeError(
            f"Adapter class '{class_name}' not found in '{module_path}'"
        )


def _validate_model_name(model_name: str) -> str:
    if not isinstance(model_name, str) or not model_name.strip():
        raise ModelNotRecognizedException("Model name must be a non-empty string.")
    model_name = model_name.strip()
    if not _MODEL_PATTERN.fullmatch(model_name):
        raise ModelNotRecognizedException(
            f"Invalid model name format: '{model_name}'. "
            "Only letters, digits, dots, hyphens, underscores, colons, "
            "and forward-slashes are permitted."
        )
    return model_name


def _resolve_base_model(model_name: str) -> str:
    if model_name.startswith("ollama/"):
        model_name = model_name[len("ollama/"):]
    return model_name.split(":", 1)[0]


def _is_cloud_allowed() -> bool:
    return _CLOUD_ACCESS_PERMITTED


def reconfigure_cloud_access(allow: bool) -> bool:
    
    global _CLOUD_ACCESS_PERMITTED

    if allow and _OLLAMA_ONLY_FROZEN:
        raise RuntimeError(
            "reconfigure_cloud_access(allow=True) refused: "
            "OLLAMA_ONLY was set at process startup. "
            "Restart with OLLAMA_ONLY=0 to enable cloud models."
        )

    with _ADAPTER_CACHE_LOCK:
        previous = _CLOUD_ACCESS_PERMITTED
        _CLOUD_ACCESS_PERMITTED = bool(allow)
        _ADAPTER_CACHE.clear()

    return previous


# ---------------------------------------------------------------------------
# SGLang routing
# ---------------------------------------------------------------------------

def _build_sglang_adapter(tier: str) -> Any:
    
    try:
        from adapters.sglang_adapter import create_sglang_adapter_from_tier  # noqa: PLC0415
        adapter = create_sglang_adapter_from_tier(tier)

        # Health check: warn if server is unreachable at build time
        if not adapter.health_check():
            _logger.warning(
                "[AdapterFactory] SGLang server at %s is NOT reachable. "
                "Calls will fail until the server is started. "
                "Launch: python -m sglang.launch_server --model %s --port <port>",
                adapter._base_url,
                adapter._model_id,
            )
        else:
            _logger.info(
                "[AdapterFactory] SGLang server healthy: tier=%s model=%s url=%s",
                tier, adapter._model_id, adapter._base_url,
            )

        return _SGLangCallable(adapter)

    except RuntimeError as exc:
        raise ModelNotRecognizedException(
            f"SGLang adapter build failed for tier '{tier}': {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Main factory
# ---------------------------------------------------------------------------

class AdapterFactory:

    @staticmethod
    def build_llm(model_name: str):
        
        model_name = _validate_model_name(model_name)
        _ensure_patches()

        with _ADAPTER_CACHE_LOCK:
            cached = _cache_get(model_name)
            if cached is not None:
                return cached

        build_lock = _get_model_build_lock(model_name)
        with build_lock:
            with _ADAPTER_CACHE_LOCK:
                cached = _cache_get(model_name)
                if cached is not None:
                    return cached

            # -------------------------------------------------------------------
            # Route A: SGLang GPU inference (sglang/fast, sglang/deep, etc.)
            # -------------------------------------------------------------------
            if model_name.startswith(_SGLANG_TIER_PREFIX):
                tier = model_name[len(_SGLANG_TIER_PREFIX):]
                if tier not in _VALID_SGLANG_TIERS:
                    raise ModelNotRecognizedException(
                        f"Unknown SGLang tier '{tier}'. "
                        f"Valid tiers: {sorted(_VALID_SGLANG_TIERS)}"
                    )
                instance = _build_sglang_adapter(tier)
                with _ADAPTER_CACHE_LOCK:
                    _cache_put(model_name, instance)
                return instance

            # -------------------------------------------------------------------
            # Route B: Explicit cloud prefix (anthropic:* / openai:*)
            # -------------------------------------------------------------------
            from adapters.cloud_adapter import is_cloud_model, create_cloud_adapter  # noqa: PLC0415

            if is_cloud_model(model_name):
                instance = create_cloud_adapter(model_name)
                instance_callable = _CloudCallable(instance)
                with _ADAPTER_CACHE_LOCK:
                    _cache_put(model_name, instance_callable)
                return instance_callable

            base_model = _resolve_base_model(model_name)

            # -------------------------------------------------------------------
            # Route C1: Qwen3-VL — special constructor (no model_name arg)
            # -------------------------------------------------------------------
            if base_model in ("qwen3-vl", "qwen3-vl:8b", "qwen3-vl:32b",
                              "qwen3-vl:2b", "qwen3-vl:30b"):
                try:
                    from adapters.qwen3_vl_adapter import get_qwen3_vl  # noqa: PLC0415
                    vl_instance = get_qwen3_vl()
                    vl_callable = _Qwen3VLCallable(vl_instance)
                    with _ADAPTER_CACHE_LOCK:
                        _cache_put(model_name, vl_callable)
                    return vl_callable
                except ImportError as exc:
                    _logger.warning("[AdapterFactory] Qwen3VL import failed: %s", exc)

            # -------------------------------------------------------------------
            # Route C2: GUI-Actor — special constructor (no model_name arg)
            # -------------------------------------------------------------------
            if base_model in ("gui-actor", "gui-actor-7b"):
                try:
                    from adapters.gui_actor_adapter import get_gui_actor  # noqa: PLC0415
                    actor_instance = get_gui_actor()
                    actor_callable = _GUIActorCallable(actor_instance)
                    with _ADAPTER_CACHE_LOCK:
                        _cache_put(model_name, actor_callable)
                    return actor_callable
                except ImportError as exc:
                    _logger.warning("[AdapterFactory] GUIActor import failed: %s", exc)

            # -------------------------------------------------------------------
            # Route C: Local adapter (Ollama)
            # -------------------------------------------------------------------
            local_path = _LOCAL_REGISTRY.get(base_model)
            if local_path is not None:
                AdapterClass = _import_class(local_path)
                instance = AdapterClass(model_name=model_name)
                with _ADAPTER_CACHE_LOCK:
                    _cache_put(model_name, instance)
                return instance

            # -------------------------------------------------------------------
            # Route D: Legacy cloud via PureLLMWrapper
            # -------------------------------------------------------------------
            if base_model in _CLOUD_REGISTRY:
                if not _is_cloud_allowed():
                    raise ModelNotRecognizedException(
                        f"Model '{model_name}' is a cloud model, but OLLAMA_ONLY is "
                        "enforced (default). To enable cloud models, start the system "
                        "with OLLAMA_ONLY=0 in the environment BEFORE importing this "
                        "module, or use an explicit 'anthropic:<model>' / "
                        f"'openai:<model>' prefix.\n"
                        f"  Local models: {sorted(_LOCAL_REGISTRY.keys())}\n"
                        f"  SGLang tiers: {sorted(_VALID_SGLANG_TIERS)} (prefix: sglang/)"
                    )

                from adapters.pure_llm_wrapper import PureLLMWrapper  # noqa: PLC0415

                instance = PureLLMWrapper(model_name=base_model)
                with _ADAPTER_CACHE_LOCK:
                    _cache_put(model_name, instance)
                return instance

            # -------------------------------------------------------------------
            # Route E: Unknown
            # -------------------------------------------------------------------
            raise ModelNotRecognizedException(
                f"Model '{model_name}' is not registered.\n"
                f"  SGLang tiers:  sglang/{{fast|deep|vision|coder}} (set PROJECTZEO_USE_SGLANG=1)\n"
                f"  Local models:  {sorted(_LOCAL_REGISTRY.keys())}\n"
                f"  Cloud prefix:  anthropic:<model> | openai:<model>\n"
                f"  Legacy cloud:  {sorted(_CLOUD_REGISTRY)} (require OLLAMA_ONLY=0)\n"
                "Add the model to the appropriate registry in adapters/factory.py."
            )

    @staticmethod
    async def get_action(
        model_name: str,
        messages: List[Dict[str, Any]],
        objective: str,
        session_id: str,
    ):
        adapter = AdapterFactory.build_llm(model_name)
        return await adapter.get_next_action(
            messages=messages,
            objective=objective,
            session_id=session_id,
        )


# ---------------------------------------------------------------------------
# Public tier-routing accessor used by ConsequenceReasoner
# ---------------------------------------------------------------------------

def get_reasoning_client(tier: str = "fast") -> Any:
    
    from config.model_config import is_gpu_mode  # noqa: PLC0415

    if is_gpu_mode():
        try:
            return AdapterFactory.build_llm(f"sglang/{tier}")
        except Exception as exc:
            _logger.warning(
                "[AdapterFactory] get_reasoning_client: SGLang build failed for "
                "tier=%s, falling back to local: %s", tier, exc,
            )

    # CPU fallback: local Ollama model
    local_model = os.environ.get("PROJECTZEO_LOCAL_MODEL", "qwen2.5-vl")
    try:
        return AdapterFactory.build_llm(local_model)
    except Exception as exc:
        _logger.error(
            "[AdapterFactory] get_reasoning_client: local model build also failed: %s", exc
        )
        raise


# ---------------------------------------------------------------------------
# Callable wrappers
# ---------------------------------------------------------------------------

class _CloudCallable:
    """Wrap a cloud adapter so it satisfies the llm_callable protocol."""

    def __init__(self, adapter) -> None:
        self._adapter = adapter
        self.model_name: str = getattr(adapter, "model_name", "cloud")

    def __call__(self, messages, objective=None, session_id=None):
        return self._adapter(messages, objective=objective, session_id=session_id)

    def get_llm_callable(self):
        return self

    def __getattr__(self, name: str):
        return getattr(self._adapter, name)


class _SGLangCallable:
    

    def __init__(self, adapter) -> None:
        self._adapter = adapter
        self.model_name: str = getattr(adapter, "_model_id", "sglang")

    def __call__(self, messages, objective=None, session_id=None):
        return self._adapter(messages, objective=objective, session_id=session_id)

    def get_llm_callable(self):
        return self

    def with_thinking(self, enabled: bool) -> "_SGLangCallable":
        """Return a thinking-mode toggled copy of this callable."""
        return _SGLangCallable(self._adapter.with_thinking(enabled))

    def health_check(self) -> bool:
        return self._adapter.health_check()

    def get_stats(self) -> Dict[str, Any]:
        return self._adapter.get_stats()

    def __getattr__(self, name: str):
        return getattr(self._adapter, name)


# Module-level aliases for backward compatibility
build_llm = AdapterFactory.build_llm


class _Qwen3VLCallable:
    """Wrap Qwen3VLAdapter so it satisfies the llm_callable protocol."""

    def __init__(self, adapter) -> None:
        self._adapter = adapter
        self.model_name: str = getattr(adapter, "_model", "qwen3-vl")

    def __call__(self, messages, objective=None, session_id=None):
        return self._adapter.llm_call(
            messages, objective=objective or "", session_id=session_id or ""
        )

    def get_llm_callable(self):
        return self

    def with_thinking(self, enabled: bool) -> "_Qwen3VLCallable":
        """Toggle thinking mode on the underlying adapter."""
        try:
            self._adapter._thinking = enabled
        except Exception:
            pass
        return self

    def health_check(self) -> bool:
        return self._adapter.is_available()

    def get_stats(self) -> Dict[str, Any]:
        return self._adapter.health_check()

    def __getattr__(self, name: str):
        return getattr(self._adapter, name)


class _GUIActorCallable:
    """Wrap GUIActorAdapter so it satisfies the llm_callable protocol."""

    def __init__(self, adapter) -> None:
        self._adapter = adapter
        self.model_name: str = getattr(adapter, "_model", "gui-actor-7b")

    def __call__(self, messages, objective=None, session_id=None):
        # GUI-Actor is a grounding tool, not an LLM — return empty for text calls
        _logger.debug("[GUIActorCallable] text-only call — GUI-Actor is a grounding tool")
        return ""

    def get_llm_callable(self):
        return self

    def ground(self, screenshot, description: str):
        return self._adapter.ground(screenshot, description)

    def health_check(self) -> bool:
        return self._adapter.is_available()

    def __getattr__(self, name: str):
        return getattr(self._adapter, name)
get_action = AdapterFactory.get_action
