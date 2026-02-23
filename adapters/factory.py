from __future__ import annotations

import importlib
import threading
import re
from typing import Dict, Any, List, Type

from operate.exceptions import ModelNotRecognizedException
from adapters.apis_safety_layer import apply_patches


# =========================================================
# LOCAL MODEL REGISTRY
# Maps base model name → fully-qualified adapter class path.
# These are served by dedicated local adapters (Ollama, etc.)
# =========================================================
_LOCAL_REGISTRY: Dict[str, str] = {
    "qwen2.5-vl": "adapters.qwen_ollama_adapter.QwenOllamaAdapter",
    # FIX RB-7: llava was misclassified in _CLOUD_REGISTRY. LLaVA is served
    # by local Ollama and never routes through cloud APIs. Moving it here
    # ensures OLLAMA_ONLY=1 mode can use llava without hitting the cloud-block
    # guard, and that llava users are not incorrectly denied at runtime.
    #
    # QwenOllamaAdapter is provider-agnostic: the model name ("llava") is
    # passed through directly to ollama.Client().chat(), so no adapter-level
    # changes are needed. The only requirement is that 'llava' is pulled
    # locally via `ollama pull llava` before use.
    "llava": "adapters.qwen_ollama_adapter.QwenOllamaAdapter",
    # Extension point — add new local models here:
    # "llama3.2-vision": "adapters.llama_ollama_adapter.LlamaOllamaAdapter",
}

# =========================================================
# CLOUD MODEL REGISTRY
# Base model names routed through PureLLMWrapper.
# PureLLMWrapper resolves these to the correct cloud API function.
# =========================================================
_CLOUD_REGISTRY = {
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
    "qwen-vl",       # Qwen cloud (DashScope) — not local Ollama
    # NOTE: "llava" removed from cloud registry (FIX RB-7 — see LOCAL_REGISTRY above)
    # Extension point — add new cloud model base names here.
}

# =========================================================
# INTERNALS
# =========================================================

_PATCHES_APPLIED = False
_PATCH_LOCK = threading.Lock()

# Allows: letters, digits, dots, hyphens, underscores, colons, forward-slash.
# Slash is needed for 'ollama/qwen2.5-vl' format.
_MODEL_PATTERN = re.compile(r"^[a-zA-Z0-9.\-_:/]+$")


# SI-8 / H8 FIX: Bound the adapter cache with LRU eviction.
#
# Bug: _ADAPTER_CACHE was an unbounded plain dict. Different version tags
# (e.g. "qwen2.5-vl:7b-instruct" vs "qwen2.5-vl:latest") created separate
# cache entries, each with its own ThreadPoolExecutor. In long-running
# processes with many model variants the cache grew without bound, leaking
# thread-pool resources proportional to distinct model name strings seen.
#
# Fix: use an OrderedDict capped at _ADAPTER_CACHE_MAX_SIZE entries. On
# insertion beyond the cap, the least-recently-used entry (first in dict
# order) is evicted. The build-lock design is preserved; eviction uses
# _ADAPTER_CACHE_LOCK for thread safety.
_ADAPTER_CACHE_MAX_SIZE: int = 10

from collections import OrderedDict as _OrderedDict
_ADAPTER_CACHE: "_OrderedDict[str, Any]" = _OrderedDict()
_ADAPTER_CACHE_LOCK = threading.Lock()           # guards _ADAPTER_CACHE reads/writes
_ADAPTER_BUILD_LOCKS: Dict[str, threading.Lock] = {}  # per-model construction mutex
_BUILD_LOCKS_LOCK = threading.Lock()             # guards _ADAPTER_BUILD_LOCKS itself


def _cache_put(model_name: str, instance: Any) -> None:
    """
    SI-8 / H8 FIX: Insert into the LRU adapter cache, evicting the oldest
    entry when the cache exceeds _ADAPTER_CACHE_MAX_SIZE.

    Must be called with _ADAPTER_CACHE_LOCK already held.
    """
    _ADAPTER_CACHE[model_name] = instance
    # Move to end (most-recently-used position in OrderedDict)
    _ADAPTER_CACHE.move_to_end(model_name)
    # Evict oldest (least-recently-used) entries beyond the cap
    while len(_ADAPTER_CACHE) > _ADAPTER_CACHE_MAX_SIZE:
        _ADAPTER_CACHE.popitem(last=False)


def _cache_get(model_name: str) -> "Any | None":
    """
    SI-8 / H8 FIX: Retrieve from the LRU adapter cache, promoting the entry
    to most-recently-used position.

    Must be called with _ADAPTER_CACHE_LOCK already held.
    Returns None if not present.
    """
    instance = _ADAPTER_CACHE.get(model_name)
    if instance is not None:
        _ADAPTER_CACHE.move_to_end(model_name)
    return instance


import os as _os_module
_raw_ollama_only = _os_module.environ.get("OLLAMA_ONLY", "1").strip().lower()

_CLOUD_ACCESS_PERMITTED: bool = _raw_ollama_only not in ("1", "true", "yes")
del _raw_ollama_only  # remove intermediate from module namespace


def _get_model_build_lock(model_name: str) -> threading.Lock:
    """Return the per-model build lock, creating it if it doesn't exist."""
    with _BUILD_LOCKS_LOCK:
        if model_name not in _ADAPTER_BUILD_LOCKS:
            _ADAPTER_BUILD_LOCKS[model_name] = threading.Lock()
        return _ADAPTER_BUILD_LOCKS[model_name]


def _ensure_patches() -> None:
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return
    with _PATCH_LOCK:
        if not _PATCHES_APPLIED:
            apply_patches()
            _PATCHES_APPLIED = True


def _import_class(path: str) -> Type:
    """Dynamically import a class from a dotted module path."""
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
    """
    Normalise a model name to its registry lookup key.

    Handles two common variant formats:
      1. Provider prefix:  'ollama/qwen2.5-vl'        → 'qwen2.5-vl'
      2. Version tag:      'qwen2.5-vl:7b-instruct'   → 'qwen2.5-vl'
    """
    # Strip optional 'ollama/' provider prefix
    if model_name.startswith("ollama/"):
        model_name = model_name[len("ollama/"):]

    # Strip version tag (everything after the first ':')
    return model_name.split(":", 1)[0]


def _is_cloud_allowed() -> bool:
    return _CLOUD_ACCESS_PERMITTED


class AdapterFactory:

    @staticmethod
    def build_llm(model_name: str):

        model_name = _validate_model_name(model_name)
        _ensure_patches()

        # Fast path: return cached adapter without acquiring build lock.
        # SI-8 / H8 FIX: Use _cache_get() which also updates LRU order.
        with _ADAPTER_CACHE_LOCK:
            cached = _cache_get(model_name)
            if cached is not None:
                return cached

        # Slow path: acquire per-model build lock.
        build_lock = _get_model_build_lock(model_name)
        with build_lock:
            # Double-checked locking: another thread may have completed
            # construction while we waited for the build lock.
            with _ADAPTER_CACHE_LOCK:
                cached = _cache_get(model_name)
                if cached is not None:
                    return cached

            base_model = _resolve_base_model(model_name)

            # --- Route 1: Local adapter ---
            local_path = _LOCAL_REGISTRY.get(base_model)
            if local_path is not None:
                AdapterClass = _import_class(local_path)
                instance = AdapterClass(model_name=model_name)
                with _ADAPTER_CACHE_LOCK:
                    _cache_put(model_name, instance)
                return instance

            # --- Route 2: Cloud adapter via PureLLMWrapper ---
            if base_model in _CLOUD_REGISTRY:
                if not _is_cloud_allowed():
                    raise ModelNotRecognizedException(
                        f"Model '{model_name}' is a cloud model, but OLLAMA_ONLY is "
                        "enforced (default). To enable cloud models, start the system "
                        "with --allow-cloud or set OLLAMA_ONLY=0 in the environment "
                        "BEFORE importing this module.\n"
                        f"  Local models: {sorted(_LOCAL_REGISTRY.keys())}"
                    )

                # Lazy import so Ollama-only boots never touch cloud code
                from adapters.pure_llm_wrapper import PureLLMWrapper  # noqa: PLC0415

                instance = PureLLMWrapper(model_name=base_model)
                with _ADAPTER_CACHE_LOCK:
                    _cache_put(model_name, instance)
                return instance

            # --- Route 3: Unknown ---
            raise ModelNotRecognizedException(
                f"Model '{model_name}' is not registered.\n"
                f"  Local models:  {sorted(_LOCAL_REGISTRY.keys())}\n"
                f"  Cloud models:  {sorted(_CLOUD_REGISTRY)} (require --allow-cloud)\n"
                "Add the model to the appropriate registry in adapters/factory.py."
            )

    @staticmethod
    async def get_action(
        model_name: str,
        messages: List[Dict[str, Any]],
        objective: str,
        session_id: str,
    ):
        """
        Convenience coroutine: resolve adapter and call get_next_action().
        Reuses the cached adapter — does NOT reconstruct on every call.
        """
        adapter = AdapterFactory.build_llm(model_name)
        return await adapter.get_next_action(
            messages=messages,
            objective=objective,
            session_id=session_id,
        )


# Module-level aliases for backward compatibility
build_llm = AdapterFactory.build_llm
get_action = AdapterFactory.get_action
