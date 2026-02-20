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
    # Extension point — add new local models here:
    # "llama3.2-vision": "adapters.llama_ollama_adapter.LlamaOllamaAdapter",
    # "llava":           "adapters.llava_ollama_adapter.LlavaOllamaAdapter",
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
    "llava",          # LLaVA via legacy ollama path in PureLLMWrapper
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

# §R3: module-level adapter cache — one instance per model name.
_ADAPTER_CACHE: Dict[str, Any] = {}
_ADAPTER_CACHE_LOCK = threading.Lock()


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


class AdapterFactory:

    @staticmethod
    def build_llm(model_name: str):
        """
        Build and cache an LLM adapter for the given model name.

        Routing priority:
          1. Cached instance → return immediately.
          2. Local registry match → instantiate registered local adapter.
          3. Cloud registry match → instantiate PureLLMWrapper.
          4. Unknown → raise ModelNotRecognizedException.
        """
        model_name = _validate_model_name(model_name)
        _ensure_patches()

        # §R3: return cached adapter if already built
        with _ADAPTER_CACHE_LOCK:
            if model_name in _ADAPTER_CACHE:
                return _ADAPTER_CACHE[model_name]

        base_model = _resolve_base_model(model_name)

        # --- Route 1: Local adapter ---
        local_path = _LOCAL_REGISTRY.get(base_model)
        if local_path is not None:
            AdapterClass = _import_class(local_path)
            instance = None
            try:
                instance = AdapterClass(model_name=model_name)
            except Exception:
                # EVO-3: ensure no partial entry leaks into cache on failure
                with _ADAPTER_CACHE_LOCK:
                    _ADAPTER_CACHE.pop(model_name, None)
                raise

            with _ADAPTER_CACHE_LOCK:
                _ADAPTER_CACHE[model_name] = instance
            return instance

        # --- Route 2: Cloud adapter via PureLLMWrapper ---
        if base_model in _CLOUD_REGISTRY:
            # Lazy import so Ollama-only boots never touch cloud code
            from adapters.pure_llm_wrapper import PureLLMWrapper  # noqa: PLC0415

            instance = None
            try:
                instance = PureLLMWrapper(model_name=base_model)
            except Exception:
                with _ADAPTER_CACHE_LOCK:
                    _ADAPTER_CACHE.pop(model_name, None)
                raise

            with _ADAPTER_CACHE_LOCK:
                _ADAPTER_CACHE[model_name] = instance
            return instance

        # --- Route 3: Unknown ---
        raise ModelNotRecognizedException(
            f"Model '{model_name}' is not registered.\n"
            f"  Local models:  {sorted(_LOCAL_REGISTRY.keys())}\n"
            f"  Cloud models:  {sorted(_CLOUD_REGISTRY)}\n"
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
        Reuses the cached adapter — does NOT reconstruct on every call (§R3).
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
