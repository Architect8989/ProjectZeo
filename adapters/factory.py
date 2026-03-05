from __future__ import annotations

import importlib
import threading
import re
from typing import Dict, Any, List, Type

from operate.exceptions import ModelNotRecognizedException
from adapters.apis_safety_layer import apply_patches


_LOCAL_REGISTRY: Dict[str, str] = {
    "qwen2.5-vl": "adapters.qwen_ollama_adapter.QwenOllamaAdapter",
    "llava": "adapters.llava_ollama_adapter.LLaVAOllamaAdapter",
    "llava-llama3": "adapters.llava_ollama_adapter.LLaVAOllamaAdapter",
    "llava-phi3": "adapters.llava_ollama_adapter.LLaVAOllamaAdapter",
}

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
    "qwen-vl",
}

# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

_PATCHES_APPLIED = False
_PATCH_LOCK = threading.Lock()

# Allows: letters, digits, dots, hyphens, underscores, colons, forward-slash.
_MODEL_PATTERN = re.compile(r"^[a-zA-Z0-9.\-_:/]+$")

_ADAPTER_CACHE_MAX_SIZE: int = 10
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
    """
    Reconfigure cloud access at runtime.

    Raises RuntimeError if allow=True and OLLAMA_ONLY was set at startup.
    Returns the previous value of _CLOUD_ACCESS_PERMITTED.
    """
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
            # Route 0 (NEW): Explicit cloud prefix — anthropic:* and openai:*
            #
            # These bypass the OLLAMA_ONLY gate because the operator has made
            # the intent explicit by choosing a prefixed model name.  The cloud
            # adapter still requires the relevant API key env var to be set;
            # construction will raise RuntimeError if the key is absent.
            # -------------------------------------------------------------------
            from adapters.cloud_adapter import is_cloud_model, create_cloud_adapter  # noqa: PLC0415

            if is_cloud_model(model_name):
                instance = create_cloud_adapter(model_name)
                # Wrap in a thin callable so the llm_callable contract is met.
                instance_callable = _CloudCallable(instance)
                with _ADAPTER_CACHE_LOCK:
                    _cache_put(model_name, instance_callable)
                return instance_callable

            base_model = _resolve_base_model(model_name)

            # --- Route 1: Local adapter ---
            local_path = _LOCAL_REGISTRY.get(base_model)
            if local_path is not None:
                AdapterClass = _import_class(local_path)
                instance = AdapterClass(model_name=model_name)
                with _ADAPTER_CACHE_LOCK:
                    _cache_put(model_name, instance)
                return instance

            # --- Route 2: Legacy cloud via PureLLMWrapper ---
            if base_model in _CLOUD_REGISTRY:
                if not _is_cloud_allowed():
                    raise ModelNotRecognizedException(
                        f"Model '{model_name}' is a cloud model, but OLLAMA_ONLY is "
                        "enforced (default). To enable cloud models, start the system "
                        "with OLLAMA_ONLY=0 in the environment BEFORE importing this "
                        "module, or use an explicit 'anthropic:<model>' / "
                        f"'openai:<model>' prefix.\n"
                        f"  Local models: {sorted(_LOCAL_REGISTRY.keys())}"
                    )

                from adapters.pure_llm_wrapper import PureLLMWrapper  # noqa: PLC0415

                instance = PureLLMWrapper(model_name=base_model)
                with _ADAPTER_CACHE_LOCK:
                    _cache_put(model_name, instance)
                return instance

            # --- Route 3: Unknown ---
            raise ModelNotRecognizedException(
                f"Model '{model_name}' is not registered.\n"
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
# Thin callable wrapper for cloud adapters
#
# Cloud adapters implement __call__(messages, objective, session_id) directly.
# The rest of the system expects an object that is callable AND has
# model_name / get_llm_callable() attributes (for ExecutionPlanner introspection).
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

    # Forward attribute access to underlying adapter for any other consumers
    def __getattr__(self, name: str):
        return getattr(self._adapter, name)


# Module-level aliases for backward compatibility
build_llm = AdapterFactory.build_llm
get_action = AdapterFactory.get_action
