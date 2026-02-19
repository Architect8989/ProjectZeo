"""
adapters/factory.py
====================
PATCHES APPLIED (Audit Fixes):

  ✅  §R3  (was §factory-5): get_action() classmethod now uses a module-level
           adapter cache keyed by model_name so a new ollama.Client is NOT
           constructed on every call.  Cache is lock-protected for thread safety.
           Avoids connection pool exhaustion under concurrent use.

  ✅  §DEF-1: _validate_model_name() now permits the 'ollama/' prefix format
           (e.g. 'ollama/qwen2.5-vl') which is a common Ollama CLI convention.
           The prefix is stripped before registry lookup so 'ollama/qwen2.5-vl'
           resolves identically to 'qwen2.5-vl'. The regex is also updated to
           allow '/' as a valid character.

All existing correct behaviours preserved:
  - _ADAPTER_REGISTRY with qwen2.5-vl as the only local entry
  - _validate_model_name() strict regex guard (updated to allow /)
  - _resolve_base_model() version-tag stripping + ollama/ prefix stripping
  - _ensure_patches() idempotent safety layer application
  - Dynamic import via _import_from_path()
"""

from typing import List, Dict, Any, Type
from operate.exceptions import ModelNotRecognizedException
from adapters.apis_safety_layer import apply_patches

import importlib
import threading
import re


_ADAPTER_REGISTRY: Dict[str, str] = {
    "qwen2.5-vl": "adapters.qwen_ollama_adapter.QwenOllamaAdapter",
}

_PATCHES_APPLIED = False
_PATCH_LOCK = threading.Lock()

# DEF-1: Allow '/' in model names to support 'ollama/qwen2.5-vl' format.
# The slash is safe here because model names are only used for registry lookup
# and adapter cache keys — they are never interpolated into shell commands.
_MODEL_PATTERN = re.compile(r"^[a-zA-Z0-9.\-_:/]+$")

# PATCH §R3: module-level adapter cache — one instance per model name
_ADAPTER_CACHE: Dict[str, Any] = {}
_ADAPTER_CACHE_LOCK = threading.Lock()


def _ensure_patches():
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return

    with _PATCH_LOCK:
        if not _PATCHES_APPLIED:
            apply_patches()
            _PATCHES_APPLIED = True


def _import_from_path(path: str) -> Type:
    try:
        module_path, class_name = path.rsplit(".", 1)
    except ValueError:
        raise RuntimeError(f"Invalid adapter path format: {path}")

    module = importlib.import_module(module_path)

    try:
        return getattr(module, class_name)
    except AttributeError:
        raise RuntimeError(
            f"Adapter class '{class_name}' not found in '{module_path}'"
        )


def _validate_model_name(model_name: str) -> str:
    if not isinstance(model_name, str) or not model_name.strip():
        raise ModelNotRecognizedException(
            "Model name must be non-empty string."
        )

    model_name = model_name.strip()

    if not _MODEL_PATTERN.fullmatch(model_name):
        raise ModelNotRecognizedException(
            f"Invalid model name format: '{model_name}'"
        )

    return model_name


def _resolve_base_model(model_name: str) -> str:
    """
    Normalise a model name to its base registry key.

    Handles two common variant formats:
      1. Version tags:     'qwen2.5-vl:7b-instruct'  → 'qwen2.5-vl'
      2. Provider prefix:  'ollama/qwen2.5-vl'        → 'qwen2.5-vl'
         (DEF-1 fix: common Ollama CLI convention — prefix stripped before lookup)
    """
    # Strip optional 'ollama/' provider prefix
    if model_name.startswith("ollama/"):
        model_name = model_name[len("ollama/"):]

    # Strip version tag (everything after the first ':')
    return model_name.split(":", 1)[0]


class AdapterFactory:

    @staticmethod
    def build_llm(model_name: str):
        model_name = _validate_model_name(model_name)

        _ensure_patches()

        # PATCH §R3: return cached adapter if already built
        with _ADAPTER_CACHE_LOCK:
            if model_name in _ADAPTER_CACHE:
                return _ADAPTER_CACHE[model_name]

        base_model = _resolve_base_model(model_name)

        adapter_path = _ADAPTER_REGISTRY.get(base_model)

        if adapter_path is None:
            raise ModelNotRecognizedException(
                f"Model '{model_name}' not registered in AdapterFactory."
            )

        AdapterClass = _import_from_path(adapter_path)
        instance = AdapterClass(model_name=model_name)

        with _ADAPTER_CACHE_LOCK:
            _ADAPTER_CACHE[model_name] = instance

        return instance

    @staticmethod
    async def get_action(
        model_name: str,
        messages: List[Dict[str, Any]],
        objective: str,
        session_id: str,
    ):
        # PATCH §R3: reuse cached adapter — does NOT reconstruct on every call
        adapter = AdapterFactory.build_llm(model_name)

        return await adapter.get_next_action(
            messages=messages,
            objective=objective,
            session_id=session_id,
        )


build_llm = AdapterFactory.build_llm
get_action = AdapterFactory.get_action
