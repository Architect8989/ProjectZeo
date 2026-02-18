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

_ADAPTER_CACHE: Dict[str, Any] = {}
_CACHE_LOCK = threading.Lock()

_MODEL_PATTERN = re.compile(r"^[a-zA-Z0-9.\-_:]+$")


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

    if not _MODEL_PATTERN.match(model_name):
        raise ModelNotRecognizedException(
            f"Invalid model name format: '{model_name}'"
        )

    return model_name


def _resolve_base_model(model_name: str) -> str:
    # Accept format like: qwen2.5-vl:7b-instruct
    base = model_name.split(":", 1)[0]
    return base


class AdapterFactory:

    @staticmethod
    def build_llm(model_name: str):
        model_name = _validate_model_name(model_name)

        _ensure_patches()

        base_model = _resolve_base_model(model_name)

        adapter_path = _ADAPTER_REGISTRY.get(base_model)

        if adapter_path is None:
            raise ModelNotRecognizedException(
                f"Model '{model_name}' not registered in AdapterFactory."
            )

        with _CACHE_LOCK:
            if model_name in _ADAPTER_CACHE:
                return _ADAPTER_CACHE[model_name]

            AdapterClass = _import_from_path(adapter_path)
            instance = AdapterClass(model_name=model_name)
            _ADAPTER_CACHE[model_name] = instance
            return instance

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


build_llm = AdapterFactory.build_llm
get_action = AdapterFactory.get_action
