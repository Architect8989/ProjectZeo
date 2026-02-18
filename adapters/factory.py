from typing import List, Dict, Any
from operate.exceptions import ModelNotRecognizedException

from adapters.apis_safety_layer import apply_patches


_ADAPTER_REGISTRY: Dict[str, str] = {
    "qwen2.5-vl": "adapters.qwen_ollama_adapter.QwenOllamaAdapter",
}


_PATCHES_APPLIED = False


def _ensure_patches():
    global _PATCHES_APPLIED
    if not _PATCHES_APPLIED:
        apply_patches()
        _PATCHES_APPLIED = True


def _import_from_path(path: str):
    try:
        module_path, class_name = path.rsplit(".", 1)
    except ValueError:
        raise RuntimeError(f"Invalid adapter path format: {path}")

    module = __import__(module_path, fromlist=[class_name])

    if not hasattr(module, class_name):
        raise RuntimeError(
            f"Adapter class '{class_name}' not found in '{module_path}'"
        )

    return getattr(module, class_name)


class AdapterFactory:

    @staticmethod
    def build_llm(model_name: str):
        if not isinstance(model_name, str) or not model_name.strip():
            raise ModelNotRecognizedException(
                "Model name must be non-empty string."
            )

        model_name = model_name.strip()

        _ensure_patches()

        adapter_path = None

        for prefix, path in _ADAPTER_REGISTRY.items():
            if model_name.startswith(prefix):
                adapter_path = path
                break

        if adapter_path is None:
            raise ModelNotRecognizedException(
                f"Model '{model_name}' not registered in AdapterFactory."
            )

        AdapterClass = _import_from_path(adapter_path)

        return AdapterClass(model_name=model_name)

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
