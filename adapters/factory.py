from typing import List, Dict, Any
from operate.exceptions import ModelNotRecognizedException

# Correct package import (prevents ModuleNotFoundError at startup)
from .apis_safety_layer import apply_patches


# ==================================================
# MODEL REGISTRY
# ==================================================
# Maps model prefix → adapter class path
# Allows version flexibility while keeping strict mapping

_ADAPTER_REGISTRY: Dict[str, str] = {
    "qwen2.5-vl": "adapters.qwen_ollama_adapter.QwenOllamaAdapter",
}


# ==================================================
# PATCH GUARD
# ==================================================

_PATCHES_APPLIED = False


def _ensure_patches() -> None:
    global _PATCHES_APPLIED
    if not _PATCHES_APPLIED:
        apply_patches()
        _PATCHES_APPLIED = True


# ==================================================
# INTERNAL IMPORT UTIL
# ==================================================

def _import_from_path(path: str):
    """
    Lazy class importer.
    Prevents heavy adapter imports at module load time.
    """
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


# ==================================================
# FACTORY
# ==================================================

class AdapterFactory:
    """
    Clean adapter factory.
    - Containment patches enforced
    - Prefix-based model resolution
    - Strict contract: returns (operation_list, error_object)
    """

    @staticmethod
    def build_llm(model_name: str):
        """
        Instantiate adapter for given model.
        """

        if not isinstance(model_name, str) or not model_name.strip():
            raise ModelNotRecognizedException(
                "Model name must be non-empty string."
            )

        model_name = model_name.strip()

        # Activate containment patches exactly once
        _ensure_patches()

        adapter_path = None

        # Prefix-based resolution
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
        """
        Unified async execution interface.
        Returns:
            (operation_list, error_object)
        """

        adapter = AdapterFactory.build_llm(model_name)

        return await adapter.get_next_action(
            messages=messages,
            objective=objective,
            session_id=session_id,
        )


# ==================================================
# MODULE EXPORTS
# ==================================================

build_llm = AdapterFactory.build_llm
get_action = AdapterFactory.get_action
