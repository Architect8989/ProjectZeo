from typing import Callable, List, Dict, Any
from operate.exceptions import ModelNotRecognizedException


# ==================================================
# MODEL REGISTRY
# ==================================================
# Maps model name → adapter class path
# Fully isolates operate.models.apis

_ADAPTER_REGISTRY: Dict[str, str] = {
    "qwen2.5-vl:7b-instruct": "adapters.qwen_ollama_adapter.QwenOllamaAdapter",
    "qwen2.5-vl:3b-instruct": "adapters.qwen_ollama_adapter.QwenOllamaAdapter",
}


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
    - No operate.models.apis dependency
    - No fallback logic
    - One model → one adapter
    - Strict contract: returns (operation_list, error_object)
    """

    # --------------------------------------------------
    # BUILDER
    # --------------------------------------------------

    @staticmethod
    def build_llm(model_name: str):
        """
        Instantiate adapter for given model.
        """

        adapter_path = _ADAPTER_REGISTRY.get(model_name)

        if adapter_path is None:
            raise ModelNotRecognizedException(
                f"Model '{model_name}' not registered in AdapterFactory."
            )

        AdapterClass = _import_from_path(adapter_path)

        # Instantiate once per request — kernel can cache if needed
        return AdapterClass(model_name=model_name)

    # --------------------------------------------------
    # EXECUTION ENTRYPOINT
    # --------------------------------------------------

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

        # Adapter must enforce immutability and exception safety internally
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
