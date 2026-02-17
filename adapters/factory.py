from typing import Callable, List, Dict, Any

from operate.exceptions import ModelNotRecognizedException


# ==================================================
# MODEL REGISTRY
# ==================================================
# Maps model names → adapter class paths
# Keeps apis.py isolated and optional

_ADAPTER_REGISTRY: Dict[str, str] = {
    # Pure local Qwen-VL adapter
    "qwen2.5-vl:7b-instruct": "adapters.qwen_ollama_adapter.QwenOllamaAdapter",
    "qwen2.5-vl:3b-instruct": "adapters.qwen_ollama_adapter.QwenOllamaAdapter",
}


# ==================================================
# INTERNAL UTIL
# ==================================================

def _import_from_path(path: str):
    """
    Lazy class importer.
    Prevents heavy imports at module load time.
    """
    module_path, class_name = path.rsplit(".", 1)

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
    No apis.py usage.
    No fallback logic.
    One model → one adapter.
    """

    # --------------------------------------------------
    # PURE ADAPTER BUILDER
    # --------------------------------------------------

    @staticmethod
    def build_llm(model_name: str):
        """
        Returns adapter instance.
        """

        adapter_path = _ADAPTER_REGISTRY.get(model_name)

        if adapter_path is None:
            raise ModelNotRecognizedException(
                f"Model '{model_name}' not registered in AdapterFactory."
            )

        AdapterClass = _import_from_path(adapter_path)

        return AdapterClass(model_name=model_name)

    # --------------------------------------------------
    # DIRECT ACTION FLOW
    # --------------------------------------------------

    @staticmethod
    async def get_action(
        model_name: str,
        messages: List[Dict[str, Any]],
        objective: str,
        session_id: str,
    ):
        """
        Direct execution entrypoint.
        Returns (operation_list, error_object)
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
