from typing import Callable, List, Dict, Any

from operate.exceptions import ModelNotRecognizedException
from adapters.pure_llm_wrapper import PureLLMWrapper


# ==================================================
# MODEL REGISTRY (STRING → FUNCTION NAME)
# ==================================================

_MODEL_REGISTRY: Dict[str, str] = {
    "gpt-4o": "call_gpt_4o",
    "qwen-vl": "call_qwen_vl_with_ocr",
    "gpt-4o-with-ocr": "call_gpt_4o_with_ocr",
    "o1-with-ocr": "call_o1_with_ocr",
    "claude-3": "call_claude_3_with_ocr",
    "gemini-pro-vision": "call_gemini_pro_vision",
    "llava": "call_ollama_llava",
    "gpt-4.1-with-ocr": "call_gpt_4_1_with_ocr",
    "gpt-4o-labeled": "call_gpt_4o_labeled",
}


# ==================================================
# INTERNAL RESOLUTION
# ==================================================

def _resolve_provider(model_name: str) -> Callable:
    fn_name = _MODEL_REGISTRY.get(model_name)
    if fn_name is None:
        raise ModelNotRecognizedException(
            f"Model '{model_name}' not recognized."
        )

    # Lazy import — prevents heavy dependency loading at module import time
    from operate.models import apis

    if not hasattr(apis, fn_name):
        raise RuntimeError(
            f"Provider '{fn_name}' not found in operate.models.apis"
        )

    return getattr(apis, fn_name)


# ==================================================
# FACTORY
# ==================================================

class AdapterFactory:
    """
    Adapter Factory for dynamic model selection.
    Responsible for:
        - Registry validation
        - Raw model access (legacy)
        - Kernel-facing normalized builder
    """

    # --------------------------------------------------
    # RAW MODEL ACCESS (Legacy Support)
    # --------------------------------------------------

    @staticmethod
    def create_llm_callable(model_name: str) -> Callable:
        return _resolve_provider(model_name)

    # --------------------------------------------------
    # NORMALIZED KERNEL-FACING BUILDER
    # --------------------------------------------------

    @staticmethod
    def build_llm(model_name: str) -> Callable:
        if model_name not in _MODEL_REGISTRY:
            raise ModelNotRecognizedException(
                f"Model '{model_name}' not recognized."
            )

        # PureLLMWrapper internally resolves provider via apis
        return PureLLMWrapper(model_name)

    # --------------------------------------------------
    # BACKWARD-COMPATIBLE ACTION FLOW
    # --------------------------------------------------

    @staticmethod
    def get_action(
        model_name: str,
        messages: List[Dict[str, Any]],
        objective: str,
        session_id: str,
    ) -> Any:

        if model_name not in _MODEL_REGISTRY:
            raise ModelNotRecognizedException(
                f"Model '{model_name}' not recognized."
            )

        from operate.models import apis  # lazy import

        if not hasattr(apis, "get_next_action"):
            raise RuntimeError("get_next_action not available in apis")

        return apis.get_next_action(
            model_name,
            messages,
            objective,
            session_id,
        )


# ==================================================
# MODULE-LEVEL EXPORTS
# ==================================================

build_llm = AdapterFactory.build_llm
create_llm_callable = AdapterFactory.create_llm_callable
get_action = AdapterFactory.get_action
