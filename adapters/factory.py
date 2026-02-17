from typing import Callable, List, Dict, Any

from operate.models.apis import (
    get_next_action,
    call_gpt_4o,
    call_qwen_vl_with_ocr,
    call_gpt_4o_with_ocr,
    call_o1_with_ocr,
    call_claude_3_with_ocr,
    call_gemini_pro_vision,
    call_ollama_llava,
    call_gpt_4_1_with_ocr,
    call_gpt_4o_labeled,
)

from operate.exceptions import ModelNotRecognizedException
from adapters.pure_llm_wrapper import PureLLMWrapper


# ==================================================
# MODEL REGISTRY
# ==================================================

_MODEL_REGISTRY: Dict[str, Callable] = {
    "gpt-4o": call_gpt_4o,
    "qwen-vl": call_qwen_vl_with_ocr,
    "gpt-4o-with-ocr": call_gpt_4o_with_ocr,
    "o1-with-ocr": call_o1_with_ocr,
    "claude-3": call_claude_3_with_ocr,
    "gemini-pro-vision": call_gemini_pro_vision,
    "llava": call_ollama_llava,
    "gpt-4.1-with-ocr": call_gpt_4_1_with_ocr,  # FIXED KEY
    "gpt-4o-labeled": call_gpt_4o_labeled,
}


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
        model_fn = _MODEL_REGISTRY.get(model_name)
        if model_fn is None:
            raise ModelNotRecognizedException(
                f"Model '{model_name}' not recognized."
            )
        return model_fn

    # --------------------------------------------------
    # NORMALIZED KERNEL-FACING BUILDER
    # --------------------------------------------------

    @staticmethod
    def build_llm(model_name: str) -> Callable:
        if model_name not in _MODEL_REGISTRY:
            raise ModelNotRecognizedException(
                f"Model '{model_name}' not recognized."
            )

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

        return get_next_action(
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
