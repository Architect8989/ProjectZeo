# adapters/factory.py

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


# ==================================================
# MODEL REGISTRY (No branching sprawl)
# ==================================================

_MODEL_REGISTRY = {
    "gpt-4o": call_gpt_4o,
    "qwen-vl": call_qwen_vl_with_ocr,
    "gpt-4o-with-ocr": call_gpt_4o_with_ocr,
    "o1-with-ocr": call_o1_with_ocr,
    "claude-3": call_claude_3_with_ocr,
    "gemini-pro-vision": call_gemini_pro_vision,
    "llava": call_ollama_llava,
    "gpt-4_1-with-ocr": call_gpt_4_1_with_ocr,
    "gpt-4o-labeled": call_gpt_4o_labeled,
}


# ==================================================
# FACTORY
# ==================================================

class AdapterFactory:
    """
    Adapter Factory for dynamic model selection.
    """

    @staticmethod
    def create_llm_callable(model_name: str) -> Callable:
        """
        Returns raw model function from registry.
        """
        model_fn = _MODEL_REGISTRY.get(model_name)
        if not model_fn:
            raise ModelNotRecognizedException(
                f"Model '{model_name}' not recognized!"
            )
        return model_fn

    @staticmethod
    def build_llm(model_name: str) -> Callable:
        """
        Unified kernel-facing LLM builder.

        Returns a normalized callable with consistent signature:

            llm_callable(messages, objective=None, session_id=None)

        This prevents signature mismatch across models.
        """

        model_fn = AdapterFactory.create_llm_callable(model_name)

        def llm_callable(
            messages: List[Dict[str, Any]],
            objective: str = None,
            session_id: str = None,
        ):
            """
            Normalized wrapper around all model APIs.
            """
            return model_fn(
                messages=messages,
                objective=objective,
                session_id=session_id,
            )

        return llm_callable

    @staticmethod
    def get_action(model_name: str, messages, objective, session_id):
        """
        Backward-compatible action fetcher.
        """
        try:
            llm_callable = AdapterFactory.build_llm(model_name)
            return get_next_action(
                llm_callable,
                messages,
                objective,
                session_id,
            )
        except ModelNotRecognizedException as e:
            print(f"Error: {str(e)}")
            raise
