# adapters/pure_llm_wrapper.py

import asyncio
import inspect
from typing import Callable, List, Dict, Any

from operate.models import apis


class PureLLMWrapper:
    """
    Isolation layer between kernel and apis.py.
    Normalizes all model functions to unified interface.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    def _resolve_model_function(self) -> Callable:
        registry = {
            "gpt-4o": apis.call_gpt_4o,
            "qwen-vl": apis.call_qwen_vl_with_ocr,
            "gpt-4o-with-ocr": apis.call_gpt_4o_with_ocr,
            "o1-with-ocr": apis.call_o1_with_ocr,
            "claude-3": apis.call_claude_3_with_ocr,
            "gemini-pro-vision": apis.call_gemini_pro_vision,
            "llava": apis.call_ollama_llava,
            "gpt-4_1-with-ocr": apis.call_gpt_4_1_with_ocr,
            "gpt-4o-labeled": apis.call_gpt_4o_labeled,
        }

        fn = registry.get(self.model_name)
        if not fn:
            raise ValueError(f"Unsupported model: {self.model_name}")

        return fn

    def __call__(
        self,
        messages: List[Dict[str, Any]],
        objective: str = None,
        session_id: str = None,
    ) -> Any:

        model_fn = self._resolve_model_function()

        try:
            if inspect.iscoroutinefunction(model_fn):
                return asyncio.run(
                    model_fn(messages, objective, self.model_name)
                )

            # Try full signature
            try:
                return model_fn(messages, objective, self.model_name)
            except TypeError:
                pass

            # Try reduced signature
            try:
                return model_fn(messages, objective)
            except TypeError:
                pass

            # Try minimal signature
            return model_fn(messages)

        except Exception as e:
            raise RuntimeError(
                f"Model execution failed for {self.model_name}: {e}"
          )
