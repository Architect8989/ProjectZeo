import asyncio
import inspect
import copy
import json
from typing import Callable, List, Dict, Any, Optional

from operate.models import apis


class PureLLMWrapper:
    """
    Isolation layer between kernel and apis.py.

    Guarantees:
        - No input mutation
        - Unified callable interface
        - Safe coroutine handling
        - Deterministic failure behavior
    """

    def __init__(self, model_name: str):
        from adapters.apis_safety_layer
        import apply_patches

    apply_patches()  # activate containment layer

    self.model_name = model_name

    # ==================================================
    # MODEL RESOLUTION
    # ==================================================

    def _resolve_model_function(self) -> Callable:
        registry = {
            "gpt-4o": apis.call_gpt_4o,
            "qwen-vl": apis.call_qwen_vl_with_ocr,
            "gpt-4o-with-ocr": apis.call_gpt_4o_with_ocr,
            "o1-with-ocr": apis.call_o1_with_ocr,
            "claude-3": apis.call_claude_3_with_ocr,
            "gemini-pro-vision": apis.call_gemini_pro_vision,
            "llava": apis.call_ollama_llava,
            "gpt-4.1-with-ocr": apis.call_gpt_4_1_with_ocr,
            "gpt-4o-labeled": apis.call_gpt_4o_labeled,
        }

        fn = registry.get(self.model_name)
        if not fn:
            raise ValueError(f"Unsupported model: {self.model_name}")

        return fn

    # ==================================================
    # PUBLIC CALL INTERFACE
    # ==================================================

    def __call__(
        self,
        messages: List[Dict[str, Any]],
        objective: Optional[str] = None,
        session_id: Optional[str] = None,
        screen_image: Optional[str] = None,
    ) -> Any:

        model_fn = self._resolve_model_function()

        # ----------------------------
        # 1. Freeze inputs
        # ----------------------------

        original_snapshot = copy.deepcopy(messages)
        messages_copy = copy.deepcopy(messages)

        # ----------------------------
        # 2. Execute safely
        # ----------------------------

        try:
            result = self._execute_model(
                model_fn,
                messages_copy,
                objective,
                session_id,
                screen_image,
            )
        except Exception as e:
            raise RuntimeError(
                f"Model execution failed for {self.model_name}: {e}"
            ) from e

        # ----------------------------
        # 3. Mutation detection
        # ----------------------------

        if messages_copy != original_snapshot:
            raise RuntimeError(
                f"Model '{self.model_name}' mutated input messages."
            )

        # ----------------------------
        # 4. Normalize output
        # ----------------------------

        return self._normalize_output(result)

    # ==================================================
    # INTERNAL EXECUTION LOGIC
    # ==================================================

    def _execute_model(
        self,
        model_fn: Callable,
        messages: List[Dict[str, Any]],
        objective: Optional[str],
        session_id: Optional[str],
        screen_image: Optional[str],
    ) -> Any:

        if inspect.iscoroutinefunction(model_fn):
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    # Running inside existing loop (e.g. notebook)
                    return asyncio.create_task(
                        self._call_with_signature(
                            model_fn,
                            messages,
                            objective,
                            session_id,
                            screen_image,
                        )
                    )
            except RuntimeError:
                pass

            return asyncio.run(
                self._call_with_signature(
                    model_fn,
                    messages,
                    objective,
                    session_id,
                    screen_image,
                )
            )

        return self._call_with_signature(
            model_fn,
            messages,
            objective,
            session_id,
            screen_image,
        )

    def _call_with_signature(
        self,
        model_fn: Callable,
        messages: List[Dict[str, Any]],
        objective: Optional[str],
        session_id: Optional[str],
        screen_image: Optional[str],
    ) -> Any:

        sig = inspect.signature(model_fn)
        params = sig.parameters

        if "screen_image" in params:
            return model_fn(
                messages,
                objective=objective,
                session_id=session_id,
                screen_image=screen_image,
            )

        if "session_id" in params:
            return model_fn(
                messages,
                objective=objective,
                session_id=session_id,
            )

        if "objective" in params:
            return model_fn(messages, objective)

        return model_fn(messages)

    # ==================================================
    # OUTPUT NORMALIZATION
    # ==================================================

    def _normalize_output(self, result: Any) -> Any:

        if result is None:
            raise RuntimeError(
                f"Model '{self.model_name}' returned None."
            )

        if isinstance(result, dict):
            return result

        if isinstance(result, list):
            return result  # allow action arrays (llava, etc.)

        if isinstance(result, str):
            try:
                parsed = json.loads(result)
            except Exception:
                raise RuntimeError(
                    f"Model '{self.model_name}' returned non-JSON string."
                )

            return parsed

        raise RuntimeError(
            f"Model '{self.model_name}' returned unsupported type: "
            f"{type(result)}"
            )
