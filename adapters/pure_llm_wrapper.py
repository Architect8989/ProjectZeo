from __future__ import annotations

import asyncio
import inspect
import copy
import json
from typing import Callable, List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from operate.models import apis


class PureLLMWrapper:
    """
    Unified wrapper for all cloud / legacy LLM providers.

    Exposes the same get_next_action() interface as QwenOllamaAdapter
    so the kernel treats all LLM providers identically.
    """

    _patch_applied = False

    # RB-07 FIX: _executor is now an INSTANCE attribute (created in __init__),
    # not a class attribute.
    #
    # Bug: _executor = ThreadPoolExecutor(max_workers=4) as a class attribute
    # means ALL instances share the same executor. In multi-adapter scenarios
    # (e.g. during replan that replaces the planner, creating a new adapter
    # instance), the first instance's __del__ or shutdown() call would destroy
    # the executor for all other live instances. Any in-flight coroutine
    # running in the second instance would then receive RuntimeError from the
    # destroyed pool.
    #
    # Fix: instantiate a fresh executor per instance in __init__, with a
    # matching __del__ for graceful cleanup. Each adapter now has independent
    # lifecycle management.

    def __init__(self, model_name: str):
        if not PureLLMWrapper._patch_applied:
            from adapters.apis_safety_layer import apply_patches  # noqa: PLC0415
            apply_patches()
            PureLLMWrapper._patch_applied = True

        self.model_name = model_name

        # RB-07 FIX: Instance-level executor with independent lifecycle.
        self._executor = ThreadPoolExecutor(max_workers=4)

    def __del__(self):
        """Gracefully shut down the instance executor on garbage collection."""
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass

    # ==================================================
    # ADAPTER INTERFACE — matches QwenOllamaAdapter
    # ==================================================

    async def get_next_action(
        self,
        messages: List[Dict[str, Any]],
        objective: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        """
        Public adapter interface — identical signature to QwenOllamaAdapter.
        Returns (ops, None) on success, (None, exception) on error.
        """
        try:
            result = self(
                messages=messages,
                objective=objective,
                session_id=session_id,
            )
            return result, None
        except Exception as exc:
            return None, exc

    # ==================================================
    # MODEL RESOLUTION
    # ==================================================

    def _resolve_model_function(self) -> Callable:
        """
        Map model_name to the corresponding API function.

        EXTENSION: add new cloud models here. The function must be exported
        from operate/models/apis.py.

        NOTE (RB-NEW-02 FIX): "llava" is intentionally absent from this
        registry. llava is a local Ollama model and must be routed exclusively
        through QwenOllamaAdapter (the local adapter). Including it here would
        create a shadow cloud routing path through PureLLMWrapper even when
        OLLAMA_ONLY=1 enforcement is active. If "llava" reaches this method,
        it means the factory misconfigured the adapter — raise ValueError
        immediately so the misconfiguration is visible rather than silently
        routing through the wrong backend.
        """
        registry: Dict[str, Callable] = {
            "gpt-4":              apis.call_gpt_4o,
            "gpt-4o":             apis.call_gpt_4o,
            "qwen-vl":            apis.call_qwen_vl_with_ocr,
            "gpt-4o-with-ocr":   apis.call_gpt_4o_with_ocr,
            # SI-08 FIX: Removed duplicate key "gpt-4.1-with-ocr" that appeared
            # twice in this dict. Python silently uses the last definition when a
            # dict literal contains duplicate keys. While both entries mapped to
            # the same function (call_gpt_4_1_with_ocr), the duplication signals
            # an unmaintained registry where a future silent overwrite could
            # introduce a hard-to-detect routing bug.
            "gpt-4.1-with-ocr":  apis.call_gpt_4_1_with_ocr,
            "o1-with-ocr":       apis.call_o1_with_ocr,
            "claude-3":          apis.call_claude_3_with_ocr,
            "claude-3-opus":     apis.call_claude_3_with_ocr,
            "claude-3-sonnet":   apis.call_claude_3_with_ocr,
            "gemini-pro-vision": apis.call_gemini_pro_vision,
            # RB-NEW-02 FIX: "llava" removed — see docstring above.
            "gpt-4o-labeled":    apis.call_gpt_4o_labeled,
            "gpt-4-with-som":    apis.call_gpt_4o_labeled,
            "gpt-4-with-ocr":    apis.call_gpt_4o_with_ocr,
        }

        fn = registry.get(self.model_name)
        if fn is None:
            raise ValueError(
                f"Unsupported model: '{self.model_name}'. "
                f"Known models: {sorted(registry.keys())}"
            )
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

        original_snapshot = copy.deepcopy(messages)
        messages_copy = copy.deepcopy(messages)

        try:
            result = self._execute_model(
                model_fn, messages_copy, objective, session_id, screen_image
            )
        except Exception as exc:
            raise RuntimeError(
                f"Model execution failed for '{self.model_name}': {exc}"
            ) from exc

        if messages_copy != original_snapshot:
            raise RuntimeError(
                f"Model '{self.model_name}' mutated input messages (immutability violation)."
            )

        return self._normalize_output(result)

    # ==================================================
    # INTERNAL EXECUTION
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
            coro = self._call_with_signature(
                model_fn, messages, objective, session_id, screen_image
            )
            return self._run_coroutine_safely(coro)

        result = self._call_with_signature(
            model_fn, messages, objective, session_id, screen_image
        )

        if inspect.iscoroutine(result):
            return self._run_coroutine_safely(result)

        return result

    def _run_coroutine_safely(self, coro):
        try:
            asyncio.get_running_loop()
            inside_loop = True
        except RuntimeError:
            inside_loop = False

        if not inside_loop:
            return asyncio.run(coro)

        # Inside a running loop — run in isolated thread to avoid deadlock
        def _run_in_isolated_loop(coroutine):
            new_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                return new_loop.run_until_complete(coroutine)
            finally:
                new_loop.close()

        # RB-NEW-01 FIX: Use self._executor (instance attribute) not
        # PureLLMWrapper._executor (class attribute that does not exist).
        # The RB-07 fix converted _executor to an instance attribute in __init__,
        # but left this call-site referencing the class. Without this fix,
        # any cloud-model call on an async call-stack raises:
        #   AttributeError: type object 'PureLLMWrapper' has no attribute '_executor'
        future = self._executor.submit(_run_in_isolated_loop, coro)
        return future.result()

    # ==================================================
    # SIGNATURE HANDLING
    # ==================================================

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
            return model_fn(messages, objective=objective, session_id=session_id)
        if "objective" in params:
            return model_fn(messages, objective)
        return model_fn(messages)

    # ==================================================
    # OUTPUT NORMALIZATION
    # ==================================================

    def _normalize_output(self, result: Any) -> Any:
        if result is None:
            raise RuntimeError(f"Model '{self.model_name}' returned None.")

        if isinstance(result, (dict, list)):
            return result

        if isinstance(result, str):
            try:
                return json.loads(result)
            except Exception as exc:
                raise RuntimeError(
                    f"Model '{self.model_name}' returned non-JSON string."
                ) from exc

        raise RuntimeError(
            f"Model '{self.model_name}' returned unsupported type: {type(result)}"
        )
