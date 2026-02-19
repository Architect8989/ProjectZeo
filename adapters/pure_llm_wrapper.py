"""
adapters/pure_llm_wrapper.py
=============================
PATCH AUDIT FIXES:

  ❌  §1.2: `from operate.models import apis` → ImportError because
            operate/models/apis.py did not exist.
            FIX: The canonical apis.py has now been created at
            operate/models/apis.py (see that file's patch notes).
            This file now imports from that module correctly.

  ✅  All other logic unchanged — executor, coroutine isolation,
      signature inspection, output normalisation.
"""

from __future__ import annotations

import asyncio
import inspect
import copy
import json
from typing import Callable, List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

# PATCH: was `from operate.models import apis` which caused ImportError.
# operate/models/apis.py now exists and re-exports all legacy callables.
from operate.models import apis  # noqa: E402


class PureLLMWrapper:

    _patch_applied = False
    _executor = ThreadPoolExecutor(max_workers=4)

    def __init__(self, model_name: str):
        if not PureLLMWrapper._patch_applied:
            from adapters.apis_safety_layer import apply_patches  # noqa: PLC0415
            apply_patches()
            PureLLMWrapper._patch_applied = True

        self.model_name = model_name

    # ==================================================
    # MODEL RESOLUTION
    # ==================================================

    def _resolve_model_function(self) -> Callable:
        registry = {
            "gpt-4o":             apis.call_gpt_4o,
            "qwen-vl":            apis.call_qwen_vl_with_ocr,
            "gpt-4o-with-ocr":    apis.call_gpt_4o_with_ocr,
            "o1-with-ocr":        apis.call_o1_with_ocr,
            "claude-3":           apis.call_claude_3_with_ocr,
            "gemini-pro-vision":  apis.call_gemini_pro_vision,
            "llava":              apis.call_ollama_llava,
            "gpt-4.1-with-ocr":   apis.call_gpt_4_1_with_ocr,
            "gpt-4o-labeled":     apis.call_gpt_4o_labeled,
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

        original_snapshot = copy.deepcopy(messages)
        messages_copy = copy.deepcopy(messages)

        try:
            result = self._execute_model(
                model_fn,
                messages_copy,
                objective,
                session_id,
                screen_image,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Model execution failed for {self.model_name}: {exc}"
            ) from exc

        if messages_copy != original_snapshot:
            raise RuntimeError(
                f"Model '{self.model_name}' mutated input messages."
            )

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

        def _run_in_isolated_loop(coroutine):
            new_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                return new_loop.run_until_complete(coroutine)
            finally:
                new_loop.close()

        future = PureLLMWrapper._executor.submit(
            _run_in_isolated_loop,
            coro,
        )

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
