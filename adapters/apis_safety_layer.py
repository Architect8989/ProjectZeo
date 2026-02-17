# adapters/apis_safety_layer.py

"""
External hardening layer for operate.models.apis

Goals:
- Prevent message mutation
- Prevent silent None returns
- Disable cloud fallback cascade
- Stop recursion patterns
- Enforce deterministic failure
- Avoid modifying original 6k-line file
"""

import copy
import inspect
import functools
from operate.models import apis

_PATCHED = False


# ============================================================
# Public Entry
# ============================================================

def apply_patches():
    global _PATCHED
    if _PATCHED:
        return

    _PATCHED = True

    _patch_all_providers()
    _disable_cloud_fallback()
    _guard_dispatch()


# ============================================================
# Core Provider Wrapper
# ============================================================

def _wrap_provider(fn):

    if getattr(fn, "_apis_safety_wrapped", False):
        return fn

    is_async = inspect.iscoroutinefunction(fn)

    def validate_no_mutation(original, caller_messages, name):
        if caller_messages != original:
            raise RuntimeError(
                f"[APIS-SAFETY] Provider mutated caller message history: {name}"
            )

    def validate_result(result, name):
        if result is None:
            raise RuntimeError(
                f"[APIS-SAFETY] {name} returned None"
            )
        if not isinstance(result, (dict, list)):
            raise RuntimeError(
                f"[APIS-SAFETY] {name} returned invalid type: {type(result)}"
            )

    if is_async:

        @functools.wraps(fn)
        async def async_wrapper(messages, *args, **kwargs):
            caller_snapshot = copy.deepcopy(messages)
            safe_messages = copy.deepcopy(messages)

            try:
                result = await fn(safe_messages, *args, **kwargs)
            except Exception as e:
                raise RuntimeError(
                    f"[APIS-SAFETY] {fn.__name__} failed: {e}"
                ) from e

            validate_no_mutation(caller_snapshot, messages, fn.__name__)
            validate_result(result, fn.__name__)

            return result

        async_wrapper._apis_safety_wrapped = True
        return async_wrapper

    else:

        @functools.wraps(fn)
        def sync_wrapper(messages, *args, **kwargs):
            caller_snapshot = copy.deepcopy(messages)
            safe_messages = copy.deepcopy(messages)

            try:
                result = fn(safe_messages, *args, **kwargs)
            except Exception as e:
                raise RuntimeError(
                    f"[APIS-SAFETY] {fn.__name__} failed: {e}"
                ) from e

            validate_no_mutation(caller_snapshot, messages, fn.__name__)
            validate_result(result, fn.__name__)

            return result

        sync_wrapper._apis_safety_wrapped = True
        return sync_wrapper


# ============================================================
# Patch All Providers
# ============================================================

def _patch_all_providers():

    provider_names = [
        "call_gpt_4o",
        "call_qwen_vl_with_ocr",
        "call_gemini_pro_vision",
        "call_gpt_4o_with_ocr",
        "call_gpt_4_1_with_ocr",
        "call_o1_with_ocr",
        "call_gpt_4o_labeled",
        "call_ollama_llava",
        "call_claude_3_with_ocr",
    ]

    for name in provider_names:
        if hasattr(apis, name):
            original = getattr(apis, name)
            wrapped = _wrap_provider(original)
            setattr(apis, name, wrapped)


# ============================================================
# Disable Cloud Fallback
# ============================================================

def _disable_cloud_fallback():

    if hasattr(apis, "gpt_4_fallback"):

        def hard_fail_fallback(*args, **kwargs):
            raise RuntimeError(
                "[APIS-SAFETY] Cloud fallback disabled"
            )

        apis.gpt_4_fallback = hard_fail_fallback


# ============================================================
# Guard Dispatcher
# ============================================================

def _guard_dispatch():

    if not hasattr(apis, "get_next_action"):
        return

    original = apis.get_next_action

    if getattr(original, "_apis_safety_wrapped", False):
        return

    @functools.wraps(original)
    async def guarded(model, messages, objective, session_id):
        result = await original(model, messages, objective, session_id)

        if result is None:
            raise RuntimeError(
                "[APIS-SAFETY] get_next_action returned None"
            )

        if not isinstance(result, (dict, list)):
            raise RuntimeError(
                "[APIS-SAFETY] get_next_action returned invalid type"
            )

        return result

    guarded._apis_safety_wrapped = True
    apis.get_next_action = guarded
