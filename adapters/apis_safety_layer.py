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
from operate.models import apis

_PATCHED = False


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
    is_async = inspect.iscoroutinefunction(fn)

    def validate_no_mutation(original, current, name):
        if original != current:
            raise RuntimeError(
                f"[APIS-SAFETY] Provider mutated message history: {name}"
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

        async def async_wrapper(messages, *args, **kwargs):
            snapshot = copy.deepcopy(messages)
            safe_messages = copy.deepcopy(messages)

            try:
                result = await fn(safe_messages, *args, **kwargs)
            except Exception as e:
                raise RuntimeError(
                    f"[APIS-SAFETY] {fn.__name__} failed: {e}"
                ) from e

            validate_no_mutation(snapshot, messages, fn.__name__)
            validate_result(result, fn.__name__)
            return result

        return async_wrapper

    else:

        def sync_wrapper(messages, *args, **kwargs):
            snapshot = copy.deepcopy(messages)
            safe_messages = copy.deepcopy(messages)

            try:
                result = fn(safe_messages, *args, **kwargs)
            except Exception as e:
                raise RuntimeError(
                    f"[APIS-SAFETY] {fn.__name__} failed: {e}"
                ) from e

            validate_no_mutation(snapshot, messages, fn.__name__)
            validate_result(result, fn.__name__)
            return result

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

    def hard_fail_fallback(*args, **kwargs):
        raise RuntimeError(
            "[APIS-SAFETY] Cloud fallback disabled"
        )

    if hasattr(apis, "gpt_4_fallback"):
        apis.gpt_4_fallback = hard_fail_fallback


# ============================================================
# Guard get_next_action
# ============================================================

def _guard_dispatch():

    if not hasattr(apis, "get_next_action"):
        return

    original = apis.get_next_action

    async def guarded(model, messages, objective, session_id):
        result = await original(model, messages, objective, session_id)

        if result is None:
            raise RuntimeError(
                "[APIS-SAFETY] get_next_action returned None"
            )

        return result

    apis.get_next_action = guarded
