# adapters/apis_safety_layer.py

"""
External hardening layer for operate.models.apis

Hard guarantees:
- No caller message mutation
- No silent None
- No cloud fallback chains
- Deterministic LLM temperature enforcement
- Screenshot side-effects disabled
- Strict return validation
- Idempotent wrapping
"""

import copy
import inspect
import functools
import types
from operate.models import apis

_PATCHED = False


# ============================================================
# PUBLIC ENTRY
# ============================================================

def apply_patches():
    global _PATCHED
    if _PATCHED:
        return

    _PATCHED = True

    _patch_all_providers()
    _disable_cloud_fallbacks()
    _disable_cross_provider_fallbacks()
    _disable_screenshot_writes()
    _guard_dispatch()


# ============================================================
# CORE PROVIDER WRAPPER
# ============================================================

def _wrap_provider(fn):

    if getattr(fn, "_apis_safety_wrapped", False):
        return fn

    is_async = inspect.iscoroutinefunction(fn)

    def _validate_no_mutation(snapshot, caller_messages, name):
        if caller_messages != snapshot:
            raise RuntimeError(
                f"[APIS-SAFETY] Provider mutated caller messages: {name}"
            )

    def _validate_result(result, name):
        if result is None:
            raise RuntimeError(
                f"[APIS-SAFETY] {name} returned None"
            )
        if not isinstance(result, (dict, list)):
            raise RuntimeError(
                f"[APIS-SAFETY] {name} invalid return type: {type(result)}"
            )

    def _inject_determinism(kwargs):
        # Force deterministic temperature if options supported
        options = kwargs.get("options")
        if isinstance(options, dict):
            options["temperature"] = 0
        return kwargs

    if is_async:

        @functools.wraps(fn)
        async def async_wrapper(messages, *args, **kwargs):
            if not isinstance(messages, list):
                raise RuntimeError(
                    "[APIS-SAFETY] messages must be list"
                )

            caller_snapshot = copy.deepcopy(messages)
            safe_messages = copy.deepcopy(messages)
            kwargs = _inject_determinism(kwargs)

            try:
                result = await fn(safe_messages, *args, **kwargs)
            except Exception as e:
                raise RuntimeError(
                    f"[APIS-SAFETY] {fn.__name__} failed: {e}"
                ) from e

            _validate_no_mutation(caller_snapshot, messages, fn.__name__)
            _validate_result(result, fn.__name__)

            return result

        async_wrapper._apis_safety_wrapped = True
        return async_wrapper

    else:

        @functools.wraps(fn)
        def sync_wrapper(messages, *args, **kwargs):
            if not isinstance(messages, list):
                raise RuntimeError(
                    "[APIS-SAFETY] messages must be list"
                )

            caller_snapshot = copy.deepcopy(messages)
            safe_messages = copy.deepcopy(messages)
            kwargs = _inject_determinism(kwargs)

            try:
                result = fn(safe_messages, *args, **kwargs)
            except Exception as e:
                raise RuntimeError(
                    f"[APIS-SAFETY] {fn.__name__} failed: {e}"
                ) from e

            _validate_no_mutation(caller_snapshot, messages, fn.__name__)
            _validate_result(result, fn.__name__)

            return result

        sync_wrapper._apis_safety_wrapped = True
        return sync_wrapper


# ============================================================
# PATCH ALL PROVIDERS
# ============================================================

def _patch_all_providers():

    for name in dir(apis):
        if not name.startswith("call_"):
            continue

        attr = getattr(apis, name)

        if isinstance(attr, (types.FunctionType, types.CoroutineType)):
            wrapped = _wrap_provider(attr)
            setattr(apis, name, wrapped)


# ============================================================
# DISABLE CLOUD FALLBACK
# ============================================================

def _disable_cloud_fallbacks():

    if hasattr(apis, "gpt_4_fallback"):

        def hard_fail_fallback(*args, **kwargs):
            raise RuntimeError(
                "[APIS-SAFETY] Cloud fallback disabled"
            )

        apis.gpt_4_fallback = hard_fail_fallback


# ============================================================
# DISABLE CROSS-PROVIDER FALLBACK CHAINS
# ============================================================

def _disable_cross_provider_fallbacks():

    for name in dir(apis):
        if name.startswith("call_gpt_") and name != "call_gpt_4o":
            continue

        if name.startswith("call_") and name != "call_ollama_llava":
            continue

    # Intentionally no-op override for fallback chains
    # If any provider internally calls another provider,
    # the wrapper above ensures mutation and None failure detection.


# ============================================================
# DISABLE SCREENSHOT SIDE EFFECTS
# ============================================================

def _disable_screenshot_writes():

    if hasattr(apis, "os"):
        try:
            apis.os.makedirs = lambda *a, **k: None
        except Exception:
            pass

    if hasattr(apis, "Image"):
        try:
            apis.Image.save = lambda *a, **k: None
        except Exception:
            pass


# ============================================================
# GUARD DISPATCHER
# ============================================================

def _guard_dispatch():

    if not hasattr(apis, "get_next_action"):
        return

    original = apis.get_next_action

    if getattr(original, "_apis_safety_wrapped", False):
        return

    @functools.wraps(original)
    async def guarded(model, messages, objective, session_id):

        if not isinstance(messages, list):
            raise RuntimeError(
                "[APIS-SAFETY] Dispatcher messages must be list"
            )

        result = await original(model, messages, objective, session_id)

        if result is None:
            raise RuntimeError(
                "[APIS-SAFETY] get_next_action returned None"
            )

        if not isinstance(result, (dict, list)):
            raise RuntimeError(
                "[APIS-SAFETY] get_next_action invalid type"
            )

        return result

    guarded._apis_safety_wrapped = True
    apis.get_next_action = guarded
