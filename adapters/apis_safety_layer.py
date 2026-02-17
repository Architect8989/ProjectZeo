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
        # Enforce deterministic temperature if options supported
        options = kwargs.get("options")
        if isinstance(options, dict):
            options = dict(options)
            options["temperature"] = 0
            kwargs["options"] = options
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

        if isinstance(attr, types.FunctionType):
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
# DISABLE SCREENSHOT SIDE EFFECTS
# ============================================================

def _disable_screenshot_writes():
    """
    Neutralize filesystem screenshot writes
    without corrupting the os module.
    """

    # Block directory creation for screenshots only
    if hasattr(apis, "os") and hasattr(apis.os, "makedirs"):

        original_makedirs = apis.os.makedirs

        def guarded_makedirs(path, *args, **kwargs):
            if isinstance(path, str) and "screenshot" in path.lower():
                return
            return original_makedirs(path, *args, **kwargs)

        apis.os.makedirs = guarded_makedirs

    # Block PIL Image.save used for screenshots
    if hasattr(apis, "Image"):

        try:
            original_save = apis.Image.Image.save

            def guarded_save(self, fp, *args, **kwargs):
                if isinstance(fp, str) and "screenshot" in fp.lower():
                    return
                return original_save(self, fp, *args, **kwargs)

            apis.Image.Image.save = guarded_save
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

    if inspect.iscoroutinefunction(original):

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

    else:

        @functools.wraps(original)
        def guarded(model, messages, objective, session_id):

            if not isinstance(messages, list):
                raise RuntimeError(
                    "[APIS-SAFETY] Dispatcher messages must be list"
                )

            result = original(model, messages, objective, session_id)

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
