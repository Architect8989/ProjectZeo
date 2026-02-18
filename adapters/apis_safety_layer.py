# adapters/apis_safety_layer.py

"""
External hardening layer for operate.models.apis

Hard guarantees:
- No caller message mutation
- No silent None
- No cloud fallback chains
- Deterministic LLM temperature enforcement
- Screenshot side-effects disabled (ALL write vectors)
- Strict return validation
- Idempotent wrapping
"""

import copy
import inspect
import functools
import types
import importlib
import builtins
import pathlib

_PATCHED = False


# ============================================================
# RESOLVE APIS MODULE
# ============================================================

def _resolve_apis_module():

    candidates = [
        "operate.models.apis",
        "operate.legacy.apis",
    ]

    for path in candidates:
        try:
            return importlib.import_module(path)
        except Exception:
            continue

    raise RuntimeError(
        "[APIS-SAFETY] Unable to resolve operate apis module"
    )


apis = _resolve_apis_module()


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
        options = dict(kwargs.get("options") or {})
        options["temperature"] = 0
        kwargs["options"] = options
        return kwargs

    if is_async:

        @functools.wraps(fn)
        async def async_wrapper(messages, *args, **kwargs):

            if not isinstance(messages, list):
                raise RuntimeError("[APIS-SAFETY] messages must be list")

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
                raise RuntimeError("[APIS-SAFETY] messages must be list")

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
# PATCH PROVIDERS
# ============================================================

def _patch_all_providers():

    for name in dir(apis):

        if not name.startswith("call_"):
            continue

        attr = getattr(apis, name)

        if isinstance(attr, types.FunctionType):
            setattr(apis, name, _wrap_provider(attr))


# ============================================================
# DISABLE CLOUD FALLBACK
# ============================================================

def _disable_cloud_fallbacks():

    if hasattr(apis, "gpt_4_fallback"):

        def hard_fail_fallback(*args, **kwargs):
            raise RuntimeError("[APIS-SAFETY] Cloud fallback disabled")

        apis.gpt_4_fallback = hard_fail_fallback


# ============================================================
# DISABLE SCREENSHOT SIDE EFFECTS (FULL HARDENING)
# ============================================================

def _disable_screenshot_writes():

    def _is_screenshot_path(path):
        if not isinstance(path, (str, pathlib.Path)):
            return False
        return "screenshot" in str(path).lower()

    # ---- Guard os.makedirs ----
    if hasattr(apis, "os") and hasattr(apis.os, "makedirs"):

        original_makedirs = apis.os.makedirs

        def guarded_makedirs(path, *args, **kwargs):
            if _is_screenshot_path(path):
                return
            return original_makedirs(path, *args, **kwargs)

        apis.os.makedirs = guarded_makedirs

    # ---- Guard PIL Image.save ----
    if hasattr(apis, "Image"):
        try:
            original_save = apis.Image.Image.save

            def guarded_save(self, fp, *args, **kwargs):
                if _is_screenshot_path(fp):
                    return
                return original_save(self, fp, *args, **kwargs)

            apis.Image.Image.save = guarded_save
        except Exception as e:
            raise RuntimeError(
                f"[APIS-SAFETY] Failed to patch PIL save: {e}"
            )

    # ---- Guard builtins.open ----
    original_open = builtins.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if "w" in mode or "a" in mode or "x" in mode:
            if _is_screenshot_path(file):
                raise RuntimeError(
                    "[APIS-SAFETY] Screenshot file write blocked"
                )
        return original_open(file, mode, *args, **kwargs)

    builtins.open = guarded_open

    # ---- Guard pathlib writes ----
    original_write_bytes = pathlib.Path.write_bytes
    original_write_text = pathlib.Path.write_text

    def guarded_write_bytes(self, data, *args, **kwargs):
        if _is_screenshot_path(self):
            raise RuntimeError(
                "[APIS-SAFETY] Screenshot file write blocked"
            )
        return original_write_bytes(self, data, *args, **kwargs)

    def guarded_write_text(self, data, *args, **kwargs):
        if _is_screenshot_path(self):
            raise RuntimeError(
                "[APIS-SAFETY] Screenshot file write blocked"
            )
        return original_write_text(self, data, *args, **kwargs)

    pathlib.Path.write_bytes = guarded_write_bytes
    pathlib.Path.write_text = guarded_write_text


# ============================================================
# GUARD DISPATCH
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
