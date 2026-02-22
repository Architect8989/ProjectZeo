import copy
import inspect
import functools
import types
import importlib
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


_apis_module = None


def _get_apis() -> object:
    """Return the cached apis module, resolving it on first call."""
    global _apis_module
    if _apis_module is None:
        _apis_module = _resolve_apis_module()
    return _apis_module


# ============================================================
# PUBLIC ENTRY
# ============================================================

def apply_patches():
    # DETERMINISM FIX (double-wrap): Store the applied flag as an attribute on
    # the target apis module rather than as a module-level bool on this safety
    # layer. If this module is reloaded (importlib.reload) the local _PATCHED
    # bool resets to False and patches would be applied again, double-wrapping
    # already-wrapped functions.
    #
    # Storing the flag on the target module survives reloads of this module
    # because the apis module object identity is preserved across reloads of
    # the safety layer. Even if the apis module itself is reloaded, _wrap_provider
    # guards against double-wrapping via the _apis_safety_wrapped attribute on
    # each function object.
    try:
        apis = _get_apis()
    except RuntimeError:
        import sys
        print(
            "[APIS-SAFETY] Warning: could not resolve operate apis module — "
            "provider patches skipped (OK for Ollama-only path)",
            file=sys.stderr,
        )
        return

    if getattr(apis, "_safety_patches_applied", False):
        return
    setattr(apis, "_safety_patches_applied", True)

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
    """
    HARD-5: Patch both the wrapper module (operate.models.apis) AND the
    legacy implementation module (operate.legacy.apis).

    The previous implementation only patched the thin wrapper layer. Any code
    that imports from operate.legacy.apis directly (bypassing the wrapper)
    received no safety-layer enforcement. Cloud API functions in the legacy
    module were unpatched — immutability enforcement, temperature injection,
    and validation did not apply to them.

    Fix: apply _wrap_provider() to both modules' call_* functions.
    """
    modules_to_patch = []

    # Primary (wrapper) module — always present
    modules_to_patch.append(_get_apis())

    # Legacy implementation module — best-effort
    try:
        legacy = importlib.import_module("operate.legacy.apis")
        modules_to_patch.append(legacy)
    except Exception:
        pass  # Legacy module absent — acceptable on Ollama-only installs

    for _m in modules_to_patch:
        for name in dir(_m):
            if not name.startswith("call_"):
                continue
            attr = getattr(_m, name)
            if isinstance(attr, types.FunctionType):
                setattr(_m, name, _wrap_provider(attr))


# ============================================================
# DISABLE CLOUD FALLBACK
# ============================================================

def _disable_cloud_fallbacks():

    if hasattr(_get_apis(), "gpt_4_fallback"):

        def hard_fail_fallback(*args, **kwargs):
            raise RuntimeError("[APIS-SAFETY] Cloud fallback disabled")

        _get_apis().gpt_4_fallback = hard_fail_fallback


# ============================================================
# DISABLE SCREENSHOT SIDE EFFECTS
# ============================================================

def _disable_screenshot_writes():

    def _is_screenshot_path(path):
        if not isinstance(path, (str, pathlib.Path)):
            return False
        return "screenshot" in str(path).lower()

    _m = _get_apis()

    # ---- Guard os.makedirs within the apis module ----
    if hasattr(_m, "os") and hasattr(_m.os, "makedirs"):

        original_makedirs = _m.os.makedirs

        def guarded_makedirs(path, *args, **kwargs):
            if _is_screenshot_path(path):
                return
            return original_makedirs(path, *args, **kwargs)

        _m.os.makedirs = guarded_makedirs

    # ---- Guard PIL Image.save within the apis module ----
    if hasattr(_get_apis(), "Image"):
        try:
            original_save = _m.Image.Image.save

            def guarded_save(self, fp, *args, **kwargs):
                if _is_screenshot_path(fp):
                    return
                return original_save(self, fp, *args, **kwargs)

            _m.Image.Image.save = guarded_save
        except Exception as e:
            raise RuntimeError(
                f"[APIS-SAFETY] Failed to patch PIL save: {e}"
            )

    # ---- Guard builtins.open within the apis module namespace only ----
    import builtins as _builtins_mod
    _real_open = _builtins_mod.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if "w" in mode or "a" in mode or "x" in mode:
            if _is_screenshot_path(file):
                raise RuntimeError(
                    "[APIS-SAFETY] Screenshot file write blocked"
                )
        return _real_open(file, mode, *args, **kwargs)

    # SI-05 FIX: Do NOT set _m.open = guarded_open on the wrapper module.
    #
    # operate.models.apis (the thin wrapper module) contains NO direct open()
    # calls — it delegates all I/O to operate.legacy.apis via _legacy(). Setting
    # _m.open on the wrapper module is structurally inert: no code in that module
    # namespace calls open(), so the guard never fires. The previous dual-patch
    # approach created a false sense of security while the actual enforcement
    # relied entirely on the legacy-module patch below.
    #
    # Corrected policy: apply guarded_open ONLY to operate.legacy.apis, which is
    # the module that actually calls open() for screenshot file reads/writes.
    # The wrapper module namespace is NOT patched — no benefit, no confusion.
    try:
        legacy = importlib.import_module("operate.legacy.apis")
        if not getattr(legacy, "_open_safety_guarded", False):
            legacy.open = guarded_open
            setattr(legacy, "_open_safety_guarded", True)
    except Exception:
        pass  # Legacy module absent — acceptable on Ollama-only installs


# ============================================================
# GUARD DISPATCH
# ============================================================

def _guard_dispatch():

    _m2 = _get_apis()
    if not hasattr(_m2, "get_next_action"):
        return

    original = _m2.get_next_action

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
    _m2.get_next_action = guarded
