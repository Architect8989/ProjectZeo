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


# PATCH (audit): Removed module-level `apis = _resolve_apis_module()`.
# The original code ran _resolve_apis_module() at import time which caused the
# entire process to crash if operate.legacy.apis had uninstallable dependencies
# (ultralytics, easyocr).  Resolution is now deferred to apply_patches() so
# the process can start even on a raw OS where those packages are absent.
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
    global _PATCHED
    if _PATCHED:
        return

    _PATCHED = True

    try:
        # Eagerly resolve here so errors surface at patch time, not usage time.
        _get_apis()
    except RuntimeError:
        # Neither operate.models.apis nor operate.legacy.apis is importable
        # (e.g. raw OS without ultralytics/easyocr).  The safety patches that
        # wrap those modules are therefore not applicable.  Log a warning and
        # return — the Ollama path does not use these modules at all, so the
        # process remains functional.
        import sys
        print(
            "[APIS-SAFETY] Warning: could not resolve operate apis module — "
            "provider patches skipped (OK for Ollama-only path)",
            file=sys.stderr,
        )
        return

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

    for name in dir(_get_apis()):

        if not name.startswith("call_"):
            continue

        _m = _get_apis()
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
# DISABLE SCREENSHOT SIDE EFFECTS (FULL HARDENING)
# ============================================================

def _disable_screenshot_writes():

    def _is_screenshot_path(path):
        if not isinstance(path, (str, pathlib.Path)):
            return False
        return "screenshot" in str(path).lower()

    # GAP-2 FIX: _m was referenced throughout this function but never assigned
    # in this scope. _patch_all_providers() assigned _m as a local variable
    # inside its own loop — it never propagated here. On cloud-model paths,
    # this caused NameError("name '_m' is not defined") at runtime.
    # Fix: resolve the module once at function entry via _get_apis().
    _m = _get_apis()

    # ---- Guard os.makedirs ----
    if hasattr(_m, "os") and hasattr(_m.os, "makedirs"):

        original_makedirs = _m.os.makedirs

        def guarded_makedirs(path, *args, **kwargs):
            if _is_screenshot_path(path):
                return
            return original_makedirs(path, *args, **kwargs)

        _m.os.makedirs = guarded_makedirs

    # ---- Guard PIL Image.save ----
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
    # HRD-07: Use _real_open to reference the unpatched built-in open so
    # that guarded_open can call through to it without recursion.
    import builtins as _builtins_mod
    _real_open = _builtins_mod.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if "w" in mode or "a" in mode or "x" in mode:
            if _is_screenshot_path(file):
                raise RuntimeError(
                    "[APIS-SAFETY] Screenshot file write blocked"
                )
        return _real_open(file, mode, *args, **kwargs)

    # HRD-07 FIX: The original implementation patched builtins.open globally
    # for the entire process lifetime. This is an uncontrolled global side
    # effect that blocks any third-party library writing to a path containing
    # "screenshot" (e.g. ~/screenshots/backup.png).
    #
    # Revised approach: inject guarded_open into the apis module's own
    # namespace. Python name resolution checks the module's global namespace
    # before builtins, so direct calls to `open()` within the apis module
    # will use guarded_open while all other modules continue using the real
    # builtins.open. This scopes the protection to exactly the risk surface.
    _m.open = guarded_open  # module-local open shadows builtins within _m

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

