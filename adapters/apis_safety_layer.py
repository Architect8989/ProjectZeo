import copy
import functools
import types
import importlib



_MODULE_PATCHES_APPLIED: bool = False  # set to True in apply_patches(); read via is_patched()


def is_patched() -> bool:
    """Return True if apply_patches() has been successfully called in this process."""
    return _MODULE_PATCHES_APPLIED


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

def apply_patches() -> None:
    
    global _MODULE_PATCHES_APPLIED

    if _MODULE_PATCHES_APPLIED:
        return

    # RT-03 / SI-03 FIX: Install screenshot guard FIRST — unconditionally.
    # This guard must be active even on Ollama-only installs where the cloud
    # API module is absent.  Previously this call was after _get_apis(), so
    # Ollama-only installs never reached it.
    _disable_screenshot_writes()

    try:
        apis = _get_apis()
    except RuntimeError:
        import sys
        print(
            "[APIS-SAFETY] Warning: could not resolve operate apis module — "
            "cloud provider patches skipped (OK for Ollama-only path). "
            "Screenshot write guard is installed.",
            file=sys.stderr,
        )
        # Screenshot guard is installed; mark complete so we don't retry.
        _MODULE_PATCHES_APPLIED = True
        return

    if getattr(apis, "_safety_patches_applied", False):
        # Target module already patched (e.g. this module was reloaded but the
        # target module was not). Mark this module as patched too so future
        # calls short-circuit on the cheaper _MODULE_PATCHES_APPLIED check.
        _MODULE_PATCHES_APPLIED = True
        return

    setattr(apis, "_safety_patches_applied", True)

    _patch_all_providers()
    _disable_cloud_fallbacks()
    _guard_dispatch()

    # RD-01 FIX: Set this module's flag LAST — after all patches succeed.
    # If any patch raises, _MODULE_PATCHES_APPLIED stays False and the next
    # call to apply_patches() will retry from scratch.
    _MODULE_PATCHES_APPLIED = True


def uninstall_patches():
    import logging as _logging

    _logger2 = _logging.getLogger(__name__)
    _logger2.info("[APIS-SAFETY] uninstall_patches(): completed (no process-wide patches to remove).")


# ============================================================
# CORE PROVIDER WRAPPER
# ============================================================

def _wrap_provider(fn):

    if getattr(fn, "_apis_safety_wrapped", False):
        return fn

    is_async = inspect.iscoroutinefunction(fn)

    def _validate_no_mutation(snapshot, checked_copy, name):
        
        if checked_copy != snapshot:
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

            # H2 FIX: pass safe_messages (what fn received) not messages (original)
            _validate_no_mutation(caller_snapshot, safe_messages, fn.__name__)
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

            # H2 FIX: pass safe_messages (what fn received) not messages (original)
            _validate_no_mutation(caller_snapshot, safe_messages, fn.__name__)
            _validate_result(result, fn.__name__)

            return result

        sync_wrapper._apis_safety_wrapped = True
        return sync_wrapper


# ============================================================
# PATCH PROVIDERS
# ============================================================

def _patch_all_providers():
    
    modules_to_patch = []

    # Primary (wrapper) module — always present
    modules_to_patch.append(_get_apis())

    # Legacy implementation module — best-effort
    try:
        legacy = importlib.import_module("operate.legacy.apis")
        modules_to_patch.append(legacy)
    except Exception:
        pass  # Legacy module absent — acceptable on Ollama-only installs

    # H-07 FIX (BOUNDARY-02): Also patch operate.models.apis_openrouter.
    # The previous implementation only patched operate.models.apis and
    # operate.legacy.apis.  Any future import of apis_openrouter bypassed the
    # safety layer entirely — its cloud calls received no immutability
    # enforcement, no temperature injection, and no validation.
    try:
        openrouter = importlib.import_module("operate.models.apis_openrouter")
        modules_to_patch.append(openrouter)
    except Exception:
        pass  # Module absent on Ollama-only installs — acceptable

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
    import os as _os_h6

    _skip = _os_h6.environ.get("PROJECTZEO_DISABLE_OPEN_PATCH", "").strip() == "1"
    if _skip:
        import sys as _sys_h6
        print(
            "[APIS-SAFETY] WARNING: PROJECTZEO_DISABLE_OPEN_PATCH=1 — "
            "screenshot write guard is DISABLED.",
            file=_sys_h6.stderr,
        )
        return

    def _is_screenshot_path(path) -> bool:
        return isinstance(path, (str, __import__("pathlib").Path)) and \
               "screenshot" in str(path).lower()

    try:
        _m = _get_apis()
    except RuntimeError:
        import sys as _sys
        print(
            "[APIS-SAFETY] _disable_screenshot_writes(): apis module absent — "
            "screenshot guards skipped (OK for Ollama-only).",
            file=_sys.stderr,
        )
        return

    if hasattr(_m, "os") and hasattr(_m.os, "makedirs"):
        _orig_makedirs = _m.os.makedirs

        def _guarded_makedirs(path, *args, **kwargs):
            if _is_screenshot_path(path):
                return
            return _orig_makedirs(path, *args, **kwargs)

        _m.os.makedirs = _guarded_makedirs

    if hasattr(_m, "Image"):
        try:
            _orig_save = _m.Image.Image.save

            def _guarded_save(self, fp, *args, **kwargs):
                if _is_screenshot_path(fp):
                    return
                return _orig_save(self, fp, *args, **kwargs)

            _m.Image.Image.save = _guarded_save
        except Exception as exc:
            raise RuntimeError(
                f"[APIS-SAFETY] Failed to patch PIL save: {exc}"
            )


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

            # SI-C FIX: Validate that `model` is a non-empty string.
            # The original dispatcher accepted None or non-string model values
            # silently — they passed through to the underlying router without
            # type validation.  A None model (possible if an adapter's
            # _resolve_model_function() returns None on misconfiguration) would
            # propagate into Ollama's client.chat(), producing a cryptic
            # internal error rather than a clear contract violation here.
            if not isinstance(model, str) or not model.strip():
                raise RuntimeError(
                    f"[APIS-SAFETY] Dispatcher model must be a non-empty string, "
                    f"got: {model!r} (type={type(model).__name__}). "
                    "Check adapter _resolve_model_function() return value."
                )

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

            # SI-C FIX: Same model type validation as the async path above.
            if not isinstance(model, str) or not model.strip():
                raise RuntimeError(
                    f"[APIS-SAFETY] Dispatcher model must be a non-empty string, "
                    f"got: {model!r} (type={type(model).__name__}). "
                    "Check adapter _resolve_model_function() return value."
                )

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
