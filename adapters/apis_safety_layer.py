import copy
import inspect
import functools
import types
import importlib
import pathlib



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
    
    import builtins as _builtins_mod
    import logging as _logging

    _uninstall_logger = _logging.getLogger(__name__)

    original = getattr(_builtins_mod, "_original_open_pre_safety_patch", None)

    if original is None:
        # Patches were never installed — nothing to do.
        _uninstall_logger.debug(
            "[APIS-SAFETY] uninstall_patches(): no patch installed, nothing to restore."
        )
        return

    # Restore builtins.open to the true original
    _builtins_mod.open = original

    # Clean up sentinel attributes atomically
    for _attr in ("_original_open_pre_safety_patch", "_safety_open_installed"):
        try:
            delattr(_builtins_mod, _attr)
        except AttributeError:
            pass

    # Post-uninstall verification: confirm builtins.open IS the original.
    # If the identity check fails, a concurrent thread may have re-patched it
    # between our restore and this check — log an explicit ERROR so the
    # condition is surfaced in structured logs rather than silently accepted.
    if _builtins_mod.open is not original:
        _uninstall_logger.error(
            "[APIS-SAFETY] uninstall_patches(): builtins.open identity check FAILED "
            "after restoration. A concurrent patch may have re-installed itself. "
            "Current open=%r, expected original=%r. "
            "Process-wide file I/O may still be intercepted.",
            _builtins_mod.open,
            original,
        )
    else:
        _uninstall_logger.info(
            "[APIS-SAFETY] uninstall_patches(): builtins.open successfully restored "
            "to original=%r. All screenshot-write interception is deactivated.",
            original,
        )


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
    # H6 FIX: Document builtins.open monkey-patch scope and add
    # PROJECTZEO_DISABLE_OPEN_PATCH=1 escape hatch.
    #
    # WHAT THIS PATCH DOES:
    #   Replaces builtins.open (the built-in open() callable available to ALL
    #   Python modules in this process) with a guarded version that raises
    #   RuntimeError when code originating from operate.models.apis or
    #   operate.legacy.apis attempts to open a file whose path contains the
    #   substring "screenshot" in write/append/exclusive-create mode.
    #
    # WHY THIS IS NECESSARY:
    #   The cloud API adapters (gpt4v, claude-vision) historically wrote raw
    #   screenshots to disk as a side-effect of building vision payloads.
    #   Blocking them at builtins.open prevents the writes regardless of how
    #   the adapter module organises its internal code.
    #
    # SCOPE — PROCESS-WIDE (critical to understand):
    #   The guarded open() is installed process-wide.  It intercepts write-mode
    #   opens to any path containing "screenshot" from ANY module.  The guard
    #   inspects the call-stack to confirm the calling frame originates from an
    #   apis module before blocking; all other callers pass through unchanged.
    #   The stack inspection adds a small overhead on every write to a
    #   "screenshot"-named path regardless of the calling module.
    #
    # KNOWN CONFLICT RISK:
    #   Third-party automation libraries (playwright, pyautogui, mss) that
    #   legitimately write screenshot files may be affected if their call stack
    #   happens to include an apis module frame.  In practice, Ollama-only
    #   deployments never import cloud API modules, so no writes are blocked.
    #
    # ESCAPE HATCH (H6):
    #   Set PROJECTZEO_DISABLE_OPEN_PATCH=1 to skip builtins.open replacement.
    #   Use ONLY if the patch causes confirmed third-party library conflicts.
    #
    #   WARNING: Disabling this patch removes screenshot-write isolation for
    #   cloud API adapters.  Do NOT use when OLLAMA_ONLY != 1.
    #
    # UNINSTALL:
    #   uninstall_patches() restores the original builtins.open on shutdown.

    import os as _os_h6

    # H6 ESCAPE HATCH — opt out of builtins.open replacement if explicitly
    # requested by the operator.
    _skip_open_patch: bool = (
        _os_h6.environ.get("PROJECTZEO_DISABLE_OPEN_PATCH", "").strip() == "1"
    )
    if _skip_open_patch:
        import sys as _sys_h6
        print(
            "[APIS-SAFETY] WARNING H6: PROJECTZEO_DISABLE_OPEN_PATCH=1 — "
            "builtins.open screenshot write guard is DISABLED. "
            "Screenshot isolation for cloud API adapters is NOT enforced. "
            "Do NOT use this setting when cloud API access is enabled.",
            file=_sys_h6.stderr,
        )

    # RT-A2 FIX (P1): The builtins.open patch and legacy module patch MUST be
    # installed BEFORE _get_apis() is called.  In the original code the
    # _m = _get_apis() call appeared first; if it raised (Ollama-only install
    # where both apis modules are absent) the entire rest of the function was
    # skipped, leaving builtins.open unpatched — violating the stated
    # screenshot-write isolation guarantee for stripped deployments.
    #
    # Fix: install builtins.open patch unconditionally first, THEN attempt
    # the os.makedirs and PIL patches which require the apis module object.
    # An ImportError from _get_apis() is caught and logged; the builtins.open
    # patch has already been installed at that point and remains active.

    def _is_screenshot_path(path):
        if not isinstance(path, (str, pathlib.Path)):
            return False
        return "screenshot" in str(path).lower()

    # ---- STEP 1: Install builtins.open patch UNCONDITIONALLY ----
    # This must happen regardless of whether the apis module is available.
    import builtins as _builtins_mod
    _real_open = getattr(_builtins_mod, "_original_open_pre_safety_patch", None)
    if _real_open is None:
        _real_open = _builtins_mod.open
        _builtins_mod._original_open_pre_safety_patch = _real_open

    _APIS_MODULE_PATTERNS = (
        "operate.legacy.apis",
        "operate.models.apis",
        "operate/legacy/apis",
        "operate/models/apis",
    )

    def guarded_open(file, mode="r", *args, **kwargs):
        if not ("w" in str(mode) or "a" in str(mode) or "x" in str(mode)):
            return _real_open(file, mode, *args, **kwargs)
        if not _is_screenshot_path(file):
            return _real_open(file, mode, *args, **kwargs)
        # IH-06 FIX: Restrict stack inspection scope.
        # (1) Only run the check when DISPLAY is set — headless backends and
        #     CI environments should not be affected.
        # (2) Exempt known legitimate screenshot writers (pyautogui, mss,
        #     operate.utils.screenshot) from the block.  These write to paths
        #     containing "screenshot" as a normal part of their operation and
        #     must not be caught by the cloud-API guard.
        import os as _os_ih06
        if not _os_ih06.environ.get("DISPLAY"):
            return _real_open(file, mode, *args, **kwargs)
        _LEGITIMATE_SCREENSHOT_PATTERNS = (
            "pyautogui",
            "mss",
            "operate/utils/screenshot",
            "operate.utils.screenshot",
            "operate\\utils\\screenshot",
        )
        try:
            stack = inspect.stack()
            _has_legitimate_caller = any(
                any(
                    pat in (frame_info.filename or "") or
                    pat in (frame_info.frame.f_globals.get("__name__") or "")
                    for pat in _LEGITIMATE_SCREENSHOT_PATTERNS
                )
                for frame_info in stack
            )
            # If a legitimate screenshot module is anywhere in the call stack,
            # pass through without scanning for cloud API patterns.
            if _has_legitimate_caller:
                return _real_open(file, mode, *args, **kwargs)
            for frame_info in stack:
                filename = frame_info.filename or ""
                module = (frame_info.frame.f_globals.get("__name__") or "")
                if any(pat in filename or pat in module for pat in _APIS_MODULE_PATTERNS):
                    raise RuntimeError(
                        "[APIS-SAFETY] Screenshot file write blocked "
                        f"(origin: {module or filename})"
                    )
        except RuntimeError:
            raise
        except Exception:
            pass
        return _real_open(file, mode, *args, **kwargs)

    if not _skip_open_patch and not getattr(_builtins_mod, "_safety_open_installed", False):
        _builtins_mod.open = guarded_open
        _builtins_mod._safety_open_installed = True

    # Also patch the legacy module attribute for backwards compatibility
    try:
        legacy = importlib.import_module("operate.legacy.apis")
        if not _skip_open_patch and not getattr(legacy, "_open_safety_guarded", False):
            legacy.open = guarded_open
            setattr(legacy, "_open_safety_guarded", True)
    except Exception:
        pass  # Legacy module absent — acceptable on Ollama-only installs

    # ---- STEP 2: Apply apis-module-specific patches (require _m) ----
    # These patches are best-effort; absence of the apis module is acceptable
    # on Ollama-only installs.  The builtins.open guard above already covers
    # the critical path for stripped deployments.
    try:
        _m = _get_apis()
    except RuntimeError:
        import sys as _sys
        print(
            "[APIS-SAFETY] _disable_screenshot_writes(): apis module absent — "
            "os.makedirs and PIL Image.save patches skipped (OK for Ollama-only). "
            "builtins.open screenshot guard is installed.",
            file=_sys.stderr,
        )
        return

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
