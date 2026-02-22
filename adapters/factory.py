from __future__ import annotations

import importlib
import threading
import re
from typing import Dict, Any, List, Type

from operate.exceptions import ModelNotRecognizedException
from adapters.apis_safety_layer import apply_patches


# =========================================================
# LOCAL MODEL REGISTRY
# Maps base model name → fully-qualified adapter class path.
# These are served by dedicated local adapters (Ollama, etc.)
# =========================================================
_LOCAL_REGISTRY: Dict[str, str] = {
    "qwen2.5-vl": "adapters.qwen_ollama_adapter.QwenOllamaAdapter",
    # Extension point — add new local models here:
    # "llama3.2-vision": "adapters.llama_ollama_adapter.LlamaOllamaAdapter",
    # "llava":           "adapters.llava_ollama_adapter.LlavaOllamaAdapter",
}

# =========================================================
# CLOUD MODEL REGISTRY
# Base model names routed through PureLLMWrapper.
# PureLLMWrapper resolves these to the correct cloud API function.
# =========================================================
_CLOUD_REGISTRY = {
    "gpt-4",
    "gpt-4o",
    "gpt-4-with-som",
    "gpt-4-with-ocr",
    "gpt-4o-with-ocr",
    "gpt-4.1-with-ocr",
    "gpt-4o-labeled",
    "o1-with-ocr",
    "claude-3",
    "claude-3-opus",
    "claude-3-sonnet",
    "gemini-pro-vision",
    "qwen-vl",       # Qwen cloud (DashScope) — not local Ollama
    "llava",          # LLaVA via legacy ollama path in PureLLMWrapper
    # Extension point — add new cloud model base names here.
}

# =========================================================
# INTERNALS
# =========================================================

_PATCHES_APPLIED = False
_PATCH_LOCK = threading.Lock()

# Allows: letters, digits, dots, hyphens, underscores, colons, forward-slash.
# Slash is needed for 'ollama/qwen2.5-vl' format.
_MODEL_PATTERN = re.compile(r"^[a-zA-Z0-9.\-_:/]+$")

# §R3: module-level adapter cache — one instance per model name.
# FIX-05 (SI-04): Per-model construction locks prevent the double-construction
# race where two concurrent threads both find a cache miss, both construct an
# adapter (duplicating warmup side effects: Ollama client creation, thread pool
# allocation), and the second writer silently overwrites the first — leaking
# the first adapter's ThreadPoolExecutor and any other resources it holds.
#
# Pattern: _ADAPTER_CACHE stores finished instances; _ADAPTER_BUILD_LOCKS
# stores per-model RLock objects.  A thread acquiring a model lock, finding
# no cached instance, builds one, stores it, and releases.  A second concurrent
# thread blocks on the lock and finds the cached instance on release.
_ADAPTER_CACHE: Dict[str, Any] = {}
_ADAPTER_CACHE_LOCK = threading.Lock()           # guards _ADAPTER_CACHE reads/writes
_ADAPTER_BUILD_LOCKS: Dict[str, threading.Lock] = {}  # per-model construction mutex
_BUILD_LOCKS_LOCK = threading.Lock()             # guards _ADAPTER_BUILD_LOCKS itself

# HAR-08 / FIX RB-2: Freeze the cloud-isolation enforcement decision at module
# import time.
#
# CORRECTNESS INVARIANT:
#   This freeze is ONLY correct if os.environ["OLLAMA_ONLY"] is set or cleared
#   BEFORE this module is imported. run.py is responsible for this — it parses
#   --allow-cloud from sys.argv at the very top of its file (before any import
#   statements) and mutates os.environ unconditionally.
#
#   If this module is imported before run.py sets the env var — e.g. by an
#   alternative entry point, a test harness, or any import at the top of
#   another module that transitively imports this one — the freeze will capture
#   the default "1" (cloud-denied), which is the SAFE default.  Tests that need
#   cloud routing MUST set os.environ["OLLAMA_ONLY"] = "0" before importing.
#
# CORRECTNESS DEPENDENCY:
#   run.py ensures os.environ["OLLAMA_ONLY"] is set/cleared BEFORE this line
#   runs. Any alternative entry point must replicate that setup.
#
# SECURITY INVARIANT:
#   After this line, no runtime mutation of os.environ["OLLAMA_ONLY"] can
#   bypass cloud isolation. The frozen bool is immutable for the process
#   lifetime.
import os as _os_module
_raw_ollama_only = _os_module.environ.get("OLLAMA_ONLY", "1").strip().lower()
# OLLAMA_ONLY=1/true/yes → cloud denied (Ollama-only mode, safe default)
# OLLAMA_ONLY=0/false/no → cloud permitted (--allow-cloud path)
_OLLAMA_ONLY_ENFORCEMENT_FROZEN: bool = _raw_ollama_only not in ("1", "true", "yes")
del _raw_ollama_only  # remove intermediate from module namespace


def _get_model_build_lock(model_name: str) -> threading.Lock:
    """Return the per-model build lock, creating it if it doesn't exist."""
    with _BUILD_LOCKS_LOCK:
        if model_name not in _ADAPTER_BUILD_LOCKS:
            _ADAPTER_BUILD_LOCKS[model_name] = threading.Lock()
        return _ADAPTER_BUILD_LOCKS[model_name]


def _ensure_patches() -> None:
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return
    with _PATCH_LOCK:
        if not _PATCHES_APPLIED:
            apply_patches()
            _PATCHES_APPLIED = True


def _import_class(path: str) -> Type:
    """Dynamically import a class from a dotted module path."""
    try:
        module_path, class_name = path.rsplit(".", 1)
    except ValueError:
        raise RuntimeError(f"Invalid adapter path format: {path!r}")

    module = importlib.import_module(module_path)

    try:
        return getattr(module, class_name)
    except AttributeError:
        raise RuntimeError(
            f"Adapter class '{class_name}' not found in '{module_path}'"
        )


def _validate_model_name(model_name: str) -> str:
    if not isinstance(model_name, str) or not model_name.strip():
        raise ModelNotRecognizedException("Model name must be a non-empty string.")

    model_name = model_name.strip()

    if not _MODEL_PATTERN.fullmatch(model_name):
        raise ModelNotRecognizedException(
            f"Invalid model name format: '{model_name}'. "
            "Only letters, digits, dots, hyphens, underscores, colons, "
            "and forward-slashes are permitted."
        )

    return model_name


def _resolve_base_model(model_name: str) -> str:
    """
    Normalise a model name to its registry lookup key.

    Handles two common variant formats:
      1. Provider prefix:  'ollama/qwen2.5-vl'        → 'qwen2.5-vl'
      2. Version tag:      'qwen2.5-vl:7b-instruct'   → 'qwen2.5-vl'
    """
    # Strip optional 'ollama/' provider prefix
    if model_name.startswith("ollama/"):
        model_name = model_name[len("ollama/"):]

    # Strip version tag (everything after the first ':')
    return model_name.split(":", 1)[0]


def _is_cloud_allowed() -> bool:
    """
    FIX RB-2 / H-01: Cloud access is BLOCKED by default.

    Returns the value of _OLLAMA_ONLY_ENFORCEMENT_FROZEN which was captured
    once at module import time. This makes the enforcement boundary immune to
    post-startup os.environ mutations.

    CORRECTNESS DEPENDENCY (FIX RB-2):
        The freeze is only correct if run.py (or the test harness) set
        os.environ["OLLAMA_ONLY"] BEFORE this module was imported.
        See the HAR-08 comment at the module level for the full invariant.

    Returns True if cloud routing is permitted, False if Ollama-only is enforced.
    """
    return _OLLAMA_ONLY_ENFORCEMENT_FROZEN


class AdapterFactory:

    @staticmethod
    def build_llm(model_name: str):
        """
        Build and cache an LLM adapter for the given model name.

        Routing priority:
          1. Cached instance → return immediately (fast path, no lock contention).
          2. Per-model build lock → acquire, re-check cache (double-checked locking).
          3. Local registry match → instantiate registered local adapter.
          4. Cloud registry match + cloud allowed → PureLLMWrapper.
          5. Cloud registry match + cloud DENIED → ModelNotRecognizedException.
          6. Unknown → ModelNotRecognizedException.

        FIX-05 (SI-04): The original code checked the cache under the global
        lock, released it, constructed the adapter (potentially expensive),
        then re-acquired the global lock to write.  Two concurrent threads
        entering simultaneously both saw a cache miss, both constructed adapter
        instances (duplicating Ollama client warmup, ThreadPoolExecutor
        allocation, etc.), and the second write silently overwrote the first —
        leaking the first adapter's executor permanently.

        Fix: per-model build locks ensure only one thread constructs each model.
        The global cache lock is held only for the O(1) dict read/write steps.
        """
        model_name = _validate_model_name(model_name)
        _ensure_patches()

        # Fast path: return cached adapter without acquiring build lock.
        with _ADAPTER_CACHE_LOCK:
            if model_name in _ADAPTER_CACHE:
                return _ADAPTER_CACHE[model_name]

        # Slow path: acquire per-model build lock.
        build_lock = _get_model_build_lock(model_name)
        with build_lock:
            # Double-checked locking: another thread may have completed
            # construction while we waited for the build lock.
            with _ADAPTER_CACHE_LOCK:
                if model_name in _ADAPTER_CACHE:
                    return _ADAPTER_CACHE[model_name]

            base_model = _resolve_base_model(model_name)

            # --- Route 1: Local adapter ---
            local_path = _LOCAL_REGISTRY.get(base_model)
            if local_path is not None:
                AdapterClass = _import_class(local_path)
                instance = AdapterClass(model_name=model_name)
                with _ADAPTER_CACHE_LOCK:
                    _ADAPTER_CACHE[model_name] = instance
                return instance

            # --- Route 2: Cloud adapter via PureLLMWrapper ---
            if base_model in _CLOUD_REGISTRY:
                # FIX RB-2 / H-01: Cloud is BLOCKED by default.
                # _is_cloud_allowed() returns the value frozen at import time.
                # For this to return True, run.py must have cleared OLLAMA_ONLY
                # from os.environ BEFORE this module was imported.
                if not _is_cloud_allowed():
                    raise ModelNotRecognizedException(
                        f"Model '{model_name}' is a cloud model, but OLLAMA_ONLY is "
                        "enforced (default). To enable cloud models, start the system "
                        "with --allow-cloud or set OLLAMA_ONLY=0 in the environment "
                        "BEFORE importing this module.\n"
                        f"  Local models: {sorted(_LOCAL_REGISTRY.keys())}"
                    )

                # Lazy import so Ollama-only boots never touch cloud code
                from adapters.pure_llm_wrapper import PureLLMWrapper  # noqa: PLC0415

                instance = PureLLMWrapper(model_name=base_model)
                with _ADAPTER_CACHE_LOCK:
                    _ADAPTER_CACHE[model_name] = instance
                return instance

            # --- Route 3: Unknown ---
            raise ModelNotRecognizedException(
                f"Model '{model_name}' is not registered.\n"
                f"  Local models:  {sorted(_LOCAL_REGISTRY.keys())}\n"
                f"  Cloud models:  {sorted(_CLOUD_REGISTRY)} (require --allow-cloud)\n"
                "Add the model to the appropriate registry in adapters/factory.py."
            )

    @staticmethod
    async def get_action(
        model_name: str,
        messages: List[Dict[str, Any]],
        objective: str,
        session_id: str,
    ):
        """
        Convenience coroutine: resolve adapter and call get_next_action().
        Reuses the cached adapter — does NOT reconstruct on every call (§R3).
        """
        adapter = AdapterFactory.build_llm(model_name)
        return await adapter.get_next_action(
            messages=messages,
            objective=objective,
            session_id=session_id,
        )


# Module-level aliases for backward compatibility
build_llm = AdapterFactory.build_llm
get_action = AdapterFactory.get_action
