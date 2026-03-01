import os
import sys

def _early_parse_args() -> bool:
    """
    Parse --allow-cloud from sys.argv BEFORE any imports execute.

    Returns True if --allow-cloud is present, False otherwise.

    This is intentionally minimal — full argument validation happens in
    _parse_args() after all imports are complete.
    """
    return "--allow-cloud" in sys.argv


# Resolve the operator's cloud intent at the earliest possible moment.
_allow_cloud_early = _early_parse_args()

if not _allow_cloud_early:
    # Default: Ollama-only. Set explicitly (don't rely on default in factory.py)
    # so the freeze always sees "1" when --allow-cloud is absent.
    os.environ["OLLAMA_ONLY"] = "1"
else:
    # Explicit --allow-cloud: set OLLAMA_ONLY="0" so factory.py freeze
    # sees a falsy value and permits cloud routing.
    #
    # RB-NEW-02 FIX: The previous code used os.environ.pop("OLLAMA_ONLY", None).
    # factory.py reads: os.environ.get("OLLAMA_ONLY", "1").strip().lower()
    # When the key is absent, get() returns the default "1" (cloud-blocked),
    # making --allow-cloud permanently non-functional (cloud always blocked).
    #
    # Fix: set "0" explicitly so the freeze captures the correct value and
    # _OLLAMA_ONLY_ENFORCEMENT_FROZEN = True (cloud permitted).
    os.environ["OLLAMA_ONLY"] = "0"
    # Warn early so operators see this even if something crashes during import.
    print(
        "[run.py] WARNING: --allow-cloud is set. Cloud API routing is ENABLED. "
        "Ensure API keys are intentionally configured.",
        file=sys.stderr,
    )

# ===========================================================================
# Imports — factory.py freeze now captures the correct OLLAMA_ONLY value.
# ===========================================================================

import asyncio
import threading
from typing import Any

from adapters.factory import build_llm
from main import main
from config.timeouts import LLM_THREAD_TIMEOUT_SECONDS


# ---------------------------------------------------------------------------
# CLI ARGUMENT PARSING (full)
# ---------------------------------------------------------------------------

def _parse_args():
    """
    Full argument parser. Called after imports.

    Recognised flags:
        --allow-cloud   Permit cloud model routing (processed pre-import above).
        --interactive   BUG-10 FIX: Print mode transitions to stdout and accept
                        intents directly from stdin. Sets PROJECTZEO_INTERACTIVE=1.
        --status        BUG-10 FIX: Print current system status and exit immediately.

    Positional argument (required unless --status):
        model_name      First non-flag argument is treated as the model name.
                        Can also be supplied via LLM_MODEL env var.

    Returns (model_name, allow_cloud, interactive, status_only)
    """
    args = sys.argv[1:]
    allow_cloud = "--allow-cloud" in args
    interactive = "--interactive" in args
    status_only = "--status" in args
    positional = [a for a in args if not a.startswith("--")]

    model: str | None = None
    if positional:
        model = positional[0].strip() or None

    if not model:
        model = os.getenv("LLM_MODEL", "").strip() or None

    if not status_only and not model:
        raise RuntimeError(
            "No model specified. "
            "Pass model as CLI argument or set LLM_MODEL environment variable.\n"
            "Example: python run.py qwen2.5-vl:7b-instruct\n"
            "         LLM_MODEL=qwen2.5-vl:7b-instruct python run.py\n"
            "         python run.py --status  (check current system status)"
        )

    return model, allow_cloud, interactive, status_only


def _print_status() -> None:
    """
    BUG-10 FIX: Read temp/ sidecar files and print human-readable status.
    Called when --status flag is present.
    """
    import json as _json
    import pathlib as _pathlib
    import datetime as _dt

    _root = _pathlib.Path(__file__).resolve().parent
    _temp = _root / "temp"

    print("ProjectZeo — System Status")
    print("=" * 42)

    _result_path = _temp / "task_result.json"
    if _result_path.exists():
        try:
            with open(_result_path) as f:
                r = _json.load(f)
            status = "SUCCEEDED" if r.get("success") else "FAILED"
            print(f"Last task   : {status}")
            print(f"  Intent    : {r.get('intent', '(unknown)')[:72]}")
            print(f"  Completed : {r.get('completed_at', '(unknown)')}")
            if r.get("error"):
                print(f"  Error     : {r['error'][:72]}")
            if r.get("steps_completed") is not None:
                print(f"  Steps     : {r['steps_completed']}")
        except Exception as e:
            print(f"Last task   : (unreadable — {e})")
    else:
        print("Last task   : No completed task on record")

    _success_path = _temp / "arm_success.json"
    _failure_path = _temp / "arm_failure.json"
    if _success_path.exists():
        try:
            with open(_success_path) as f:
                s = _json.load(f)
            ts = _dt.datetime.fromtimestamp(s.get("timestamp", 0)).strftime("%Y-%m-%d %H:%M:%S")
            print(f"Last arm    : SUCCESS at {ts}")
        except Exception:
            print("Last arm    : SUCCESS (timestamp unreadable)")
    elif _failure_path.exists():
        try:
            with open(_failure_path) as f:
                s = _json.load(f)
            print(f"Last arm    : FAILED — {s.get('reason', '?')}")
        except Exception:
            print("Last arm    : FAILED (details unreadable)")
    else:
        print("Last arm    : No arm event recorded")

    _intent_path = _root / "arm_system.intent"
    if _intent_path.exists():
        try:
            content = _intent_path.read_text(encoding="utf-8").strip()[:72]
            print(f"Pending     : {content!r}")
        except Exception:
            print("Pending     : (unreadable)")
    else:
        print("Pending     : None")

    print("=" * 42)
    print("Tip: write intent to arm_system.intent  OR  run with --interactive")


# ---------------------------------------------------------------------------
# THREAD-SAFE COROUTINE EXECUTOR
# ---------------------------------------------------------------------------

def _run_coroutine_threadsafe(coro) -> Any:
    """
    Execute coroutine in a fresh event loop inside a dedicated thread.
    Enforces hard timeout derived from shared config.
    """
    result_container: dict = {}
    error_container: dict = {}

    def _thread_target():
        try:
            result_container["result"] = asyncio.run(coro)
        except Exception as e:
            error_container["error"] = e

    t = threading.Thread(target=_thread_target, daemon=True)
    t.start()
    t.join(timeout=LLM_THREAD_TIMEOUT_SECONDS)

    if t.is_alive():
        raise RuntimeError(
            f"LLM thread timed out after {LLM_THREAD_TIMEOUT_SECONDS} seconds"
        )

    if "error" in error_container:
        raise error_container["error"]

    return result_container.get("result")


# ---------------------------------------------------------------------------
# LLM CALLABLE FACTORY
# ---------------------------------------------------------------------------

def _make_llm_callable(adapter):
    """
    Wrap async adapter into a safe synchronous callable
    compatible with ExecutionPlanner.

    PATCH §R1: asyncio.get_running_loop() catch only re-routes on
    the specific RuntimeError from no running loop. Genuine errors propagate.
    """

    if not hasattr(adapter, "get_next_action"):
        raise RuntimeError("Adapter missing get_next_action()")

    def _call(messages, objective=None, session_id=None):

        async def _invoke():
            return await adapter.get_next_action(
                messages=messages,
                objective=objective,
                session_id=session_id,
            )

        try:
            _inside_loop = False
            try:
                asyncio.get_running_loop()
                _inside_loop = True
            except RuntimeError:
                _inside_loop = False

            if _inside_loop:
                result = _run_coroutine_threadsafe(_invoke())
            else:
                result = asyncio.run(_invoke())

        except Exception as e:
            raise RuntimeError(f"LLM adapter invocation failed: {e}") from e

        # PATCH §R2: explicit contract enforcement
        if isinstance(result, tuple) and len(result) == 2:
            ops, err = result
            if err:
                raise RuntimeError(f"LLM adapter error: {err}")
        elif isinstance(result, list):
            ops = result
        else:
            raise RuntimeError(
                f"LLM adapter returned unexpected type: {type(result)!r}. "
                "Expected (List, Exception|None) tuple or List."
            )

        if not isinstance(ops, list):
            raise RuntimeError("LLM adapter returned invalid operation list")

        return ops

    return _call


# ---------------------------------------------------------------------------
# STARTUP DEPENDENCY VALIDATOR
# ---------------------------------------------------------------------------

def _validate_runtime_dependencies(model_name: str) -> None:
    """
    Consolidated, authoritative pre-flight dependency check.

    Called once in run.py before any project code is imported.
    Exits with code 1 on FATAL issues. Prints WARNINGs and continues.

    Checks (in order):
      1.  Required directories (temp/, memory/, logs/)
      2.  psutil                     (process watchdog — FATAL)
      3.  pyautogui + display access (all UI actions — FATAL)
      4.  pyautogui display check    (X11/Wayland reachable — FATAL)
      5.  xdotool on Linux           (snapshot/window — FATAL)
      6.  wmctrl on Linux            (window activation — WARNING)
      7.  DISPLAY / WAYLAND_DISPLAY  (display server — FATAL)
      8.  Wayland tooling            (ydotool/AT-SPI — WARNING)
      9.  pyyaml                     (policy.yaml load — WARNING)
      10. playwright + chromium      (browser DOM automation — WARNING)
      11. EasyOCR                    (label-click resolution — WARNING)
      12. Ollama daemon reachable    (LLM — FATAL)
      13. Requested model pulled     (LLM — FATAL)
      14. Text model availability    (planner performance — WARNING)
    """
    import shutil as _shutil
    import platform as _platform

    errors = []
    warnings = []

    # ------------------------------------------------------------------
    # 0. Create required directories — silent, never fatal
    # ------------------------------------------------------------------
    _project_root = os.path.dirname(os.path.abspath(__file__))
    for _req_dir in ("temp", "memory/snapshots", "memory/playbooks", "logs"):
        _abs_dir = os.path.join(_project_root, _req_dir)
        try:
            os.makedirs(_abs_dir, exist_ok=True)
        except OSError as _mkdir_err:
            warnings.append(
                f"  [WARNING] Could not create required directory {_abs_dir!r}: {_mkdir_err}. "
                "Snapshot persistence and transition logging will be disabled for this session."
            )

    # ------------------------------------------------------------------
    # 1. psutil — required by RuntimeWatchdog
    # ------------------------------------------------------------------
    try:
        import psutil as _  # noqa: F401
    except ImportError:
        errors.append(
            "  [MISSING] psutil is not installed.\n"
            "    Fix: pip install psutil"
        )

    # ------------------------------------------------------------------
    # 2. pyautogui — required for ALL UI actions (click, type, press, scroll)
    # ------------------------------------------------------------------
    _pyautogui_ok = False
    try:
        import pyautogui as _pya
        _pya.size()   # also verifies X11/display is reachable
        _pyautogui_ok = True
    except ImportError:
        errors.append(
            "  [MISSING] pyautogui is not installed.\n"
            "    Fix: pip install pyautogui\n"
            "    All UI actions (click, type, press, scroll) will fail without it."
        )
    except Exception as _pya_err:
        errors.append(
            f"  [DISPLAY] pyautogui cannot access display: {_pya_err}.\n"
            "    On headless systems start a virtual display first:\n"
            "      Xvfb :99 -screen 0 1920x1080x24 &\n"
            "      export DISPLAY=:99\n"
            "    OR run ProjectZeo from inside a graphical desktop session."
        )

    # ------------------------------------------------------------------
    # 3 & 4. Linux-specific: xdotool, wmctrl, DISPLAY, Wayland
    # ------------------------------------------------------------------
    if _platform.system() == "Linux":

        # xdotool — FATAL: required for window title snapshot on X11
        if _shutil.which("xdotool") is None:
            errors.append(
                "  [MISSING] xdotool is not installed (required for snapshot capture on Linux).\n"
                "    Fix (Debian/Ubuntu): sudo apt-get install xdotool\n"
                "    Fix (Fedora/RHEL):   sudo dnf install xdotool\n"
                "    Fix (Arch):          sudo pacman -S xdotool"
            )

        # wmctrl — WARNING: best-effort window activation
        if _shutil.which("wmctrl") is None:
            warnings.append(
                "  [WARNING] wmctrl is not installed. "
                "Window-focus restoration will be best-effort only.\n"
                "    Fix: sudo apt-get install wmctrl"
            )

        # DISPLAY / WAYLAND_DISPLAY — FATAL if both absent
        # (pyautogui check above usually catches this, but guard for the
        # pyautogui-not-installed path where _pya.size() was never called)
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            errors.append(
                "  [NO DISPLAY] DISPLAY and WAYLAND_DISPLAY are both unset on Linux.\n"
                "    pyautogui requires a running X11 or Wayland display.\n"
                "    To run headlessly:\n"
                "      Xvfb :99 -screen 0 1920x1080x24 &\n"
                "      export DISPLAY=:99\n"
                "    OR launch ProjectZeo from inside a graphical desktop session."
            )

        # Wayland tooling — WARNING when no Wayland-capable tool is present
        _xdg_session = os.environ.get("XDG_SESSION_TYPE", "").lower()
        _wayland_disp = os.environ.get("WAYLAND_DISPLAY", "")
        # Also detect Wayland when XDG_SESSION_TYPE is unset (e.g. launched via
        # sudo) by checking WAYLAND_DISPLAY directly.
        _is_wayland = _xdg_session == "wayland" or bool(_wayland_disp)
        if _is_wayland:
            _has_ydotool = _shutil.which("ydotool") is not None
            _has_wmctrl  = _shutil.which("wmctrl") is not None
            _has_pyatspi = False
            try:
                import pyatspi  # noqa: F401
                _has_pyatspi = True
            except ImportError:
                pass
            if not _has_ydotool and not _has_pyatspi and not _has_wmctrl:
                warnings.append(
                    "  [WARNING] Wayland session detected but no Wayland-compatible "
                    "window management tool found (ydotool / AT-SPI2 / wmctrl).\n"
                    "    Snapshot/restore will degrade to cursor-only (no window focus).\n"
                    "    Fix (best):  sudo apt-get install ydotool && ydotoold &\n"
                    "    Fix (alt):   pip install pyatspi  and\n"
                    "                 gsettings set org.gnome.desktop.interface "
                    "toolkit-accessibility true\n"
                    "    Fix (switch): log into a GNOME-on-Xorg session instead of Wayland."
                )

    # ------------------------------------------------------------------
    # 5. pyyaml — WARNING: without it policy.yaml never loads
    # ------------------------------------------------------------------
    try:
        import yaml  # noqa: F401
    except ImportError:
        warnings.append(
            "  [WARNING] pyyaml is not installed — policy.yaml will NOT be loaded.\n"
            "    PolicyEngine will run with the built-in default allowlist only.\n"
            "    Fix: pip install pyyaml"
        )

    # ------------------------------------------------------------------
    # 6. playwright + chromium binary — WARNING: browser tasks fall back to
    #    pyautogui coordinate clicks without it (coordinate staleness on SPAs)
    # ------------------------------------------------------------------
    try:
        import playwright  # noqa: F401
        # playwright Python package present — check for a usable browser binary
        _chromium_found = (
            _shutil.which("chromium-browser") is not None
            or _shutil.which("chromium") is not None
            or _shutil.which("google-chrome") is not None
            or _shutil.which("google-chrome-stable") is not None
        )
        if not _chromium_found:
            warnings.append(
                "  [WARNING] playwright is installed but no Chromium/Chrome binary found.\n"
                "    Browser DOM automation will fall back to pyautogui coordinate clicks.\n"
                "    Fix: playwright install chromium\n"
                "    OR:  sudo apt-get install chromium-browser"
            )
    except ImportError:
        warnings.append(
            "  [WARNING] playwright is not installed — browser DOM automation disabled.\n"
            "    Web tasks will use pyautogui coordinate clicks (SPA-unfriendly).\n"
            "    Fix: pip install playwright && playwright install chromium"
        )

    # ------------------------------------------------------------------
    # 7. EasyOCR — WARNING: label-based clicks fail until OCR is warm
    # ------------------------------------------------------------------
    try:
        import easyocr  # noqa: F401
    except ImportError:
        warnings.append(
            "  [WARNING] easyocr is not installed — label-based clicks will fail.\n"
            "    The agent can only click by pixel coordinates, not by UI label text.\n"
            "    Fix: pip install easyocr\n"
            "    Note: first EasyOCR init downloads ~500 MB of model weights (~5-10 min)."
        )

    # ------------------------------------------------------------------
    # 8. Ollama daemon reachability — FATAL
    # ------------------------------------------------------------------
    _ollama_ok = False
    try:
        import ollama as _ollama
        _ollama.Client().list()
        _ollama_ok = True
    except Exception as _ollama_err:
        errors.append(
            "  [UNREACHABLE] Ollama daemon is not running or not installed: "
            + str(_ollama_err) + "\n"
            "    Fix: install Ollama from https://ollama.com "
            "and ensure the daemon is running.\n"
            "    On Linux/macOS: ollama serve"
        )

    # ------------------------------------------------------------------
    # 9. Model availability — FATAL (only when Ollama is reachable)
    # ------------------------------------------------------------------
    if _ollama_ok:
        try:
            import ollama as _ollama  # noqa: F811
            _existing_models = {m.model for m in _ollama.Client().list().models}
            _base = model_name.split(":")[0]
            _found = any(
                model_name in m or _base in m
                for m in _existing_models
            )
            if not _found:
                errors.append(
                    "  [NOT PULLED] Model '" + model_name + "' is not available locally.\n"
                    "    Fix: ollama pull " + model_name
                )
        except Exception:
            pass  # Daemon check already covered above

    # ------------------------------------------------------------------
    # 10. Text model availability — WARNING (performance only)
    # ------------------------------------------------------------------
    if _ollama_ok:
        try:
            import ollama as _ollama  # noqa: F811
            _existing = {m.model for m in _ollama.Client().list().models}
            _text_candidate = model_name.replace("-vl:", ":").replace("-vl", "")
            _text_base = _text_candidate.split(":")[0]
            _text_found = any(
                _text_candidate in m or _text_base in m
                for m in _existing
            )
            _is_vision_model = "-vl" in model_name or "vision" in model_name.lower()
            if _is_vision_model and not _text_found:
                warnings.append(
                    "  [WARNING] No dedicated text model found for '" + model_name + "'.\n"
                    "    ExecutionPlanner will use the vision model for text-only planning.\n"
                    "    This works correctly but is slower and uses more VRAM.\n"
                    "    Fix: ollama pull " + _text_candidate + "\n"
                    "    Or:  set LLM_TEXT_MODEL env var to an explicit text model name."
                )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Emit results
    # ------------------------------------------------------------------
    if warnings:
        print("\n[STARTUP] Dependency warnings (non-fatal):", file=sys.stderr)
        for w in warnings:
            print(w, file=sys.stderr)

    if errors:
        print(
            "\n[STARTUP] FATAL: Required dependencies are missing or unreachable.\n"
            "ProjectZeo cannot start until these are resolved:\n",
            file=sys.stderr,
        )
        for e in errors:
            print(e, file=sys.stderr)
        print("", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model_name, allow_cloud, interactive, status_only = _parse_args()

    # BUG-10 FIX: --status flag — print status from temp/ sidecars and exit.
    if status_only:
        _print_status()
        sys.exit(0)

    # FIX F-02: Persist the resolved model name so all downstream text-only
    # Ollama calls see the operator-specified model.
    os.environ["LLM_MODEL"] = model_name

    # BUG-10 FIX: --interactive mode — signal to main.py to print mode
    # transitions to stdout and enable the interactive stdin prompt.
    if interactive:
        os.environ["PROJECTZEO_INTERACTIVE"] = "1"
        print(
            "[ProjectZeo] Interactive mode enabled. "
            "Type your intent and press Enter to arm the agent. "
            "Mode transitions will be printed to stdout.",
            flush=True,
        )

    # FIX P0-1/P0-2/P0-3/P0-4: Validate all hard runtime dependencies before
    # importing any project code that would crash on missing system tools.
    _validate_runtime_dependencies(model_name)

    # OLLAMA_ONLY was already set/cleared above before factory import.
    # The factory freeze has already captured the correct value.
    # Do NOT mutate os.environ["OLLAMA_ONLY"] again here — factory.py's
    # module-level freeze is immutable; any mutation after import is silently
    # ignored by the enforcement boundary.

    adapter = build_llm(model_name)
    llm_callable = _make_llm_callable(adapter)
    main(llm_callable, model_name=model_name)

    # GAP-5 FIX: After main() returns, print the task result so the operator
    # can see the outcome without digging through JSONL log files.
    import json as _json, pathlib as _pathlib
    _result_path = _pathlib.Path(__file__).resolve().parent / "temp" / "task_result.json"
    if _result_path.exists():
        try:
            with open(_result_path) as _f:
                _r = _json.load(_f)
            _ok = _r.get("success", False)
            _label = "SUCCEEDED" if _ok else "FAILED"
            print(f"\n[ProjectZeo] Task {_label}: {_r.get('intent', '?')[:80]}", flush=True)
            if _r.get("error"):
                print(f"[ProjectZeo] Error: {_r['error'][:120]}", flush=True)
        except Exception:
            pass
