import os
import sys
import subprocess

def _early_parse_args() -> bool:
    
    return "--allow-cloud" in sys.argv


# Resolve the operator's cloud intent at the earliest possible moment.
_allow_cloud_early = _early_parse_args()

if not _allow_cloud_early:
    
    os.environ["OLLAMA_ONLY"] = "1"
else:
    
    os.environ["OLLAMA_ONLY"] = "0"
    # Warn early so operators see this even if something crashes during import.
    print(
        "[run.py] WARNING: --allow-cloud is set. Cloud API routing is ENABLED. "
        "Ensure API keys are intentionally configured.",
        file=sys.stderr,
    )



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

            
            if _has_ydotool:
                _daemon_ok = False
                try:
                    _probe = subprocess.run(
                        ["ydotool", "mousemove", "--relative", "-x", "0", "-y", "0"],
                        capture_output=True,
                        timeout=3,
                    )
                    _daemon_ok = (_probe.returncode == 0)
                except Exception:
                    _daemon_ok = False

                if not _daemon_ok:
                    # Optionally attempt auto-start
                    _auto_start = os.environ.get(
                        "PROJECTZEO_START_YDOTOOLD", ""
                    ).strip().lower() in ("1", "true", "yes")

                    if _auto_start:
                        print(
                            "[STARTUP] BLOCKER-1: ydotoold daemon not running. "
                            "PROJECTZEO_START_YDOTOOLD=1 — attempting auto-start: "
                            "sudo ydotoold &",
                            file=sys.stderr,
                        )
                        try:
                            subprocess.Popen(
                                ["sudo", "ydotoold"],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                            import time as _time_mod
                            _time_mod.sleep(2.0)  # allow daemon to bind socket
                            # Re-probe after auto-start
                            _probe2 = subprocess.run(
                                ["ydotool", "mousemove", "--relative",
                                 "-x", "0", "-y", "0"],
                                capture_output=True,
                                timeout=3,
                            )
                            if _probe2.returncode == 0:
                                print(
                                    "[STARTUP] BLOCKER-1: ydotoold auto-start succeeded.",
                                    file=sys.stderr,
                                )
                                _daemon_ok = True
                            else:
                                errors.append(
                                    "  [FATAL] ydotoold auto-start attempted but daemon "
                                    "still not responding.\n"
                                    "    Manual fix: sudo ydotoold &\n"
                                    "    Or via systemd: "
                                    "systemctl --user enable --now ydotool.service"
                                )
                        except Exception as _as_err:
                            errors.append(
                                f"  [FATAL] ydotoold auto-start failed: {_as_err}\n"
                                "    Manual fix: sudo ydotoold &\n"
                                "    Or via systemd: "
                                "systemctl --user enable --now ydotool.service"
                            )
                    else:
                        errors.append(
                            "  [FATAL] Wayland session: ydotool is installed but the\n"
                            "    ydotoold daemon is NOT running.  All UI input (click,\n"
                            "    type, hotkey) will silently fail until the daemon starts.\n"
                            "\n"
                            "    Quickest fix:\n"
                            "      sudo ydotoold &\n"
                            "\n"
                            "    Persistent fix (systemd user service):\n"
                            "      systemctl --user enable --now ydotool.service\n"
                            "\n"
                            "    Auto-start on next ProjectZeo launch:\n"
                            "      export PROJECTZEO_START_YDOTOOLD=1\n"
                            "\n"
                            "    Or switch to a GNOME-on-Xorg session (login gear → "
                            "'Ubuntu on Xorg') to avoid Wayland input limitations."
                        )

    
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

    
    if _ollama_ok:
        try:
            import ollama as _ollama_prewarm
            print(
                f"[STARTUP] GAP-10: Pre-warming Ollama model {model_name!r} "
                "(first inference loads weights; may take 1-5 min on cold start)...",
                file=sys.stderr,
            )
            _pw_client = _ollama_prewarm.Client()
            _pw_response = _pw_client.chat(
                model=model_name,
                messages=[{"role": "user", "content": "Hello. Reply with one word."}],
                options={"temperature": 0, "num_predict": 5},
            )
            _pw_content = ""
            if hasattr(_pw_response, "message") and hasattr(_pw_response.message, "content"):
                _pw_content = _pw_response.message.content[:40]
            elif isinstance(_pw_response, dict):
                _pw_content = (_pw_response.get("message") or {}).get("content", "")[:40]
            print(
                f"[STARTUP] GAP-10: Model pre-warm complete (response: {_pw_content!r}). "
                "VisionRuntime will use warm-start inferences.",
                file=sys.stderr,
            )
        except Exception as _pw_err:
            # Non-fatal: warmup failures print a warning but do not block startup.
            # The model may still load during VisionRuntime warmup — just slower.
            print(
                f"[STARTUP] GAP-10 WARNING: Model pre-warm failed: {_pw_err}. "
                "VisionRuntime warmup may take longer than PROJECTZEO_WARMUP_TIMEOUT_SECONDS. "
                f"Consider increasing: export PROJECTZEO_WARMUP_TIMEOUT_SECONDS=900",
                file=sys.stderr,
            )


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

    
    if interactive:
        os.environ["PROJECTZEO_INTERACTIVE"] = "1"
        print(
            "[ProjectZeo] Interactive mode enabled. "
            "Type your intent and press Enter to arm the agent. "
            "Mode transitions will be printed to stdout.",
            flush=True,
        )

    
    _validate_runtime_dependencies(model_name)

    

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
