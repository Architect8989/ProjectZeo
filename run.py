import os
import sys
import subprocess


def _early_parse_args() -> tuple:
    """
    Pre-import scan of sys.argv for flags that must be set before module imports.
    Returns (allow_cloud: bool, model_name_or_none: str|None).
    """
    allow_cloud = "--allow-cloud" in sys.argv

    # Also treat any anthropic:* or openai:* model as cloud-allowed.
    for arg in sys.argv[1:]:
        if not arg.startswith("--"):
            if arg.startswith("anthropic:") or arg.startswith("openai:"):
                allow_cloud = True
            break

    # Support --model <name> flag in addition to positional arg.
    model = None
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg in ("--model", "-m") and i < len(sys.argv):
            model = sys.argv[i + 1] if i < len(sys.argv) else None
            if model and (model.startswith("anthropic:") or model.startswith("openai:")):
                allow_cloud = True

    return allow_cloud, model


_allow_cloud_early, _model_early = _early_parse_args()

if not _allow_cloud_early:
    os.environ["OLLAMA_ONLY"] = "1"
else:
    os.environ["OLLAMA_ONLY"] = "0"
    print(
        "[run.py] WARNING: --allow-cloud is set (or a cloud model prefix was detected). "
        "Cloud API routing is ENABLED. Ensure API keys are intentionally configured.",
        file=sys.stderr,
    )


import asyncio
import threading
from typing import Any

from adapters.factory import build_llm
from adapters.cloud_adapter import is_cloud_model
from main import main
from config.timeouts import LLM_THREAD_TIMEOUT_SECONDS


def _parse_args():
    args = sys.argv[1:]
    allow_cloud = "--allow-cloud" in args
    interactive = "--interactive" in args
    status_only = "--status" in args

    positional = [a for a in args if not a.startswith("--")]

    model: str | None = None

    # --model <name> flag takes precedence over positional arg
    for i, arg in enumerate(args):
        if arg in ("--model", "-m") and i + 1 < len(args):
            model = args[i + 1].strip() or None
            break

    if not model and positional:
        model = positional[0].strip() or None

    if not model:
        model = os.getenv("LLM_MODEL", "").strip() or None

    if not status_only and not model:
        raise RuntimeError(
            "No model specified.\n"
            "Pass model as CLI argument, --model flag, or LLM_MODEL env var.\n"
            "\n"
            "Local (Ollama):\n"
            "  python run.py qwen2.5-vl:7b-instruct\n"
            "\n"
            "Cloud (Anthropic — requires ANTHROPIC_API_KEY):\n"
            "  python run.py anthropic:claude-sonnet-4-20250514\n"
            "\n"
            "Cloud (OpenAI — requires OPENAI_API_KEY):\n"
            "  python run.py openai:gpt-4o\n"
            "\n"
            "Status check:\n"
            "  python run.py --status"
        )

    if model and (model.startswith("anthropic:") or model.startswith("openai:")):
        allow_cloud = True

    return model, allow_cloud, interactive, status_only


def _print_status() -> None:
    import json as _json
    import pathlib as _pathlib
    import datetime as _dt

    _root = _pathlib.Path(__file__).resolve().parent
    _temp = _root / "temp"

    # D-7 FIX: Intent file now lives in ~/.projectzeo/
    from core.intent_listener import IntentListener as _IL
    _intent_file = _pathlib.Path(_IL.INTENT_FILE)
    _arm_success = _pathlib.Path(_IL._SIDECAR_DIR) / "arm_success.json"
    _arm_failure = _pathlib.Path(_IL._SIDECAR_DIR) / "arm_failure.json"

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

    if _arm_success.exists():
        try:
            with open(_arm_success) as f:
                s = _json.load(f)
            ts = _dt.datetime.fromtimestamp(s.get("timestamp", 0)).strftime("%Y-%m-%d %H:%M:%S")
            print(f"Last arm    : SUCCESS at {ts}")
        except Exception:
            print("Last arm    : SUCCESS (timestamp unreadable)")
    elif _arm_failure.exists():
        try:
            with open(_arm_failure) as f:
                s = _json.load(f)
            print(f"Last arm    : FAILED — {s.get('reason', '?')}")
        except Exception:
            print("Last arm    : FAILED (details unreadable)")
    else:
        print("Last arm    : No arm event recorded")

    if _intent_file.exists():
        try:
            content = _intent_file.read_text(encoding="utf-8").strip()[:72]
            print(f"Pending     : {content!r}")
        except Exception:
            print("Pending     : (unreadable)")
    else:
        print("Pending     : None")

    print("=" * 42)
    print(f"Intent file : {_intent_file}")
    print("Tip: write intent to the intent file above  OR  run with --interactive")


def _run_coroutine_threadsafe(coro) -> Any:
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


def _make_llm_callable(adapter, model_name: str):
    """
    Build the llm_callable that is passed to main().

    Cloud adapters (AnthropicCloudAdapter, OpenAICloudAdapter via _CloudCallable)
    implement __call__(messages, objective, session_id) directly and do NOT have
    an async get_next_action() method.  For these, call __call__ directly without
    going through the async thread bounce.

    Local Ollama adapters require the async get_next_action() + thread path.
    """

    # Cloud path: direct synchronous call.
    if is_cloud_model(model_name):
        def _cloud_call(messages, objective=None, session_id=None):
            try:
                result = adapter(messages, objective=objective, session_id=session_id)
            except Exception as e:
                raise RuntimeError(f"Cloud LLM adapter invocation failed: {e}") from e

            # Cloud adapters return a string; wrap in the list format expected by
            # operate.py / plan parsing code that expects a list of operations.
            # The per_step_reasoner and planner both handle str responses natively
            # through their _parse_action / _parse_step_array methods.
            if isinstance(result, str):
                return result
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                return [result]
            return str(result)

        return _cloud_call

    # Local Ollama path: async get_next_action() via thread.
    if not hasattr(adapter, "get_next_action"):
        raise RuntimeError(
            f"Adapter for model {model_name!r} is missing get_next_action(). "
            "Ensure the adapter class implements the llm_callable contract."
        )

    def _ollama_call(messages, objective=None, session_id=None):
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

    return _ollama_call


def _validate_runtime_dependencies(model_name: str) -> None:
    import shutil as _shutil
    import platform as _platform

    errors = []
    warnings = []

    _project_root = os.path.dirname(os.path.abspath(__file__))
    for _req_dir in ("temp", "memory/snapshots", "memory/playbooks", "logs"):
        _abs_dir = os.path.join(_project_root, _req_dir)
        try:
            os.makedirs(_abs_dir, exist_ok=True)
        except OSError as _mkdir_err:
            warnings.append(
                f"  [WARNING] Could not create required directory {_abs_dir!r}: {_mkdir_err}. "
                "Snapshot persistence and transition logging will be disabled."
            )

    # Cloud models do not require Ollama; skip local Ollama checks entirely.
    _is_cloud = is_cloud_model(model_name)

    if not _is_cloud:
        try:
            import psutil as _  # noqa: F401
        except ImportError:
            errors.append(
                "  [MISSING] psutil is not installed.\n"
                "    Fix: pip install psutil"
            )

    _pyautogui_ok = False
    try:
        import pyautogui as _pya
        _pya.size()
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
            "      export DISPLAY=:99"
        )

    if _platform.system() == "Linux":
        if _shutil.which("xdotool") is None:
            errors.append(
                "  [MISSING] xdotool is not installed (required for snapshot capture on Linux).\n"
                "    Fix (Debian/Ubuntu): sudo apt-get install xdotool"
            )

        if _shutil.which("wmctrl") is None:
            warnings.append(
                "  [WARNING] wmctrl is not installed. Window-focus restoration will be "
                "best-effort only.\n    Fix: sudo apt-get install wmctrl"
            )

        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            errors.append(
                "  [NO DISPLAY] DISPLAY and WAYLAND_DISPLAY are both unset on Linux.\n"
                "    pyautogui requires a running X11 or Wayland display.\n"
                "      Xvfb :99 -screen 0 1920x1080x24 &\n"
                "      export DISPLAY=:99"
            )

        _xdg_session = os.environ.get("XDG_SESSION_TYPE", "").lower()
        _wayland_disp = os.environ.get("WAYLAND_DISPLAY", "")
        _is_wayland = _xdg_session == "wayland" or bool(_wayland_disp)
        if _is_wayland:
            _has_ydotool = _shutil.which("ydotool") is not None
            _has_pyatspi = False
            try:
                import pyatspi  # noqa: F401
                _has_pyatspi = True
            except ImportError:
                pass
            _has_wmctrl = _shutil.which("wmctrl") is not None
            if not _has_ydotool and not _has_pyatspi and not _has_wmctrl:
                warnings.append(
                    "  [WARNING] Wayland session detected but no Wayland-compatible "
                    "input tool found.\n"
                    "    Fix: sudo apt-get install ydotool && "
                    "systemctl --user enable --now ydotool.service"
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
                    errors.append(
                        "  [FATAL] Wayland session: ydotool is installed but the\n"
                        "    ydotoold daemon is NOT running. All UI input will fail.\n"
                        "\n"
                        "    Quick fix (manual):  ydotoold &\n"
                        "    Persistent fix:      systemctl --user enable --now ydotool.service"
                    )

    try:
        import yaml  # noqa: F401
    except ImportError:
        warnings.append(
            "  [WARNING] pyyaml is not installed — policy.yaml will NOT be loaded.\n"
            "    Fix: pip install pyyaml"
        )

    try:
        import playwright  # noqa: F401
        _chromium_found = any(
            _shutil.which(b) is not None
            for b in ("chromium-browser", "chromium", "google-chrome", "google-chrome-stable")
        )
        if not _chromium_found:
            warnings.append(
                "  [WARNING] playwright installed but no Chromium binary found.\n"
                "    Fix: playwright install chromium"
            )
    except ImportError:
        warnings.append(
            "  [WARNING] playwright is not installed — browser DOM automation disabled.\n"
            "    Fix: pip install playwright && playwright install chromium"
        )

    # Cloud adapter dependency checks
    if _is_cloud:
        if model_name.startswith("anthropic:"):
            try:
                import anthropic  # noqa: F401
            except ImportError:
                errors.append(
                    "  [MISSING] anthropic package not installed.\n"
                    "    Fix: pip install anthropic"
                )
            if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
                errors.append(
                    "  [MISSING] ANTHROPIC_API_KEY environment variable is not set.\n"
                    "    Fix: export ANTHROPIC_API_KEY=sk-ant-..."
                )
        elif model_name.startswith("openai:"):
            try:
                import openai  # noqa: F401
            except ImportError:
                errors.append(
                    "  [MISSING] openai package not installed.\n"
                    "    Fix: pip install openai"
                )
            if not os.environ.get("OPENAI_API_KEY", "").strip():
                errors.append(
                    "  [MISSING] OPENAI_API_KEY environment variable is not set.\n"
                    "    Fix: export OPENAI_API_KEY=sk-..."
                )
    else:
        # Ollama checks
        _ollama_ok = False
        try:
            import ollama as _ollama
            _ollama.Client().list()
            _ollama_ok = True
        except Exception as _ollama_err:
            errors.append(
                "  [UNREACHABLE] Ollama daemon is not running or not installed: "
                + str(_ollama_err) + "\n"
                "    Fix: install Ollama from https://ollama.com and run: ollama serve"
            )

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
                        f"  [NOT PULLED] Model '{model_name}' is not available locally.\n"
                        f"    Fix: ollama pull {model_name}"
                    )
            except Exception:
                pass

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
                        f"  [WARNING] No dedicated text model found for '{model_name}'.\n"
                        "    ExecutionPlanner will use the vision model for text-only planning.\n"
                        f"    Fix: ollama pull {_text_candidate}"
                    )
            except Exception:
                pass

        if _ollama_ok:
            try:
                import ollama as _ollama_prewarm
                print(
                    f"[STARTUP] Pre-warming Ollama model {model_name!r} "
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
                    f"[STARTUP] Model pre-warm complete (response: {_pw_content!r}).",
                    file=sys.stderr,
                )
            except Exception as _pw_err:
                print(
                    f"[STARTUP] WARNING: Model pre-warm failed: {_pw_err}. "
                    "VisionRuntime warmup may take longer.",
                    file=sys.stderr,
                )

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


if __name__ == "__main__":
    model_name, allow_cloud, interactive, status_only = _parse_args()

    if status_only:
        _print_status()
        sys.exit(0)

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
    llm_callable = _make_llm_callable(adapter, model_name)
    main(llm_callable, model_name=model_name)

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
