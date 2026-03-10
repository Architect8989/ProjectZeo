from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="[bootstrap] %(levelname)s  %(message)s",
    stream=sys.stdout,
)
_log = logging.getLogger("bootstrap")

ROOT = Path(__file__).parent.resolve()

def detect_hardware() -> Dict:
    hw: Dict = {
        "os": platform.system(),
        "arch": platform.machine(),
        "python": sys.version.split()[0],
        "gpu": False,
        "gpu_name": "",
        "vram_gb": 0,
        "cuda_version": "",
        "ollama": bool(shutil.which("ollama")),
        "docker": bool(shutil.which("docker")),
    }

    try:
        import torch
        hw["gpu"] = torch.cuda.is_available()
        if hw["gpu"]:
            hw["gpu_name"] = torch.cuda.get_device_name(0)
            hw["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1
            )
            hw["cuda_version"] = torch.version.cuda or ""
    except ImportError:
        try:
            smi = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if smi.returncode == 0 and smi.stdout.strip():
                parts = smi.stdout.strip().split(",")
                hw["gpu"] = True
                hw["gpu_name"] = parts[0].strip()
                hw["vram_gb"] = round(int(parts[1].strip()) / 1024, 1)
        except Exception:
            pass

    return hw

def hardware_tier(hw: Dict) -> str:
    if not hw["gpu"]:
        return "CPU"
    if hw["vram_gb"] >= 24:
        return "GPU24"
    if hw["vram_gb"] >= 8:
        return "GPU16"
    return "CPU"

_BASE_PACKAGES = [
    "ollama>=0.4.0", "httpx>=0.25.2", "Pillow>=10.1.0", "numpy>=1.26.1",
    "mss>=9.0.1", "pyautogui>=0.9.54", "pyyaml>=6.0", "tqdm>=4.66.1",
    "requests>=2.31.0", "psutil>=5.9.0", "imagehash>=4.3.1",
    "anthropic>=0.40.0,<1.0.0", "openai>=1.13.0",
    "aiohttp>=3.9.1",
    "easyocr>=1.7.1", "playwright>=1.42.0",
    "langgraph>=0.1.0", "langchain-core>=0.1.0",
    "mem0ai>=0.1.0", "qdrant-client>=1.7.0", "cognee>=0.1.0",
    "mctx>=0.0.5",
    "pymdp>=0.0.7.1",
    "pynput>=1.7.6",
    "matplotlib>=3.8.1", "open-interpreter>=0.3.0",
]

_GPU16_PACKAGES = [
    "torch>=2.2.0",
    "torchvision",
    "sglang[all]>=0.4.0",
]

_GPU24_PACKAGES = _GPU16_PACKAGES + [
    "vllm>=0.4.0",
    "transformers>=4.40.0",
    "accelerate>=0.27.0",
]

def pip_install(packages: List[str], label: str = "") -> bool:
    if not packages:
        return True
    tag = f"[{label}] " if label else ""
    _log.info("%sInstalling %d package(s)...", tag, len(packages))
    cmd = [sys.executable, "-m", "pip", "install", "--quiet",
           "--disable-pip-version-check"] + packages
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        _log.error("%sPip failed:\n%s", tag, result.stderr[-2000:])
        return False
    return True

def install_dependencies(tier: str) -> None:
    _log.info("─── Installing dependencies (tier=%s) ───", tier)

    if not pip_install(_BASE_PACKAGES, "base"):
        _log.warning("Some base packages failed — continuing (may affect features)")

    if tier in ("GPU16", "GPU24"):
        pkgs = _GPU24_PACKAGES if tier == "GPU24" else _GPU16_PACKAGES
        if not pip_install(pkgs, "gpu"):
            _log.warning("GPU packages failed — system will run in CPU mode")

    _log.info("Dependencies installed.")

_OLLAMA_API = "http://localhost:11434"

_REQUIRED_MODELS = [
    ("qwen2.5-vl:7b",      7,  "primary_vl"),
    ("llama-guard3:8b",    0,  "safety"),
]
_GPU_MODELS = [
    ("qwen3-vl:8b",       10,  "primary_vl_gpu"),
]

def ollama_running() -> bool:
    try:
        import httpx
        r = httpx.get(f"{_OLLAMA_API}/api/tags", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False

def start_ollama() -> bool:
    if ollama_running():
        _log.info("Ollama already running.")
        return True

    if not shutil.which("ollama"):
        _log.error(
            "Ollama not found. Install from https://ollama.com/download then re-run."
        )
        return False

    _log.info("Starting Ollama daemon...")
    if platform.system() == "Windows":
        subprocess.Popen(["ollama", "serve"], creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    for i in range(20):
        time.sleep(1)
        if ollama_running():
            _log.info("Ollama started (%.0fs).", i + 1)
            return True

    _log.error("Ollama did not start within 20s.")
    return False

def model_available(tag: str) -> bool:
    try:
        import httpx
        r = httpx.get(f"{_OLLAMA_API}/api/tags", timeout=5.0)
        if r.status_code == 200:
            models = r.json().get("models", [])
            return any(m.get("name", "").startswith(tag.split(":")[0]) for m in models)
    except Exception:
        pass
    return False

def pull_model(tag: str) -> bool:
    if model_available(tag):
        _log.info("Model already present: %s", tag)
        return True

    _log.info("Pulling model %s (this may take a while)...", tag)
    try:
        result = subprocess.run(
            ["ollama", "pull", tag],
            capture_output=False,
            text=True,
            timeout=1800,
        )
        if result.returncode == 0:
            _log.info("Model pulled: %s", tag)
            return True
        _log.error("Failed to pull %s (exit %d)", tag, result.returncode)
    except subprocess.TimeoutExpired:
        _log.error("Model pull timed out: %s", tag)
    except Exception as exc:
        _log.error("Model pull error (%s): %s", tag, exc)
    return False

def ensure_models(hw: Dict, tier: str) -> Dict[str, bool]:
    _log.info("─── Ensuring required models ───")
    results: Dict[str, bool] = {}

    models_to_pull = list(_REQUIRED_MODELS)
    if tier in ("GPU16", "GPU24"):
        models_to_pull += _GPU_MODELS

    for tag, min_vram, role in models_to_pull:
        if min_vram > 0 and hw["vram_gb"] < min_vram and not hw["gpu"]:
            _log.info("Skipping %s (requires %.0fGB VRAM, tier=%s)", tag, min_vram, tier)
            results[tag] = False
            continue
        results[tag] = pull_model(tag)

    return results

_QDRANT_PORT = int(os.environ.get("PROJECTZEO_QDRANT_PORT", "6333"))
_QDRANT_DATA = os.path.expanduser(
    os.environ.get("PROJECTZEO_QDRANT_DATA", "~/.projectzeo/qdrant")
)

def qdrant_running() -> bool:
    try:
        import httpx
        r = httpx.get(f"http://localhost:{_QDRANT_PORT}/readyz", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False

def start_qdrant() -> bool:
    if qdrant_running():
        _log.info("Qdrant already running on port %d.", _QDRANT_PORT)
        return True

    os.makedirs(_QDRANT_DATA, exist_ok=True)

    if shutil.which("docker"):
        _log.info("Starting Qdrant via Docker...")
        try:
            subprocess.run(["docker", "rm", "-f", "projectzeo-qdrant"],
                           capture_output=True, timeout=10)
            subprocess.Popen([
                "docker", "run", "-d",
                "--name", "projectzeo-qdrant",
                "--restart", "unless-stopped",
                "-p", f"{_QDRANT_PORT}:{_QDRANT_PORT}",
                "-v", f"{_QDRANT_DATA}:/qdrant/storage",
                "qdrant/qdrant:latest",
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for i in range(30):
                time.sleep(1)
                if qdrant_running():
                    _log.info("Qdrant (Docker) started (%.0fs).", i + 1)
                    return True
        except Exception as exc:
            _log.warning("Qdrant Docker start failed: %s", exc)

    if shutil.which("qdrant"):
        _log.info("Starting Qdrant binary...")
        subprocess.Popen(
            ["qdrant", "--storage-dir", _QDRANT_DATA],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        for i in range(15):
            time.sleep(1)
            if qdrant_running():
                _log.info("Qdrant binary started (%.0fs).", i + 1)
                return True

    _log.info(
        "Qdrant server not available — Mem0/Cognee will use in-memory mode. "
        "For persistence: docker run -p 6333:6333 qdrant/qdrant"
    )
    return False

def build_env(hw: Dict, tier: str, model_results: Dict, qdrant_ok: bool) -> Dict[str, str]:

    if tier in ("GPU16", "GPU24") and model_results.get("qwen3-vl:8b"):
        primary_model = "qwen3-vl:8b"
    elif model_results.get("qwen2.5-vl:7b"):
        primary_model = "qwen2.5-vl:7b"
    else:
        primary_model = "qwen2.5-vl"

    llamaguard_url = ""
    if tier in ("GPU16", "GPU24"):
        llamaguard_url = ""

    env: Dict[str, str] = {
        "PROJECTZEO_REQUIRE_GII":         "1",
        "PROJECTZEO_REQUIRE_LLAMAGUARD":  "1" if model_results.get("llama-guard3:8b") else "0",
        "PROJECTZEO_GII_MODE":            "2",
        "PROJECTZEO_USE_MILESTONES":      "1",
        "PROJECTZEO_USE_WORLD_MODEL":     "1",
        "PROJECTZEO_USE_SELF_MODEL":      "1",
        "PROJECTZEO_USE_LANGGRAPH":       "1",

        "PROJECTZEO_DEFAULT_MODEL":       primary_model,

        "PROJECTZEO_LLAMAGUARD_OLLAMA":   "llama-guard3:8b"
                                          if model_results.get("llama-guard3:8b") else "",

        "PROJECTZEO_QDRANT_URL":          f"http://localhost:{_QDRANT_PORT}" if qdrant_ok else "",
        "PROJECTZEO_MEM0_ENABLED":        "1",
        "PROJECTZEO_COGNEE_ENABLED":      "1" if qdrant_ok else "0",

        "PROJECTZEO_ARPO_ENABLED":        "1",
        "PROJECTZEO_UI_EVOL_ENABLED":     "1",
        "PROJECTZEO_ARPO_EWC":            "1",

        "PROJECTZEO_ACTIVE_INFERENCE":    "1",

        "PROJECTZEO_ATSPI_ENABLED":       "1" if hw["os"] == "Linux" else "0",

        "PROJECTZEO_USE_SGLANG":          "1" if tier in ("GPU16", "GPU24") else "0",
        "PROJECTZEO_USE_VJEPA":           "0",
        "PROJECTZEO_USE_GROUNDING_DINO":  "0",

        "PROJECTZEO_CONSTITUTION_PATH":   str(ROOT / "docs" / "authority_constitution.md"),

        "OLLAMA_HOST":                    _OLLAMA_API,
        "OLLAMA_ONLY":                    "1",
    }

    if tier in ("GPU16", "GPU24"):
        env["OLLAMA_ONLY"] = "0"

    return env

def write_env_file(env: Dict[str, str]) -> Path:
    env_path = ROOT / ".env"
    existing: Dict[str, str] = {}

    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                existing[k.strip()] = v.strip()

    merged = {**existing, **env}

    lines = ["# ProjectZeo auto-generated environment — regenerated by bootstrap.py\n"]
    for k, v in sorted(merged.items()):
        lines.append(f"{k}={v}\n")

    env_path.write_text("".join(lines))
    _log.info(".env written → %s", env_path)
    return env_path

def apply_env(env: Dict[str, str]) -> None:
    for k, v in env.items():
        os.environ[k] = v

def validate_subsystems() -> Dict[str, str]:
    results: Dict[str, str] = {}
    sys.path.insert(0, str(ROOT))

    checks = [
        ("adapters.factory",              "AdapterFactory",           None),
        ("adapters.constitutional_wrapper","ConstitutionalWrapper",    None),
        ("core.gii.gii_controller",       "GIIController",            None),
        ("core.gii.gii_loop",             "GIILoop",                  None),
        ("core.safety.llamaguard_classifier","classify_with_llamaguard",None),
        ("core.safety.scaffold_audit",    "ScaffoldAudit",            None),
        ("core.learning.self_refine",     "SelfRefineEngine",         None),
        ("core.learning.grpo_trainer",    "GRPOTrainer",              None),
        ("core.memory.application_memory","ApplicationMemory",        None),
        ("core.perception.atspi_bridge",  "ATSPIBridge",              None),
        ("core.planner.htn_planner",      "HTNPlanner",               None),
        ("core.agents.langgraph_pipeline","LangGraphPipeline",        None),
        ("adapters.vjepa_adapter",        "VJEPAWorldModel",          None),
        ("adapters.grounding_adapter",    "GroundingAdapter",         None),
        ("adapters.uitars2_adapter",      "UITARS2Adapter",           None),
    ]

    for module_path, symbol, _ in checks:
        try:
            import importlib
            mod = importlib.import_module(module_path)
            getattr(mod, symbol)
            results[module_path] = "ok"
        except ImportError as e:
            results[module_path] = f"warn: missing dep — {e}"
        except Exception as e:
            results[module_path] = f"fail: {e}"

    return results

def print_status(hw: Dict, tier: str, model_results: Dict,
                 qdrant_ok: bool, validation: Dict) -> None:

    ok = sum(1 for v in validation.values() if v == "ok")
    warn = sum(1 for v in validation.values() if v.startswith("warn"))
    fail = sum(1 for v in validation.values() if v.startswith("fail"))

    _log.info("")
    _log.info("═══ ProjectZeo Bootstrap Status ═══")
    _log.info("Hardware  : %s  (tier=%s, GPU=%s %s, VRAM=%.0fGB)",
              hw["os"], tier, hw["gpu"], hw["gpu_name"], hw["vram_gb"])
    _log.info("Ollama    : %s", "✓ running" if ollama_running() else "✗ not running")
    _log.info("Qdrant    : %s", "✓ running" if qdrant_ok else "○ embedded/none")

    _log.info("Models    :")
    for tag, ok_flag in model_results.items():
        _log.info("  %-30s %s", tag, "✓" if ok_flag else "✗ not pulled")

    _log.info("Subsystems: %d ok / %d warn / %d fail", ok, warn, fail)
    for mod, status in validation.items():
        icon = "✓" if status == "ok" else ("⚠" if status.startswith("warn") else "✗")
        if status != "ok":
            _log.info("  %s %-50s %s", icon, mod, status)

    _log.info("")

def patch_llamaguard_for_ollama() -> None:
    ollama_model = os.environ.get("PROJECTZEO_LLAMAGUARD_OLLAMA", "").strip()
    if not ollama_model:
        return

    try:
        from core.safety import llamaguard_classifier as _lg
        _orig_init = _lg.LlamaGuardClassifier._init_backend

        def _patched_init(self_inner):
            _orig_init(self_inner)
            if self_inner._backend != "disabled":
                return
            _model_tag = ollama_model
            try:
                import httpx as _hx
                _r = _hx.get(f"{_OLLAMA_API}/api/tags", timeout=3.0)
                if _r.status_code == 200:
                    _tags = [m.get("name", "") for m in _r.json().get("models", [])]
                    if any(_model_tag.split(":")[0] in t for t in _tags):
                        self_inner._ollama_model_tag = _model_tag
                        self_inner._backend = "ollama"
                        _log.info(
                            "[bootstrap] LlamaGuard wired to Ollama model: %s", _model_tag
                        )
            except Exception as _e:
                _log.debug("[bootstrap] LlamaGuard Ollama probe failed: %s", _e)

        _lg.LlamaGuardClassifier._init_backend = _patched_init

        _orig_classify = _lg.LlamaGuardClassifier._call_backend

        def _patched_classify(self_inner, prompt: str) -> str:
            if self_inner._backend != "ollama":
                return _orig_classify(self_inner, prompt)
            try:
                import httpx as _hx
                payload = {
                    "model": self_inner._ollama_model_tag,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                }
                r = _hx.post(
                    f"{_OLLAMA_API}/api/chat",
                    json=payload,
                    timeout=_hx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0),
                )
                r.raise_for_status()
                return r.json()["message"]["content"]
            except Exception as _e:
                raise RuntimeError(f"LlamaGuard Ollama call failed: {_e}") from _e

        _lg.LlamaGuardClassifier._call_backend = _patched_classify

    except Exception as exc:
        _log.warning("LlamaGuard Ollama patch failed (non-fatal): %s", exc)

def setup(
    skip_install: bool = False,
    skip_models: bool = False,
    force: bool = False,
    model: str = "",
) -> bool:
    _log.info("═══ ProjectZeo Bootstrap ═══")

    _log.info("─── Detecting hardware ───")
    hw = detect_hardware()
    tier = hardware_tier(hw)
    _log.info(
        "OS=%s  Arch=%s  Python=%s  GPU=%s  VRAM=%.0fGB  Tier=%s",
        hw["os"], hw["arch"], hw["python"], hw["gpu_name"] or "none",
        hw["vram_gb"], tier,
    )

    if not skip_install:
        install_dependencies(tier)
    else:
        _log.info("─── Skipping dependency install (skip_install=True) ───")

    _log.info("─── Starting Ollama ───")
    ollama_ok = start_ollama()
    if not ollama_ok:
        _log.warning("Ollama unavailable — system will run in limited mode.")

    model_results: Dict[str, bool] = {}
    if ollama_ok and not skip_models:
        model_results = ensure_models(hw, tier)
    else:
        _log.info("─── Skipping model pulls ───")

    _log.info("─── Starting Qdrant ───")
    qdrant_ok = start_qdrant()

    _log.info("─── Configuring environment ───")
    env = build_env(hw, tier, model_results, qdrant_ok)
    if model:
        env["PROJECTZEO_DEFAULT_MODEL"] = model
    write_env_file(env)
    apply_env(env)

    patch_llamaguard_for_ollama()

    _log.info("─── Validating subsystems ───")
    validation = validate_subsystems()

    print_status(hw, tier, model_results, qdrant_ok, validation)

    hard_fails = [k for k, v in validation.items() if v.startswith("fail")]
    if hard_fails and not force:
        _log.error("Hard failures detected:")
        for f in hard_fails:
            _log.error("  ✗ %s: %s", f, validation[f])
        return False

    _log.info("═══ Bootstrap complete — system ready ═══")
    return True

def bootstrap(args) -> int:
    if args.status:
        hw = detect_hardware()
        tier = hardware_tier(hw)
        qdrant_ok = qdrant_running()
        validation = validate_subsystems()
        print_status(hw, tier, {}, qdrant_ok, validation)
        return 0

    ok = setup(
        skip_install=args.skip_install,
        skip_models=args.skip_models,
        force=args.force,
        model=args.model or "",
    )
    return 0 if ok else 1

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ProjectZeo bootstrap — setup only. To run the system: python run.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bootstrap.py                   # Setup only (run.py calls this automatically)
  python bootstrap.py --status          # Print hardware/service status and exit
  python bootstrap.py --skip-models     # Setup without pulling models
  python bootstrap.py --skip-install    # Setup without pip install
  python run.py                         # Normal entry point (calls setup if needed)
        """,
    )
    parser.add_argument("--model",        default="",          help="Override default VL model")
    parser.add_argument("--skip-install", action="store_true", help="Skip pip install")
    parser.add_argument("--skip-models",  action="store_true", help="Skip Ollama model pulls")
    parser.add_argument("--force",        action="store_true", help="Continue despite failures")
    parser.add_argument("--status",       action="store_true", help="Status check only")
    args = parser.parse_args()

    sys.exit(bootstrap(args))

if __name__ == "__main__":
    main()
