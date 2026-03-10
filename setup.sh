#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# ProjectZeo GII — Complete First-Run Setup Script
# Blueprint: All 19 algorithms, 6 restoration tiers, 9-tier safety stack
#
# Usage:
#   bash setup.sh                    # Full setup
#   bash setup.sh --skip-models      # Skip Ollama model pulls (large downloads)
#   bash setup.sh --skip-weights     # Skip OmniParser weight download
#   bash setup.sh --skip-docker      # Skip Docker services
#   bash setup.sh --gpu              # Install GPU dependencies (CUDA 12.1+)
#
# What this script does:
#   1. Creates required directories
#   2. Installs system dependencies (apt)
#   3. Installs Python dependencies (pip)
#   4. Installs Playwright browser
#   5. Pulls Ollama models
#   6. Downloads OmniParser weights (optional)
#   7. Starts Docker services (Qdrant + FalkorDB)
#   8. Runs connectivity checks
#   9. Creates .env from .env.example (if not exists)
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail
IFS=$'\n\t'

# ── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }
header()  { echo -e "\n${BOLD}${BLUE}═══ $* ═══${NC}\n"; }

# ── Parse arguments ──────────────────────────────────────────────────────────
SKIP_MODELS=false
SKIP_WEIGHTS=false
SKIP_DOCKER=false
INSTALL_GPU=false

for arg in "$@"; do
    case $arg in
        --skip-models)  SKIP_MODELS=true ;;
        --skip-weights) SKIP_WEIGHTS=true ;;
        --skip-docker)  SKIP_DOCKER=true ;;
        --gpu)          INSTALL_GPU=true ;;
        --help|-h)
            echo "Usage: bash setup.sh [--skip-models] [--skip-weights] [--skip-docker] [--gpu]"
            exit 0
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Step 1: Required directories ─────────────────────────────────────────────
header "Step 1: Creating required directories"

DIRS=(
    "temp"
    "data/qdrant"
    "data/falkordb"
    "data/redis"
    "scripts"
    "$HOME/.projectzeo"
    "$HOME/.projectzeo/memory"
    "$HOME/.projectzeo/weights"
    "$HOME/.projectzeo/trajectories"
    "$HOME/.projectzeo/grpo"
    "$HOME/.projectzeo/agent_q"
    "$HOME/.projectzeo/training"
    "$HOME/.projectzeo/criu_dumps"
    "$HOME/.projectzeo/sppo"
)

for dir in "${DIRS[@]}"; do
    mkdir -p "$dir"
    success "Created: $dir"
done

# ── Step 2: System dependencies ──────────────────────────────────────────────
header "Step 2: System dependencies"

OS="$(uname -s)"

if [[ "$OS" == "Linux" ]]; then
    info "Detected Linux — installing system packages..."
    if command -v apt-get &>/dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y \
            xdotool \
            wmctrl \
            at-spi2-core \
            python3-pyatspi \
            xclip \
            xsel \
            ffmpeg \
            curl \
            wget \
            git \
            build-essential \
            libssl-dev \
            libffi-dev \
            python3-dev \
            2>/dev/null || warn "Some apt packages failed (non-fatal)"

        # CRIU (optional — requires privileged mode)
        if ! command -v criu &>/dev/null; then
            info "Installing CRIU (process checkpointing)..."
            sudo apt-get install -y criu 2>/dev/null || \
                warn "CRIU not available via apt — Tier-4 restoration disabled (non-fatal)"
        else
            success "CRIU already installed: $(criu --version 2>/dev/null | head -1)"
        fi

        # bwrap (bubblewrap) for sandbox fallback
        if ! command -v bwrap &>/dev/null; then
            sudo apt-get install -y bubblewrap 2>/dev/null || \
                warn "bwrap not available — subprocess sandbox fallback will be used"
        fi

        success "System packages installed"
    else
        warn "apt-get not found — skipping system package install. Install manually: xdotool wmctrl at-spi2-core criu"
    fi
elif [[ "$OS" == "Darwin" ]]; then
    info "Detected macOS — installing via Homebrew..."
    if command -v brew &>/dev/null; then
        brew install xdotool 2>/dev/null || true
        brew install ffmpeg 2>/dev/null || true
        success "Homebrew packages installed"
    else
        warn "Homebrew not found. Install from https://brew.sh"
    fi
fi

# ── Step 3: Python dependencies ──────────────────────────────────────────────
header "Step 3: Python dependencies"

info "Installing base requirements..."
pip install -r requirements.txt --quiet || {
    warn "Some packages failed — trying individual installs..."
    # Install critical packages individually
    CRITICAL_PKGS=(
        "anthropic>=0.40.0,<1.0.0"
        "openai>=1.13.0"
        "ollama>=0.4.0"
        "pillow>=10.1.0"
        "numpy>=1.26.1"
        "psutil>=5.9.0"
        "pyautogui>=0.9.54"
        "pyyaml>=6.0"
        "requests>=2.31.0"
        "playwright>=1.42.0"
        "langgraph>=0.1.0"
        "langchain-core>=0.1.0"
        "mem0ai>=0.1.0"
        "faiss-cpu>=1.7.4"
        "imagehash>=4.3.1"
        "watchdog>=3.0.0"
        "scipy>=1.11.0"
        "transformers>=4.40.0"
        "docker>=7.0.0"
    )
    for pkg in "${CRITICAL_PKGS[@]}"; do
        pip install "$pkg" --quiet 2>/dev/null || warn "Failed: $pkg"
    done
}
success "Python requirements installed"

# GPU dependencies (optional)
if [[ "$INSTALL_GPU" == "true" ]]; then
    header "Step 3b: GPU dependencies (CUDA 12.1+)"
    warn "GPU install requires CUDA 12.1+ and ≥16GB VRAM"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
    pip install trl unsloth peft bitsandbytes accelerate --quiet
    pip install spikingjelly --quiet
    pip install pymdp --quiet
    success "GPU packages installed"
fi

# pymdp for Active Inference (CPU install)
info "Installing pymdp (Active Inference / FEP)..."
pip install pymdp --quiet 2>/dev/null || warn "pymdp install failed — Active Inference FEP will use approximation"

# SpikingJelly for SNN (CPU install)
info "Installing SpikingJelly (SNN event processor)..."
pip install spikingjelly --quiet 2>/dev/null || warn "SpikingJelly install failed — using leaky-integrator fallback"

# graphiti-core
info "Installing graphiti-core..."
pip install graphiti-core --quiet 2>/dev/null || warn "graphiti-core install failed — Graphiti KG unavailable"

# ── Step 4: Playwright browser ───────────────────────────────────────────────
header "Step 4: Playwright browser"

if command -v playwright &>/dev/null || python3 -c "import playwright" 2>/dev/null; then
    info "Installing Playwright Chromium..."
    python3 -m playwright install chromium 2>/dev/null || playwright install chromium 2>/dev/null || \
        warn "Playwright browser install failed — browser CDP restoration disabled"
    success "Playwright Chromium ready"
else
    warn "Playwright not installed — skipping browser install"
fi

# ── Step 5: Ollama models ────────────────────────────────────────────────────
if [[ "$SKIP_MODELS" == "true" ]]; then
    warn "Skipping Ollama model pulls (--skip-models)"
else
    header "Step 5: Ollama models"

    if ! command -v ollama &>/dev/null; then
        error "Ollama not installed! Install from https://ollama.com then re-run setup."
        warn "Skipping model pulls..."
    else
        # Check if Ollama is running
        if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
            info "Starting Ollama daemon..."
            ollama serve &>/dev/null &
            sleep 3
        fi

        MODELS=(
            "qwen2.5-vl:7b-instruct"   # Primary vision-language model
            "nomic-embed-text"           # Semantic embeddings for memory
        )

        OPTIONAL_MODELS=(
            "llama-guard3:8b"            # Tier-4 safety (LlamaGuard)
        )

        for model in "${MODELS[@]}"; do
            info "Pulling required model: $model (this may take several minutes)..."
            ollama pull "$model" || warn "Failed to pull $model"
            success "Model ready: $model"
        done

        for model in "${OPTIONAL_MODELS[@]}"; do
            info "Pulling optional model: $model..."
            ollama pull "$model" 2>/dev/null || warn "Optional model $model not pulled (non-fatal)"
        done
    fi
fi

# ── Step 6: OmniParser weights ───────────────────────────────────────────────
if [[ "$SKIP_WEIGHTS" == "true" ]]; then
    warn "Skipping OmniParser weight download (--skip-weights)"
else
    header "Step 6: OmniParser V2 + GUI-Actor weights"
    if [[ -f "scripts/install_weights.sh" ]]; then
        bash scripts/install_weights.sh || warn "Weight download failed (non-fatal — VL model fallback active)"
    else
        warn "scripts/install_weights.sh not found — create it or download weights manually"
    fi
fi

# ── Step 7: Docker services ──────────────────────────────────────────────────
if [[ "$SKIP_DOCKER" == "true" ]]; then
    warn "Skipping Docker services (--skip-docker)"
else
    header "Step 7: Docker services (Qdrant + FalkorDB)"

    if ! command -v docker &>/dev/null; then
        warn "Docker not installed — memory vector DB and Graphiti KG will use fallbacks"
        warn "Install Docker: https://docs.docker.com/get-docker/"
    else
        if [[ ! -f "docker-compose.yml" ]]; then
            warn "docker-compose.yml not found — skipping service start"
        else
            info "Starting Qdrant, FalkorDB, Redis..."
            docker compose up -d 2>/dev/null || docker-compose up -d 2>/dev/null || \
                warn "Docker Compose failed — start manually: docker-compose up -d"

            # Wait for health
            info "Waiting for services to be healthy..."
            sleep 5

            # Check Qdrant
            if curl -sf http://localhost:6333/healthz &>/dev/null; then
                success "Qdrant healthy at http://localhost:6333"
            else
                warn "Qdrant not responding yet — may still be starting"
            fi

            # Check FalkorDB
            if command -v redis-cli &>/dev/null; then
                if redis-cli -p 6379 ping 2>/dev/null | grep -q PONG; then
                    success "FalkorDB (Redis-compat) healthy at port 6379"
                else
                    warn "FalkorDB not responding yet"
                fi
            fi
        fi
    fi
fi

# ── Step 8: Environment file ─────────────────────────────────────────────────
header "Step 8: Environment configuration"

if [[ ! -f ".env" ]]; then
    if [[ -f ".env.example" ]]; then
        cp .env.example .env
        success "Created .env from .env.example"
        warn "IMPORTANT: Edit .env to set your API keys and configuration!"
    else
        warn ".env.example not found — create .env manually"
    fi
else
    info ".env already exists — not overwriting"
fi

# ── Step 9: Connectivity checks ──────────────────────────────────────────────
header "Step 9: Connectivity checks"

# Python import checks
CHECKS=(
    "import anthropic; print('anthropic OK')"
    "import ollama; print('ollama OK')"
    "import pyautogui; print('pyautogui OK')"
    "import numpy; print('numpy OK')"
    "import PIL; print('Pillow OK')"
    "import psutil; print('psutil OK')"
    "import yaml; print('pyyaml OK')"
    "import imagehash; print('imagehash OK')"
    "import watchdog; print('watchdog OK')"
    "import mem0; print('mem0ai OK')"
    "import faiss; print('faiss-cpu OK')"
    "import scipy; print('scipy OK')"
    "import transformers; print('transformers OK')"
)

for check in "${CHECKS[@]}"; do
    pkg=$(echo "$check" | awk '{print $2}' | tr -d ';')
    if python3 -c "$check" 2>/dev/null; then
        success "$pkg"
    else
        warn "MISSING: $pkg — $(echo "$check" | sed 's/^import //;s/;.*//')"
    fi
done

# Optional checks
OPTIONAL_CHECKS=(
    "import pymdp; print('pymdp (Active Inference) OK')"
    "import spikingjelly; print('spikingjelly (SNN) OK')"
    "import graphiti; print('graphiti-core OK')"
    "import cognee; print('cognee OK')"
    "import docker; print('docker-sdk OK')"
    "import langgraph; print('langgraph OK')"
    "import torch; print(f'PyTorch OK (CUDA={__import__(\"torch\").cuda.is_available()})')"
)

echo ""
info "Optional components:"
for check in "${OPTIONAL_CHECKS[@]}"; do
    if python3 -c "$check" 2>/dev/null; then
        success "  $check" | sed 's/.*print(//;s/).*//'
    else
        warn "  NOT available: $(echo "$check" | sed 's/^import //;s/;.*//')"
    fi
done

# ── Done ─────────────────────────────────────────────────────────────────────
header "Setup Complete"

echo -e "${GREEN}${BOLD}ProjectZeo GII setup finished!${NC}"
echo ""
echo "Next steps:"
echo "  1. Edit .env to add your API keys"
echo "  2. Start the agent:"
echo "     # Local Ollama:"
echo "     python run.py qwen2.5-vl:7b-instruct"
echo ""
echo "     # Anthropic Claude:"
echo "     python run.py anthropic:claude-sonnet-4-20250514 --allow-cloud"
echo ""
echo "     # OpenAI GPT-4o:"
echo "     python run.py openai:gpt-4o --allow-cloud"
echo ""
echo "  3. Write an intent to arm the agent:"
echo "     echo 'Open a text editor and type Hello World' > temp/arm.intent"
echo ""
echo "  4. Check status:"
echo "     python run.py --status"
echo ""

if [[ "$SKIP_DOCKER" == "false" ]] && command -v docker &>/dev/null; then
    echo "Services running:"
    echo "  Qdrant:   http://localhost:6333/dashboard"
    echo "  FalkorDB: http://localhost:3000"
    echo ""
fi

echo -e "${YELLOW}${BOLD}IMPORTANT: For GPU self-improvement loop:${NC}"
echo "  bash setup.sh --gpu  (requires CUDA 12.1+ and ≥16GB VRAM)"
