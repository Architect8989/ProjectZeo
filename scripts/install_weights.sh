#!/usr/bin/env bash

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

info()    { echo -e "${BLUE}[weights]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }

# ── Target directory ─────────────────────────────────────────────────────────
WEIGHTS_DIR="${PROJECTZEO_WEIGHTS_DIR:-$HOME/.projectzeo/weights}"
OMNIPARSER_DIR="$WEIGHTS_DIR/omniparser"
DINO_DIR="$WEIGHTS_DIR/grounding_dino"
GUI_ACTOR_DIR="$WEIGHTS_DIR/gui_actor"

mkdir -p "$OMNIPARSER_DIR" "$DINO_DIR" "$GUI_ACTOR_DIR"

# ── Argument parsing ─────────────────────────────────────────────────────────
OMNIPARSER_ONLY=false
DINO_ONLY=false
CHECK_ONLY=false

for arg in "$@"; do
    case $arg in
        --omniparser-only) OMNIPARSER_ONLY=true ;;
        --dino-only)       DINO_ONLY=true ;;
        --check)           CHECK_ONLY=true ;;
    esac
done

# ── Check existing weights ───────────────────────────────────────────────────
check_weights() {
    echo ""
    info "Checking installed weights..."

    # OmniParser V2
    if ls "$OMNIPARSER_DIR"/*.safetensors &>/dev/null 2>&1 || \
       ls "$OMNIPARSER_DIR"/*.bin &>/dev/null 2>&1 || \
       [[ -f "$OMNIPARSER_DIR/config.json" ]]; then
        success "OmniParser V2: $OMNIPARSER_DIR"
    else
        warn "OmniParser V2: NOT FOUND ($OMNIPARSER_DIR is empty)"
    fi

    # Grounding DINO
    if ls "$DINO_DIR"/*.pth &>/dev/null 2>&1 || \
       ls "$DINO_DIR"/*.safetensors &>/dev/null 2>&1 || \
       ls "$DINO_DIR"/*.bin &>/dev/null 2>&1; then
        success "Grounding DINO: $DINO_DIR"
    else
        warn "Grounding DINO: NOT FOUND ($DINO_DIR is empty)"
    fi

    # GUI-Actor
    if [[ -f "$GUI_ACTOR_DIR/config.json" ]] || \
       ls "$GUI_ACTOR_DIR"/*.bin &>/dev/null 2>&1; then
        success "GUI-Actor: $GUI_ACTOR_DIR"
    else
        warn "GUI-Actor: NOT FOUND ($GUI_ACTOR_DIR is empty)"
    fi

    echo ""
}

if [[ "$CHECK_ONLY" == "true" ]]; then
    check_weights
    exit 0
fi

# ── Download function ─────────────────────────────────────────────────────────
download_file() {
    local url="$1"
    local dest="$2"
    local name="$(basename "$dest")"

    if [[ -f "$dest" ]]; then
        info "Already exists: $name (skipping)"
        return 0
    fi

    info "Downloading: $name"
    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "$dest" "$url" || {
            error "Download failed: $url"
            rm -f "$dest"
            return 1
        }
    elif command -v curl &>/dev/null; then
        curl -fL --progress-bar -o "$dest" "$url" || {
            error "Download failed: $url"
            rm -f "$dest"
            return 1
        }
    else
        error "Neither wget nor curl found. Install one and retry."
        return 1
    fi
    success "Downloaded: $name"
}

# ── Python HuggingFace download ───────────────────────────────────────────────
hf_snapshot_download() {
    local repo_id="$1"
    local local_dir="$2"
    local description="$3"

    info "Downloading $description from Hugging Face: $repo_id"
    info "Target: $local_dir"

    python3 - <<PYTHON
import sys, os
try:
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="$repo_id",
        local_dir="$local_dir",
        ignore_patterns=["*.pt", "*.gguf", "optimizer*"],
        local_files_only=False,
    )
    print(f"[OK] Downloaded $repo_id to $local_dir")
except ImportError:
    print("[WARN] huggingface_hub not installed, trying pip install...")
    os.system("pip install huggingface_hub -q")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="$repo_id",
        local_dir="$local_dir",
        ignore_patterns=["*.pt", "*.gguf", "optimizer*"],
    )
except Exception as e:
    print(f"[ERROR] Download failed: {e}", file=sys.stderr)
    sys.exit(1)
PYTHON
}

# ── OmniParser V2 ─────────────────────────────────────────────────────────────
install_omniparser() {
    info "=== Installing OmniParser V2 ==="

    # Check if already installed
    if [[ -f "$OMNIPARSER_DIR/config.json" ]]; then
        success "OmniParser V2 already installed at $OMNIPARSER_DIR"
        return 0
    fi

    # Primary: Microsoft OmniParser on HuggingFace
    info "Downloading OmniParser V2 from microsoft/OmniParser-v2.0..."
    warn "This is approximately 2-4 GB — may take several minutes on slow connections"

    hf_snapshot_download \
        "microsoft/OmniParser-v2.0" \
        "$OMNIPARSER_DIR" \
        "OmniParser V2" && {
        success "OmniParser V2 installed at $OMNIPARSER_DIR"
        return 0
    }

    # Fallback: Try the older v1 if v2 fails
    warn "V2 download failed, trying OmniParser v1..."
    hf_snapshot_download \
        "microsoft/OmniParser" \
        "$OMNIPARSER_DIR" \
        "OmniParser v1 (fallback)" && {
        success "OmniParser v1 installed at $OMNIPARSER_DIR"
        return 0
    }

    error "OmniParser download failed. VL model fallback will be used."
    return 1
}

# ── Grounding DINO + SAM2 ─────────────────────────────────────────────────────
install_dino() {
    info "=== Installing Grounding DINO ==="

    if ls "$DINO_DIR"/*.pth &>/dev/null 2>&1; then
        success "Grounding DINO already installed"
        return 0
    fi

    # Grounding DINO 1.5 Pro config and weights
    DINO_CONFIG_URL="https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    DINO_WEIGHTS_URL="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"

    mkdir -p "$DINO_DIR"
    download_file "$DINO_CONFIG_URL" "$DINO_DIR/GroundingDINO_SwinT_OGC.py" || true
    download_file "$DINO_WEIGHTS_URL" "$DINO_DIR/groundingdino_swint_ogc.pth" || {
        warn "Grounding DINO download failed (non-fatal) — OmniParser fallback active"
        return 1
    }

    success "Grounding DINO installed at $DINO_DIR"
}

# ── GUI-Actor ─────────────────────────────────────────────────────────────────
install_gui_actor() {
    info "=== Installing GUI-Actor ==="

    if [[ -f "$GUI_ACTOR_DIR/config.json" ]]; then
        success "GUI-Actor already installed"
        return 0
    fi

    # GUI-Actor from HuggingFace
    hf_snapshot_download \
        "showlab/GUI-Actor-2B-v0.1" \
        "$GUI_ACTOR_DIR" \
        "GUI-Actor 2B" && {
        success "GUI-Actor installed at $GUI_ACTOR_DIR"
        return 0
    }

    warn "GUI-Actor download failed — OmniParser+VL model fallback active"
    return 1
}

# ── Create weight manifest ───────────────────────────────────────────────────
write_manifest() {
    cat > "$WEIGHTS_DIR/manifest.json" <<EOF
{
  "installed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "weights_dir": "$WEIGHTS_DIR",
  "components": {
    "omniparser": {
      "path": "$OMNIPARSER_DIR",
      "installed": $(ls "$OMNIPARSER_DIR"/*.json &>/dev/null 2>&1 && echo "true" || echo "false")
    },
    "grounding_dino": {
      "path": "$DINO_DIR",
      "installed": $(ls "$DINO_DIR"/*.pth &>/dev/null 2>&1 && echo "true" || echo "false")
    },
    "gui_actor": {
      "path": "$GUI_ACTOR_DIR",
      "installed": $(ls "$GUI_ACTOR_DIR"/*.json &>/dev/null 2>&1 && echo "true" || echo "false")
    }
  }
}
EOF
    success "Wrote weight manifest: $WEIGHTS_DIR/manifest.json"
}

# ── Create omniparser path config ────────────────────────────────────────────
write_omniparser_config() {
    python3 - <<PYTHON
import json, os, pathlib

weights_dir = os.path.expanduser("$WEIGHTS_DIR")
config = {
    "weights_dir": weights_dir,
    "omniparser_dir": os.path.join(weights_dir, "omniparser"),
    "dino_dir": os.path.join(weights_dir, "grounding_dino"),
    "gui_actor_dir": os.path.join(weights_dir, "gui_actor"),
    "dino_config": os.path.join(weights_dir, "grounding_dino", "GroundingDINO_SwinT_OGC.py"),
    "dino_checkpoint": os.path.join(weights_dir, "grounding_dino", "groundingdino_swint_ogc.pth"),
}

config_path = pathlib.Path(os.path.expanduser("~/.projectzeo/weights_config.json"))
config_path.parent.mkdir(parents=True, exist_ok=True)
config_path.write_text(json.dumps(config, indent=2))
print(f"[OK] Weights config written: {config_path}")
PYTHON
}

# ── Main execution ────────────────────────────────────────────────────────────
main() {
    echo ""
    info "ProjectZeo GII — Weight Installer"
    info "Target directory: $WEIGHTS_DIR"
    echo ""

    if [[ "$DINO_ONLY" == "true" ]]; then
        install_dino
    elif [[ "$OMNIPARSER_ONLY" == "true" ]]; then
        install_omniparser
    else
        # Install all
        install_omniparser || true
        install_dino       || true
        install_gui_actor  || true
    fi

    write_manifest
    write_omniparser_config
    check_weights

    echo ""
    success "Weight installation complete!"
    echo ""
    info "To verify weights are loaded correctly:"
    echo "  python3 -c \"from core.perception.omniparser import OmniParser; p = OmniParser(); print(p.status())\""
    echo ""
    info "If weights are missing, ProjectZeo uses VL model (Qwen2.5-VL) as fallback."
    echo "  Set PROJECTZEO_WEIGHTS_DIR in .env to customize the weights directory."
}

main

