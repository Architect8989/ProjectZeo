#!/usr/bin/env bash


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
INTENT_DIR="$PROJECT_ROOT/temp"
INTENT_FILE="$INTENT_DIR/arm_system.intent"

mkdir -p "$INTENT_DIR"

if [[ $# -ge 1 ]]; then
    INTENT_TEXT="$*"
else
    echo "Enter intent (task for ProjectZeo to perform):"
    read -r INTENT_TEXT
fi

if [[ -z "$INTENT_TEXT" ]]; then
    echo "Error: intent text must be non-empty" >&2
    exit 1
fi

if [[ ${#INTENT_TEXT} -gt 4096 ]]; then
    echo "Error: intent text exceeds 4096 byte limit" >&2
    exit 1
fi

# Atomic write: write to temp, then rename
TMP_FILE="$(mktemp "$INTENT_DIR/.intent_XXXXXX")"
echo -n "$INTENT_TEXT" > "$TMP_FILE"
mv "$TMP_FILE" "$INTENT_FILE"

echo "[deploy_intent] Intent written to: $INTENT_FILE"
echo "[deploy_intent] Content: $INTENT_TEXT"
echo "[deploy_intent] ProjectZeo will arm when it next polls (within 100ms)."
