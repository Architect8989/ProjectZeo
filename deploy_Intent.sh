set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTENT_FILE="$SCRIPT_DIR/arm_system.intent"

# Sidecar dir (informational only — written by IntentListener, not here)
SIDECAR_DIR="$SCRIPT_DIR/temp"

# ------------------------------------------------------------------
# Collect intent text
# ------------------------------------------------------------------

if [[ $# -ge 1 ]]; then
    INTENT_TEXT="$*"
elif [[ ! -t 0 ]]; then
    # Non-interactive stdin (pipe, heredoc)
    INTENT_TEXT="$(cat)"
else
    printf "Enter intent (task for ProjectZeo to perform):\n> "
    read -r INTENT_TEXT
fi

INTENT_STRIPPED="$(printf '%s' "$INTENT_TEXT" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"

if [[ -z "$INTENT_STRIPPED" ]]; then
    printf "[deploy_intent] ERROR: intent text must be non-empty.\n" >&2
    exit 1
fi

# Byte-length check (IntentListener.INTENT_MAX_BYTES = 4096)
INTENT_BYTES="$(printf '%s' "$INTENT_TEXT" | wc -c)"
if [[ "$INTENT_BYTES" -gt 4096 ]]; then
    printf "[deploy_intent] ERROR: intent text is %d bytes; limit is 4096 bytes.\n" \
        "$INTENT_BYTES" >&2
    exit 1
fi



UPPER_INTENT="$(printf '%s' "$INTENT_TEXT" | cut -c1-4 | tr '[:lower:]' '[:upper:]')"
if [[ "$UPPER_INTENT" != "ARM:" ]]; then
    INTENT_TEXT="ARM: $INTENT_TEXT"
fi


mkdir -p "$(dirname "$INTENT_FILE")"

TMP_FILE="$(mktemp "$SCRIPT_DIR/.intent_XXXXXX")"

# Trap ensures temp file is removed if the script exits early.
trap 'rm -f "$TMP_FILE"' EXIT

printf '%s' "$INTENT_TEXT" > "$TMP_FILE"


chmod 644 "$TMP_FILE"

# Atomic rename — the intent file is never visible with a wrong mode.
mv "$TMP_FILE" "$INTENT_FILE"


if [[ ! -f "$INTENT_FILE" ]]; then
    printf "[deploy_intent] ERROR: intent file not found after write: %s\n" \
        "$INTENT_FILE" >&2
    exit 1
fi

if [[ ! -r "$INTENT_FILE" ]]; then
    printf "[deploy_intent] ERROR: intent file not readable after write: %s\n" \
        "$INTENT_FILE" >&2
    exit 1
fi

# ------------------------------------------------------------------
# Success output
# ------------------------------------------------------------------

printf "[deploy_intent] Intent written successfully.\n"
printf "  Path    : %s\n" "$INTENT_FILE"
printf "  Content : %s\n" "$INTENT_TEXT"
printf "  Size    : %d bytes\n" "$INTENT_BYTES"
printf "[deploy_intent] ProjectZeo will arm within ~100ms (IntentListener poll interval).\n"
printf "[deploy_intent] Monitor: tail -f %s/arm_failure.json %s/arm_success.json\n" \
    "$SIDECAR_DIR" "$SIDECAR_DIR"
