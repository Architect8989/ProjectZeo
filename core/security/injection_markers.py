import unicodedata
import re


def normalize_for_injection_check(text: str) -> str:
    
    if not isinstance(text, str):
        return ""
    # NFKD decomposes characters into base + combining marks.
    # Encoding to ASCII with 'ignore' then strips any non-ASCII residuals.
    normalized = unicodedata.normalize("NFKD", text)
    ascii_bytes = normalized.encode("ascii", errors="ignore")
    return ascii_bytes.decode("ascii").lower()


# H-8 FIX: HIGH-FALSE-POSITIVE MARKERS REMOVED — "system:", "user:", "assistant:"
#
# Root cause: these three bare markers matched legitimate UI strings:
#   - "system:"     → matches "File System:", "Operating System:", "Subsystem:"
#   - "user:"       → matches "Username:", "User: admin", any dialog with "User:"
#   - "assistant:"  → matches any chat UI label showing "Assistant:"
#
# When contains_injection_marker() returned True for legitimate content, the
# safety layer silently suppressed that entity text from LLM prompts. The LLM
# then received an incomplete world description and stagnated — with only a
# logger.warning() as the diagnostic. Tasks on system-administration UIs
# (terminal, file managers, user-management dialogs) were disproportionately
# affected because these UIs display the high-FP words constantly.
#
# Fix: the bare markers are replaced with two-tier detection:
#
#   Tier 1 — INJECTION_MARKERS (frozenset):
#     Contains only zero-false-positive strings: explicit multi-word phrases
#     and delimiters that cannot appear in normal UI text. Substring matching
#     is safe for these.
#
#   Tier 2 — _WORD_BOUNDARY_MARKERS (frozenset):
#     Contains the formerly high-FP single-word role markers. These are now
#     matched with a word-boundary regex: the marker must appear at the START
#     of the text OR immediately after a newline/sentence-end character. This
#     prevents "File System: C:\\" from triggering while still catching
#     "system: ignore all previous instructions" (role-injection preamble).
#
# contains_injection_marker() is updated to apply both tiers.
INJECTION_MARKERS: frozenset = frozenset({
    # Classic override phrases (zero false-positive risk)
    "ignore previous instructions",
    "ignore all previous",
    "ignore prior instructions",
    "disregard previous",
    "disregard all previous",
    "disregard prior instructions",
    "disregard the above",
    "forget previous instructions",
    "forget all previous",
    "override previous instructions",
    "override the above",

    # "New instruction" injection patterns
    "new instruction:",
    "new instructions:",
    "updated instruction:",
    "revised instruction:",
    "important instruction:",

    # System prompt / role injection — explicit multi-word forms (safe)
    "system prompt:",
    "new system prompt:",
    "<|system|>",
    "[system]",
    "###system",
    "### system",

    # LLM template delimiters used in fine-tuned / instruction-tuned models
    "</s>",
    "[inst]",
    "</inst>",
    "[/inst]",
    "<s>[inst]",
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",

    # Jailbreak prefix patterns
    "act as",
    "pretend you are",
    "pretend to be",
    "you are now",
    "from now on",
    "your new role",
    "your task is now",

    # Escaping / structural attacks
    "```system",
    "---system",
    "=== system",
})

# Tier-2: formerly high-FP role-label markers.
# Matched only at line/sentence start (word-boundary pattern) to avoid matching
# legitimate UI labels like "Username:", "File System:", "Assistant Manager:".
_WORD_BOUNDARY_MARKERS: frozenset = frozenset({
    "system:",
    "user:",
    "assistant:",
    "human:",
})

# Precompiled pattern: marker appears at start of string, after a newline,
# after a sentence-end (. ! ?), or after common separator chars (| ; {}).
# The (?:...) group anchors to these positions before the literal marker text.
_BOUNDARY_PATTERN = re.compile(
    r"(?:^|[\n.!?;|{])\s*(" +
    "|".join(re.escape(m) for m in sorted(_WORD_BOUNDARY_MARKERS)) +
    r")",
    re.IGNORECASE,
)


def contains_injection_marker(text: str) -> bool:
    """
    H-8 FIX: Two-tier injection detection.

    Tier 1: substring check against INJECTION_MARKERS (zero-false-positive).
    Tier 2: word-boundary regex check against _WORD_BOUNDARY_MARKERS (role labels
            matched only at sentence/line start, not as substrings of longer words).

    Returns True if any injection pattern is detected, False otherwise.
    """
    normalized = normalize_for_injection_check(text)

    # Tier 1: fast substring scan
    if any(marker in normalized for marker in INJECTION_MARKERS):
        return True

    # Tier 2: context-anchored scan for role-label markers
    if _BOUNDARY_PATTERN.search(normalized):
        return True

    return False
