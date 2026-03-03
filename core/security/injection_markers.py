import unicodedata
import re


def normalize_for_injection_check(text: str) -> str:
    if not isinstance(text, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in nfkd if unicodedata.category(ch) != "Mn")
    nfkc = unicodedata.normalize("NFKC", stripped)
    ascii_bytes = nfkc.encode("ascii", errors="ignore")
    return ascii_bytes.decode("ascii").lower()


INJECTION_MARKERS: frozenset = frozenset({
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

    "new instruction:",
    "new instructions:",
    "updated instruction:",
    "revised instruction:",
    "important instruction:",

    "system prompt:",
    "new system prompt:",
    "<|system|>",
    "[system]",
    "###system",
    "### system",

    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",

    "[inst]",
    "</inst>",
    "[/inst]",
    "<s>[inst]",
    "</s>",

    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",

    "<start_of_turn>",
    "<end_of_turn>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",

    "act as",
    "pretend you are",
    "pretend to be",
    "you are now",
    "from now on",
    "your new role",
    "your task is now",

    "do anything now",
    "developer mode",
    "jailbreak mode",
    "switch mode",
    "enable developer",
    "bypass restrictions",
    "ignore your training",
    "ignore your guidelines",
    "without restrictions",
    "no restrictions",
    "unrestricted mode",

    "actually, let me reconsider",
    "wait, new instruction",
    "correction: ignore",

    "```system",
    "---system",
    "=== system",

    "```python\nimport os",
    "```bash\nrm",
    "```shell\nrm",
    "exec(base64",
    "eval(base64",

    "human turn:",
    "ai turn:",
    "bot response:",
    "[end of conversation]",
    "[new conversation]",
    "--- end of system ---",
    "--- begin user ---",
})


_WORD_BOUNDARY_MARKERS: frozenset = frozenset({
    "system:",
    "user:",
    "assistant:",
    "human:",
    "bot:",
})


_BOUNDARY_PATTERN = re.compile(
    r"(?:^|[\n.!?;|{])\s*("
    + "|".join(re.escape(m) for m in sorted(_WORD_BOUNDARY_MARKERS))
    + r")",
    re.IGNORECASE,
)


def contains_injection_marker(text: str) -> bool:
    normalised = normalize_for_injection_check(text)
    if not normalised:
        return False

    if any(marker in normalised for marker in INJECTION_MARKERS):
        return True

    if _BOUNDARY_PATTERN.search(normalised):
        return True

    return False
