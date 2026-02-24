INJECTION_MARKERS: frozenset = frozenset({
    # Classic override phrases
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

    # System prompt / role injection
    "system:",
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
    "assistant:",
    "human:",
    "user:",

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

