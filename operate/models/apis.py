from __future__ import annotations

import os as _os


_raw = _os.environ.get("OLLAMA_ONLY", "1").strip().lower()
if _raw in ("1", "true", "yes"):
    raise ImportError(
        "operate.models.apis cannot be imported when OLLAMA_ONLY is set. "
        "Use QwenOllamaAdapter (adapters/qwen_ollama_adapter.py) for local inference. "
        "Cloud API wrapper functions are disabled when OLLAMA_ONLY=1."
    )
del _raw


def _legacy():
    """Lazy import of legacy apis — called at function invocation time only."""
    from operate.legacy import apis as _apis  # noqa: PLC0415
    return _apis


# ---------------------------------------------------------------------------
# Synchronous helpers
# ---------------------------------------------------------------------------

def call_gpt_4o(messages):
    return _legacy().call_gpt_4o(messages)


def call_gemini_pro_vision(messages, objective):
    return _legacy().call_gemini_pro_vision(messages, objective)


def call_ollama_llava(messages):
    return _legacy().call_ollama_llava(messages)


def clean_json(content: str) -> str:
    return _legacy().clean_json(content)


def get_last_assistant_message(messages):
    return _legacy().get_last_assistant_message(messages)


def confirm_system_prompt(messages, objective, model):
    return _legacy().confirm_system_prompt(messages, objective, model)


# ---------------------------------------------------------------------------
# Async helpers — preserved as coroutines (PureLLMWrapper detects via inspect)
# ---------------------------------------------------------------------------

async def call_qwen_vl_with_ocr(messages, objective, model):
    return await _legacy().call_qwen_vl_with_ocr(messages, objective, model)


async def call_gpt_4o_with_ocr(messages, objective, model):
    return await _legacy().call_gpt_4o_with_ocr(messages, objective, model)


async def call_gpt_4_1_with_ocr(messages, objective, model):
    return await _legacy().call_gpt_4_1_with_ocr(messages, objective, model)


async def call_o1_with_ocr(messages, objective, model):
    return await _legacy().call_o1_with_ocr(messages, objective, model)


# §Evo3: was missing — caused AttributeError when 'claude-3' model was requested
async def call_claude_3_with_ocr(messages, objective, model):
    return await _legacy().call_claude_3_with_ocr(messages, objective, model)


async def call_gpt_4o_labeled(messages, objective, model):
    return await _legacy().call_gpt_4o_labeled(messages, objective, model)
