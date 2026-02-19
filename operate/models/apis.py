"""
operate/models/apis.py
======================
PATCHES APPLIED (Audit Fixes):

  ✅  §Evo3 (was §apis-2): call_claude_3_with_ocr was referenced in
           pure_llm_wrapper.py registry but not exported here.
           Specifying 'claude-3' as the model raised AttributeError at
           runtime.  Now exported.

  ✅  §1.2 (original): File was ABSENT from the codebase.
           adapters/pure_llm_wrapper.py imports `from operate.models import apis`
           which raised ImportError at startup, making PureLLMWrapper dead code.

DESIGN:
  - Re-exports all callable API functions from operate/legacy/apis.py under the
    canonical operate.models.apis namespace so existing imports resolve.
  - Does NOT duplicate logic — every function is a thin delegation to legacy.
  - Lazy-import guard: importing this module is safe even when cloud keys
    are absent. Functions only fail at *call time* if a key is missing.
  - Preserves full async contract expected by PureLLMWrapper._call_with_signature().
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# LAZY DELEGATION — every name is imported from legacy on first access.
# This keeps module-level side-effects to zero.
# ---------------------------------------------------------------------------

def _lazy_legacy():
    """Return the legacy apis module, imported on first call."""
    from operate.legacy import apis as _legacy  # noqa: PLC0415
    return _legacy


# ---------------------------------------------------------------------------
# Synchronous helpers
# ---------------------------------------------------------------------------

def call_gpt_4o(messages):
    return _lazy_legacy().call_gpt_4o(messages)


def call_gemini_pro_vision(messages, objective):
    return _lazy_legacy().call_gemini_pro_vision(messages, objective)


def call_ollama_llava(messages):
    return _lazy_legacy().call_ollama_llava(messages)


def clean_json(content: str) -> str:
    return _lazy_legacy().clean_json(content)


def get_last_assistant_message(messages):
    return _lazy_legacy().get_last_assistant_message(messages)


def confirm_system_prompt(messages, objective, model):
    return _lazy_legacy().confirm_system_prompt(messages, objective, model)


# ---------------------------------------------------------------------------
# Async helpers (preserved as coroutines — PureLLMWrapper detects via inspect)
# ---------------------------------------------------------------------------

async def call_qwen_vl_with_ocr(messages, objective, model):
    return await _lazy_legacy().call_qwen_vl_with_ocr(messages, objective, model)


async def call_gpt_4o_with_ocr(messages, objective, model):
    return await _lazy_legacy().call_gpt_4o_with_ocr(messages, objective, model)


async def call_gpt_4_1_with_ocr(messages, objective, model):
    return await _lazy_legacy().call_gpt_4_1_with_ocr(messages, objective, model)


async def call_o1_with_ocr(messages, objective, model):
    return await _lazy_legacy().call_o1_with_ocr(messages, objective, model)


# PATCH §Evo3: export call_claude_3_with_ocr — was missing, caused AttributeError
# when pure_llm_wrapper.py tried to resolve model "claude-3".
async def call_claude_3_with_ocr(messages, objective, model):
    return await _lazy_legacy().call_claude_3_with_ocr(messages, objective, model)


# ---------------------------------------------------------------------------
# gpt-4o-labeled (used by pure_llm_wrapper registry for "gpt-4o-labeled")
# ---------------------------------------------------------------------------

async def call_gpt_4o_labeled(messages, objective, model):
    return await _lazy_legacy().call_gpt_4o_labeled(messages, objective, model)
