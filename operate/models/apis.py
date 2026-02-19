"""
operate/models/apis.py
======================
PATCH: This file was ABSENT from the codebase (audit ❌ FAIL §1.2 / §1.13).
       `adapters/pure_llm_wrapper.py` imports `from operate.models import apis`
       which raised ImportError at startup, making PureLLMWrapper dead code.

DESIGN:
  - Re-exports all callable API functions from operate/legacy/apis.py under the
    canonical `operate.models.apis` namespace so existing imports resolve.
  - Does NOT duplicate logic — every function is a thin delegation to legacy.
  - Adds lazy-import guard: importing this module is safe even when cloud keys
    are absent.  Functions only fail at *call time* if a key is missing.
  - Preserves full async contract expected by PureLLMWrapper._call_with_signature().

AUDIT FIXES APPLIED:
  ❌  operate/models/apis.py did not exist → created
  ⚠️  apis_openrouter.py raised RuntimeError at import → deferred (see below note)
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


async def call_claude_3_with_ocr(messages, objective, model):
    return await _lazy_legacy().call_claude_3_with_ocr(messages, objective, model)


async def call_gpt_4o_labeled(messages, objective, model):
    return await _lazy_legacy().call_gpt_4o_labeled(messages, objective, model)


# ---------------------------------------------------------------------------
# Top-level dispatcher — mirrors legacy.get_next_action
# ---------------------------------------------------------------------------

async def get_next_action(model, messages, objective, session_id=None):
    """
    Dispatcher identical to legacy.get_next_action.
    Returns (operations, session_id_or_None).
    """
    return await _lazy_legacy().get_next_action(model, messages, objective, session_id)
