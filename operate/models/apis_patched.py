"""
operate/models/apis.py
=======================
Canonical API namespace for all LLM model call functions.

DESIGN:
  - Re-exports all callable API functions from operate/legacy/apis.py
    under the canonical operate.models.apis namespace.
  - Fully LAZY: importing this module has zero side effects.
    No SDK is loaded until the specific function is called.
  - Provider-agnostic: zero references to any cloud SDK here.

EXTENSION:
  To add a new model function:
    1. Add implementation in operate/legacy/apis.py.
    2. Add a delegation wrapper here.
    3. Register the model in adapters/factory.py _CLOUD_REGISTRY.
    4. Add to adapters/pure_llm_wrapper.py _resolve_model_function() registry.

PATCHES APPLIED:
  ✅  §1.2 (prior): File created (was absent — ImportError in pure_llm_wrapper).
  ✅  §Evo3 (prior): call_claude_3_with_ocr added (was missing — AttributeError).
"""

from __future__ import annotations


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
