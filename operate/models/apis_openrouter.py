from __future__ import annotations

import os
import json
import asyncio
import uuid
import httpx
from typing import List, Tuple, Dict, Optional

# ============================================================
# HARD CONSTRAINTS
# ============================================================
# - OpenRouter ONLY
# - No OpenAI SDK
# - No Gemini SDK
# - No silent fallback
# - Fail hard on any deviation
# ============================================================

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Allowlist ONLY
ALLOWED_MODELS = {
    "openai/gpt-4o-mini",
    "qwen/qwen2.5-vl-72b-instruct",
    "anthropic/claude-3.5-sonnet",
}

# ============================================================
# INTERNAL GUARDS
# ============================================================

SYSTEM_GUARD_PROMPT = (
    "You are a computer-operating agent.\n"
    "Return ONLY valid JSON.\n"
    "Top-level JSON schema:\n"
    "{\n"
    '  "operations": [\n'
    '    {"operation": "...", "thought": "...", "...": "..."}\n'
    "  ]\n"
    "}\n"
    "No prose.\n"
    "No markdown.\n"
    "No explanations.\n"
)

REQUEST_TIMEOUT_SECONDS = 60
MAX_RESPONSE_BYTES = 2_000_000  # 2MB hard ceiling


def _get_api_key() -> str:
    """
    PATCH §1.13: Deferred key resolution — checked at call time, NOT at import.

    Previously the module-level guard:
        if not OPENROUTER_API_KEY:
            raise RuntimeError(...)
    crashed the entire process on import when no cloud key was set.
    The Ollama-only path never needs OpenRouter — crashing on import is wrong.
    """
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. "
            "Export it before using the OpenRouter backend."
        )
    return key


def _build_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {_get_api_key()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost",
        "X-Title": "Self-Operating-Computer",
    }


# ============================================================
# CORE ENTRYPOINT (SOC CONTRACT)
# ============================================================

async def get_next_action(
    model: str,
    messages: List[Dict],
    objective: str,
    session_id: Optional[str] = None,
) -> Tuple[List[Dict], str]:
    """
    Must return:
      - operations: List[dict]
      - session_id: str

    PATCH: _get_api_key() is called here (call time), not at import.
    """

    if model not in ALLOWED_MODELS:
        raise ValueError(f"Model not supported by OpenRouter engine: {model}")

    system_guard = {
        "role": "system",
        "content": SYSTEM_GUARD_PROMPT,
    }

    payload = {
        "model": model,
        "messages": [system_guard] + messages,
        "temperature": 0,  # HAR-02: deterministic — matches Ollama paths (was 0.2)
        "max_tokens": 1024,
    }

    headers = _build_headers()  # raises RuntimeError if key absent

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        resp = await client.post(
            OPENROUTER_BASE_URL,
            headers=headers,
            json=payload,
        )

    if resp.status_code != 200:
        raise RuntimeError(
            f"OpenRouter API failure {resp.status_code}: {resp.text}"
        )

    # PATCH: enforce byte limit before attempting json.loads to prevent OOM
    raw_bytes = resp.content
    if len(raw_bytes) > MAX_RESPONSE_BYTES:
        raise RuntimeError("OpenRouter response too large")

    raw_text = raw_bytes.decode("utf-8", errors="replace")

    try:
        data = json.loads(raw_text)
    except Exception:
        raise RuntimeError(f"Non-JSON response from OpenRouter: {raw_text[:500]}")

    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"Malformed OpenRouter response: {data}")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        raise RuntimeError(f"Model did not return JSON:\n{content}")

    operations = parsed.get("operations")
    if not isinstance(operations, list):
        raise RuntimeError(f"Invalid operations payload: {parsed}")

    new_session_id = session_id or _generate_session_id()

    return operations, new_session_id


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _generate_session_id() -> str:
    # PATCH: previously used asyncio.get_event_loop().time() which is deprecated
    # in Python 3.10+ and raises DeprecationWarning.  Use uuid4 instead.
    return f"soc-openrouter-{uuid.uuid4().hex[:12]}"
