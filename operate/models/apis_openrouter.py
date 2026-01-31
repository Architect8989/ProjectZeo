# operate/models/apis_openrouter.py

import os
import json
import asyncio
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

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Allowlist ONLY
ALLOWED_MODELS = {
    "openai/gpt-4o-mini",
    "qwen/qwen2.5-vl-72b-instruct",
    "anthropic/claude-3.5-sonnet",
}

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    # Recommended by OpenRouter
    "HTTP-Referer": "https://localhost",
    "X-Title": "Self-Operating-Computer",
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
        "temperature": 0.2,
        "max_tokens": 1024,
    }

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        resp = await client.post(
            OPENROUTER_BASE_URL,
            headers=HEADERS,
            json=payload,
        )

    if resp.status_code != 200:
        raise RuntimeError(
            f"OpenRouter API failure {resp.status_code}: {resp.text}"
        )

    raw_text = resp.text
    if len(raw_text.encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise RuntimeError("OpenRouter response too large")

    try:
        data = resp.json()
    except Exception:
        raise RuntimeError(f"Non-JSON response from OpenRouter: {raw_text}")

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
    return f"soc-openrouter-{asyncio.get_event_loop().time()}"
