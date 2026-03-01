"""
adapters/llava_ollama_adapter.py
=================================
GAP-3 FIX: LLaVA-specific Ollama adapter.

Root cause of GAP-3:
    factory.py mapped "llava" → QwenOllamaAdapter.
    QwenOllamaAdapter uses Qwen2.5-VL-specific prompt format:
      - Structured JSON schema in system prompt targeting Qwen2.5-VL tokens
      - get_system_prompt() / get_user_prompt() from operate/models/prompts.py
        embed Qwen-specific XML-style structured output markers
    LLaVA (1.5/1.6/next) uses plain multi-modal chat format with a simpler
    instruction-following prompt. Passing Qwen prompts to LLaVA causes:
      - Malformed JSON outputs (LLaVA ignores structure hints)
      - Constant parse errors in _parse_and_normalize_json()
      - Effectively every action decision fails → stagnation → REPLAN → FAIL

Fix: separate LLaVAOllamaAdapter with:
  - LLaVA-compatible system and user prompts (plain English instruction)
  - Same external interface (get_next_action) as QwenOllamaAdapter
  - Registered in factory._LOCAL_REGISTRY for all "llava" model variants
  - factory.py updated to route llava:* → LLaVAOllamaAdapter
"""
from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
import re
import sys
import threading
import time
import asyncio
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import httpx
import ollama

from operate.utils.screenshot import capture_screen_with_cursor, compress_screenshot

logger = logging.getLogger(__name__)

# Reuse the shared inference lock from vision_runtime (same GPU contention concern)
try:
    from core.vision.vision_runtime import get_inference_lock as _get_inference_lock
    _INFERENCE_LOCK = _get_inference_lock()
except ImportError:
    _INFERENCE_LOCK = threading.Lock()

# Reuse the shared VisionRuntime frame to avoid duplicate screenshots
_SHARED_VISION_RUNTIME = None


def set_shared_vision_runtime(runtime) -> None:
    global _SHARED_VISION_RUNTIME
    _SHARED_VISION_RUNTIME = runtime


# ---------------------------------------------------------------------------
# LLaVA-specific prompt templates
# LLaVA (1.5/1.6/next) is instruction-tuned on plain multi-turn dialogue.
# It responds best to clear, direct commands with explicit JSON schema hints
# embedded in the user message (not separate XML structural prompts).
# ---------------------------------------------------------------------------

_LLAVA_SYSTEM_PROMPT = """\
You are an intelligent computer automation agent. You control a real computer \
by issuing JSON action commands based on screenshots.

You MUST respond with ONLY a JSON array of action objects. No explanations, \
no markdown, no text outside the JSON array.

Available action types:
  {"operation": "click", "x": 0.5, "y": 0.4}          -- click at fractional screen coords
  {"operation": "write", "content": "hello"}             -- type text
  {"operation": "press", "keys": ["ctrl", "c"]}          -- key combination
  {"operation": "scroll", "direction": "down", "clicks": 3}
  {"operation": "command", "command": "ls -la"}          -- run shell command
  {"operation": "file_create", "path": "/tmp/f.txt", "content": "..."}
  {"operation": "done"}                                  -- task completed

Rules:
- x and y coordinates are 0.0-1.0 fractions of screen width/height
- Always use {"operation": "done"} as the LAST action when the task is complete
- Respond with ONLY the JSON array, nothing else
"""

_LLAVA_USER_TEMPLATE = """\
Current objective: {objective}

Look at this screenshot of the current screen state. What single action \
should be taken next to make progress toward the objective?

Respond with ONLY a JSON array, for example:
[{{"operation": "click", "x": 0.5, "y": 0.3}}]
"""


class LLaVAOllamaAdapter:
    """
    Local LLaVA adapter via Ollama.

    Compatible with llava:7b, llava:13b, llava-llama3, llava:1.6, etc.
    Uses plain instruction-following prompts suited to LLaVA's training format.
    """

    MAX_HISTORY_TURNS = 6  # LLaVA context windows are typically shorter

    def __init__(self, model_name: str = "llava:7b"):
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        self.model_name = model_name.strip()
        self._client = ollama.Client(
            timeout=httpx.Timeout(
                connect=10.0,
                read=180.0,   # LLaVA 13b can be slow on CPU
                write=5.0,
                pool=2.0,
            )
        )
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def get_next_action(
        self,
        messages: List[dict],
        objective: str,
        session_id: Optional[str] = None,
    ) -> Tuple[Optional[List[dict]], Optional[Exception]]:
        try:
            ops = await self._call_llava(messages, objective)
            return ops, None
        except Exception as exc:
            return None, exc

    async def _call_llava(
        self,
        messages: List[dict],
        objective: str,
    ) -> List[dict]:
        # Get screenshot (prefer shared VisionRuntime frame)
        img_base64: Optional[str] = None
        jpeg_tmp_name: Optional[str] = None
        _cleanup = contextlib.ExitStack()

        try:
            _vr = _SHARED_VISION_RUNTIME
            if _vr is not None:
                try:
                    img_base64 = _vr.get_latest_frame_jpeg_b64(max_age_seconds=5.0)
                except Exception:
                    img_base64 = None

            if img_base64 is None:
                _rtf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                raw_tmp = _rtf.name
                _rtf.close()
                _cleanup.callback(lambda p=raw_tmp: os.unlink(p) if os.path.exists(p) else None)

                _jtf = tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False)
                jpeg_tmp_name = _jtf.name
                _jtf.close()
                _cleanup.callback(lambda p=jpeg_tmp_name: os.unlink(p) if os.path.exists(p) else None)

                capture_screen_with_cursor(raw_tmp)
                compress_screenshot(raw_tmp, jpeg_tmp_name)
                with open(jpeg_tmp_name, "rb") as f:
                    img_base64 = base64.b64encode(f.read()).decode("utf-8")

            # Build conversation — LLaVA uses simple role/content messages
            history: List[Dict[str, Any]] = []
            for msg in messages[-(self.MAX_HISTORY_TURNS):]:
                role = msg.get("role")
                if role not in ("user", "assistant"):
                    continue
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    history.append({"role": role, "content": content.strip()})
                elif isinstance(content, list):
                    # Extract text parts only (drop old images from history)
                    parts = [
                        p.get("text", "")
                        for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    ]
                    text = "\n".join(parts).strip()
                    if text:
                        history.append({"role": role, "content": text})

            user_prompt = _LLAVA_USER_TEMPLATE.format(objective=objective.strip())

            ollama_messages: List[Dict[str, Any]] = [
                {"role": "system", "content": _LLAVA_SYSTEM_PROMPT},
            ]
            ollama_messages.extend(history)
            # LLaVA image injection: same format as Qwen — images list on the user turn
            ollama_messages.append({
                "role": "user",
                "content": user_prompt,
                "images": [img_base64],
            })

            loop = asyncio.get_running_loop()

            def _blocking():
                with _INFERENCE_LOCK:
                    return self._client.chat(
                        model=self.model_name,
                        messages=ollama_messages,
                        options={"temperature": 0},
                    )

            response = await loop.run_in_executor(self._executor, _blocking)
            content = self._extract_content(response)
            operations = self._parse_json(content)

            if not isinstance(operations, list):
                raise RuntimeError("LLM output must be a JSON array")

            return [op for op in operations if isinstance(op, dict) and "operation" in op]

        finally:
            _cleanup.close()

    @staticmethod
    def _extract_content(response: Any) -> str:
        if hasattr(response, "message") and hasattr(response.message, "content"):
            return response.message.content
        if isinstance(response, dict):
            return response.get("message", {}).get("content", "")
        raise RuntimeError(f"Cannot extract LLaVA response content: {type(response)}")

    @staticmethod
    def _parse_json(text: str) -> List[dict]:
        text = text.strip()
        # Strip markdown fences
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE).strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
        except Exception:
            pass

        for pattern in (r"(\[.*\])", r"(\{.*\})"):
            m = re.search(pattern, text, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(1))
                    return [parsed] if isinstance(parsed, dict) else parsed
                except Exception:
                    pass

        raise RuntimeError(f"No valid JSON in LLaVA response: {text[:200]!r}")
