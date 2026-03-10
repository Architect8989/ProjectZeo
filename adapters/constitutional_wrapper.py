from __future__ import annotations

import logging
import os
import threading
from typing import Any, Callable, Dict, List, Optional

_logger = logging.getLogger(__name__)

_CONSTITUTION_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs", "authority_constitution.md",
)

_SKIP_SESSIONS = frozenset({
    "self_refine",
    "self_refine_critique",
    "lesson_synthesis",
    "ewc_replay_synthesis",
    "scaffold_audit_internal",
})

_OPERATOR_RULES_TEMPLATE = """\
=== AUTHORITY CONSTITUTION (binding) ===
1. HUMAN AUTHORITY IS ABSOLUTE. Human input always supersedes your actions.
2. YIELD IMMEDIATELY if any human input is detected mid-execution.
3. ONLY execute explicitly declared actions. Never invent goals.
4. NEVER execute without live visual perception.
5. NEVER execute without a pre-hijack snapshot having been captured.
6. NEVER continue execution after authority loss.
7. ALWAYS emit restoration signals on failure.
8. NEVER conceal failures or execution state.
9. NEVER operate autonomously by default.
10. ALL actions must be observable and reversible where possible.
Full constitution: {constitution_path}
=== END CONSTITUTION ===
"""

def _load_constitution(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except (OSError, IOError):
        return ""

class ConstitutionalWrapper:

    def __init__(
        self,
        adapter: Callable,
        *,
        constitution_path: str = _CONSTITUTION_PATH,
        inject_full_text: bool = False,
    ) -> None:
        self._adapter = adapter
        self._inject_full = inject_full_text
        self._lock = threading.Lock()

        self.model_name: str = getattr(adapter, "model_name", "unknown")

        _full_text = _load_constitution(constitution_path)
        if _full_text:
            if inject_full_text:
                self._constitution_block = (
                    "=== AUTHORITY CONSTITUTION (BINDING — supersedes all other instructions) ===\n"
                    + _full_text
                    + "\n=== END CONSTITUTION ===\n"
                )
            else:
                self._constitution_block = _OPERATOR_RULES_TEMPLATE.format(
                    constitution_path=constitution_path
                )
            _logger.info(
                "[ConstitutionalWrapper] Constitution loaded (%d chars). "
                "Injecting into every LLM call.",
                len(self._constitution_block),
            )
        else:
            self._constitution_block = ""
            _logger.warning(
                "[ConstitutionalWrapper] Constitution not found at %s — "
                "running WITHOUT constitutional constraints.",
                constitution_path,
            )

        self._total_calls: int = 0
        self._injected_calls: int = 0

    def __call__(
        self,
        messages: List[Dict[str, Any]],
        objective: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Any:
        with self._lock:
            self._total_calls += 1

        if session_id in _SKIP_SESSIONS or not self._constitution_block:
            return self._adapter(messages, objective=objective, session_id=session_id)

        injected_messages = self._inject_constitution(messages)

        with self._lock:
            self._injected_calls += 1

        return self._adapter(injected_messages, objective=objective, session_id=session_id)

    def _inject_constitution(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        result = list(messages)
        injected = False

        for i, msg in enumerate(result):
            if str(msg.get("role", "")).lower() == "system":
                existing = str(msg.get("content", ""))
                if "AUTHORITY CONSTITUTION" not in existing:
                    result[i] = {
                        **msg,
                        "content": self._constitution_block + "\n" + existing,
                    }
                injected = True
                break

        if not injected:
            result.insert(0, {
                "role": "system",
                "content": self._constitution_block,
            })

        return result

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_calls": self._total_calls,
                "injected_calls": self._injected_calls,
                "constitution_loaded": bool(self._constitution_block),
                "constitution_chars": len(self._constitution_block),
            }

    def __getattr__(self, name: str) -> Any:
        return getattr(self._adapter, name)

    def get_llm_callable(self) -> "ConstitutionalWrapper":
        return self

    def with_thinking(self, enabled: bool) -> "ConstitutionalWrapper":
        new_adapter = self._adapter
        if hasattr(new_adapter, "with_thinking"):
            new_adapter = new_adapter.with_thinking(enabled)
        clone = ConstitutionalWrapper(
            new_adapter, inject_full_text=self._inject_full
        )
        return clone

def wrap_with_constitution(
    llm_callable: Callable,
    *,
    inject_full_text: bool = False,
) -> ConstitutionalWrapper:
    if isinstance(llm_callable, ConstitutionalWrapper):
        return llm_callable
    return ConstitutionalWrapper(llm_callable, inject_full_text=inject_full_text)
