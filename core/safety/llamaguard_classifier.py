from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Hazard taxonomy
# ─────────────────────────────────────────────────────────────────────────────

class HazardCategory(str, Enum):
    VIOLENT_CRIMES            = "S1"
    NON_VIOLENT_CRIMES        = "S2"
    SEX_RELATED_CRIMES        = "S3"
    CHILD_EXPLOITATION        = "S4"
    DEFAMATION                = "S5"
    SPECIALIZED_ADVICE        = "S6"
    PRIVACY_VIOLATIONS        = "S7"
    IP_VIOLATIONS             = "S8"
    INDISCRIMINATE_WEAPONS    = "S9"
    HATE_SPEECH               = "S10"
    SUICIDE_SELF_HARM         = "S11"
    SEXUAL_CONTENT            = "S12"
    ELECTIONS_MISINFORMATION  = "S13"
    CODE_CYBERATTACKS         = "S14"

# Categories that must be immediately blocked regardless of context
_BLOCK_CATEGORIES: frozenset = frozenset({
    HazardCategory.VIOLENT_CRIMES,
    HazardCategory.CHILD_EXPLOITATION,
    HazardCategory.INDISCRIMINATE_WEAPONS,
    HazardCategory.SUICIDE_SELF_HARM,
    HazardCategory.CODE_CYBERATTACKS,
})

# Categories that require human confirmation
_CONFIRM_CATEGORIES: frozenset = frozenset({
    HazardCategory.NON_VIOLENT_CRIMES,
    HazardCategory.SPECIALIZED_ADVICE,
    HazardCategory.PRIVACY_VIOLATIONS,
})

# Categories that produce a warning but allow execution
_WARN_CATEGORIES: frozenset = frozenset({
    HazardCategory.DEFAMATION,
    HazardCategory.IP_VIOLATIONS,
    HazardCategory.HATE_SPEECH,
    HazardCategory.ELECTIONS_MISINFORMATION,
    HazardCategory.SEXUAL_CONTENT,
    HazardCategory.SEX_RELATED_CRIMES,
})

# ─────────────────────────────────────────────────────────────────────────────
# FAIL-CLOSED: operation categories when LlamaGuard is required but unavailable
# ─────────────────────────────────────────────────────────────────────────────

# Operations blocked outright when classifier is down (can cause irreversible harm)
_FAIL_CLOSED_BLOCK_OPS: frozenset = frozenset({
    "file_delete", "file_move", "file_write", "file_modify",
    "rm", "rmdir", "format", "wipe",
})

# Operations requiring human confirmation when classifier is down
_FAIL_CLOSED_CONFIRM_OPS: frozenset = frozenset({
    "command", "install", "file_create", "navigate", "submit",
    "send", "post", "upload", "download",
})

# Operations allowed even when classifier is down (inherently observable/reversible)
_FAIL_CLOSED_ALLOW_OPS: frozenset = frozenset({
    "click", "scroll", "move_mouse", "focus", "screenshot",
    "observe", "wait", "done", "verify", "read", "type", "write",
})


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LlamaGuardResult:
    """Result of a LlamaGuard classification call."""

    verdict: str                     # "safe" | "unsafe"
    categories: List[str]            # Triggered S-codes e.g. ["S1", "S14"]
    action: str                      # "ALLOW" | "BLOCK" | "REQUIRE_HUMAN_CONFIRMATION" | "WARN"
    reason: str
    latency_seconds: float = 0.0

    @property
    def is_safe(self) -> bool:
        return self.verdict == "safe"

    @property
    def is_blocked(self) -> bool:
        return self.action == "BLOCK"

    @property
    def requires_confirmation(self) -> bool:
        return self.action == "REQUIRE_HUMAN_CONFIRMATION"


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────────

_LLAMAGUARD_SYSTEM = """\
[INST] Task: Check if there is unsafe content in the 'Agent' messages in conversations according our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
S1: Violent Crimes
S2: Non-Violent Crimes
S3: Sex-Related Crimes
S4: Child Sexual Exploitation
S5: Defamation
S6: Specialized Advice
S7: Privacy Violations
S8: Intellectual Property Violations
S9: Indiscriminate Weapons (CBRN)
S10: Hate Speech
S11: Suicide / Self-Harm
S12: Sexual Content
S13: Elections / Voting
S14: Code enabling cyberattacks
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>
Agent: {agent_content}
<END CONVERSATION>

Provide your safety assessment for the Agent message.
First line must be "safe" or "unsafe".
If unsafe, list triggered categories on the second line (comma-separated S codes). [/INST]"""


def _build_llamaguard_prompt(action_dict: Dict) -> str:
    op = str(action_dict.get("operation", "unknown"))

    if op == "command":
        content = f"Execute shell command: {action_dict.get('command', '')}"
    elif op == "file_create":
        path = action_dict.get("path", "")
        file_content = str(action_dict.get("content", ""))[:500]
        content = f"Create file at {path} with content:\n{file_content}"
    elif op == "file_delete":
        content = f"Delete file or directory: {action_dict.get('path', '')}"
    elif op in ("write", "type"):
        typed = str(action_dict.get("content") or action_dict.get("text", ""))[:300]
        content = f"Type/write text: {typed}"
    elif op == "install":
        tool = action_dict.get("tool", {})
        if isinstance(tool, dict):
            cmds = tool.get("install_commands", [])
            content = f"Install software with commands: {'; '.join(str(c) for c in cmds[:3])}"
        else:
            content = f"Install: {tool}"
    elif op == "click":
        text = str(action_dict.get("text", ""))
        content = f"Click UI element: {text}"
    elif op in ("navigate", "browse"):
        content = f"Navigate to URL/path: {action_dict.get('url', action_dict.get('path', ''))}"
    else:
        content = f"Perform action: {op} — {json.dumps(action_dict, default=str)[:200]}"

    thought = str(action_dict.get("thought", ""))
    if thought:
        content += f"\n\nReasoning: {thought[:200]}"

    return _LLAMAGUARD_SYSTEM.format(agent_content=content)


# ─────────────────────────────────────────────────────────────────────────────
# LlamaGuardClassifier
# ─────────────────────────────────────────────────────────────────────────────

class LlamaGuardClassifier:
    """
    LlamaGuard-3-8B classifier for Tier 4 safety gate.

    Supports four backends (in priority order):
      1. SGLang server  (PROJECTZEO_LLAMAGUARD_ENDPOINT or PROJECTZEO_LLAMAGUARD_URL)
      2. llama-cpp-python GGUF  (PROJECTZEO_LLAMAGUARD_GGUF)
      3. HuggingFace transformers  (PROJECTZEO_LLAMAGUARD_HF=1)
      4. Ollama  (PROJECTZEO_LLAMAGUARD_OLLAMA)

    FAIL-CLOSED behaviour:
      When PROJECTZEO_REQUIRE_LLAMAGUARD=1 (default) and no backend is
      available, the classifier denies destructive operations and requires
      human confirmation for irreversible ones instead of silently allowing
      everything.
    """

    _instance: Optional["LlamaGuardClassifier"] = None
    _instance_lock = threading.Lock()

    # Startup warning emitted at most once per process
    _FAILCLOSED_WARNING_EMITTED = False

    def __init__(self) -> None:
        self._backend: Optional[str] = None
        self._model = None
        self._client = None
        self._url: Optional[str] = None
        self._timeout: float = float(os.environ.get("PROJECTZEO_LLAMAGUARD_TIMEOUT", "30"))

        self._total_classifications: int = 0
        self._total_blocks: int = 0
        self._total_confirms: int = 0
        self._total_failclosed: int = 0
        self._total_latency: float = 0.0

        # Whether this classifier is enabled at all
        self._enabled = (
            os.environ.get("PROJECTZEO_LLAMAGUARD_ENABLED", "1").strip()
            not in ("0", "false", "no")
        )

        # Whether absence of a backend should be fail-closed (default: YES)
        self._require = (
            os.environ.get("PROJECTZEO_REQUIRE_LLAMAGUARD", "1").strip()
            not in ("0", "false", "no")
        )

        if self._enabled:
            self._init_backend()

        if self._enabled and self._backend == "disabled" and self._require:
            if not LlamaGuardClassifier._FAILCLOSED_WARNING_EMITTED:
                LlamaGuardClassifier._FAILCLOSED_WARNING_EMITTED = True
                import sys
                print(
                    "\n[SAFETY WARNING] LlamaGuard-3 Tier 4 classifier is REQUIRED "
                    "(PROJECTZEO_REQUIRE_LLAMAGUARD=1) but NO backend is configured.\n"
                    "  → Destructive operations (file_delete, file_move, file_write) will be DENIED.\n"
                    "  → Command/install operations will require human confirmation.\n"
                    "  → Set one of: PROJECTZEO_LLAMAGUARD_ENDPOINT, PROJECTZEO_LLAMAGUARD_GGUF,\n"
                    "                PROJECTZEO_LLAMAGUARD_HF=1, or PROJECTZEO_LLAMAGUARD_OLLAMA\n"
                    "    to restore full Tier 4 coverage.",
                    file=sys.stderr,
                )

        _logger.info(
            "[LlamaGuardClassifier] Initialised. backend=%s enabled=%s require=%s fail_closed=%s",
            self._backend, self._enabled, self._require,
            (self._enabled and self._backend == "disabled" and self._require),
        )

    @classmethod
    def get_instance(cls) -> "LlamaGuardClassifier":
        if cls._instance is not None:
            return cls._instance
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def classify(self, action_dict: Dict) -> LlamaGuardResult:
        """
        Classify an action dict.

        Returns a LlamaGuardResult with verdict, categories, and recommended action.
        Fail-closed when backend unavailable and REQUIRE_LLAMAGUARD=1.
        """
        # Classifier entirely disabled — pass through
        if not self._enabled:
            return LlamaGuardResult(
                verdict="safe", categories=[], action="ALLOW",
                reason="LlamaGuard disabled (PROJECTZEO_LLAMAGUARD_ENABLED=0)",
                latency_seconds=0.0,
            )

        # Backend unavailable: apply fail-closed policy
        if self._backend in (None, "disabled"):
            return self._fail_closed_result(action_dict)

        start = time.monotonic()
        self._total_classifications += 1

        try:
            prompt = _build_llamaguard_prompt(action_dict)
            raw_output = self._run_inference(prompt)
            result = self._parse_output(raw_output)
            result.latency_seconds = time.monotonic() - start
            self._total_latency += result.latency_seconds

            if result.is_blocked:
                self._total_blocks += 1
                _logger.warning(
                    "[LlamaGuardClassifier] BLOCK: op=%s categories=%s reason=%s",
                    action_dict.get("operation"), result.categories, result.reason[:80],
                )
            elif result.requires_confirmation:
                self._total_confirms += 1
                _logger.info(
                    "[LlamaGuardClassifier] CONFIRM_REQUIRED: op=%s categories=%s",
                    action_dict.get("operation"), result.categories,
                )

            return result

        except Exception as exc:
            latency = time.monotonic() - start
            op = str(action_dict.get("operation", "")).lower()
            _logger.warning(
                "[LlamaGuardClassifier] Classification error for op=%s: %s",
                op, exc,
            )

            # FAIL-CLOSED on exception for destructive operations
            if op in _FAIL_CLOSED_BLOCK_OPS:
                self._total_blocks += 1
                return LlamaGuardResult(
                    verdict="unsafe",
                    categories=[],
                    action="BLOCK",
                    reason=(
                        f"LlamaGuard classification failed ({type(exc).__name__}). "
                        f"Fail-closed BLOCK for destructive op '{op}'."
                    ),
                    latency_seconds=latency,
                )

            if op in _FAIL_CLOSED_CONFIRM_OPS:
                self._total_confirms += 1
                return LlamaGuardResult(
                    verdict="unsafe",
                    categories=[],
                    action="REQUIRE_HUMAN_CONFIRMATION",
                    reason=(
                        f"LlamaGuard classification failed ({type(exc).__name__}). "
                        f"Fail-closed CONFIRM for op '{op}'."
                    ),
                    latency_seconds=latency,
                )

            # Harmless op — allow with warning
            return LlamaGuardResult(
                verdict="safe",
                categories=[],
                action="ALLOW",
                reason=f"Classification error (fail-open for non-destructive op): {exc}",
                latency_seconds=latency,
            )

    def _fail_closed_result(self, action_dict: Dict) -> LlamaGuardResult:
        """
        Called when the classifier is required but no backend is available.
        Applies a conservative policy based on operation type.
        """
        op = str(action_dict.get("operation", "")).lower()
        self._total_failclosed += 1

        if not self._require:
            # Fail-open allowed — return ALLOW with a debug note
            return LlamaGuardResult(
                verdict="safe",
                categories=[],
                action="ALLOW",
                reason="LlamaGuard backend unavailable (REQUIRE_LLAMAGUARD=0, failing open)",
                latency_seconds=0.0,
            )

        # REQUIRE_LLAMAGUARD=1: enforce conservative policy
        if op in _FAIL_CLOSED_BLOCK_OPS:
            _logger.warning(
                "[LlamaGuardClassifier] FAIL-CLOSED DENY: op=%s (no backend, require=True)",
                op,
            )
            return LlamaGuardResult(
                verdict="unsafe",
                categories=[],
                action="BLOCK",
                reason=(
                    f"LlamaGuard Tier 4 backend unavailable. "
                    f"Fail-closed: op '{op}' is categorised as destructive — BLOCKED. "
                    "Configure PROJECTZEO_LLAMAGUARD_ENDPOINT to restore Tier 4 coverage."
                ),
                latency_seconds=0.0,
            )

        if op in _FAIL_CLOSED_CONFIRM_OPS:
            _logger.info(
                "[LlamaGuardClassifier] FAIL-CLOSED CONFIRM: op=%s (no backend, require=True)",
                op,
            )
            return LlamaGuardResult(
                verdict="unsafe",
                categories=[],
                action="REQUIRE_HUMAN_CONFIRMATION",
                reason=(
                    f"LlamaGuard Tier 4 backend unavailable. "
                    f"Fail-closed: op '{op}' requires human confirmation. "
                    "Configure PROJECTZEO_LLAMAGUARD_ENDPOINT to restore Tier 4 coverage."
                ),
                latency_seconds=0.0,
            )

        # Explicitly allowed ops or unknown ops → ALLOW with warning
        _logger.debug(
            "[LlamaGuardClassifier] FAIL-CLOSED ALLOW: op=%s (no backend, allow-listed op)",
            op,
        )
        return LlamaGuardResult(
            verdict="safe",
            categories=[],
            action="ALLOW",
            reason=(
                f"LlamaGuard backend unavailable. Op '{op}' allowed by fail-closed "
                "allow-list (observational/reversible). Tier 4 is degraded."
            ),
            latency_seconds=0.0,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Backend dispatch
    # ─────────────────────────────────────────────────────────────────────────

    def _run_inference(self, prompt: str) -> str:
        if self._backend == "sglang":
            return self._infer_sglang(prompt)
        elif self._backend == "llamacpp":
            return self._infer_llamacpp(prompt)
        elif self._backend == "hf":
            return self._infer_hf(prompt)
        elif self._backend == "ollama":
            return self._call_ollama(prompt)
        raise RuntimeError(f"Unknown backend: {self._backend!r}")

    def _call_ollama(self, prompt: str) -> str:
        model_tag = getattr(self, "_ollama_model_tag", "llama-guard3:8b")
        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        try:
            import httpx as _hx
            payload = {
                "model": model_tag,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            }
            r = _hx.post(
                f"{ollama_host}/api/chat",
                json=payload,
                timeout=_hx.Timeout(connect=5.0, read=self._timeout, write=5.0, pool=5.0),
            )
            r.raise_for_status()
            return r.json()["message"]["content"]
        except Exception as exc:
            raise RuntimeError(f"LlamaGuard Ollama call failed: {exc}") from exc

    def _infer_sglang(self, prompt: str) -> str:
        import json as _json
        payload = {
            "model": "meta-llama/Llama-Guard-3-8B",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 64,
            "temperature": 0.0,
        }
        response = self._client.post(
            f"{self._url}/v1/chat/completions",
            content=_json.dumps(payload),
            timeout=self._timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(f"SGLang returned {response.status_code}: {response.text[:200]}")
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _infer_llamacpp(self, prompt: str) -> str:
        output = self._model(
            prompt,
            max_tokens=64,
            temperature=0.0,
            echo=False,
        )
        return output["choices"][0]["text"]

    def _infer_hf(self, prompt: str) -> str:
        output = self._model(prompt, max_new_tokens=64, temperature=None, do_sample=False)
        if isinstance(output, list) and output:
            return output[0].get("generated_text", "")[-64:]
        return str(output)

    # ─────────────────────────────────────────────────────────────────────────
    # Output parsing
    # ─────────────────────────────────────────────────────────────────────────

    def _parse_output(self, raw: str) -> LlamaGuardResult:
        lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
        if not lines:
            return LlamaGuardResult(
                verdict="safe", categories=[], action="ALLOW",
                reason="Empty response — treated as safe",
            )

        verdict = lines[0].lower()
        if "unsafe" not in verdict:
            return LlamaGuardResult(
                verdict="safe", categories=[], action="ALLOW", reason="LlamaGuard: safe",
            )

        # Parse triggered categories from second line
        categories: List[str] = []
        if len(lines) > 1:
            raw_cats = lines[1].upper()
            for token in raw_cats.replace(",", " ").split():
                tok = token.strip().rstrip(".")
                if tok.startswith("S") and tok[1:].isdigit():
                    categories.append(tok)

        cat_enums: set = set()
        for c in categories:
            try:
                cat_enums.add(HazardCategory(c))
            except ValueError:
                pass  # Unknown category code — ignore

        if cat_enums & _BLOCK_CATEGORIES:
            blocked = sorted(str(c.value) for c in (cat_enums & _BLOCK_CATEGORIES))
            return LlamaGuardResult(
                verdict="unsafe",
                categories=categories,
                action="BLOCK",
                reason=(
                    f"LlamaGuard3 BLOCK: action violates safety categories "
                    f"{blocked}. This action cannot be approved."
                ),
            )

        if cat_enums & _CONFIRM_CATEGORIES:
            confirm_cats = sorted(str(c.value) for c in (cat_enums & _CONFIRM_CATEGORIES))
            return LlamaGuardResult(
                verdict="unsafe",
                categories=categories,
                action="REQUIRE_HUMAN_CONFIRMATION",
                reason=(
                    f"LlamaGuard3: action may involve sensitive content "
                    f"({confirm_cats}). Human confirmation required."
                ),
            )

        if cat_enums & _WARN_CATEGORIES:
            warn_cats = sorted(str(c.value) for c in (cat_enums & _WARN_CATEGORIES))
            _logger.warning(
                "[LlamaGuardClassifier] WARN: action contains potentially sensitive "
                "content in categories %s — allowing with warning.", warn_cats,
            )
            return LlamaGuardResult(
                verdict="unsafe",
                categories=categories,
                action="WARN",
                reason=f"LlamaGuard3 warning: categories {warn_cats}",
            )

        return LlamaGuardResult(
            verdict="unsafe",
            categories=categories,
            action="ALLOW",
            reason=f"LlamaGuard3 unsafe but non-actionable categories: {categories}",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Backend initialisation
    # ─────────────────────────────────────────────────────────────────────────

    def _init_backend(self) -> None:
        # 1. SGLang / OpenAI-compatible endpoint
        # Accept both ENDPOINT and URL env vars (ENDPOINT takes priority)
        url = (
            os.environ.get("PROJECTZEO_LLAMAGUARD_ENDPOINT", "").strip()
            or os.environ.get("PROJECTZEO_LLAMAGUARD_URL", "").strip()
        )
        if url:
            try:
                import httpx
                self._client = httpx.Client(
                    headers={"Content-Type": "application/json"},
                    timeout=httpx.Timeout(connect=5.0, read=self._timeout, write=5.0, pool=5.0),
                )
                # Try /health endpoint first; fall back to /v1/models
                healthy = False
                for health_path in ("/health", "/v1/models"):
                    try:
                        resp = self._client.get(f"{url}{health_path}", timeout=5.0)
                        if resp.status_code in (200, 404):  # 404 = server alive, wrong path
                            healthy = True
                            break
                    except Exception:
                        pass

                if healthy:
                    self._url = url
                    self._backend = "sglang"
                    _logger.info(
                        "[LlamaGuardClassifier] SGLang/OpenAI backend active at %s", url
                    )
                    return
                else:
                    _logger.warning(
                        "[LlamaGuardClassifier] Endpoint %s is not reachable. "
                        "Trying other backends.", url,
                    )
            except Exception as exc:
                _logger.debug("[LlamaGuardClassifier] SGLang init failed: %s", exc)

        # 2. llama-cpp-python GGUF
        gguf_path = os.environ.get("PROJECTZEO_LLAMAGUARD_GGUF", "").strip()
        if gguf_path and os.path.exists(gguf_path):
            try:
                from llama_cpp import Llama
                n_gpu = int(os.environ.get("PROJECTZEO_LLAMAGUARD_GPU_LAYERS", "0"))
                self._model = Llama(
                    model_path=gguf_path,
                    n_ctx=4096,
                    n_threads=4,
                    n_gpu_layers=n_gpu,
                    verbose=False,
                )
                self._backend = "llamacpp"
                _logger.info(
                    "[LlamaGuardClassifier] llama-cpp-python backend: %s (gpu_layers=%d)",
                    gguf_path, n_gpu,
                )
                return
            except Exception as exc:
                _logger.debug("[LlamaGuardClassifier] llama-cpp init failed: %s", exc)

        # 3. HuggingFace transformers
        if os.environ.get("PROJECTZEO_LLAMAGUARD_HF", "0").strip() == "1":
            try:
                from transformers import pipeline
                self._model = pipeline(
                    "text-generation",
                    model="meta-llama/Llama-Guard-3-8B",
                    device_map="auto",
                )
                self._backend = "hf"
                _logger.info("[LlamaGuardClassifier] HuggingFace transformers backend active.")
                return
            except Exception as exc:
                _logger.debug("[LlamaGuardClassifier] HuggingFace init failed: %s", exc)

        # 4. Ollama
        ollama_model = os.environ.get("PROJECTZEO_LLAMAGUARD_OLLAMA", "").strip()
        if ollama_model:
            try:
                import httpx as _hx
                r = _hx.get("http://localhost:11434/api/tags", timeout=3.0)
                if r.status_code == 200:
                    tags = [m.get("name", "") for m in r.json().get("models", [])]
                    if any(ollama_model.split(":")[0] in t for t in tags):
                        self._ollama_model_tag = ollama_model
                        self._backend = "ollama"
                        _logger.info(
                            "[LlamaGuardClassifier] Ollama backend: %s", ollama_model
                        )
                        return
                    else:
                        _logger.warning(
                            "[LlamaGuardClassifier] Ollama model '%s' not found in tags. "
                            "Run: ollama pull %s", ollama_model, ollama_model,
                        )
            except Exception as exc:
                _logger.debug("[LlamaGuardClassifier] Ollama check failed: %s", exc)

        # No backend found
        self._backend = "disabled"
        _logger.info(
            "[LlamaGuardClassifier] No backend configured — Tier 4 in fail-%s mode.\n"
            "  Configure with one of:\n"
            "    PROJECTZEO_LLAMAGUARD_ENDPOINT=http://<host>:<port>  (SGLang server)\n"
            "    PROJECTZEO_LLAMAGUARD_GGUF=/path/to/llama-guard-3-8b.gguf\n"
            "    PROJECTZEO_LLAMAGUARD_HF=1  (HuggingFace transformers, GPU recommended)\n"
            "    PROJECTZEO_LLAMAGUARD_OLLAMA=llama-guard3:8b  (Ollama)",
            "closed" if self._require else "open",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Utility / stats
    # ─────────────────────────────────────────────────────────────────────────

    def is_enabled(self) -> bool:
        return self._enabled and self._backend not in (None, "disabled")

    def is_fail_closed(self) -> bool:
        """True when the classifier is degraded but still applying fail-closed policy."""
        return (
            self._enabled
            and self._backend in (None, "disabled")
            and self._require
        )

    def get_stats(self) -> Dict:
        avg = (
            self._total_latency / self._total_classifications
            if self._total_classifications > 0 else 0.0
        )
        return {
            "backend": self._backend,
            "enabled": self._enabled,
            "require": self._require,
            "fail_closed": self.is_fail_closed(),
            "total_classifications": self._total_classifications,
            "total_blocks": self._total_blocks,
            "total_confirms": self._total_confirms,
            "total_failclosed_intercepts": self._total_failclosed,
            "avg_latency_seconds": round(avg, 3),
        }

    def __repr__(self) -> str:
        return (
            f"<LlamaGuardClassifier backend={self._backend!r} "
            f"enabled={self._enabled} fail_closed={self.is_fail_closed()}>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Module-level convenience function
# ─────────────────────────────────────────────────────────────────────────────

def classify_with_llamaguard(action_dict: Dict) -> LlamaGuardResult:
    """Classify an action dict using the global LlamaGuard singleton."""
    return LlamaGuardClassifier.get_instance().classify(action_dict)
