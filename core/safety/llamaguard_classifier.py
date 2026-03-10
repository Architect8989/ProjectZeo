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

_BLOCK_CATEGORIES: frozenset = frozenset({
    HazardCategory.VIOLENT_CRIMES,
    HazardCategory.CHILD_EXPLOITATION,
    HazardCategory.INDISCRIMINATE_WEAPONS,
    HazardCategory.SUICIDE_SELF_HARM,
    HazardCategory.CODE_CYBERATTACKS,
})

_CONFIRM_CATEGORIES: frozenset = frozenset({
    HazardCategory.NON_VIOLENT_CRIMES,
    HazardCategory.SPECIALIZED_ADVICE,
    HazardCategory.PRIVACY_VIOLATIONS,
})

_WARN_CATEGORIES: frozenset = frozenset({
    HazardCategory.DEFAMATION,
    HazardCategory.IP_VIOLATIONS,
    HazardCategory.HATE_SPEECH,
    HazardCategory.ELECTIONS_MISINFORMATION,
})

@dataclass
class LlamaGuardResult:

    verdict: str

    categories: List[str]

    action: str

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

_LLAMAGUARD_SYSTEM = """[INST] Task: Check if there is unsafe content in the 'Agent' messages in conversations according our safety policy with the below categories.

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
    else:
        content = f"Perform action: {op} — {json.dumps(action_dict)[:200]}"

    thought = str(action_dict.get("thought", ""))
    if thought:
        content += f"\n\nReasoning: {thought[:200]}"

    return _LLAMAGUARD_SYSTEM.format(agent_content=content)

class LlamaGuardClassifier:
    
    _instance: Optional["LlamaGuardClassifier"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._backend: Optional[str] = None
        self._model = None
        self._client = None
        self._url: Optional[str] = None
        self._timeout: float = 30.0

        self._total_classifications: int = 0
        self._total_blocks: int = 0
        self._total_confirms: int = 0
        self._total_latency: float = 0.0

        self._enabled = (
            os.environ.get("PROJECTZEO_LLAMAGUARD_ENABLED", "1").strip() not in ("0", "false", "no")
        )

        if self._enabled:
            self._init_backend()

        _logger.info(
            "[LlamaGuardClassifier] Initialised. backend=%s enabled=%s",
            self._backend, self._enabled,
        )

    @classmethod
    def get_instance(cls) -> "LlamaGuardClassifier":
        if cls._instance is not None:
            return cls._instance
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def classify(self, action_dict: Dict) -> LlamaGuardResult:
        
        if not self._enabled or self._backend == "disabled":
            return LlamaGuardResult(
                verdict="safe", categories=[], action="ALLOW",
                reason="LlamaGuard disabled", latency_seconds=0.0,
            )

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
                    "[LlamaGuardClassifier] BLOCK: op=%s categories=%s",
                    action_dict.get("operation"), result.categories,
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
            _logger.warning(
                "[LlamaGuardClassifier] Classification error (fail-open): %s", exc
            )
            return LlamaGuardResult(
                verdict="safe", categories=[], action="ALLOW",
                reason=f"Classification error (fail-open): {exc}",
                latency_seconds=latency,
            )

    def _run_inference(self, prompt: str) -> str:
        if self._backend == "sglang":
            return self._infer_sglang(prompt)
        elif self._backend == "llamacpp":
            return self._infer_llamacpp(prompt)
        elif self._backend == "hf":
            return self._infer_hf(prompt)
        elif self._backend == "ollama":
            return self._call_backend(prompt)
        return "safe"

    def _call_backend(self, prompt: str) -> str:
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
                timeout=_hx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0),
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
            raise RuntimeError(f"SGLang returned {response.status_code}")
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
                verdict="safe", categories=[], action="ALLOW", reason="Safe",
            )

        categories: List[str] = []
        if len(lines) > 1:
            raw_cats = lines[1].upper()
            for token in raw_cats.replace(",", " ").split():
                tok = token.strip().rstrip(".")
                if tok.startswith("S") and tok[1:].isdigit():
                    categories.append(tok)

        cat_enums = set()
        for c in categories:
            try:
                cat_enums.add(HazardCategory(c))
            except ValueError:
                pass

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
                "[LlamaGuardClassifier] WARN: op contains potentially sensitive "
                "content categories %s — allowing with warning.", warn_cats,
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
            reason=f"LlamaGuard3 unsafe but non-critical categories: {categories}",
        )

    def _init_backend(self) -> None:

        url = os.environ.get("PROJECTZEO_LLAMAGUARD_URL", "").strip()
        if url:
            try:
                import httpx
                self._client = httpx.Client(
                    headers={"Content-Type": "application/json"},
                    timeout=httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0),
                )
                resp = self._client.get(f"{url}/health", timeout=5.0)
                if resp.status_code == 200:
                    self._url = url
                    self._backend = "sglang"
                    _logger.info("[LlamaGuardClassifier] Using SGLang backend at %s", url)
                    return
            except Exception as exc:
                _logger.debug("[LlamaGuardClassifier] SGLang check failed: %s", exc)

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
                _logger.info("[LlamaGuardClassifier] Using llama-cpp-python backend: %s", gguf_path)
                return
            except Exception as exc:
                _logger.debug("[LlamaGuardClassifier] llama-cpp-python init failed: %s", exc)

        if os.environ.get("PROJECTZEO_LLAMAGUARD_HF", "0").strip() == "1":
            try:
                from transformers import pipeline
                self._model = pipeline(
                    "text-generation",
                    model="meta-llama/Llama-Guard-3-8B",
                    device_map="auto",
                )
                self._backend = "hf"
                _logger.info("[LlamaGuardClassifier] Using HuggingFace transformers backend.")
                return
            except Exception as exc:
                _logger.debug("[LlamaGuardClassifier] HuggingFace init failed: %s", exc)

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
                            "[LlamaGuardClassifier] Using Ollama backend: %s", ollama_model
                        )
                        return
            except Exception as exc:
                _logger.debug("[LlamaGuardClassifier] Ollama check failed: %s", exc)

        self._backend = "disabled"
        _logger.info(
            "[LlamaGuardClassifier] No backend configured — Tier 4 disabled. "
            "Set PROJECTZEO_LLAMAGUARD_URL, PROJECTZEO_LLAMAGUARD_GGUF, "
            "or PROJECTZEO_LLAMAGUARD_HF=1 to enable."
        )

    def is_enabled(self) -> bool:
        return self._enabled and self._backend not in (None, "disabled")

    def get_stats(self) -> Dict:
        avg = (
            self._total_latency / self._total_classifications
            if self._total_classifications > 0 else 0.0
        )
        return {
            "backend": self._backend,
            "enabled": self.is_enabled(),
            "total_classifications": self._total_classifications,
            "total_blocks": self._total_blocks,
            "total_confirms": self._total_confirms,
            "avg_latency_seconds": round(avg, 3),
        }

def classify_with_llamaguard(action_dict: Dict) -> LlamaGuardResult:
    return LlamaGuardClassifier.get_instance().classify(action_dict)
