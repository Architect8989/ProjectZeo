"""
core/safety/piguard.py — PIGuard Prompt Injection Filter (Layer 5, Research §6.2)

PIGuard is the first fully open-sourced safeguard against prompt injection
attacks for agentic systems (ACL 2025). This module provides:

  1. A DeBERTa-based classifier that detects prompt injection in tool responses,
     web content, email bodies, and document text.
  2. A lightweight heuristic fallback that works without the model weights,
     using the existing injection_markers.py frozenset plus new agentic patterns.

The key improvement over the existing LlamaGuard Tier 4 (which exists in
ProjectZeo's llamaguard_classifier.py) is:
  - PIGuard is specifically fine-tuned for agentic injection (tool responses,
    indirect prompt injection in content the agent reads)
  - Trained with MOF (Margin-based Over-defense Fix) to avoid blocking
    legitimate tool responses due to keyword bias
  - Runs BEFORE content enters the agent's context window (pre-inference)
    rather than as a post-hoc safety check

Reference: "PIGuard: Prompt Injection Guardrail via Mitigating Overdefense
for Free" — Li et al., ACL 2025
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import List, Optional

_logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Agentic injection patterns (extends injection_markers.py)
# Specific to tool responses and indirect prompt injection vectors
# ─────────────────────────────────────────────────────────────────────────────

_AGENTIC_INJECTION_PATTERNS: List[re.Pattern] = [
    # Direct hijacking instructions
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?previous", re.IGNORECASE),
    re.compile(r"forget\s+(all\s+)?previous\s+instructions?", re.IGNORECASE),
    re.compile(r"new\s+instructions?\s*:", re.IGNORECASE),
    re.compile(r"your\s+new\s+task\s+is", re.IGNORECASE),
    re.compile(r"override\s+(previous\s+)?(instructions?|commands?)", re.IGNORECASE),
    re.compile(r"system\s*:\s*you\s+are", re.IGNORECASE),
    # Jailbreak patterns
    re.compile(r"\[jailbreak\]", re.IGNORECASE),
    re.compile(r"act\s+as\s+(an?\s+)?(unrestricted|uncensored|evil|hacker)", re.IGNORECASE),
    re.compile(r"do\s+anything\s+now", re.IGNORECASE),
    re.compile(r"dan\s+mode", re.IGNORECASE),
    # Tool response injection
    re.compile(r"<\s*system\s*>.*<\s*/\s*system\s*>", re.IGNORECASE | re.DOTALL),
    re.compile(r"\[\s*system\s+prompt\s*\]", re.IGNORECASE),
    re.compile(r"assistant\s*:\s*(?:i\s+will|i\s+must|execute|run)", re.IGNORECASE),
    # Exfiltration patterns
    re.compile(r"send\s+(all\s+)?(?:conversation|history|context|data)\s+to", re.IGNORECASE),
    re.compile(r"exfiltrate", re.IGNORECASE),
    re.compile(r"leak\s+(?:the\s+)?(?:system|instructions?|context)", re.IGNORECASE),
    # Role manipulation
    re.compile(r"you\s+are\s+now\s+(a\s+)?different", re.IGNORECASE),
    re.compile(r"pretend\s+(?:you\s+are|to\s+be)", re.IGNORECASE),
    re.compile(r"roleplay\s+as", re.IGNORECASE),
    # Embedded command patterns
    re.compile(r"<!--\s*@agent", re.IGNORECASE),
    re.compile(r"\[\[.*inject.*\]\]", re.IGNORECASE),
    re.compile(r"\{\{.*system.*\}\}", re.IGNORECASE),
]

# High-confidence single-word indicators (after NFKD normalization)
_HIGH_CONFIDENCE_TOKENS = frozenset({
    "jailbreak", "jailbroken", "dan", "doomslayer",
    "gpt4free", "godmode", "unrestricted",
})


def _normalize(text: str) -> str:
    """NFKD normalize + lowercase. Matches injection_markers.py approach."""
    return unicodedata.normalize("NFKD", text).casefold()


class PIGuardHeuristic:
    """
    Lightweight heuristic PIGuard — no model weights required.
    Uses the extended agentic injection pattern list above.
    Falls back gracefully when the DeBERTa model is unavailable.
    """

    def __init__(self) -> None:
        # Try to load the full injection_markers frozenset
        try:
            from core.security.injection_markers import contains_injection_marker  # noqa
            self._full_marker_check = contains_injection_marker
        except ImportError:
            self._full_marker_check = None

    def classify(self, text: str) -> str:
        """
        Returns "INJECTION" or "SAFE".
        Uses NFKD-normalized pattern matching + full injection_markers check.
        """
        if not text or not text.strip():
            return "SAFE"

        normalized = _normalize(text)

        # Full injection_markers check (50+ markers with Unicode normalization)
        if self._full_marker_check is not None:
            try:
                if self._full_marker_check(normalized):
                    _logger.info("[PIGuard] Full marker match detected.")
                    return "INJECTION"
            except Exception:
                pass

        # Agentic injection patterns
        for pat in _AGENTIC_INJECTION_PATTERNS:
            if pat.search(normalized):
                _logger.info("[PIGuard] Agentic injection pattern matched: %s", pat.pattern[:60])
                return "INJECTION"

        # High-confidence token check
        words = set(re.findall(r"\w+", normalized))
        if words & _HIGH_CONFIDENCE_TOKENS:
            matched = words & _HIGH_CONFIDENCE_TOKENS
            _logger.info("[PIGuard] High-confidence injection token: %s", matched)
            return "INJECTION"

        return "SAFE"


class PIGuard:
    """
    PIGuard classifier with DeBERTa model support.

    When the PIGuard DeBERTa model is available (via transformers + model
    weights), it uses the neural classifier. Otherwise falls back to the
    heuristic classifier above.

    To install the neural model:
        pip install transformers torch
        # Model weights from: https://huggingface.co/li-et-al/PIGuard-DeBERTa
    """

    def __init__(
        self,
        model_name: str = "li-et-al/piguard-deberta",
        device: str = "cpu",
        threshold: float = 0.5,
    ) -> None:
        self._threshold = threshold
        self._neural_available = False
        self._pipeline = None
        self._heuristic = PIGuardHeuristic()

        # Attempt to load the neural model
        try:
            from transformers import pipeline  # type: ignore
            self._pipeline = pipeline(
                "text-classification",
                model=model_name,
                device=device,
                truncation=True,
                max_length=512,
            )
            self._neural_available = True
            _logger.info("[PIGuard] Neural DeBERTa classifier loaded: %s", model_name)
        except Exception as e:
            _logger.info(
                "[PIGuard] Neural model unavailable (%s) — using heuristic classifier.", e
            )

    def classify(self, text: str) -> str:
        """
        Returns "INJECTION" or "SAFE".
        Neural model takes precedence over heuristic when available.
        """
        if not text or not text.strip():
            return "SAFE"

        if self._neural_available and self._pipeline is not None:
            try:
                result = self._pipeline(text[:2048])
                if isinstance(result, list) and result:
                    label = str(result[0].get("label", "SAFE")).upper()
                    score = float(result[0].get("score", 0.0))
                    if "INJECTION" in label and score >= self._threshold:
                        _logger.info("[PIGuard] Neural: INJECTION (score=%.3f)", score)
                        return "INJECTION"
                    return "SAFE"
            except Exception as e:
                _logger.debug("[PIGuard] Neural classify error, falling back: %s", e)

        return self._heuristic.classify(text)


def create_piguard(use_neural: bool = False, **kwargs) -> PIGuard:
    """
    Factory function. Returns a PIGuard instance.

    Args:
        use_neural: Attempt to load the DeBERTa neural model. If False,
                    uses heuristic only (lower latency, no model download).
    """
    guard = PIGuard(**kwargs) if use_neural else PIGuard.__new__(PIGuard)
    if not use_neural:
        guard._threshold = 0.5
        guard._neural_available = False
        guard._pipeline = None
        guard._heuristic = PIGuardHeuristic()
    return guard
