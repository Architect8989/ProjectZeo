"""
adapters/gui_actor_adapter.py
==============================
Microsoft GUI-Actor-7B Grounding Adapter for ProjectZeo GII.

Blueprint Reference: §2.1.4 (arXiv:2506.03143), §3.2.2

GUI-Actor replaces coordinate-based grounding with attention-based action heads.
Instead of predicting (x, y) pixel coordinates directly, it identifies the
semantically correct UI element through cross-attention between the instruction
and the screenshot, then maps attention peaks to screen positions.

Advantages over OmniParser coordinate grounding:
  - Resolution-agnostic: handles DPI scaling, windowed/full-screen, multi-monitor
  - Native confidence scores from attention peaks
  - Uncertainty quantification: low-confidence peaks trigger verification subgoal
  - SOTA on ScreenSpot-Pro benchmark with Qwen2.5-VL backbone

Serving options (tried in order):
  1. Remote vLLM server (PROJECTZEO_GUI_ACTOR_URL) — production
  2. HuggingFace Transformers (local GPU, microsoft/GUI-Actor-7B-Qwen2.5-VL)
  3. Coordinate-extraction fallback (OmniParser / AT-SPI bounding boxes)

Key interface:
  adapter.ground(screenshot, instruction) → {"x": int, "y": int, "confidence": float, ...}

Environment variables:
    PROJECTZEO_GUI_ACTOR_URL    — vLLM/SGLang endpoint, e.g. http://localhost:8080
    PROJECTZEO_GUI_ACTOR_MODEL  — model ID (default: microsoft/GUI-Actor-7B-Qwen2.5-VL)
    PROJECTZEO_GUI_ACTOR_PORT   — port for local vLLM server (default: 8080)
    PROJECTZEO_GUI_ACTOR_CONF   — minimum confidence threshold (default: 0.70)
    PROJECTZEO_GUI_ACTOR_LOCAL  — "1" to force local HF loading
    PROJECTZEO_GUI_ACTOR_TIMEOUT — request timeout in seconds (default: 30)
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_MODEL       = "microsoft/GUI-Actor-7B-Qwen2.5-VL"
_DEFAULT_PORT        = int(os.environ.get("PROJECTZEO_GUI_ACTOR_PORT", "8080"))
_DEFAULT_URL         = os.environ.get(
    "PROJECTZEO_GUI_ACTOR_URL",
    f"http://localhost:{_DEFAULT_PORT}"
)
_MODEL_ID            = os.environ.get("PROJECTZEO_GUI_ACTOR_MODEL", _DEFAULT_MODEL)
_CONF_THRESHOLD      = float(os.environ.get("PROJECTZEO_GUI_ACTOR_CONF", "0.70"))
_FORCE_LOCAL         = os.environ.get("PROJECTZEO_GUI_ACTOR_LOCAL", "0") == "1"
_REQUEST_TIMEOUT     = float(os.environ.get("PROJECTZEO_GUI_ACTOR_TIMEOUT", "30"))
_MAX_IMAGE_PIXELS    = 1344 * 1344   # GUI-Actor native resolution
_CACHE_GROUND_TTL    = 2.0           # seconds to cache identical grounding results

# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GroundingResult:
    """Result of GUI-Actor grounding operation."""
    x:              int                  # Screen X coordinate (centre of target)
    y:              int                  # Screen Y coordinate (centre of target)
    confidence:     float                # Attention peak confidence 0.0–1.0
    element_type:   str = ""            # Detected element type
    element_label:  str = ""            # Detected element text/label
    bbox:           Optional[Tuple[int,int,int,int]] = None  # (x1,y1,x2,y2) if available
    backend:        str = "unknown"      # Which backend produced this
    latency_ms:     float = 0.0

    @property
    def is_confident(self) -> bool:
        return self.confidence >= _CONF_THRESHOLD

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x":            self.x,
            "y":            self.y,
            "confidence":   round(self.confidence, 4),
            "element_type": self.element_type,
            "element_label": self.element_label,
            "bbox":         list(self.bbox) if self.bbox else None,
            "backend":      self.backend,
            "latency_ms":   round(self.latency_ms, 1),
        }


# ─────────────────────────────────────────────────────────────────────────────
# System prompt for GUI-Actor API
# ─────────────────────────────────────────────────────────────────────────────

_GROUNDING_SYSTEM = """\
You are GUI-Actor, a coordinate-free GUI element grounding model.
(Microsoft Research, arXiv:2506.03143)

Given a screenshot and a natural language instruction describing a UI element,
identify the exact screen location of that element using cross-attention.

Respond ONLY with a JSON object:
{
  "x": <screen X coordinate as integer (pixel from left)>,
  "y": <screen Y coordinate as integer (pixel from top)>,
  "confidence": <0.0-1.0, attention peak confidence>,
  "element_type": "<button|input|link|text|image|checkbox|dropdown|dialog|unknown>",
  "element_label": "<visible text or aria-label of the element>",
  "bbox": [<x1>, <y1>, <x2>, <y2>]
}

RULES:
- x, y must be the CENTER of the target element
- confidence reflects certainty: 1.0=absolutely certain, 0.0=cannot find
- If element is not visible, return confidence=0.0 and x=0, y=0
- bbox must enclose the complete element bounds
- DO NOT guess: low confidence is correct when element is ambiguous
"""


# ─────────────────────────────────────────────────────────────────────────────
# Backend implementations
# ─────────────────────────────────────────────────────────────────────────────

class _RemoteBackend:
    """
    Calls a vLLM/SGLang server running GUI-Actor-7B via OpenAI-compatible API.
    This is the production path for GPU-served inference.
    """

    def __init__(self, base_url: str, model_id: str, timeout: float) -> None:
        self._url     = base_url.rstrip("/")
        self._model   = model_id
        self._timeout = timeout
        self._available: Optional[bool] = None
        self._lock    = threading.Lock()

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import httpx
            r = httpx.get(f"{self._url}/health", timeout=5.0)
            self._available = r.status_code < 500
        except Exception:
            try:
                import httpx
                r = httpx.get(f"{self._url}/v1/models", timeout=5.0)
                self._available = r.status_code == 200
            except Exception:
                self._available = False
        _logger.info("[GUI-Actor Remote] Available: %s at %s", self._available, self._url)
        return self._available

    def ground(self, image_b64: str, instruction: str, screen_size: Tuple[int,int]) -> GroundingResult:
        import httpx

        t0 = time.perf_counter()
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _GROUNDING_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                        {
                            "type": "text",
                            "text": (
                                f"Screen size: {screen_size[0]}x{screen_size[1]} pixels.\n"
                                f"INSTRUCTION: {instruction}\n\n"
                                "Ground this element and return JSON."
                            ),
                        },
                    ],
                },
            ],
            "max_tokens": 256,
            "temperature": 0.0,
        }

        try:
            r = httpx.post(
                f"{self._url}/v1/chat/completions",
                json=payload,
                timeout=self._timeout,
            )
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"]
            latency = (time.perf_counter() - t0) * 1000
            return _parse_grounding_response(raw, "remote_vllm", latency, screen_size)
        except Exception as exc:
            _logger.warning("[GUI-Actor Remote] Request failed: %s", exc)
            raise


class _LocalHFBackend:
    """
    Loads GUI-Actor-7B locally via HuggingFace Transformers.
    Requires: transformers, torch, and GPU with ~15GB VRAM (4-bit quant: ~5GB).
    """

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id
        self._model    = None
        self._processor = None
        self._lock      = threading.Lock()
        self._loaded    = False

    def _load(self) -> bool:
        if self._loaded:
            return True
        with self._lock:
            if self._loaded:
                return True
            try:
                from transformers import AutoProcessor, AutoModelForCausalLM
                import torch

                _logger.info("[GUI-Actor Local] Loading %s ...", self._model_id)
                device = "cuda" if torch.cuda.is_available() else "cpu"

                quant_kwargs: Dict[str, Any] = {}
                if device == "cuda":
                    try:
                        from transformers import BitsAndBytesConfig
                        quant_kwargs = {
                            "quantization_config": BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.float16,
                            )
                        }
                    except ImportError:
                        pass

                self._processor = AutoProcessor.from_pretrained(
                    self._model_id, trust_remote_code=True
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    self._model_id,
                    trust_remote_code=True,
                    device_map="auto" if device == "cuda" else None,
                    **quant_kwargs,
                )
                self._model.eval()
                self._loaded = True
                _logger.info("[GUI-Actor Local] Model loaded on %s", device)
                return True
            except Exception as exc:
                _logger.error("[GUI-Actor Local] Failed to load model: %s", exc)
                return False

    def is_available(self) -> bool:
        try:
            import transformers, torch  # noqa: F401
            return True
        except ImportError:
            return False

    def ground(self, image_b64: str, instruction: str, screen_size: Tuple[int,int]) -> GroundingResult:
        if not self._load():
            raise RuntimeError("GUI-Actor local model failed to load")

        import torch
        from PIL import Image

        t0 = time.perf_counter()

        # Decode image
        img_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        prompt = (
            f"<|im_start|>system\n{_GROUNDING_SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Screen size: {screen_size[0]}x{screen_size[1]} pixels.\n"
            f"INSTRUCTION: {instruction}\n"
            f"Ground this element and return JSON.<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        with self._lock:
            inputs = self._processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            )
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self._processor.tokenizer.eos_token_id,
                )
            raw = self._processor.tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

        latency = (time.perf_counter() - t0) * 1000
        return _parse_grounding_response(raw, "local_hf", latency, screen_size)


class _FallbackBackend:
    """
    Coordinate-extraction fallback using entity list from VisionRuntime.
    Used when neither remote nor local GUI-Actor is available.
    Provides best-effort grounding with lower accuracy and no attention confidence.
    """

    def is_available(self) -> bool:
        return True  # Always available as last resort

    def ground_from_entities(
        self,
        entities: List[Dict[str, Any]],
        instruction: str,
        screen_size: Tuple[int,int],
    ) -> GroundingResult:
        """Match instruction to entity list using keyword overlap."""
        if not entities:
            w, h = screen_size
            return GroundingResult(
                x=w // 2, y=h // 2,
                confidence=0.10,
                backend="fallback_no_entities",
            )

        instr_lower = instruction.lower()
        instr_words = set(re.findall(r"\w+", instr_lower))
        instr_words -= {"click", "the", "on", "a", "an", "button", "press",
                        "select", "choose", "find", "locate", "tap"}

        best_score = -1.0
        best_entity: Optional[Dict[str, Any]] = None

        for entity in entities:
            label = str(entity.get("text") or entity.get("label") or "").lower()
            etype = str(entity.get("type", "")).lower()
            entity_words = set(re.findall(r"\w+", label + " " + etype))

            if not entity_words or not instr_words:
                continue

            overlap = len(instr_words & entity_words) / max(len(instr_words), 1)
            # Boost button elements for click instructions
            if "click" in instr_lower and etype in ("button", "link"):
                overlap *= 1.3

            if overlap > best_score:
                best_score = overlap
                best_entity = entity

        if best_entity is None or best_score < 0.15:
            # No match: return centre of screen
            w, h = screen_size
            return GroundingResult(
                x=w // 2, y=h // 2,
                confidence=0.05,
                backend="fallback_no_match",
            )

        # Extract coordinates from entity bounding box
        bbox = best_entity.get("bbox") or best_entity.get("bounding_box")
        if bbox:
            try:
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    conf = min(0.65, 0.20 + best_score * 0.45)
                    return GroundingResult(
                        x=cx, y=cy,
                        confidence=conf,
                        element_type=best_entity.get("type", ""),
                        element_label=best_entity.get("text") or best_entity.get("label", ""),
                        bbox=(x1, y1, x2, y2),
                        backend="fallback_entity",
                    )
            except (ValueError, TypeError):
                pass

        # Use x/y if available
        ex = best_entity.get("x") or best_entity.get("cx")
        ey = best_entity.get("y") or best_entity.get("cy")
        if ex is not None and ey is not None:
            conf = min(0.60, 0.15 + best_score * 0.40)
            return GroundingResult(
                x=int(ex), y=int(ey),
                confidence=conf,
                element_type=best_entity.get("type", ""),
                element_label=best_entity.get("text") or best_entity.get("label", ""),
                backend="fallback_entity_xy",
            )

        # Last resort: screen centre
        w, h = screen_size
        return GroundingResult(
            x=w // 2, y=h // 2,
            confidence=0.10,
            backend="fallback_centre",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Response parser (shared by all backends)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_grounding_response(
    raw: str,
    backend: str,
    latency_ms: float,
    screen_size: Tuple[int, int],
) -> GroundingResult:
    """Parse GUI-Actor JSON response into GroundingResult."""
    if not raw:
        return GroundingResult(x=0, y=0, confidence=0.0, backend=backend, latency_ms=latency_ms)

    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
    if not match:
        _logger.debug("[GUI-Actor] Could not parse JSON from: %s", raw[:100])
        return GroundingResult(x=0, y=0, confidence=0.0, backend=backend, latency_ms=latency_ms)

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as exc:
        _logger.debug("[GUI-Actor] JSON decode error: %s", exc)
        return GroundingResult(x=0, y=0, confidence=0.0, backend=backend, latency_ms=latency_ms)

    x    = int(data.get("x", 0))
    y    = int(data.get("y", 0))
    conf = float(data.get("confidence", 0.0))
    conf = max(0.0, min(1.0, conf))

    # Clamp coordinates to screen bounds
    w, h = screen_size
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))

    bbox_raw = data.get("bbox")
    bbox: Optional[Tuple[int,int,int,int]] = None
    if bbox_raw and isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox_raw]
            x1 = max(0, min(w, x1))
            y1 = max(0, min(h, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            bbox = (x1, y1, x2, y2)
        except (ValueError, TypeError):
            pass

    return GroundingResult(
        x             = x,
        y             = y,
        confidence    = conf,
        element_type  = str(data.get("element_type", "")),
        element_label = str(data.get("element_label", "")),
        bbox          = bbox,
        backend       = backend,
        latency_ms    = latency_ms,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main adapter
# ─────────────────────────────────────────────────────────────────────────────

class GUIActorAdapter:
    """
    GUI-Actor-7B grounding adapter.

    Usage:
        adapter = GUIActorAdapter()
        result = adapter.ground(screenshot_pil, "click the Save button")
        if result.is_confident:
            pyautogui.click(result.x, result.y)
        else:
            # Trigger verification subgoal
    """

    def __init__(
        self,
        *,
        base_url: str = _DEFAULT_URL,
        model_id: str = _MODEL_ID,
        timeout: float = _REQUEST_TIMEOUT,
        force_local: bool = _FORCE_LOCAL,
    ) -> None:
        self._timeout     = timeout
        self._force_local = force_local
        self._lock        = threading.Lock()

        # Result cache: (instruction_hash) → (GroundingResult, timestamp)
        self._cache: Dict[str, Tuple[GroundingResult, float]] = {}

        # Initialise backends (lazy)
        self._remote: Optional[_RemoteBackend] = None
        self._local:  Optional[_LocalHFBackend] = None
        self._fallback = _FallbackBackend()
        self._active_backend: str = "uninitialized"

        if not force_local:
            self._remote = _RemoteBackend(base_url, model_id, timeout)
        if force_local or True:  # always prepare local as fallback
            self._local = _LocalHFBackend(model_id)

        _logger.info(
            "[GUI-Actor] Adapter initialised. base_url=%s model=%s force_local=%s",
            base_url, model_id, force_local
        )

    # =========================================================================
    # Public interface
    # =========================================================================

    def ground(
        self,
        screenshot: Any,
        instruction: str,
        *,
        entities: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Ground a natural language instruction to screen coordinates.

        Args:
            screenshot: PIL.Image, numpy array, base64 string, or None
            instruction: Natural language description of target element
            entities: Optional entity list from VisionRuntime (for fallback)

        Returns:
            dict with keys: x, y, confidence, element_type, element_label, bbox, backend

        Notes:
            DEFECT FIX: Previously crashed with ``ValueError: Unsupported screenshot
            type: <class 'NoneType'>`` when screenshot=None (e.g. when OperatorCycle
            calls ground() before a screenshot is available). Now returns a low-
            confidence centred fallback result instead of raising, matching the
            graceful-degradation contract expected by OperatorCycle.
        """
        t0 = time.perf_counter()

        # DEFECT FIX: guard None/invalid before _prepare_image raises.
        if screenshot is None:
            _logger.debug(
                "[GUI-Actor] ground() called with screenshot=None — "
                "returning low-confidence centre fallback."
            )
            # Try entity-based fallback even without screenshot
            if entities:
                try:
                    entity_result = self._fallback.ground_from_entities(
                        entities, instruction, screen_size=(1920, 1080)
                    )
                    if entity_result and entity_result.confidence > 0.1:
                        return entity_result.to_dict()
                except Exception:
                    pass
            fallback = GroundingResult(
                x=960, y=540,
                confidence=0.05,
                element_type="",
                element_label="",
                bbox=None,
                backend="fallback_no_screenshot",
                latency_ms=0.0,
            )
            return fallback.to_dict()

        # Prepare image
        image_b64, screen_size = self._prepare_image(screenshot)

        # Cache check
        cache_key = self._cache_key(image_b64[:64], instruction)
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached:
                result, ts = cached
                if time.time() - ts < _CACHE_GROUND_TTL:
                    _logger.debug("[GUI-Actor] Cache hit for %r", instruction[:40])
                    return result.to_dict()

        # Try backends in priority order
        result: Optional[GroundingResult] = None

        # 1. Remote vLLM server
        if not self._force_local and self._remote is not None:
            if self._remote.is_available():
                try:
                    result = self._remote.ground(image_b64, instruction, screen_size)
                    self._active_backend = "remote_vllm"
                except Exception as exc:
                    _logger.warning("[GUI-Actor] Remote backend failed: %s", exc)
                    result = None

        # 2. Local HuggingFace model
        if result is None and self._local is not None:
            if self._local.is_available():
                try:
                    result = self._local.ground(image_b64, instruction, screen_size)
                    self._active_backend = "local_hf"
                except Exception as exc:
                    _logger.warning("[GUI-Actor] Local HF backend failed: %s", exc)
                    result = None

        # 3. Fallback: entity list grounding
        if result is None:
            if entities:
                result = self._fallback.ground_from_entities(
                    entities, instruction, screen_size
                )
            else:
                # Parse entities from screenshot if possible
                result = GroundingResult(
                    x=screen_size[0] // 2,
                    y=screen_size[1] // 2,
                    confidence=0.05,
                    backend="fallback_no_backend",
                )
            self._active_backend = "fallback"

        total_ms = (time.perf_counter() - t0) * 1000

        _logger.info(
            "[GUI-Actor] ground(%r) → x=%d y=%d conf=%.2f backend=%s lat=%.0fms",
            instruction[:40], result.x, result.y,
            result.confidence, result.backend, total_ms
        )

        # Update cache
        with self._lock:
            self._cache[cache_key] = (result, time.time())
            # Bound cache size
            if len(self._cache) > 500:
                oldest = sorted(self._cache.items(), key=lambda kv: kv[1][1])[:100]
                for k, _ in oldest:
                    del self._cache[k]

        return result.to_dict()

    def ground_result(
        self,
        screenshot: Any,
        instruction: str,
        *,
        entities: Optional[List[Dict[str, Any]]] = None,
    ) -> GroundingResult:
        """Same as ground() but returns GroundingResult dataclass."""
        d = self.ground(screenshot, instruction, entities=entities)
        return GroundingResult(
            x             = d.get("x", 0),
            y             = d.get("y", 0),
            confidence    = d.get("confidence", 0.0),
            element_type  = d.get("element_type", ""),
            element_label = d.get("element_label", ""),
            bbox          = tuple(d["bbox"]) if d.get("bbox") else None,  # type: ignore
            backend       = d.get("backend", "unknown"),
            latency_ms    = d.get("latency_ms", 0.0),
        )

    @property
    def active_backend(self) -> str:
        return self._active_backend

    def health_check(self) -> Dict[str, Any]:
        """Return health status of all backends."""
        return {
            "remote_available": (
                self._remote.is_available() if self._remote else False
            ),
            "local_available": (
                self._local.is_available() if self._local else False
            ),
            "fallback_available": True,
            "active_backend": self._active_backend,
            "cache_size": len(self._cache),
        }

    # =========================================================================
    # Image preparation
    # =========================================================================

    def _prepare_image(self, screenshot: Any) -> Tuple[str, Tuple[int, int]]:
        """
        Convert any screenshot format to (base64_png_string, (width, height)).
        Handles: PIL.Image, numpy array, bytes, base64 string, file path.
        """
        try:
            from PIL import Image
        except ImportError:
            raise RuntimeError("GUI-Actor requires Pillow: pip install Pillow")

        if isinstance(screenshot, str):
            if os.path.isfile(screenshot):
                img = Image.open(screenshot).convert("RGB")
            else:
                # Assume base64
                try:
                    img_bytes = base64.b64decode(screenshot)
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                except Exception:
                    raise ValueError("screenshot string is not a valid file path or base64")
        elif isinstance(screenshot, bytes):
            img = Image.open(io.BytesIO(screenshot)).convert("RGB")
        elif hasattr(screenshot, "save"):
            # PIL Image
            img = screenshot.convert("RGB")
        else:
            try:
                import numpy as np
                arr = screenshot
                if hasattr(arr, "numpy"):
                    arr = arr.numpy()
                img = Image.fromarray(arr.astype("uint8"))
            except Exception:
                raise ValueError(f"Unsupported screenshot type: {type(screenshot)}")

        # Resize if too large (preserve aspect ratio)
        orig_size = img.size  # (width, height)
        w, h = orig_size
        if w * h > _MAX_IMAGE_PIXELS:
            ratio = (_MAX_IMAGE_PIXELS / (w * h)) ** 0.5
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            _logger.debug("[GUI-Actor] Resized image %dx%d → %dx%d", w, h, new_w, new_h)

        # Encode to base64 PNG
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=False)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        return b64, orig_size

    def _cache_key(self, image_prefix: str, instruction: str) -> str:
        import hashlib
        raw = f"{image_prefix}::{instruction}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_INSTANCE: Optional[GUIActorAdapter] = None
_INSTANCE_LOCK = threading.Lock()


def get_gui_actor(
    *,
    base_url: str = _DEFAULT_URL,
    model_id: str = _MODEL_ID,
    force_local: bool = _FORCE_LOCAL,
) -> GUIActorAdapter:
    """
    Return the singleton GUIActorAdapter instance.
    Thread-safe; creates on first call.
    """
    global _INSTANCE
    if _INSTANCE is not None:
        return _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is None:
            _INSTANCE = GUIActorAdapter(
                base_url=base_url,
                model_id=model_id,
                force_local=force_local,
            )
    return _INSTANCE


def is_gui_actor_available() -> bool:
    """
    Quick check: is GUI-Actor reachable via any backend?
    Does NOT load the model — just checks connectivity.
    """
    try:
        actor = get_gui_actor()
        health = actor.health_check()
        return health["remote_available"] or health["local_available"]
    except Exception:
        return False
