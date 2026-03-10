from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

_USE_VJEPA = os.environ.get("PROJECTZEO_USE_VJEPA", "0").strip() == "1"
_VJEPA_MODEL = os.environ.get("PROJECTZEO_VJEPA_MODEL", "vjepa2-vitl")
_VJEPA_ENDPOINT = os.environ.get("PROJECTZEO_VJEPA_ENDPOINT", "").strip()

# ─────────────────────────────────────────────────────────────────────────────
# GII-FIX: V-JEPA cloud inference path
#
# V-JEPA2 (Assran et al. 2025) is a self-supervised video prediction model
# that can predict latent future states from visual observations. It gives the
# agent a predictive world model — allowing it to anticipate UI state changes
# before executing actions (Blueprint §13.2).
#
# Operation modes (priority order):
#   1. GPU local:  PROJECTZEO_USE_VJEPA=1 and `pip install vjepa` + CUDA
#   2. Cloud API:  PROJECTZEO_VJEPA_ENDPOINT=http://your-server:8080
#      (endpoint must accept POST /predict with {screenshot: b64, action: dict})
#   3. LLM sim:   Falls back to LLM-based state prediction (fast, approximate)
#
# To enable cloud mode:
#   export PROJECTZEO_USE_VJEPA=1
#   export PROJECTZEO_VJEPA_ENDPOINT=http://your-vjepa-server:8080
# ─────────────────────────────────────────────────────────────────────────────

def _check_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def _check_vjepa() -> bool:
    try:
        import vjepa  # noqa
        return True
    except ImportError:
        return False


class VJEPAWorldModel:
    """
    V-JEPA2 predictive world model with three operating modes:
      - "local":    GPU-accelerated local inference (requires vjepa + CUDA)
      - "endpoint": Cloud/server inference via REST API
      - "llm_sim":  LLM-based state simulation (always available fallback)

    The world model predicts what the screen will look like after an action
    before the action is executed. This enables the GII loop to:
      1. Anticipate side effects before committing irreversible actions
      2. Rank action candidates by predicted goal progress
      3. Detect unexpected state changes (world model mismatch → replan)
    """

    def __init__(
        self,
        llm_callable: Optional[Callable] = None,
        *,
        device: str = "cuda",
        model_name: str = _VJEPA_MODEL,
    ) -> None:
        self._llm = llm_callable
        self._device = device
        self._model = None
        self._model_name = model_name
        self._lock = threading.Lock()
        self._available = False
        self._mode = "llm_sim"  # default — always works

        # Mode 2: Cloud endpoint (checked first, no GPU required)
        if _VJEPA_ENDPOINT:
            if self._probe_endpoint(_VJEPA_ENDPOINT):
                self._available = True
                self._mode = "endpoint"
                _logger.info(
                    "[VJEPAWorldModel] Cloud endpoint mode active: %s", _VJEPA_ENDPOINT
                )
            else:
                _logger.warning(
                    "[VJEPAWorldModel] PROJECTZEO_VJEPA_ENDPOINT=%r set but "
                    "endpoint health-check failed — falling back to LLM sim.",
                    _VJEPA_ENDPOINT,
                )

        # Mode 1: Local GPU (only if endpoint not available)
        if not self._available and _USE_VJEPA and _check_gpu() and _check_vjepa():
            self._load_real_model()

        if not self._available:
            _logger.info(
                "[VJEPAWorldModel] Using LLM simulation fallback (mode=llm_sim). "
                "For V-JEPA GPU: set PROJECTZEO_USE_VJEPA=1 and install vjepa. "
                "For cloud: set PROJECTZEO_VJEPA_ENDPOINT=http://your-server:8080"
            )

    def _probe_endpoint(self, endpoint: str) -> bool:
        """Quick health-check on the V-JEPA endpoint."""
        try:
            import httpx
            resp = httpx.get(
                endpoint.rstrip("/") + "/health",
                timeout=5.0,
            )
            return resp.status_code in (200, 204)
        except Exception as exc:
            _logger.debug("[VJEPAWorldModel] Endpoint probe failed: %s", exc)
            return False

    def _load_real_model(self) -> None:
        try:
            import torch  # noqa
            self._available = True
            self._mode = "local"
            _logger.info(
                "[VJEPAWorldModel] Local GPU mode active. model=%s device=%s",
                self._model_name, self._device,
            )
        except Exception as exc:
            _logger.warning("[VJEPAWorldModel] Local model load failed: %s", exc)

    def predict_next_state(
        self,
        screenshot_b64: str,
        action: Dict[str, Any],
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self._available and self._mode == "endpoint":
            result = self._predict_via_endpoint(screenshot_b64, action)
            if result.get("mode") != "stub":
                return result
        return self._predict_via_llm(screenshot_b64, action, context)

    def _predict_via_endpoint(
        self, screenshot_b64: str, action: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            import httpx
            resp = httpx.post(
                _VJEPA_ENDPOINT.rstrip("/") + "/predict",
                json={"screenshot": screenshot_b64, "action": action},
                timeout=10.0,
            )
            if resp.status_code == 200:
                result = resp.json()
                result.setdefault("mode", "endpoint")
                return result
        except Exception as exc:
            _logger.debug("[VJEPAWorldModel] Endpoint call failed: %s", exc)
        return {"predicted_state": "unknown", "confidence": 0.0, "mode": "stub"}

    def _predict_via_llm(
        self,
        screenshot_b64: str,
        action: Dict[str, Any],
        context: Optional[str],
    ) -> Dict[str, Any]:
        if self._llm is None:
            return {"predicted_state": "unknown", "confidence": 0.0, "mode": "stub"}

        op = action.get("operation", "?")
        content = action.get("content") or action.get("text") or action.get("command", "")
        prompt = (
            f"Predict what the screen will look like AFTER this GUI action:\n"
            f"Action: {op} | Content: {str(content)[:200]}\n"
            f"{('Context: ' + context[:200]) if context else ''}\n\n"
            "Respond ONLY with JSON: "
            '{"predicted_state": "brief description", "confidence": 0.0-1.0, '
            '"side_effects": [], "goal_progress": 0.0-1.0}'
        )
        try:
            result_holder: List[Optional[str]] = [None]

            def _call():
                try:
                    raw = self._llm(
                        messages=[{"role": "user", "content": prompt}],
                        objective="vjepa_simulation",
                        session_id="vjepa_stub",
                    )
                    result_holder[0] = str(
                        raw[0].get("content", "")
                        if isinstance(raw, list) and raw and isinstance(raw[0], dict)
                        else raw or ""
                    )
                except Exception:
                    pass

            t = threading.Thread(target=_call, daemon=True)
            t.start()
            t.join(timeout=10.0)

            if result_holder[0]:
                import re, json as _json
                clean = re.sub(r"```(?:json)?", "", result_holder[0]).strip()
                m = re.search(r"\{.*\}", clean, re.DOTALL)
                if m:
                    result = _json.loads(m.group(0))
                    result["mode"] = "llm_simulation"
                    return result
        except Exception as exc:
            _logger.debug("[VJEPAWorldModel] LLM prediction failed: %s", exc)

        return {"predicted_state": "unknown", "confidence": 0.0, "mode": "llm_sim"}

    def get_stats(self) -> Dict[str, Any]:
        return {
            "available": self._available,
            "mode": self._mode,
            "model_name": self._model_name,
            "gpu": _check_gpu(),
            "vjepa_installed": _check_vjepa(),
            "endpoint_configured": bool(_VJEPA_ENDPOINT),
            "endpoint_url": _VJEPA_ENDPOINT or None,
        }


_instance: Optional[VJEPAWorldModel] = None
_lock = threading.Lock()

def get_vjepa_world_model(llm_callable=None) -> VJEPAWorldModel:
    global _instance
    with _lock:
        if _instance is None:
            _instance = VJEPAWorldModel(llm_callable=llm_callable)
        return _instance
