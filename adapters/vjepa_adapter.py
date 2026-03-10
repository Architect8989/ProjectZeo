from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

_USE_VJEPA = os.environ.get("PROJECTZEO_USE_VJEPA", "0").strip() == "1"
_VJEPA_MODEL = os.environ.get("PROJECTZEO_VJEPA_MODEL", "vjepa2-vitl")
_VJEPA_ENDPOINT = os.environ.get("PROJECTZEO_VJEPA_ENDPOINT", "")

def _check_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def _check_vjepa() -> bool:
    try:
        import vjepa
        return True
    except ImportError:
        return False

class VJEPAWorldModel:

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
        self._mode = "stub"

        if _USE_VJEPA and _check_gpu() and _check_vjepa():
            self._load_real_model()

        if not self._available:
            _logger.info(
                "[VJEPAWorldModel] Real model unavailable — using LLM simulation fallback. "
                "Set PROJECTZEO_USE_VJEPA=1 and install vjepa to enable GPU mode."
            )

    def _load_real_model(self) -> None:
        try:
            import torch
            if _VJEPA_ENDPOINT:
                self._available = True
                self._mode = "endpoint"
                _logger.info("[VJEPAWorldModel] Using V-JEPA endpoint: %s", _VJEPA_ENDPOINT)
            else:
                _logger.info(
                    "[VJEPAWorldModel] PROJECTZEO_USE_VJEPA=1 but no endpoint or local weights. "
                    "Set PROJECTZEO_VJEPA_ENDPOINT=http://... or install vjepa package."
                )
        except Exception as exc:
            _logger.warning("[VJEPAWorldModel] Real model load failed: %s", exc)

    def predict_next_state(
        self,
        screenshot_b64: str,
        action: Dict[str, Any],
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self._available and self._mode == "endpoint":
            return self._predict_via_endpoint(screenshot_b64, action)
        return self._predict_via_llm(screenshot_b64, action, context)

    def _predict_via_endpoint(
        self, screenshot_b64: str, action: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            import httpx
            resp = httpx.post(
                _VJEPA_ENDPOINT + "/predict",
                json={"screenshot": screenshot_b64, "action": action},
                timeout=10.0,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as exc:
            _logger.debug("[VJEPAWorldModel] Endpoint call failed: %s", exc)
        return self._predict_via_llm(screenshot_b64, action, context=None)

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
                    result_holder[0] = str(raw[0].get("content", "") if isinstance(raw, list) and raw and isinstance(raw[0], dict) else raw or "")
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

        return {"predicted_state": "unknown", "confidence": 0.0, "mode": "stub"}

    def get_stats(self) -> Dict[str, Any]:
        return {
            "available": self._available,
            "mode": self._mode,
            "model_name": self._model_name,
            "gpu": _check_gpu(),
            "vjepa_installed": _check_vjepa(),
        }

_instance: Optional[VJEPAWorldModel] = None
_lock = threading.Lock()

def get_vjepa_world_model(llm_callable=None) -> VJEPAWorldModel:
    global _instance
    with _lock:
        if _instance is None:
            _instance = VJEPAWorldModel(llm_callable=llm_callable)
        return _instance
