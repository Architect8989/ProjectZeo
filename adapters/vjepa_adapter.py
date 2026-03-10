from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

_USE_VJEPA     = os.environ.get("PROJECTZEO_USE_VJEPA", "0").strip() == "1"
_VJEPA_MODEL   = os.environ.get("PROJECTZEO_VJEPA_MODEL", "vjepa2-vitl")
_VJEPA_ENDPOINT = os.environ.get("PROJECTZEO_VJEPA_ENDPOINT", "").strip()

# Surprise score above this threshold triggers consequence escalation
SURPRISE_THRESHOLD = float(os.environ.get("PROJECTZEO_VJEPA_SURPRISE_THRESHOLD", "0.65"))

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
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


def _jaccard(a: str, b: str) -> float:
    """Word-level Jaccard similarity for lightweight text comparison."""
    sa = set(re.sub(r"[^\w\s]", " ", a.lower()).split())
    sb = set(re.sub(r"[^\w\s]", " ", b.lower()).split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ─────────────────────────────────────────────────────────────────────────────
# VJEPAWorldModel
# ─────────────────────────────────────────────────────────────────────────────

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
        self._mode = "llm_sim"  # safe default — always works

        # Stats
        self._predict_count = 0
        self._high_surprise_count = 0

        # Back-reference to ConsequenceReasoner (set by GIIController)
        self._consequence_reasoner: Optional[Any] = None

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

    # ── Backend probing ───────────────────────────────────────────────────────

    def _probe_endpoint(self, endpoint: str) -> bool:
        try:
            import httpx
            resp = httpx.get(endpoint.rstrip("/") + "/health", timeout=5.0)
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

    # ── Core Prediction API ───────────────────────────────────────────────────

    def predict_next_state(
        self,
        screenshot_b64: str,
        action: Dict[str, Any],
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        
        with self._lock:
            self._predict_count += 1

        if self._available and self._mode == "endpoint":
            result = self._predict_via_endpoint(screenshot_b64, action)
            if result.get("mode") != "stub":
                return result

        return self._predict_via_llm(screenshot_b64, action, context)

    def predict_and_score(
        self,
        screenshot_b64: str,
        action: Dict[str, Any],
        context: Optional[str] = None,
        current_world_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        
        pred = self.predict_next_state(screenshot_b64, action, context)
        surprise = self.compute_surprise_signal(current_world_state or {}, pred)
        pred["surprise_score"] = round(surprise, 4)
        pred["surprise_reason"] = self._explain_surprise(surprise, pred, action)

        if surprise > SURPRISE_THRESHOLD:
            with self._lock:
                self._high_surprise_count += 1
            _logger.info(
                "[VJEPAWorldModel] High surprise (%.2f) for action %s — "
                "predicted: %s side_effects: %s",
                surprise,
                action.get("operation", "?"),
                str(pred.get("predicted_state", ""))[:80],
                pred.get("side_effects", []),
            )

        return pred

    # ── Surprise Signal ───────────────────────────────────────────────────────

    def compute_surprise_signal(
        self,
        before_world_state: Dict[str, Any],
        prediction: Dict[str, Any],
    ) -> float:
        
        surprise = 0.0

        # Component 1: Inverse confidence (low confidence = high uncertainty)
        confidence = float(prediction.get("confidence", 0.5))
        surprise += (1.0 - confidence) * 0.35

        # Component 2: Side effects penalty
        side_effects = prediction.get("side_effects", [])
        if isinstance(side_effects, list) and side_effects:
            # More side effects = more surprise
            n_effects = min(len(side_effects), 5)
            surprise += (n_effects / 5.0) * 0.30

        # Component 3: Low goal progress for a seemingly actionable action
        goal_progress = float(prediction.get("goal_progress", 0.5))
        if goal_progress < 0.2:
            surprise += (0.2 - goal_progress) / 0.2 * 0.15

        # Component 4: Dangerous keywords in predicted state
        pred_state = str(prediction.get("predicted_state", "")).lower()
        _DANGER_WORDS = frozenset({
            "delete", "remove", "lost", "error", "crash", "fail", "corrupt",
            "overwrite", "format", "wipe", "empty", "gone", "irreversible",
            "permanent", "cannot undo", "send", "submit", "deploy", "install",
        })
        n_danger = sum(1 for w in _DANGER_WORDS if w in pred_state)
        if n_danger > 0:
            surprise += min(n_danger / 3.0, 1.0) * 0.20

        return min(1.0, surprise)

    def compute_mismatch_signal(
        self,
        predicted_state: Dict[str, Any],
        actual_world_state: Dict[str, Any],
    ) -> float:
        
        pred_desc = str(predicted_state.get("predicted_state", "")).lower()
        if not pred_desc:
            return 0.0

        # Compare predicted description against actual focused_app + window_title
        actual_ctx = " ".join([
            str(actual_world_state.get("focused_app", "")),
            str(actual_world_state.get("window_title", "")),
            str(actual_world_state.get("page_title", "")),
        ]).lower()

        similarity = _jaccard(pred_desc, actual_ctx)
        mismatch = 1.0 - similarity

        # Clamp: low-information predictions can't generate strong mismatch signals
        confidence = float(predicted_state.get("confidence", 0.5))
        if confidence < 0.3:
            mismatch *= 0.5  # Don't penalise uncertain predictions

        return round(min(1.0, mismatch), 4)

    # ── Consequence Reasoner integration ─────────────────────────────────────

    def set_consequence_reasoner(self, cr: Any) -> None:
        
        self._consequence_reasoner = cr
        _logger.debug("[VJEPAWorldModel] ConsequenceReasoner registered.")

    def get_consequence_reasoner(self) -> Optional[Any]:
        return self._consequence_reasoner

    # ── Private helpers ───────────────────────────────────────────────────────

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
            return {
                "predicted_state": "unknown",
                "confidence": 0.3,
                "side_effects": [],
                "goal_progress": 0.5,
                "mode": "stub",
            }

        op      = action.get("operation", "?")
        content = action.get("content") or action.get("text") or action.get("command", "")
        prompt  = (
            f"Predict what the screen will look like AFTER this GUI action:\n"
            f"Action: {op} | Content: {str(content)[:200]}\n"
            f"{('Context: ' + context[:200]) if context else ''}\n\n"
            "Be specific about potential side effects (dialogs, notifications, "
            "data changes, navigation). Estimate goal progress honestly.\n\n"
            "Respond ONLY with JSON:\n"
            '{"predicted_state": "brief description", '
            '"confidence": 0.0-1.0, '
            '"side_effects": ["effect1", "effect2"], '
            '"goal_progress": 0.0-1.0}'
        )

        result_holder: List[Optional[str]] = [None]

        def _call() -> None:
            try:
                raw = self._llm(
                    messages=[{"role": "user", "content": prompt}],
                    objective="vjepa_simulation",
                    session_id="vjepa_predict",
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
            try:
                clean = re.sub(r"```(?:json)?", "", result_holder[0]).strip()
                m = re.search(r"\{.*\}", clean, re.DOTALL)
                if m:
                    result = json.loads(m.group(0))
                    result["mode"] = "llm_simulation"
                    # Ensure required keys
                    result.setdefault("side_effects", [])
                    result.setdefault("goal_progress", 0.5)
                    return result
            except Exception as exc:
                _logger.debug("[VJEPAWorldModel] LLM prediction parse failed: %s", exc)

        return {
            "predicted_state": "unknown",
            "confidence": 0.2,
            "side_effects": [],
            "goal_progress": 0.5,
            "mode": "llm_sim_fallback",
        }

    def _explain_surprise(
        self,
        score: float,
        pred: Dict[str, Any],
        action: Dict[str, Any],
    ) -> str:
        """Generate a human-readable explanation of why surprise is high/low."""
        if score < 0.3:
            return f"Low surprise ({score:.2f}) — action is predictable and safe."
        if score < SURPRISE_THRESHOLD:
            return (
                f"Moderate surprise ({score:.2f}) — action has some uncertainty. "
                f"Predicted: {str(pred.get('predicted_state','?'))[:60]}"
            )
        reasons = []
        if float(pred.get("confidence", 1.0)) < 0.4:
            reasons.append(f"low model confidence ({pred.get('confidence', 0):.2f})")
        side_effects = pred.get("side_effects", [])
        if side_effects:
            reasons.append(f"{len(side_effects)} side effect(s): {side_effects[:2]}")
        if float(pred.get("goal_progress", 1.0)) < 0.2:
            reasons.append(f"low goal progress ({pred.get('goal_progress', 0):.2f})")
        reason_str = "; ".join(reasons) if reasons else "multiple risk factors"
        return (
            f"HIGH surprise ({score:.2f}) for {action.get('operation','?')} — "
            f"{reason_str}. Consider safer alternative."
        )

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "available":            self._available,
                "mode":                 self._mode,
                "model_name":           self._model_name,
                "gpu":                  _check_gpu(),
                "vjepa_installed":      _check_vjepa(),
                "endpoint_configured":  bool(_VJEPA_ENDPOINT),
                "endpoint_url":         _VJEPA_ENDPOINT or None,
                "predictions_made":     self._predict_count,
                "high_surprise_count":  self._high_surprise_count,
                "surprise_threshold":   SURPRISE_THRESHOLD,
            }


# ─────────────────────────────────────────────────────────────────────────────
# Singleton factory
# ─────────────────────────────────────────────────────────────────────────────

_instance: Optional[VJEPAWorldModel] = None
_lock = threading.Lock()


def get_vjepa_world_model(llm_callable: Optional[Callable] = None) -> VJEPAWorldModel:
    """
    Return the global singleton VJEPAWorldModel, creating it if necessary.
    Pass llm_callable on first call — ignored on subsequent calls.
    """
    global _instance
    with _lock:
        if _instance is None:
            _instance = VJEPAWorldModel(llm_callable=llm_callable)
        return _instance


def reset_vjepa_singleton() -> None:
    """Reset the singleton — useful for testing."""
    global _instance
    with _lock:
        _instance = None
