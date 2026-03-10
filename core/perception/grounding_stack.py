"""
core/perception/grounding_stack.py
====================================
Six-Tier Unified GUI Grounding Dispatcher
Blueprint §6.7 — Confidence-Escalation Grounding Stack

This file is the MISSING PIECE that unifies all grounding backends into one
dispatcher. Previously, Grounding DINO + SAM 2 lived in adapters/grounding_adapter.py
but was NEVER called from the main GII loop — it was an orphan. This file wires
the full stack so gii_loop.py can call one function and get the best available result.

Tier 0 — AT-SPI Text Extraction      [FREE, deterministic, conf=1.0]
Tier 1 — OmniParser V2 YOLO+Florence [FAST, local, conf ≥ 0.70]
Tier 2 — GUI-Actor Attention Head    [FAST, local, conf ≥ 0.60 with GUI-RC vote]
Tier 3 — UI-TARS-2 / SeeClick        [MEDIUM, local, conf ≥ 0.50]
Tier 4 — Grounding DINO + SAM 2     [MEDIUM, local, conf ≥ 0.40, zero-shot]
Tier 5 — Cloud VLM fallback          [SLOW, cloud, final fallback]

Escalation logic:
    Each tier runs ONLY if the previous tier returned confidence below its threshold.
    This ensures cheap tiers handle simple elements (90%+ of cases) while expensive
    tiers handle edge-cases (novel UIs, unusual elements).

Performance impact:
    Tier 0-1: < 50ms (AT-SPI + YOLO)
    Tier 2:   < 150ms (GUI-Actor with RC voting, no server)
    Tier 3:   < 300ms (UI-TARS-2 with SGLang)
    Tier 4:   < 800ms (Grounding DINO + SAM 2, GPU)
    Tier 5:   1-5s (Cloud API call)
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

# ── Confidence thresholds per tier ───────────────────────────────────────────
_T0_CONF = float(os.environ.get("PROJECTZEO_TIER0_CONF", "0.75"))
_T1_CONF = float(os.environ.get("PROJECTZEO_TIER1_CONF", "0.70"))
_T2_CONF = float(os.environ.get("PROJECTZEO_TIER2_CONF", "0.60"))
_T3_CONF = float(os.environ.get("PROJECTZEO_TIER3_CONF", "0.50"))
_T4_CONF = float(os.environ.get("PROJECTZEO_TIER4_CONF", "0.40"))
_T5_CONF = float(os.environ.get("PROJECTZEO_TIER5_CONF", "0.35"))

_CLOUD_ENABLED = os.environ.get("PROJECTZEO_CLOUD_GROUNDING", "1").strip() != "0"
_GUI_RC_SAMPLES = int(os.environ.get("PROJECTZEO_GUI_RC_SAMPLES", "3"))

_TIER_NAMES = {
    0: "AT-SPI",
    1: "OmniParser-V2",
    2: "GUI-Actor+RC",
    3: "UI-TARS-2",
    4: "GroundingDINO+SAM2",
    5: "CloudVLM",
}


@dataclass
class GroundingResult:
    x: float            # normalised [0,1] horizontal centre
    y: float            # normalised [0,1] vertical centre
    confidence: float   # 0.0–1.0
    tier_used: int      # 0–5
    tier_name: str
    label: str = ""
    bbox: Optional[List[float]] = None
    mask: Optional[Any] = None
    latency_ms: float = 0.0
    source: str = ""


@dataclass
class StackStats:
    tier_hits: Dict[int, int] = field(default_factory=lambda: {i: 0 for i in range(6)})
    tier_misses: Dict[int, int] = field(default_factory=lambda: {i: 0 for i in range(6)})
    total_calls: int = 0
    total_escalations: int = 0
    total_failures: int = 0


class GroundingStack:
    """
    Unified six-tier grounding dispatcher.

    Each tier is tried in ascending order. If a tier returns confidence ≥ its
    threshold, the result is returned immediately. If confidence is below
    threshold, or the tier fails, escalation to the next tier occurs.
    """

    def __init__(
        self,
        atspi_bridge=None,
        omniparser=None,
        uitars_runtime=None,
        llm_callable: Optional[Callable] = None,
    ) -> None:
        self._atspi = atspi_bridge
        self._omni = omniparser
        self._uitars = uitars_runtime
        self._llm = llm_callable
        self._stats = StackStats()
        self._lock = threading.Lock()

        # Lazy-init for heavier tiers
        self._grounding_adapter = None
        self._grounding_adapter_attempted = False

        _logger.info(
            "[GroundingStack] Initialised. tiers=6 cloud=%s",
            _CLOUD_ENABLED,
        )

    def ground(
        self,
        screenshot,
        element_description: str,
        *,
        atspi_tree: Optional[Dict[str, Any]] = None,
        current_app: str = "",
        goal_context: str = "",
        omni_elements: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[GroundingResult]:
        """
        Ground an element description using the 6-tier confidence-escalation stack.

        Args:
            screenshot:          PIL Image of current screen (can be None for Tier 0)
            element_description: Natural-language description of the target element
            atspi_tree:          Optional pre-extracted AT-SPI accessibility tree dict
            current_app:         Name of focused application
            goal_context:        Current task goal (for Tier 5 cloud context)
            omni_elements:       Pre-parsed OmniParser elements (avoids re-parsing)

        Returns:
            GroundingResult or None if all tiers fail.
        """
        t_start = time.monotonic()

        with self._lock:
            self._stats.total_calls += 1

        # ── Tier 0: AT-SPI ────────────────────────────────────────────────────
        r = self._tier0_atspi(atspi_tree, element_description)
        if r:
            if r.confidence >= _T0_CONF:
                r.latency_ms = (time.monotonic() - t_start) * 1000
                self._record_hit(0)
                return r
            self._record_miss(0)

        if screenshot is None:
            return None

        # ── Tier 1: OmniParser V2 ─────────────────────────────────────────────
        r = self._tier1_omniparser(screenshot, element_description, omni_elements)
        if r:
            if r.confidence >= _T1_CONF:
                r.latency_ms = (time.monotonic() - t_start) * 1000
                self._record_hit(1)
                return r
            self._record_miss(1)
        else:
            self._record_miss(1)

        with self._lock:
            self._stats.total_escalations += 1

        # ── Tier 2: GUI-Actor + RC Consensus ──────────────────────────────────
        r = self._tier2_gui_actor(screenshot, element_description)
        if r:
            if r.confidence >= _T2_CONF:
                r.latency_ms = (time.monotonic() - t_start) * 1000
                self._record_hit(2)
                return r
            self._record_miss(2)
        else:
            self._record_miss(2)

        with self._lock:
            self._stats.total_escalations += 1

        # ── Tier 3: UI-TARS-2 / SeeClick ─────────────────────────────────────
        r = self._tier3_uitars(screenshot, element_description)
        if r:
            if r.confidence >= _T3_CONF:
                r.latency_ms = (time.monotonic() - t_start) * 1000
                self._record_hit(3)
                return r
            self._record_miss(3)
        else:
            self._record_miss(3)

        with self._lock:
            self._stats.total_escalations += 1

        # ── Tier 4: Grounding DINO + SAM 2 ───────────────────────────────────
        r = self._tier4_gdino(screenshot, element_description)
        if r:
            if r.confidence >= _T4_CONF:
                r.latency_ms = (time.monotonic() - t_start) * 1000
                self._record_hit(4)
                return r
            self._record_miss(4)
        else:
            self._record_miss(4)

        with self._lock:
            self._stats.total_escalations += 1

        # ── Tier 4b: Aguvis Pure-Vision Grounding ─────────────────────────────
        # Fallback when GroundingDINO failed: pure screenshot-based grounding
        # without accessibility tree. Useful in locked-down environments.
        # Blueprint §6.4 — Xu et al., arXiv:2412.16177
        r = self._tier4b_aguvis(screenshot, element_description, goal_context)
        if r:
            if r.confidence >= _T4_CONF:
                r.latency_ms = (time.monotonic() - t_start) * 1000
                self._record_hit(4)
                return r
            self._record_miss(4)

        with self._lock:
            self._stats.total_escalations += 1

        # ── Tier 5: Cloud VLM fallback ────────────────────────────────────────
        if _CLOUD_ENABLED:
            r = self._tier5_cloud(
                screenshot, element_description,
                goal_context=goal_context, current_app=current_app,
            )
            if r:
                if r.confidence >= _T5_CONF:
                    r.latency_ms = (time.monotonic() - t_start) * 1000
                    self._record_hit(5)
                    return r
                self._record_miss(5)
            else:
                self._record_miss(5)

        with self._lock:
            self._stats.total_failures += 1

        ms = (time.monotonic() - t_start) * 1000
        _logger.warning(
            "[GroundingStack] ALL tiers failed for %r (%.0fms).",
            element_description[:60], ms,
        )
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Tier implementations
    # ─────────────────────────────────────────────────────────────────────────

    def _tier0_atspi(
        self,
        atspi_tree: Optional[Dict[str, Any]],
        description: str,
    ) -> Optional[GroundingResult]:
        if not atspi_tree:
            return None
        try:
            elements = atspi_tree.get("elements", [])
            screen_w = atspi_tree.get("screen_width", 1920)
            screen_h = atspi_tree.get("screen_height", 1080)
            desc_lower = description.lower().strip()

            best, best_score = None, 0.0
            for elem in elements:
                name = str(elem.get("name", "")).lower()
                if name == desc_lower:
                    score = 1.0
                elif desc_lower in name or name in desc_lower:
                    score = 0.85
                else:
                    dw = set(desc_lower.split())
                    nw = set(name.split())
                    overlap = len(dw & nw)
                    score = overlap / max(len(dw), 1) * 0.75 if dw and nw else 0.0

                if score > best_score:
                    best_score, best = score, elem

            if best and best_score >= 0.55:
                bbox = best.get("bbox")
                if bbox and len(bbox) >= 4:
                    px = (bbox[0] + bbox[2]) / 2.0
                    py = (bbox[1] + bbox[3]) / 2.0
                    x = px / screen_w if px > 1.0 else px
                    y = py / screen_h if py > 1.0 else py
                    return GroundingResult(
                        x=x, y=y, confidence=best_score,
                        tier_used=0, tier_name="AT-SPI",
                        label=best.get("name", ""),
                        bbox=bbox, source="atspi",
                    )
        except Exception as exc:
            _logger.debug("[GroundingStack] Tier-0 AT-SPI: %s", exc)
        return None

    def _tier1_omniparser(
        self,
        screenshot,
        description: str,
        pre_parsed: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[GroundingResult]:
        try:
            if self._omni is None:
                try:
                    from core.perception.omniparser import OmniParserV2
                    self._omni = OmniParserV2()
                except Exception:
                    return None

            elements = pre_parsed
            if elements is None:
                elements = self._omni.parse_screenshot(screenshot)
            if not elements:
                return None

            desc_lower = description.lower()
            best, best_score = None, 0.0
            for elem in elements:
                combined = (str(elem.get("label", "")) + " " + str(elem.get("caption", ""))).lower()
                base_conf = float(elem.get("confidence", 0.5))
                if desc_lower in combined:
                    score = base_conf * 0.95
                else:
                    words = set(desc_lower.split())
                    cwords = set(combined.split())
                    score = len(words & cwords) / max(len(words), 1) * base_conf * 0.8 if words else 0.0

                if score > best_score:
                    best_score, best = score, elem

                # GII-WIRE: ProNC novel element observation (Blueprint §11.5)
                # When OmniParser reports low confidence for an element, feed
                # its label + embedding to ProNC so it can learn the class
                # incrementally without retraining.
                _label = str(elem.get("label", "")).strip()
                _elem_conf = float(elem.get("confidence", 0.5))
                if _label and _elem_conf < 0.45:
                    try:
                        from core.learning.progressive_neural_collapse import get_pronc_engine
                        _pronc = get_pronc_engine()
                        # Use bbox center as a simple positional feature vector
                        _bbox = elem.get("bbox", [])
                        if len(_bbox) >= 4:
                            _feat = [
                                (_bbox[0] + _bbox[2]) / 2.0,  # cx
                                (_bbox[1] + _bbox[3]) / 2.0,  # cy
                                float(_bbox[2] - _bbox[0]),    # width
                                float(_bbox[3] - _bbox[1]),    # height
                                _elem_conf,
                            ]
                            _pronc.observe_element(
                                label=_label,
                                features=_feat,
                                app_context=str(elem.get("source", "")),
                                confidence=_elem_conf,
                            )
                    except Exception as _pronc_exc:
                        _logger.debug("[GroundingStack] ProNC observe failed: %s", _pronc_exc)

            if best and best_score > 0.25:
                bbox = best.get("bbox", [])
                if len(bbox) >= 4:
                    iw, ih = (screenshot.size if hasattr(screenshot, "size") else (1920, 1080))
                    x = ((bbox[0] + bbox[2]) / 2.0) / iw
                    y = ((bbox[1] + bbox[3]) / 2.0) / ih
                    return GroundingResult(
                        x=x, y=y, confidence=min(1.0, best_score),
                        tier_used=1, tier_name="OmniParser-V2",
                        label=best.get("label", ""),
                        bbox=bbox, source="omniparser",
                    )
        except Exception as exc:
            _logger.debug("[GroundingStack] Tier-1 OmniParser: %s", exc)
        return None

    def _tier2_gui_actor(
        self,
        screenshot,
        description: str,
    ) -> Optional[GroundingResult]:
        """GUI-Actor with GUI-RC consensus voting."""
        try:
            actor_url = os.environ.get("PROJECTZEO_GUI_ACTOR_URL", "")
            if not actor_url:
                return None

            from core.perception.omniparser import GUIRegionConsensus
            import httpx

            def _call(img, desc, **_kw):
                try:
                    payload = {"image": _encode_b64(img), "query": desc}
                    resp = httpx.post(actor_url, json=payload, timeout=8.0)
                    if resp.status_code == 200:
                        data = resp.json()
                        pts = data.get("point") or [data.get("x"), data.get("y")]
                        if pts and pts[0] is not None:
                            return (float(pts[0]), float(pts[1]))
                except Exception:
                    pass
                return None

            rc = GUIRegionConsensus(n_samples=_GUI_RC_SAMPLES)
            result = rc.vote(_call, screenshot, description)
            if result:
                return GroundingResult(
                    x=result[0], y=result[1], confidence=0.72,
                    tier_used=2, tier_name="GUI-Actor+RC",
                    label=description, source="gui_actor",
                )
        except Exception as exc:
            _logger.debug("[GroundingStack] Tier-2 GUI-Actor: %s", exc)
        return None

    def _tier3_uitars(
        self,
        screenshot,
        description: str,
    ) -> Optional[GroundingResult]:
        try:
            # Try UI-TARS-2 runtime
            if self._uitars is None:
                try:
                    from core.vision.uitars_runtime import UITARSRuntime
                    self._uitars = UITARSRuntime()
                except Exception:
                    pass

            if self._uitars is not None and hasattr(self._uitars, "ground_element"):
                res = self._uitars.ground_element(screenshot, description)
                if res:
                    x = float(res.get("x") or res.get("cx", 0.5))
                    y = float(res.get("y") or res.get("cy", 0.5))
                    conf = float(res.get("confidence", 0.65))
                    return GroundingResult(
                        x=x, y=y, confidence=conf,
                        tier_used=3, tier_name="UI-TARS-2",
                        label=description, source="uitars",
                    )

            # SeeClick HTTP endpoint fallback
            sc_url = os.environ.get("PROJECTZEO_SEECLICK_URL", "")
            if sc_url:
                import httpx
                resp = httpx.post(
                    sc_url,
                    json={"image": _encode_b64(screenshot), "query": description},
                    timeout=10.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return GroundingResult(
                        x=float(data.get("x", 0.5)),
                        y=float(data.get("y", 0.5)),
                        confidence=float(data.get("confidence", 0.6)),
                        tier_used=3, tier_name="SeeClick",
                        label=description, source="seeclick",
                    )
        except Exception as exc:
            _logger.debug("[GroundingStack] Tier-3 UI-TARS: %s", exc)
        return None

    def _tier4_gdino(
        self,
        screenshot,
        description: str,
    ) -> Optional[GroundingResult]:
        """Grounding DINO + SAM 2 open-vocabulary detection."""
        if not self._grounding_adapter_attempted:
            self._grounding_adapter_attempted = True
            try:
                from adapters.grounding_adapter import get_grounding_adapter
                self._grounding_adapter = get_grounding_adapter()
            except Exception as exc:
                _logger.debug("[GroundingStack] Tier-4 adapter unavailable: %s", exc)

        if self._grounding_adapter is None:
            return None

        try:
            results = self._grounding_adapter.ground(screenshot, description, use_sam2=True)
            if not results:
                return None
            best = results[0]
            iw, ih = screenshot.size if hasattr(screenshot, "size") else (1920, 1080)
            x = float(best.x) / iw if float(best.x) > 1.0 else float(best.x)
            y = float(best.y) / ih if float(best.y) > 1.0 else float(best.y)
            return GroundingResult(
                x=x, y=y,
                confidence=min(1.0, float(best.confidence) * 0.95),
                tier_used=4, tier_name="GroundingDINO+SAM2",
                label=best.label,
                mask=best.mask,
                source="grounding_dino",
            )
        except Exception as exc:
            _logger.debug("[GroundingStack] Tier-4 GDINO: %s", exc)
        return None

    def _tier4b_aguvis(
        self,
        screenshot,
        description: str,
        goal_context: str = "",
    ):
        """
        Tier 4b: Aguvis pure-vision grounding.
        Blueprint §6.4 — Falls back to this when GroundingDINO unavailable.
        Operates on screenshot alone without accessibility tree.
        """
        try:
            from core.vision.aguvis_adapter import get_aguvis_adapter
            aguvis = get_aguvis_adapter(
                screen_width  = self._screen_width  if hasattr(self, "_screen_width")  else 1920,
                screen_height = self._screen_height if hasattr(self, "_screen_height") else 1080,
                llm_caller    = self._llm_callable  if hasattr(self, "_llm_callable")  else None,
            )
            if not aguvis.is_available():
                return None
            # Convert screenshot to b64 if needed
            import base64, io
            if hasattr(screenshot, "tobytes"):
                buf = io.BytesIO()
                screenshot.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
            elif isinstance(screenshot, (bytes, bytearray)):
                b64 = base64.b64encode(screenshot).decode()
            else:
                b64 = str(screenshot)

            instruction = f"{description} — {goal_context}" if goal_context else description
            ag_result = aguvis.ground(b64, instruction[:200])
            if ag_result.success and ag_result.pixel_coordinate:
                from core.perception.grounding_stack import GroundingResult as GR
                return GR(
                    element_description = ag_result.element_description or description,
                    coordinate          = ag_result.pixel_coordinate,
                    confidence          = ag_result.confidence,
                    tier_used           = 4,
                    tier_name           = "Aguvis",
                    source              = "aguvis_pure_vision",
                    bbox                = None,
                )
        except Exception as exc:
            _logger.debug("[GroundingStack] Tier-4b Aguvis: %s", exc)
        return None

    def _tier5_cloud(
        self,
        screenshot,
        description: str,

        *,
        goal_context: str = "",
        current_app: str = "",
    ) -> Optional[GroundingResult]:
        """Cloud VLM — last-resort grounding via configured LLM."""
        if self._llm is None:
            return None
        try:
            import base64
            import io
            import re
            import json

            buf = io.BytesIO()
            screenshot.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()

            goal_hint = f" Task: {goal_context[:100]}." if goal_context else ""
            app_hint = f" App: {current_app}." if current_app else ""
            prompt = (
                f"GUI grounding task.{app_hint}{goal_hint}\n"
                f"Find: '{description}'\n"
                "Respond ONLY with JSON: "
                "{\"x\": <0.0-1.0 norm cx>, \"y\": <0.0-1.0 norm cy>, \"confidence\": <0-1>}"
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            raw = self._llm(messages=messages, objective="grounding", session_id="gs_t5")
            if isinstance(raw, list) and raw:
                raw = raw[0].get("content", "") if isinstance(raw[0], dict) else str(raw[0])
            if not isinstance(raw, str):
                return None

            m = re.search(r"\{[^}]+\}", raw)
            if m:
                data = json.loads(m.group(0))
                return GroundingResult(
                    x=float(data.get("x", 0.5)),
                    y=float(data.get("y", 0.5)),
                    confidence=float(data.get("confidence", 0.5)),
                    tier_used=5, tier_name="CloudVLM",
                    label=description, source="cloud_vlm",
                )
        except Exception as exc:
            _logger.debug("[GroundingStack] Tier-5 cloud: %s", exc)
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # State management
    # ─────────────────────────────────────────────────────────────────────────

    def update_llm(self, llm_callable: Callable) -> None:
        self._llm = llm_callable

    def _record_hit(self, tier: int) -> None:
        with self._lock:
            self._stats.tier_hits[tier] = self._stats.tier_hits.get(tier, 0) + 1

    def _record_miss(self, tier: int) -> None:
        with self._lock:
            self._stats.tier_misses[tier] = self._stats.tier_misses.get(tier, 0) + 1

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_calls": self._stats.total_calls,
                "total_escalations": self._stats.total_escalations,
                "total_failures": self._stats.total_failures,
                "tier_hits": dict(self._stats.tier_hits),
                "tier_misses": dict(self._stats.tier_misses),
                "tier_names": _TIER_NAMES,
                "cloud_enabled": _CLOUD_ENABLED,
                "thresholds": {
                    0: _T0_CONF, 1: _T1_CONF, 2: _T2_CONF,
                    3: _T3_CONF, 4: _T4_CONF, 5: _T5_CONF,
                },
            }


def _encode_b64(image) -> str:
    try:
        import base64, io
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────
_instance: Optional[GroundingStack] = None
_lock = threading.Lock()


def get_grounding_stack(**kwargs) -> GroundingStack:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = GroundingStack(**kwargs)
    return _instance


def reset_grounding_stack() -> None:
    global _instance
    with _lock:
        _instance = None
