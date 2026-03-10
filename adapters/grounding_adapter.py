from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

_USE_GROUNDING = os.environ.get("PROJECTZEO_USE_GROUNDING_DINO", "0").strip() == "1"
_WEIGHTS_DIR = os.path.expanduser(
    os.environ.get("PROJECTZEO_WEIGHTS_DIR", "~/.projectzeo/weights")
)
_GDINO_WEIGHTS = os.path.join(_WEIGHTS_DIR, "groundingdino_swint_ogc.pth")
_GDINO_CONFIG  = os.path.join(_WEIGHTS_DIR, "GroundingDINO_SwinT_OGC.py")
_SAM2_WEIGHTS  = os.path.join(_WEIGHTS_DIR, "sam2_hiera_large.pt")

_BOX_THRESHOLD  = float(os.environ.get("PROJECTZEO_GDINO_THRESHOLD", "0.3"))
_TEXT_THRESHOLD = float(os.environ.get("PROJECTZEO_GDINO_TEXT_THRESHOLD", "0.25"))

@dataclass
class GroundingResult:
    label: str
    x: float
    y: float
    width: float
    height: float
    confidence: float
    mask: Optional[Any] = None

def _check_gdino() -> bool:
    try:
        import groundingdino
        return True
    except ImportError:
        return False

def _check_sam2() -> bool:
    try:
        import sam2
        return True
    except ImportError:
        return False

def _check_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

class GroundingAdapter:

    def __init__(self) -> None:
        self._gdino_model = None
        self._sam2_model = None
        self._device = "cpu"
        self._lock = threading.Lock()
        self._available = False
        self._mode = "unavailable"

        if _USE_GROUNDING or (_check_gdino() and _check_gpu()):
            self._try_init()

    def _try_init(self) -> None:
        try:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            os.makedirs(_WEIGHTS_DIR, exist_ok=True)
            self._auto_download_weights()

            if _check_gdino() and os.path.isfile(_GDINO_WEIGHTS):
                from groundingdino.util.inference import load_model
                self._gdino_model = load_model(_GDINO_CONFIG, _GDINO_WEIGHTS)
                self._available = True
                self._mode = "grounding_dino"
                _logger.info("[GroundingAdapter] Grounding DINO loaded on %s.", self._device)

            if _check_sam2() and os.path.isfile(_SAM2_WEIGHTS):
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                _sam2 = build_sam2("sam2_hiera_l", _SAM2_WEIGHTS, device=self._device)
                self._sam2_model = SAM2ImagePredictor(_sam2)
                self._mode = "grounding_dino+sam2"
                _logger.info("[GroundingAdapter] SAM 2 loaded on %s.", self._device)

        except Exception as exc:
            _logger.info(
                "[GroundingAdapter] Init skipped (GPU/weights not available): %s. "
                "Install: pip install groundingdino-py segment-anything-2",
                exc,
            )

    def _auto_download_weights(self) -> None:
        _GDINO_URL = (
            "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
            "v0.1.0-alpha/groundingdino_swint_ogc.pth"
        )
        _SAM2_URL = (
            "https://dl.fbaipublicfiles.com/segment_anything_2/072824/"
            "sam2_hiera_large.pt"
        )
        for path, url, name in [
            (_GDINO_WEIGHTS, _GDINO_URL, "Grounding DINO"),
            (_SAM2_WEIGHTS, _SAM2_URL, "SAM 2"),
        ]:
            if not os.path.isfile(path):
                _logger.info("[GroundingAdapter] Downloading %s weights...", name)
                try:
                    import urllib.request
                    urllib.request.urlretrieve(url, path)
                    _logger.info("[GroundingAdapter] %s weights saved → %s", name, path)
                except Exception as exc:
                    _logger.warning(
                        "[GroundingAdapter] Weight download failed for %s: %s. "
                        "Download manually from %s",
                        name, exc, url,
                    )

    def ground(
        self,
        image,
        text_query: str,
        *,
        use_sam2: bool = True,
    ) -> List[GroundingResult]:
        if not self._available or self._gdino_model is None:
            return []

        try:
            return self._run_grounding(image, text_query, use_sam2=use_sam2)
        except Exception as exc:
            _logger.debug("[GroundingAdapter] Grounding failed (non-fatal): %s", exc)
            return []

    def _run_grounding(
        self, image, text_query: str, *, use_sam2: bool
    ) -> List[GroundingResult]:
        import numpy as np
        from PIL import Image as PILImage

        if isinstance(image, np.ndarray):
            pil_img = PILImage.fromarray(image)
        else:
            pil_img = image
        w, h = pil_img.size

        from groundingdino.util.inference import predict as gdino_predict, annotate
        import torch

        boxes, logits, phrases = gdino_predict(
            model=self._gdino_model,
            image=pil_img,
            caption=text_query,
            box_threshold=_BOX_THRESHOLD,
            text_threshold=_TEXT_THRESHOLD,
            device=self._device,
        )

        results: List[GroundingResult] = []
        img_array = np.array(pil_img)

        for box, logit, phrase in zip(boxes, logits, phrases):
            cx, cy, bw, bh = box.tolist()
            confidence = float(logit)

            mask = None
            if use_sam2 and self._sam2_model is not None:
                try:
                    px1 = int((cx - bw / 2) * w)
                    py1 = int((cy - bh / 2) * h)
                    px2 = int((cx + bw / 2) * w)
                    py2 = int((cy + bh / 2) * h)

                    self._sam2_model.set_image(img_array)
                    masks, _, _ = self._sam2_model.predict(
                        box=np.array([[px1, py1, px2, py2]]),
                        multimask_output=False,
                    )
                    mask = masks[0] if masks is not None and len(masks) > 0 else None
                except Exception as sam_exc:
                    _logger.debug("[GroundingAdapter] SAM2 mask failed: %s", sam_exc)

            results.append(GroundingResult(
                label=phrase,
                x=cx,
                y=cy,
                width=bw,
                height=bh,
                confidence=confidence,
                mask=mask,
            ))

        return sorted(results, key=lambda r: r.confidence, reverse=True)

    def ground_to_click_coords(
        self, image, text_query: str
    ) -> Optional[Tuple[float, float]]:
        results = self.ground(image, text_query, use_sam2=False)
        if not results:
            return None
        best = results[0]
        return (best.x, best.y)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "available": self._available,
            "mode": self._mode,
            "device": self._device,
            "gdino_installed": _check_gdino(),
            "sam2_installed": _check_sam2(),
            "gpu": _check_gpu(),
            "weights_dir": _WEIGHTS_DIR,
        }

_instance: Optional[GroundingAdapter] = None
_lock = threading.Lock()

def get_grounding_adapter() -> GroundingAdapter:
    global _instance
    with _lock:
        if _instance is None:
            _instance = GroundingAdapter()
        return _instance
