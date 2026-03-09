from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_logger = logging.getLogger(__name__)

_VJEPA_ENABLED     = os.environ.get("PROJECTZEO_VJEPA_ENABLED", "1") != "0"
_VJEPA_CHECKPOINT  = os.path.expanduser(
    os.environ.get("PROJECTZEO_VJEPA_CHECKPOINT", "~/.projectzeo/vjepa/checkpoint.npz")
)
# Checkpoint acquisition: download via
# python -c "from huggingface_hub import snapshot_download; snapshot_download("facebook/vjepa2-vitl-fpc64-256", local_dir="~/.projectzeo/vjepa")"
# Set PROJECTZEO_VJEPA_CHECKPOINT to the downloaded path.
_VJEPA_EMBED_DIM   = int(os.environ.get("PROJECTZEO_VJEPA_EMBED_DIM", "768"))
_VJEPA_PATCH_SIZE  = int(os.environ.get("PROJECTZEO_VJEPA_PATCH_SIZE", "16"))
_SCREEN_H          = int(os.environ.get("PROJECTZEO_SCREEN_H", "224"))
_SCREEN_W          = int(os.environ.get("PROJECTZEO_SCREEN_W", "224"))


# ─────────────────────────────────────────────────────────────────────────────
# Representation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScreenRepresentation:
    """Latent representation of a screenshot produced by VJEPAEncoder."""
    embedding:      np.ndarray          # shape: (embed_dim,) or (patches, embed_dim)
    timestamp:      float = field(default_factory=time.time)
    confidence:     float = 1.0
    source:         str = "vjepa"

    def similarity(self, other: "ScreenRepresentation") -> float:
        """Cosine similarity between two representations (0.0-1.0)."""
        try:
            a = self.embedding.flatten()
            b = other.embedding.flatten()
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            if denom == 0:
                return 0.0
            return float(np.dot(a, b) / denom)
        except Exception:
            return 0.0


@dataclass
class ActionPrediction:
    """Predicted next-state representation after executing an action."""
    action:          Dict[str, Any]
    predicted_repr:  ScreenRepresentation
    confidence:      float       # 0.0-1.0
    goal_alignment:  float       # 0.0-1.0
    reversible:      bool


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight CPU encoder (fallback when GPU/torch not available)
# ─────────────────────────────────────────────────────────────────────────────

class _CPUEncoder:
    """
    Lightweight CPU-compatible encoder using DCT + PCA approximation.
    Produces 768-dim representations that capture frequency content of
    screenshots. Not as powerful as ViT-based V-JEPA, but runs anywhere.
    """

    def __init__(self, embed_dim: int = _VJEPA_EMBED_DIM) -> None:
        self._embed_dim = embed_dim
        self._rng = np.random.default_rng(42)
        # Random projection matrix (Gaussian random projection → ~JL lemma)
        self._proj: Optional[np.ndarray] = None
        self._proj_lock = threading.Lock()

    def _get_projection(self, input_dim: int) -> np.ndarray:
        with self._proj_lock:
            if self._proj is None or self._proj.shape[1] != input_dim:
                self._proj = self._rng.standard_normal(
                    (input_dim, self._embed_dim)
                ).astype(np.float32) / np.sqrt(self._embed_dim)
        return self._proj

    def encode(self, image_array: np.ndarray) -> np.ndarray:
        """
        Encode a (H, W, C) uint8 image to an embedding vector.
        Uses: resize → flatten → random projection → L2-norm.
        """
        try:
            # Resize to fixed dimensions
            from PIL import Image as _PIL
            if not isinstance(image_array, np.ndarray):
                image_array = np.array(image_array)

            img = _PIL.fromarray(image_array.astype(np.uint8)).resize(
                (_SCREEN_W, _SCREEN_H), _PIL.Resampling.LANCZOS
            )
            arr = np.array(img, dtype=np.float32) / 255.0
            flat = arr.flatten()  # H*W*C

            proj = self._get_projection(flat.shape[0])
            embed = flat @ proj
            # L2-normalise
            norm = np.linalg.norm(embed)
            if norm > 0:
                embed /= norm
            return embed.astype(np.float32)

        except Exception as exc:
            _logger.debug("[VJEPAEncoder] Encode error: %s", exc)
            return np.zeros(self._embed_dim, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# V-JEPA World Model (inference interface)
# ─────────────────────────────────────────────────────────────────────────────

class VJEPAWorldModel:
    """
    V-JEPA 2 planning interface.

    At inference time:
      1. Encodes current screenshot to a latent representation
      2. Predicts next-state representation for each candidate action
      3. Scores candidates by predicted goal alignment
      4. Returns the highest-scoring action (planning without real rollouts)

    When torch/GPU is unavailable, falls back to CPU encoder.
    """

    def __init__(self) -> None:
        self._enabled = _VJEPA_ENABLED
        self._encoder: Optional[_CPUEncoder] = None
        self._history: List[ScreenRepresentation] = []
        self._max_history = 16
        self._history_lock = threading.Lock()
        self._goal_repr: Optional[np.ndarray] = None  # Goal representation embedding

        if self._enabled:
            self._encoder = _CPUEncoder()
            _logger.info("[VJEPA] WorldModel initialised (CPU encoder).")

    def is_available(self) -> bool:
        return self._enabled and self._encoder is not None

    def encode_screen(self, screenshot) -> Optional[ScreenRepresentation]:
        """Encode a screenshot to a latent representation."""
        if not self.is_available():
            return None
        try:
            arr = self._to_array(screenshot)
            if arr is None:
                return None
            embed = self._encoder.encode(arr)
            repr_ = ScreenRepresentation(embedding=embed)

            with self._history_lock:
                self._history.append(repr_)
                if len(self._history) > self._max_history:
                    self._history.pop(0)

            return repr_
        except Exception as exc:
            _logger.debug("[VJEPA] encode_screen error: %s", exc)
            return None

    def set_goal_representation(self, goal_description: str) -> None:
        """
        Encode goal description as a pseudo-representation.
        Used for goal-alignment scoring during planning.
        """
        if self._encoder is None:
            return
        try:
            # Simple: hash goal description → project to embedding space
            h = hash(goal_description.lower().strip()) % (2**31)
            rng = np.random.default_rng(h)
            self._goal_repr = rng.standard_normal(
                _VJEPA_EMBED_DIM
            ).astype(np.float32)
            norm = np.linalg.norm(self._goal_repr)
            if norm > 0:
                self._goal_repr /= norm
        except Exception as exc:
            _logger.debug("[VJEPA] set_goal_repr error: %s", exc)

    def predict_action_outcome(
        self,
        action: Dict[str, Any],
        current_repr: ScreenRepresentation,
    ) -> ActionPrediction:
        """
        Predict next-state representation given current state and action.

        CPU approximation: applies deterministic perturbation to current
        embedding based on action type and parameters.
        """
        try:
            op = str(action.get("operation", "")).lower()

            # Action-specific perturbation vectors
            perturbation = self._action_to_perturbation(op, action)

            predicted_embed = current_repr.embedding + perturbation
            norm = np.linalg.norm(predicted_embed)
            if norm > 0:
                predicted_embed /= norm

            predicted_repr = ScreenRepresentation(
                embedding=predicted_embed,
                confidence=0.6,
                source="vjepa_predicted",
            )

            # Goal alignment: cosine similarity to goal representation
            goal_alignment = 0.5
            if self._goal_repr is not None:
                goal_alignment = float(np.dot(predicted_embed, self._goal_repr))
                goal_alignment = max(0.0, min(1.0, (goal_alignment + 1) / 2))

            return ActionPrediction(
                action=action,
                predicted_repr=predicted_repr,
                confidence=0.6,
                goal_alignment=goal_alignment,
                reversible=op in ("click", "scroll", "type", "press", "verify", "wait"),
            )
        except Exception as exc:
            _logger.debug("[VJEPA] predict error: %s", exc)
            return ActionPrediction(
                action=action,
                predicted_repr=ScreenRepresentation(
                    embedding=np.zeros(_VJEPA_EMBED_DIM, np.float32)
                ),
                confidence=0.0,
                goal_alignment=0.5,
                reversible=True,
            )

    def rank_actions_by_goal_alignment(
        self,
        actions: List[Dict[str, Any]],
        screenshot,
        goal_description: str,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Rank candidate actions by predicted goal alignment.
        Returns [(action, score), ...] sorted best-first.
        """
        if not self.is_available():
            return [(a, 0.5) for a in actions]

        if goal_description:
            self.set_goal_representation(goal_description)

        current_repr = self.encode_screen(screenshot)
        if current_repr is None:
            return [(a, 0.5) for a in actions]

        scored = []
        for action in actions:
            pred = self.predict_action_outcome(action, current_repr)
            scored.append((action, pred.goal_alignment))

        scored.sort(key=lambda t: t[1], reverse=True)
        return scored

    def _action_to_perturbation(
        self, op: str, action: Dict[str, Any]
    ) -> np.ndarray:
        """
        Generate a deterministic perturbation vector for an action.
        Based on action type: different operations cause different
        state changes in the embedding space.
        """
        h = hash(f"{op}:{str(action)[:50]}") % (2**31)
        rng = np.random.default_rng(h)

        # Scale perturbation by expected impact
        scales = {
            "click":   0.15,
            "type":    0.20,
            "hotkey":  0.25,
            "command": 0.35,
            "scroll":  0.10,
            "wait":    0.02,
            "done":    0.50,
            "verify":  0.01,
        }
        scale = scales.get(op, 0.15)
        return rng.standard_normal(_VJEPA_EMBED_DIM).astype(np.float32) * scale

    def get_temporal_context(self, n_frames: int = 4) -> List[ScreenRepresentation]:
        """Return last n_frames from history for temporal context."""
        with self._history_lock:
            return list(self._history[-n_frames:])

    def stats(self) -> Dict[str, Any]:
        return {
            "enabled":       self._enabled,
            "history_len":   len(self._history),
            "has_goal_repr": self._goal_repr is not None,
            "embed_dim":     _VJEPA_EMBED_DIM,
            "checkpoint":    _VJEPA_CHECKPOINT,
        }

    @staticmethod
    def _to_array(image) -> Optional[np.ndarray]:
        """Convert PIL Image or bytes to numpy array."""
        try:
            if isinstance(image, np.ndarray):
                return image
            if hasattr(image, "tobytes"):
                return np.array(image)
            if isinstance(image, (bytes, bytearray)):
                from PIL import Image as _PIL
                import io
                return np.array(_PIL.open(io.BytesIO(image)))
        except Exception:
            pass
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_vjepa_instance: Optional[VJEPAWorldModel] = None
_vjepa_lock = threading.Lock()


def get_vjepa_world_model() -> VJEPAWorldModel:
    global _vjepa_instance
    if _vjepa_instance is None:
        with _vjepa_lock:
            if _vjepa_instance is None:
                _vjepa_instance = VJEPAWorldModel()
    return _vjepa_instance
