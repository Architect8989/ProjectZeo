from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Tunables
# ─────────────────────────────────────────────────────────────────────────────
_PERSIST_PATH           = os.path.expanduser(
    os.environ.get("PROJECTZEO_PRONC_PATH", "~/.projectzeo/pronc_registry.json")
)
_NOVEL_CONFIDENCE_THRESHOLD = float(
    os.environ.get("PROJECTZEO_PRONC_NOVEL_CONF", "0.45")
)
_MIN_OBSERVATIONS_FOR_CLASS = int(
    os.environ.get("PROJECTZEO_PRONC_MIN_OBS", "5")
)
_MAX_EXEMPLARS_PER_CLASS = int(
    os.environ.get("PROJECTZEO_PRONC_MAX_EXEMPLARS", "20")
)
_MAX_CLASSES = int(
    os.environ.get("PROJECTZEO_PRONC_MAX_CLASSES", "500")
)
_NCM_DISTANCE_THRESHOLD = float(
    os.environ.get("PROJECTZEO_PRONC_NCM_THRESHOLD", "2.5")
)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Exemplar:
    """One observed instance of a GUI element class."""
    label:       str
    features:    List[float]    # Feature vector (from OmniParser embedding)
    app_context: str            # Application name when observed
    confidence:  float          # Original OmniParser confidence
    timestamp:   float = field(default_factory=time.time)


@dataclass
class ClassPrototype:
    """
    Class prototype in the NCM (Nearest Class Mean) classifier.

    centroid: mean of all exemplar feature vectors
    n_obs:    total observations seen (includes evicted exemplars)
    """
    label:       str
    centroid:    List[float]    # Mean feature vector
    n_obs:       int            = 0
    n_exemplars: int            = 0
    registered_at: float       = field(default_factory=time.time)
    last_seen:   float         = field(default_factory=time.time)
    app_contexts: List[str]    = field(default_factory=list)

    def update_centroid(self, exemplars: List[Exemplar]) -> None:
        """Recompute centroid from exemplar set."""
        if not exemplars:
            return
        n = len(exemplars)
        dim = len(exemplars[0].features)
        centroid = [0.0] * dim
        for ex in exemplars:
            for i, v in enumerate(ex.features[:dim]):
                centroid[i] += v / n
        self.centroid = centroid
        self.n_exemplars = n
        self.last_seen = time.time()

    def distance(self, features: List[float]) -> float:
        """L2 distance from features to this class centroid."""
        if not self.centroid or not features:
            return float("inf")
        dim = min(len(self.centroid), len(features))
        return math.sqrt(
            sum((self.centroid[i] - features[i]) ** 2 for i in range(dim))
        )


# ─────────────────────────────────────────────────────────────────────────────
# ProNCEngine
# ─────────────────────────────────────────────────────────────────────────────

class ProNCEngine:
    """
    Progressive Neural Collapse engine for class-incremental GUI grounding.

    Maintains a registry of known GUI element classes with NCM classification.
    Automatically promotes novel classes when enough exemplars accumulate.
    Prevents catastrophic forgetting via bounded exemplar replay.
    """

    def __init__(self, persist_path: Optional[str] = None) -> None:
        self._path      = persist_path or _PERSIST_PATH
        self._lock      = threading.Lock()

        # Registered classes: label → ClassPrototype
        self._classes:   Dict[str, ClassPrototype] = {}

        # Observation buffer: label → List[Exemplar]
        self._exemplars: Dict[str, List[Exemplar]] = defaultdict(list)

        # Candidate counter: label → observation count (before registration)
        self._candidate_counts: Dict[str, int] = defaultdict(int)

        # Stats
        self._total_observed     = 0
        self._total_registered   = 0
        self._total_predictions  = 0
        self._ncm_hits           = 0
        self._novel_hits         = 0

        self._load()
        _logger.info(
            "[ProNC] Initialized. classes=%d exemplar_labels=%d",
            len(self._classes), len(self._exemplars),
        )

    # ── Core API ──────────────────────────────────────────────────────────────

    def observe_element(
        self,
        label: str,
        features: Optional[List[float]],
        app_context: str = "",
        confidence: float = 0.0,
    ) -> None:
        """
        Record an observation of a GUI element.

        Call this from grounding_stack.py when OmniParser returns a result
        with confidence < _NOVEL_CONFIDENCE_THRESHOLD.

        After _MIN_OBSERVATIONS_FOR_CLASS calls with the same label,
        the label is automatically registered as a known class.
        """
        if not label or not features:
            return
        label = label.strip().lower()
        if not label:
            return

        with self._lock:
            self._total_observed += 1
            self._candidate_counts[label] += 1

            # Store exemplar
            ex = Exemplar(
                label=label,
                features=list(features[:256]),  # cap at 256 dims
                app_context=app_context.lower()[:40],
                confidence=confidence,
            )
            self._exemplars[label].append(ex)

            # Bound exemplar buffer per class
            if len(self._exemplars[label]) > _MAX_EXEMPLARS_PER_CLASS:
                self._exemplars[label] = self._select_diverse_exemplars(
                    self._exemplars[label], _MAX_EXEMPLARS_PER_CLASS
                )

            # Auto-promote to registered class when threshold met
            if (
                self._candidate_counts[label] >= _MIN_OBSERVATIONS_FOR_CLASS
                and label not in self._classes
            ):
                self._register_class_locked(label)

            # Update existing class if already registered
            elif label in self._classes:
                proto = self._classes[label]
                proto.n_obs += 1
                proto.last_seen = time.time()
                if app_context and app_context.lower() not in proto.app_contexts:
                    proto.app_contexts.append(app_context.lower()[:40])
                    proto.app_contexts = proto.app_contexts[-10:]  # keep last 10
                # Lazily update centroid every 5 new exemplars
                if proto.n_obs % 5 == 0:
                    proto.update_centroid(self._exemplars[label])

        self._maybe_save()

    def predict(
        self,
        features: Optional[List[float]],
        top_k: int = 1,
        app_context: str = "",
    ) -> Tuple[Optional[str], float]:
        """
        NCM-based prediction: find the nearest class mean to features.

        Returns (label, confidence) of the nearest registered class.
        Returns (None, 0.0) if no classes are registered or features is None.

        confidence is computed from: 1 - (distance / _NCM_DISTANCE_THRESHOLD)
        clamped to [0.0, 1.0].
        """
        if not features:
            return None, 0.0

        with self._lock:
            self._total_predictions += 1
            if not self._classes:
                return None, 0.0

            best_label    = None
            best_distance = float("inf")

            for label, proto in self._classes.items():
                if not proto.centroid:
                    continue
                # Boost app-contextual matches
                dist = proto.distance(features)
                if app_context and app_context.lower() in proto.app_contexts:
                    dist *= 0.85  # 15% contextual bonus

                if dist < best_distance:
                    best_distance = dist
                    best_label    = label

            if best_label is None or best_distance > _NCM_DISTANCE_THRESHOLD:
                self._novel_hits += 1
                return None, 0.0

            confidence = max(0.0, min(1.0, 1.0 - (best_distance / _NCM_DISTANCE_THRESHOLD)))
            self._ncm_hits += 1
            return best_label, round(confidence, 4)

    def register_class(self, label: str) -> bool:
        """
        Explicitly register a class.
        Returns True if registered, False if already known or no exemplars.
        """
        label = label.strip().lower()
        with self._lock:
            if label in self._classes:
                return False
            if not self._exemplars.get(label):
                _logger.debug("[ProNC] register_class: no exemplars for %r", label)
                return False
            self._register_class_locked(label)
            return True

    def consolidate(self) -> Dict[str, Any]:
        """
        Nightly consolidation: recompute all centroids and evict stale exemplars.

        Called by core/learning/nightly_consolidation.py.
        Returns stats dict.
        """
        _logger.info("[ProNC] Consolidating...")
        t0 = time.time()

        with self._lock:
            evicted_classes = 0
            updated_centroids = 0

            # Evict classes not seen in 30 days
            stale_threshold = time.time() - (30 * 86400)
            stale_labels = [
                lbl for lbl, proto in self._classes.items()
                if proto.last_seen < stale_threshold and proto.n_obs < 10
            ]
            for lbl in stale_labels:
                del self._classes[lbl]
                self._exemplars.pop(lbl, None)
                evicted_classes += 1

            # Recompute all centroids
            for label, proto in self._classes.items():
                exemplars = self._exemplars.get(label, [])
                if exemplars:
                    proto.update_centroid(exemplars)
                    updated_centroids += 1

            # Enforce max classes limit (evict lowest-obs classes)
            if len(self._classes) > _MAX_CLASSES:
                sorted_by_obs = sorted(
                    self._classes.items(),
                    key=lambda kv: kv[1].n_obs,
                )
                to_evict = len(self._classes) - _MAX_CLASSES
                for lbl, _ in sorted_by_obs[:to_evict]:
                    del self._classes[lbl]
                    self._exemplars.pop(lbl, None)
                    evicted_classes += 1

        self._save()
        elapsed = time.time() - t0
        stats = {
            "classes_total":    len(self._classes),
            "evicted_classes":  evicted_classes,
            "updated_centroids": updated_centroids,
            "elapsed_s":        round(elapsed, 2),
        }
        _logger.info("[ProNC] Consolidation complete: %s", stats)
        return stats

    def get_novel_candidates(self, min_count: int = 3) -> List[Dict[str, Any]]:
        """
        Return labels that have been observed but not yet registered as classes.
        Useful for review/debugging.
        """
        with self._lock:
            return [
                {
                    "label":    lbl,
                    "count":    cnt,
                    "promoted": lbl in self._classes,
                }
                for lbl, cnt in self._candidate_counts.items()
                if cnt >= min_count
            ]

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "registered_classes":  len(self._classes),
                "candidate_labels":    len(self._candidate_counts),
                "total_observed":      self._total_observed,
                "total_registered":    self._total_registered,
                "total_predictions":   self._total_predictions,
                "ncm_hits":            self._ncm_hits,
                "novel_hits":          self._novel_hits,
                "exemplar_labels":     len(self._exemplars),
                "novel_conf_threshold": _NOVEL_CONFIDENCE_THRESHOLD,
                "min_obs_for_class":   _MIN_OBSERVATIONS_FOR_CLASS,
            }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _register_class_locked(self, label: str) -> None:
        """Must be called with self._lock held."""
        exemplars = self._exemplars.get(label, [])
        if not exemplars:
            return

        proto = ClassPrototype(label=label)
        proto.n_obs = self._candidate_counts.get(label, len(exemplars))
        proto.update_centroid(exemplars)

        # Collect app contexts
        proto.app_contexts = list({ex.app_context for ex in exemplars if ex.app_context})[:10]

        self._classes[label] = proto
        self._total_registered += 1
        _logger.info(
            "[ProNC] New class registered: %r n_obs=%d dim=%d apps=%s",
            label, proto.n_obs, len(proto.centroid), proto.app_contexts[:3],
        )

    def _select_diverse_exemplars(
        self,
        exemplars: List[Exemplar],
        k: int,
    ) -> List[Exemplar]:
        """
        Keep the k most diverse exemplars via greedy max-distance selection.
        This is a simplified medoid selection to maintain representativeness.
        """
        if len(exemplars) <= k:
            return exemplars

        # Start with the most recent exemplar
        selected = [exemplars[-1]]
        remaining = exemplars[:-1]

        while len(selected) < k and remaining:
            # Pick the exemplar maximally distant from all selected
            best = max(
                remaining,
                key=lambda ex: min(
                    _l2_distance(ex.features, s.features) for s in selected
                ),
            )
            selected.append(best)
            remaining.remove(best)

        return selected

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)

            for lbl, pd in data.get("classes", {}).items():
                try:
                    proto = ClassPrototype(
                        label=pd["label"],
                        centroid=pd.get("centroid", []),
                        n_obs=pd.get("n_obs", 0),
                        n_exemplars=pd.get("n_exemplars", 0),
                        registered_at=pd.get("registered_at", time.time()),
                        last_seen=pd.get("last_seen", time.time()),
                        app_contexts=pd.get("app_contexts", []),
                    )
                    self._classes[lbl] = proto
                except Exception as exc:
                    _logger.debug("[ProNC] Skipping class %r on load: %s", lbl, exc)

            for lbl, exs in data.get("exemplars", {}).items():
                for ed in exs:
                    try:
                        ex = Exemplar(
                            label=ed["label"],
                            features=ed.get("features", []),
                            app_context=ed.get("app_context", ""),
                            confidence=ed.get("confidence", 0.0),
                            timestamp=ed.get("timestamp", time.time()),
                        )
                        self._exemplars[lbl].append(ex)
                    except Exception:
                        pass

            self._candidate_counts = defaultdict(
                int, data.get("candidate_counts", {})
            )
            _logger.info(
                "[ProNC] Loaded: classes=%d exemplar_labels=%d",
                len(self._classes), len(self._exemplars),
            )
        except Exception as exc:
            _logger.warning("[ProNC] Load failed: %s", exc)

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        tmp = self._path + ".tmp"
        try:
            with self._lock:
                data = {
                    "classes": {
                        lbl: {
                            "label":         proto.label,
                            "centroid":      proto.centroid,
                            "n_obs":         proto.n_obs,
                            "n_exemplars":   proto.n_exemplars,
                            "registered_at": proto.registered_at,
                            "last_seen":     proto.last_seen,
                            "app_contexts":  proto.app_contexts,
                        }
                        for lbl, proto in self._classes.items()
                    },
                    "exemplars": {
                        lbl: [
                            {
                                "label":       ex.label,
                                "features":    ex.features[:64],  # save top-64 dims
                                "app_context": ex.app_context,
                                "confidence":  ex.confidence,
                                "timestamp":   ex.timestamp,
                            }
                            for ex in exs[-_MAX_EXEMPLARS_PER_CLASS:]
                        ]
                        for lbl, exs in self._exemplars.items()
                    },
                    "candidate_counts": dict(self._candidate_counts),
                    "saved_at": time.time(),
                }
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, separators=(",", ":"))
            os.replace(tmp, self._path)
        except Exception as exc:
            _logger.debug("[ProNC] Save failed: %s", exc)

    _save_counter = 0

    def _maybe_save(self) -> None:
        """Save every 50 observations to avoid constant disk I/O."""
        self._save_counter += 1
        if self._save_counter % 50 == 0:
            threading.Thread(target=self._save, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _l2_distance(a: List[float], b: List[float]) -> float:
    dim = min(len(a), len(b))
    if dim == 0:
        return float("inf")
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(dim)))


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_instance: Optional[ProNCEngine] = None
_instance_lock = threading.Lock()


def get_pronc_engine() -> ProNCEngine:
    """Return the global singleton ProNCEngine."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = ProNCEngine()
    return _instance


def reset_pronc_singleton() -> None:
    """Reset the singleton — call between test runs."""
    global _instance
    with _instance_lock:
        _instance = None
