"""
core/perception/snn_event_processor.py

Spiking Neural Network (SNN) processor for AT-SPI accessibility event streams.

Reference:
  SpikingJelly — https://github.com/fangwei123456/spikingjelly
  Mahowald & Douglas 1991 — Silicon retina (event-driven vision)
  Schuman et al. 2017 — A Survey of Neuromorphic Computing and Neural Networks

Role:
  AT-SPI fires events (focus-changed, text-changed, state-changed, etc.) as
  discrete signals. An SNN is a natural fit — events map to spikes, membrane
  potentials accumulate, and neurons fire when UI activity exceeds threshold.

  In practice: this processes the AT-SPI event stream and produces a
  compressed "activity summary" that the GIIController uses to detect
  relevant UI changes without looking at every event individually.

Fallback:
  SpikingJelly is optional. If not installed, the module falls back to a
  simple leaky-integrator approximation that mimics SNN behaviour.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

_logger = logging.getLogger(__name__)

_TAU         = float(os.environ.get("PROJECTZEO_SNN_TAU", "0.9"))       # membrane decay
_THRESHOLD   = float(os.environ.get("PROJECTZEO_SNN_THRESHOLD", "1.5")) # spike threshold
_N_NEURONS   = int(os.environ.get("PROJECTZEO_SNN_NEURONS", "32"))      # feature neurons
_WINDOW_SEC  = float(os.environ.get("PROJECTZEO_SNN_WINDOW", "2.0"))    # time window

try:
    from spikingjelly.activation_based import neuron as _sj_neuron  # type: ignore
    _SJ_AVAILABLE = True
except ImportError:
    _sj_neuron    = None
    _SJ_AVAILABLE = False


@dataclass
class ATSPIEvent:
    event_type:  str
    source_role: str
    source_name: str
    app_name:    str
    timestamp:   float = field(default_factory=time.time)
    detail:      str   = ""


@dataclass
class ActivitySummary:
    active_apps:      List[str]
    dominant_event:   str
    spike_count:      int
    activity_level:   float   # 0.0 (idle) → 1.0 (very active)
    novel_ui_change:  bool
    window_ms:        float


class LIFNeuronLayer:
    """
    Leaky Integrate-and-Fire neuron layer.

    Fall-back when SpikingJelly is not installed. Simulates:
      V_m(t+1) = τ * V_m(t) + I(t)
      spike if V_m >= threshold → V_m = 0
    """

    def __init__(self, n: int, tau: float, threshold: float) -> None:
        self._v   = np.zeros(n)
        self._tau = tau
        self._thr = threshold

    def forward(self, current: np.ndarray) -> np.ndarray:
        """current: shape (n,). Returns spike binary vector."""
        self._v = self._tau * self._v + current
        spikes  = (self._v >= self._thr).astype(np.float32)
        self._v = self._v * (1 - spikes)  # reset on spike
        return spikes

    def reset(self) -> None:
        self._v[:] = 0.0


class SNNLayer:
    """Thin wrapper that uses SpikingJelly if available, else LIF."""

    def __init__(self, n: int) -> None:
        self._n    = n
        if _SJ_AVAILABLE:
            self._lif = _sj_neuron.LIFNode(tau=_TAU, v_threshold=_THRESHOLD)
            self._mode = "sj"
        else:
            self._lif = LIFNeuronLayer(n, _TAU, _THRESHOLD)
            self._mode = "fallback"

    def step(self, current: np.ndarray) -> np.ndarray:
        if self._mode == "sj":
            try:
                import torch  # type: ignore
                t = torch.tensor(current, dtype=torch.float32)
                out = self._lif(t).detach().numpy()
                return out.astype(np.float32)
            except Exception:
                self._mode = "fallback"
                self._lif  = LIFNeuronLayer(self._n, _TAU, _THRESHOLD)
        return self._lif.forward(current)

    def reset(self) -> None:
        if self._mode == "sj":
            try:
                self._lif.reset()
            except Exception:
                pass
        else:
            self._lif.reset()


class SNNEventProcessor:
    """
    Encodes AT-SPI event streams into neural spike trains and
    produces activity summaries for the GIIController.
    """

    _EVENT_WEIGHTS: Dict[str, float] = {
        "focus:changed":         2.0,
        "object:state-changed":  1.0,
        "object:text-changed":   1.5,
        "window:create":         2.5,
        "window:destroy":        2.0,
        "window:activate":       1.8,
        "document:load-complete": 1.2,
    }

    def __init__(self) -> None:
        self._layer = SNNLayer(_N_NEURONS)
        self._lock  = threading.Lock()
        self._buffer: Deque[ATSPIEvent] = deque(maxlen=500)
        self._spike_log: Deque[float]   = deque(maxlen=200)
        self._prev_state: Optional[np.ndarray] = None
        _logger.info("[SNN] Event processor init. neurons=%d mode=%s", _N_NEURONS, self._layer._mode)

    def ingest(self, event: ATSPIEvent) -> None:
        with self._lock:
            self._buffer.append(event)
        self._process_event(event)

    def ingest_dict(self, d: Dict[str, Any]) -> None:
        event = ATSPIEvent(
            event_type=str(d.get("type", "")),
            source_role=str(d.get("role", "")),
            source_name=str(d.get("name", ""))[:80],
            app_name=str(d.get("app", ""))[:40],
            timestamp=float(d.get("timestamp", time.time())),
            detail=str(d.get("detail", ""))[:100],
        )
        self.ingest(event)

    def _process_event(self, event: ATSPIEvent) -> None:
        current = self._encode_event(event)
        with self._lock:
            spikes = self._layer.step(current)
            n_spikes = int(spikes.sum())
            if n_spikes > 0:
                self._spike_log.append(time.time())

    def _encode_event(self, event: ATSPIEvent) -> np.ndarray:
        current = np.zeros(_N_NEURONS, dtype=np.float32)
        weight  = self._EVENT_WEIGHTS.get(event.event_type, 0.5)

        # Distribute stimulus across neurons based on event hash
        h     = hash(event.event_type + event.app_name) % _N_NEURONS
        width = max(1, _N_NEURONS // 8)
        for i in range(width):
            idx = (h + i) % _N_NEURONS
            current[idx] = weight * (1.0 - i / width)

        # Additional stimulus for novel app names
        if event.app_name:
            h2 = hash(event.app_name) % _N_NEURONS
            current[h2] += 0.3 * weight

        return current

    def summarise(self, window_sec: float = _WINDOW_SEC) -> ActivitySummary:
        now = time.time()
        with self._lock:
            recent_events = [e for e in self._buffer if now - e.timestamp <= window_sec]
            recent_spikes = [t for t in self._spike_log if now - t <= window_sec]

        if not recent_events:
            return ActivitySummary(
                active_apps=[],
                dominant_event="none",
                spike_count=0,
                activity_level=0.0,
                novel_ui_change=False,
                window_ms=window_sec * 1000,
            )

        apps = list({e.app_name for e in recent_events if e.app_name})
        event_counts: Dict[str, int] = {}
        for e in recent_events:
            event_counts[e.event_type] = event_counts.get(e.event_type, 0) + 1
        dominant = max(event_counts, key=lambda k: event_counts[k]) if event_counts else "none"

        n_spikes      = len(recent_spikes)
        activity      = min(1.0, n_spikes / max(1, window_sec * 5))
        novel_change  = any(e.event_type in ("window:create", "focus:changed") for e in recent_events)

        state = np.array([event_counts.get(etype, 0) for etype in sorted(event_counts)], dtype=float)
        novel = False
        if self._prev_state is not None and len(state) == len(self._prev_state):
            diff  = np.linalg.norm(state - self._prev_state)
            novel = diff > 2.0
        self._prev_state = state

        return ActivitySummary(
            active_apps=apps,
            dominant_event=dominant,
            spike_count=n_spikes,
            activity_level=round(activity, 3),
            novel_ui_change=novel or novel_change,
            window_ms=window_sec * 1000,
        )

    def reset(self) -> None:
        with self._lock:
            self._layer.reset()
            self._buffer.clear()
            self._spike_log.clear()

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "buffered_events": len(self._buffer),
                "total_spikes":    len(self._spike_log),
                "sj_available":    _SJ_AVAILABLE,
                "mode":            self._layer._mode,
            }


_instance: Optional[SNNEventProcessor] = None
_instance_lock = threading.Lock()


def get_snn_processor() -> SNNEventProcessor:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = SNNEventProcessor()
    return _instance
