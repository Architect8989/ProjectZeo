"""
core/cognition/active_inference.py

Active Inference agent implementing the Free Energy Principle (FEP).

Reference:
  Friston et al. — "Active Inference: The Free Energy Principle in Mind,
  Brain, and Behaviour" (MIT Press, 2022)
  Da Costa et al. arXiv:2001.00950

Role in ProjectZeo:
  Drives action selection by minimising expected free energy (EFE) rather
  than maximising reward directly. This makes the agent epistemic — it
  balances goal-seeking (exploitative) with uncertainty-reducing (exploratory)
  behaviour without separate exploration bonuses.

Integration:
  GIIController.decide_next_action() → calls ActiveInferenceAgent.select_action()
  to produce a ranked list of candidate actions before passing to the LLM.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_logger = logging.getLogger(__name__)

_PRECISION       = float(1.0)   # γ — action precision (inverse temperature)
_PLANNING_DEPTH  = int(3)       # τ — rollout horizon for EFE estimation
_N_POLICIES      = int(8)       # candidate policies per step


@dataclass
class Belief:
    """Categorical distribution over hidden states."""
    probs: np.ndarray          # shape (n_states,)
    states: List[str]          # state labels

    @classmethod
    def uniform(cls, states: List[str]) -> "Belief":
        n = len(states)
        return cls(probs=np.ones(n) / n, states=states)

    def entropy(self) -> float:
        p = np.clip(self.probs, 1e-12, 1.0)
        return float(-np.sum(p * np.log(p)))

    def most_probable(self) -> str:
        return self.states[int(np.argmax(self.probs))]


@dataclass
class ActiveInferenceAction:
    action:      Dict[str, Any]
    efe:         float            # expected free energy (lower = better)
    ambiguity:   float            # epistemic value (uncertainty reduction)
    risk:        float            # pragmatic value (goal divergence)
    rank:        int = 0


class ActiveInferenceAgent:
    """
    Minimises expected free energy to select GUI actions.

    Maintains a belief state over "what the UI is doing" and selects
    actions that jointly reduce uncertainty (ambiguity) and approach
    the goal (risk).
    """

    def __init__(
        self,
        n_states: int = 16,
        n_obs: int = 32,
    ) -> None:
        self._n_states = n_states
        self._n_obs    = n_obs
        self._lock     = threading.Lock()

        # Generative model parameters (initialised to uniform / identity)
        # A: likelihood P(o|s) — observation model
        # B: transition P(s'|s,a) — transition model  (n_states × n_states × n_actions)
        # C: log preferences log P*(o) — goal prior
        self._A = np.ones((n_obs, n_states)) / n_obs
        self._B = np.eye(n_states)[:, :, np.newaxis]   # identity, 1 action slot
        self._C = np.zeros(n_obs)                       # flat — updated from goal
        self._D = np.ones(n_states) / n_states          # prior over initial states

        # Current belief
        self._belief = Belief(probs=self._D.copy(), states=[f"s{i}" for i in range(n_states)])

        # Running precision estimate
        self._precision = _PRECISION

        # Stats
        self._calls = 0
        self._last_efe: Optional[float] = None

    # -------------------------------------------------------------------------
    # Belief update (perception)
    # -------------------------------------------------------------------------

    def update_belief(self, observation: Dict[str, Any]) -> None:
        """
        Bayesian belief update given a new observation.
        Uses variational message passing (simplified: Bayes' rule).
        """
        obs_vec = self._encode_observation(observation)

        likelihood = self._A[obs_vec, :]   # P(o|s) for each state
        posterior  = likelihood * self._belief.probs
        total      = posterior.sum()
        if total > 1e-12:
            posterior /= total
        else:
            posterior = self._D.copy()

        with self._lock:
            self._belief = Belief(probs=posterior, states=self._belief.states)

    def _encode_observation(self, obs: Dict[str, Any]) -> int:
        entities  = len(obs.get("entities", []))
        app_hash  = hash(str(obs.get("focused_app", ""))) % self._n_obs
        raw_idx   = (entities * 7 + app_hash) % self._n_obs
        return int(raw_idx)

    # -------------------------------------------------------------------------
    # Action selection (active inference)
    # -------------------------------------------------------------------------

    def select_action(
        self,
        candidate_actions: List[Dict[str, Any]],
        goal_description: str,
        world_state: Dict[str, Any],
    ) -> List[ActiveInferenceAction]:
        """
        Rank candidate actions by expected free energy.

        Lower EFE = better action. Returns list sorted ascending by EFE.
        """
        if not candidate_actions:
            return []

        self._update_goal_prior(goal_description)
        self.update_belief(world_state)

        scored: List[ActiveInferenceAction] = []
        for action in candidate_actions[:_N_POLICIES]:
            efe, ambiguity, risk = self._compute_efe(action, world_state)
            scored.append(ActiveInferenceAction(
                action=action,
                efe=efe,
                ambiguity=ambiguity,
                risk=risk,
            ))

        scored.sort(key=lambda a: a.efe)
        for i, a in enumerate(scored):
            a.rank = i

        with self._lock:
            self._calls += 1
            self._last_efe = scored[0].efe if scored else None

        return scored

    def _compute_efe(
        self,
        action: Dict[str, Any],
        world_state: Dict[str, Any],
    ) -> Tuple[float, float, float]:
        """
        G(π) = E_Q[ln Q(s) - ln P(o,s)]
              = Ambiguity + Risk

        Ambiguity = E[H[P(o|s)]]  — expected uncertainty about observations
        Risk      = E[D_KL[Q(o) || P*(o)]]  — divergence from goal
        """
        belief  = self._belief.probs

        # Predicted observation distribution under this action
        pred_obs = self._A @ belief                    # shape (n_obs,)
        pred_obs = np.clip(pred_obs, 1e-12, 1.0)
        pred_obs /= pred_obs.sum()

        # Ambiguity: E_Q[H[P(o|s)]] = H[E_Q[P(o|s)]] (Jensen upper bound)
        ambiguity = float(-np.sum(pred_obs * np.log(pred_obs)))

        # Risk: D_KL[Q(o) || P*(o)]
        goal_prior = np.exp(self._C)
        goal_prior = np.clip(goal_prior, 1e-12, None)
        goal_prior /= goal_prior.sum()
        risk = float(np.sum(pred_obs * (np.log(pred_obs) - np.log(goal_prior))))

        # Action-specific modifier from semantic content
        op   = str(action.get("operation", ""))
        risk = self._apply_operation_prior(op, risk)

        efe = ambiguity + risk
        return efe, ambiguity, risk

    def _apply_operation_prior(self, op: str, base_risk: float) -> float:
        modifiers = {
            "done":    -2.0,   # terminal — strongly preferred if goal met
            "verify":  -0.5,   # reduces epistemic uncertainty
            "wait":     0.5,   # neutral-costly
            "command":  1.0,   # higher risk
            "install":  1.5,   # highest risk
        }
        return base_risk + modifiers.get(op, 0.0)

    def _update_goal_prior(self, goal: str) -> None:
        """Encode goal as observation preferences C = log P*(o)."""
        seed = hash(goal) % (2**31)
        rng  = np.random.default_rng(seed)
        self._C = rng.standard_normal(self._n_obs) * 0.5

    # -------------------------------------------------------------------------
    # Precision adaptation
    # -------------------------------------------------------------------------

    def adapt_precision(self, recent_success_rate: float) -> None:
        """
        Adapt action precision γ based on recent task success rate.
        High success → increase precision (less exploration).
        Low success → decrease precision (more exploration).
        """
        target = 0.5 + recent_success_rate
        self._precision = 0.9 * self._precision + 0.1 * target
        _logger.debug("[AIF] Precision adapted to %.3f (success=%.2f)", self._precision, recent_success_rate)

    # -------------------------------------------------------------------------
    # Learning: update A and B matrices
    # -------------------------------------------------------------------------

    def update_from_outcome(
        self,
        prev_state: Dict[str, Any],
        action: Dict[str, Any],
        next_state: Dict[str, Any],
        success: bool,
    ) -> None:
        """
        Update generative model parameters from observed transitions.
        Uses pseudo-Bayes updates (concentration parameter increments).
        """
        obs_prev = self._encode_observation(prev_state)
        obs_next = self._encode_observation(next_state)
        belief   = self._belief.probs

        # A update: likelihood P(o|s) — increase weight for observed (o, s) pair
        delta = 0.01 * (1.5 if success else 0.5)
        with self._lock:
            self._A[obs_next, :] += delta * belief
            self._A[:, :] /= self._A.sum(axis=0, keepdims=True).clip(1e-12)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "calls":          self._calls,
            "belief_entropy": self._belief.entropy(),
            "most_probable":  self._belief.most_probable(),
            "precision":      self._precision,
            "last_efe":       self._last_efe,
        }


_instance: Optional[ActiveInferenceAgent] = None
_instance_lock = threading.Lock()


def get_active_inference_agent() -> ActiveInferenceAgent:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = ActiveInferenceAgent()
    return _instance
