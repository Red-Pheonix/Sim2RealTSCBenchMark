"""Reward transforms for the reward gap.

Mirrors the action gap's reward-transform idea (`trainer/actions/reward_transforms.py`):
the trainer folds the TRAINING reward through a transform while the reported metric
stays the *true* reward. Here a transform maps the per-agent feature matrix `Φ`
(rows = intersections, cols = `FeatureBank.components`) to a per-agent reward vector.

Every no-inference method is just *which weights* dot the bank:
  * `LinearReward(w, components)`       -- reward = -(Φ · w). Used by reward_shaping
    (heuristic w), reward_inference (`ŵ`), and dynamic_reward_shaping (the BO-tuned `w`).
  * `RandomLinearReward(components, ...)`-- resamples `w` over a simplex each episode
    (reward_random), the robustness baseline (RPG-style, Tang et al. 2021).
  * `TrueReward(w*, components)`         -- the hidden objective, used ONLY at eval.

`w >= 0` and `φ` are costs, so a larger reward (less negative) is better -- training
on `-(Φ·w)` and scoring the real objective use the exact same form.

**Normalization lives here, not in the FeatureBank.** The bank emits RAW per-intersection
costs; TSC component magnitudes differ by orders (waiting ≫ queue ≫ delay ≈ switches∈
[0,1]), which would make `-(w·φ)` scale-unstable and gridlock the Q-net. So every
transform divides `φ` by a per-component norm before dotting with `w`, putting each term
at ~O(1) so weights express *relative* importance. Training reward and the eval scorer
apply the SAME norm, so a `w` used/learned in training is consistent with how `w*` scores
-- this is a reward-construction choice, hence it sits with the transforms.
"""

import numpy as np

# Fallback per-component magnitudes, used only when a component is absent from the
# config `component_norm` (the authoritative values live in
# configs/sim2real_rewards/base.yml). Components missing from both default to 1.0.
DEFAULT_COMPONENT_NORM = {
    "queue": 10.0,
    "delay": 1.0,
    "waiting": 100.0,
    "pressure": 10.0,
    "switches": 1.0,
    # fairness = max-min spread of CUMULATIVE served throughput -> grows over the
    # episode, so a larger norm than the instantaneous costs (rough; tune per network).
    "fairness": 50.0,
    "emission": 10.0,
    "fuel": 1.0,
    "emergency_stops": 1.0,
    "collisions": 1.0,
    "safety": 1.0,
}


def _norm_vector(components, norm=None):
    norm = norm or DEFAULT_COMPONENT_NORM
    return np.array([norm.get(c, 1.0) for c in components], dtype=float)


class _LinearCost:
    """Shared base: reward = -scale * ((Φ / norm) · w), Φ a RAW per-agent cost matrix
    aligned with `components` (build `w` via `FeatureBank.weight_vector`). `norm` is the
    per-component normalizer dict from the trainer (config-driven); falls back to
    `DEFAULT_COMPONENT_NORM` when None."""

    def __init__(self, w, components, scale=1.0, norm=None):
        self.w = np.asarray(w, dtype=float)
        self.norm = _norm_vector(components, norm)
        self.scale = float(scale)

    def _normed(self, features):
        features = np.atleast_2d(np.asarray(features, dtype=float))
        return features / self.norm

    def __call__(self, features):
        return -self.scale * (self._normed(features) @ self.w)


class LinearReward(_LinearCost):
    """reward = -(Φ · w). The object behind reward_shaping (heuristic w), reward_inference
    (`ŵ`), and dynamic_reward_shaping (the BO-tuned `w`).

    `scale` is an optional positive multiplier to keep the shaped reward on a similar
    magnitude to the native proxy (helps the Q-net).
    """


class RandomLinearReward(_LinearCost):
    """reward_random: resample `w ~ Dirichlet` over the component simplex per episode.

    Robustness-to-unknown-`w` baseline -- never identifies the objective, just trains
    to do well across the simplex of plausible weightings (reward-space domain
    randomization; cf. RPG reward randomization, Tang et al. ICLR 2021). Call
    `resample(rng)` at each episode start; in between it behaves like `LinearReward`.
    """

    def __init__(self, components, alpha=1.0, scale=1.0, mask=None, norm=None):
        n = len(components)
        # Optional 0/1 mask restricting which components can get weight (e.g. only the
        # components a setting deems plausible). Defaults to all-on.
        self.mask = np.ones(n) if mask is None else np.asarray(mask, dtype=float)
        self.alpha = float(alpha)
        super().__init__(self.mask / max(self.mask.sum(), 1.0), components, scale, norm)

    def resample(self, rng):
        draw = rng.dirichlet(self.alpha * np.ones(len(self.w)))
        w = draw * self.mask
        s = w.sum()
        self.w = w / s if s > 0 else self.mask / max(self.mask.sum(), 1.0)
        return self.w


class TrueReward(_LinearCost):
    """The hidden real objective `R_real = -(Φ · w*)`, used ONLY at eval to score.

    Reads `w*` from the setting file (projected through `FeatureBank.weight_vector`).
    The agent never sees `w*`; this object scores `TEST_REAL` and logs the
    per-component breakdown so we can see *which* term a method got right.
    """

    def __init__(self, w_star, components, norm=None):
        super().__init__(w_star, components, scale=1.0, norm=norm)
        self.component_names = list(components)

    def reward(self, features):
        return self(features)

    def breakdown(self, features):
        """Summed `w*_i * (φ_i / norm_i)` per component over the given rows (a dict),
        for logging which objective term dominates."""
        weighted = self._normed(features) * self.w[None, :]
        totals = weighted.sum(axis=0)
        return {c: float(totals[i]) for i, c in enumerate(self.component_names)}
