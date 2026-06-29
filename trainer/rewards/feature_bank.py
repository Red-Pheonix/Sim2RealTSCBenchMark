"""Feature bank for the reward gap.

The reward gap models the hidden real objective as a linear combination over
interpretable, *computable* components:

    R_real(s,a) = - Σ_i w_i · φ_i(s,a)        (w_i >= 0, φ_i a per-agent COST)

`φ` is a vector of per-intersection **costs** (higher = worse), so the reward is
the negative weighted sum and a larger `R_real` is better. Every method that
trains/scores on a weight vector dots it with this same bank, so `FeatureBank` is
the single substrate behind naive (native proxy), reward_shaping, reward_random,
reward_inference (`ŵ`), dynamic_reward_shaping (BO-tuned `w`), and the eval scorer
`TrueReward` (`w*`).

Components split by where they can be computed:

  * Cross-sim (cityflow AND sumo): `queue, delay, waiting, pressure, switches`
    (`switches` = a control-effort cost: did the signal change phase this decision)
    plus `fairness` (throughput-based: max-min spread of cumulative served throughput
    across demand-active approaches, Raeis & Leon-Garcia arXiv:2107.10146; computed
    world-side and read via the `intersection_fairness` info function). A weight
    identified in real transfers to sim training for these.
  * SUMO-only (cityflow genuinely cannot compute them): `emission` (CO2), `fuel`,
    `emergency_stops`, `collisions`. These are NOT proxied or faked on the cityflow
    side -- the backing info function simply does not exist there, so the feature is
    absent (0.0). A sim-trained policy is blind to a term it cannot observe, yet the
    real (sumo) eval still scores it. That asymmetry IS the reward gap / the challenge
    (see notes/reward_gap_plan.md).
  * Special-case: `safety` (a controller-rule violation flag) is supplied by the
    phase-transition / shield trainer, not the bank.

All info-backed components are obtained through the world's normal info-function /
subscribe path (computed once per step), consumed via the same `LaneVehicleGenerator`
the core costs use -- lane-keyed signals (`emission`, `fuel`) average over the
intersection's incoming lanes; intersection-keyed signals (`fairness`,
`emergency_stops`, `collisions`) are returned per intersection directly.

**The bank does NOT decide what each simulator can compute.** The trainer (data layer,
see `base.EXTRA_INFO_FN` / `resolve_components`) resolves availability against the real
world objects and passes the resolved `available` set + `{component: info_fn}` map in.
So cityflow (which registers only `intersection_fairness`) ends up with the sumo-only
costs unavailable, but that decision lives in the trainer, not here.
"""

import numpy as np

from generator import LaneVehicleGenerator

# Core per-agent cost components, computable on cityflow AND sumo. Order is the
# canonical φ layout; weight dicts are projected onto whatever components exist.
# `switches` = 1 when the signal changed phase this decision (a control-effort cost;
# was previously mis-named `stops`).
CORE_COMPONENTS = ["queue", "delay", "waiting", "pressure", "switches"]

class FeatureBank:
    """Per-intersection cost vector `φ` over a fixed component list.

    One bank per side (sim / real). Built from the side's world + agents; reuses
    `LaneVehicleGenerator` for the lane-based costs. `features(idx, phase_changed)`
    returns the cost vector for intersection `idx` this decision.

    The bank is a pure executor and knows NOTHING about which simulator can compute
    what. The trainer (data layer) resolves that and passes:
      * `available`  -- the component names this world can actually compute;
      * `info_fns`   -- `{component: world_info_fn}` for the available info-backed
        components (fairness / emission / ...), so the bank can build a generator for
        each without inspecting `world.info_functions` itself.
    Unavailable components stay in the layout (so sim/real `φ` align) but read 0.
    """

    def __init__(self, world, agents, extra_components=None, available=None, info_fns=None):
        self.world = world
        self.agents = agents
        self.components = list(CORE_COMPONENTS)
        for c in extra_components or []:
            if c not in self.components:
                self.components.append(c)
        # Availability is decided by the trainer and handed in (default: core only).
        # A component absent here is present in the layout but always 0 -- training
        # methods must not weight it (inert); only the real-side scorer uses it.
        available = set(available) if available is not None else set(CORE_COMPONENTS)
        self.available = np.array(
            [c in available for c in self.components], dtype=bool
        )
        # Per-agent generators for the lane-based core costs (single scalar each).
        self._queue, self._delay, self._waiting, self._pressure = [], [], [], []
        for ag in agents:
            inter = ag.inter_obj
            self._queue.append(
                LaneVehicleGenerator(
                    world, inter, ["lane_waiting_count"], in_only=True, average="all"
                )
            )
            self._delay.append(
                LaneVehicleGenerator(
                    world, inter, ["lane_delay"], in_only=True, average="all"
                )
            )
            self._waiting.append(
                LaneVehicleGenerator(
                    world, inter, ["lane_waiting_time_count"], in_only=True, average="all"
                )
            )
            self._pressure.append(
                LaneVehicleGenerator(world, inter, ["pressure"], average="all")
            )
        # Per-agent generators for the info-backed components the trainer says are
        # available here, wired through the same info-function path (subscribing once).
        # `info_fns` already excludes anything this world can't compute, so the bank
        # never inspects the world or builds a generator that would fail to subscribe.
        self._extra_gens = {}
        for c, fn in (info_fns or {}).items():
            self._extra_gens[c] = [
                LaneVehicleGenerator(
                    world, ag.inter_obj, [fn], in_only=True, average="all"
                )
                for ag in agents
            ]

    def available_components(self):
        """Components this simulator can actually compute (training methods restrict
        their weight support to these)."""
        return [c for c, a in zip(self.components, self.available) if a]

    def unavailable_components(self):
        """Components present in the layout but NOT computable here (always 0) -- the
        irreducible part of the gap on this side; a sim-trained policy is blind to them."""
        return [c for c, a in zip(self.components, self.available) if not a]

    def available_mask(self):
        """Float 0/1 vector over `self.components` (1 = computable here)."""
        return self.available.astype(float)

    @staticmethod
    def _scalar(gen):
        """Collapse a generator output to a single nonneg scalar cost."""
        val = np.asarray(gen.generate(), dtype=float)
        return float(np.mean(val)) if val.size else 0.0

    def features(self, idx, phase_changed=False):
        """Cost vector `φ` for intersection `idx` (aligned with `self.components`)."""
        raw = {
            "queue": self._scalar(self._queue[idx]),
            "delay": self._scalar(self._delay[idx]),
            "waiting": self._scalar(self._waiting[idx]),
            # pressure is signed (in-out imbalance); the cost is its magnitude.
            "pressure": abs(self._scalar(self._pressure[idx])),
            # switching cost: did the executed phase change this decision.
            "switches": 1.0 if phase_changed else 0.0,
        }
        for c in self.components:
            if c in raw:
                continue
            if c in self._extra_gens:
                # info-backed cost (fairness / sumo-only) read once per step via its
                # info function: fairness is per-intersection; emission/fuel average
                # over incoming lanes.
                raw[c] = self._scalar(self._extra_gens[c][idx])
            else:
                # unavailable here (e.g. emission on cityflow), or `safety` which the
                # PT/shield trainer injects -- genuinely 0.0, not proxied.
                raw[c] = 0.0
        # RAW per-intersection costs. Scale normalization is a reward-construction
        # choice and lives in the reward transforms (LinearReward / TrueReward etc.).
        return np.array([raw[c] for c in self.components], dtype=float)

    def weight_vector(self, weight_dict):
        """Project a `{component: weight}` dict onto the bank's component order
        (missing components -> 0.0). All methods consume weights through this so a
        setting file can name only the components it cares about."""
        weight_dict = weight_dict or {}
        return np.array(
            [float(weight_dict.get(c, 0.0)) for c in self.components], dtype=float
        )
