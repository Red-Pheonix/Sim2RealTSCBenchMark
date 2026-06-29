"""reward_shaping: train on a fixed, enriched multi-component reward.

The shaping family (EcoLight-style multi-objective shaping; theoretical caveat:
potential-based shaping is policy-invariant, Ng, Harada & Russell ICML 1999, so a
genuine *objective* shift must change the reward, not just add a potential). Here we
train on a fixed `LinearReward` over feature-bank components with heuristic weights
(uniform by default, or `shaping_weights` from the setting) -- no real data, no
inference.

This is exactly where shaping earns its keep against the *unavailable-feature* part of
the gap: a term the sim cannot measure (e.g. emission in cityflow) can still be driven
down by hand-shaping an AVAILABLE, correlated proxy. Minimising `switches` (a control-
effort cost cityflow does compute) or `queue` suppresses stop-and-go and therefore real
emissions, even though the agent never sees an emission signal in sim. So `shaping_weights` is the
domain-knowledge knob: put weight on observable surrogates of the objective you can't
observe.
"""

from common.registry import Registry
from trainer.rewards.base import Sim2RealRewardsTrainer
from trainer.rewards.reward_transforms import LinearReward


@Registry.register_trainer("sim2real_rewards_shaping")
class Sim2RealRewardsShapingTrainer(Sim2RealRewardsTrainer):
    def __init__(self, logger, gpu=0, cpu=False, name="sim2real_rewards"):
        super().__init__(logger=logger, gpu=gpu, cpu=cpu, name=name)
        self.method = "reward_shaping"

    def build_reward_transform(self, feature_bank):
        cfg = self.get_sim2real_config()
        scale = float(cfg.get("reward_scale", 1.0))
        weights = cfg.get("shaping_weights")
        if weights:
            # Hand-picked proxy (domain knowledge) -- may target a sim-unobservable
            # objective via correlated available components (e.g. queue for emission).
            w = feature_bank.weight_vector(weights)
        else:
            # Uniform over the SIM-AVAILABLE components (kitchen-sink surrogate);
            # weighting a sim-unavailable term would just be inert.
            avail = feature_bank.available_components()
            w = feature_bank.weight_vector({c: 1.0 / len(avail) for c in avail})
        return LinearReward(w, feature_bank.components, scale=scale, norm=self.component_norm)
