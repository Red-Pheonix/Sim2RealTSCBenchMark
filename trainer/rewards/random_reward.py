"""reward_random: robustness to the unknown objective by reward randomization.

The no-inference robustness baseline (reward-space domain randomization; cf. RPG
reward randomization, Tang et al. ICLR 2021). Each sim episode resamples the reward
weights `w ~ Dirichlet` over the component simplex, so the policy is trained to do
well across the whole space of plausible weightings rather than betting on one.
Spends 0 real budget.
"""

import numpy as np

from common.registry import Registry
from trainer.rewards.base import Sim2RealRewardsTrainer
from trainer.rewards.reward_transforms import RandomLinearReward


@Registry.register_trainer("sim2real_rewards_random")
class Sim2RealRewardsRandomTrainer(Sim2RealRewardsTrainer):
    def __init__(self, logger, gpu=0, cpu=False, name="sim2real_rewards"):
        super().__init__(logger=logger, gpu=gpu, cpu=cpu, name=name)
        self.method = "reward_random"

    def build_reward_transform(self, feature_bank):
        cfg = self.get_sim2real_config()
        seed = int(cfg.get("reward_random_seed", self.seed or 0))
        self._rng = np.random.default_rng(seed)
        alpha = float(cfg.get("dirichlet_alpha", 1.0))
        scale = float(cfg.get("reward_scale", 1.0))
        # Only randomize over components the SIM can actually compute -- weighting a
        # sim-unavailable term (e.g. emission in cityflow) yields an all-zero reward
        # for that draw. A setting may further narrow this via `random_components`.
        mask = feature_bank.available_mask()
        mask_cfg = cfg.get("random_components")
        if mask_cfg:
            mask = mask * np.array(
                [1.0 if c in mask_cfg else 0.0 for c in feature_bank.components]
            )
        self._random_reward = RandomLinearReward(
            feature_bank.components, alpha=alpha, scale=scale, mask=mask,
            norm=self.component_norm,
        )
        return self._random_reward

    def on_episode_start(self, episode):
        # Resample the weighting at the start of every sim episode.
        self._random_reward.resample(self._rng)
