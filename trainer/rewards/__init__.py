"""Reward-gap trainers (sim = cityflow, real = sumo).

Importing this package registers the base/naive trainer and every method trainer
with the Registry, so the task dispatcher can resolve them by name.
"""

from trainer.rewards.base import Sim2RealRewardsTrainer
from trainer.rewards.shaping import Sim2RealRewardsShapingTrainer
from trainer.rewards.random_reward import Sim2RealRewardsRandomTrainer
from trainer.rewards.reward_inference import Sim2RealRewardsInferenceTrainer
from trainer.rewards.morl_grid import Sim2RealRewardsMORLGridTrainer
from trainer.rewards.dynamic_reward_shaping import Sim2RealRewardsDynamicShapingTrainer
from trainer.rewards.reward_oracle import Sim2RealRewardsOracleTrainer
from trainer.rewards.phase_transition import (
    Sim2RealRewardsShieldTrainer,
    Sim2RealRewardsPTNaiveTrainer,
)

__all__ = [
    "Sim2RealRewardsTrainer",
    "Sim2RealRewardsShapingTrainer",
    "Sim2RealRewardsRandomTrainer",
    "Sim2RealRewardsInferenceTrainer",
    "Sim2RealRewardsMORLGridTrainer",
    "Sim2RealRewardsDynamicShapingTrainer",
    "Sim2RealRewardsOracleTrainer",
    "Sim2RealRewardsShieldTrainer",
    "Sim2RealRewardsPTNaiveTrainer",
]
