"""Trainer for the soft-shield action-gap mitigation method.

Soft shield is the SOFT counterpart of the hard ``shield`` mask: instead of preventing
illegal actions, it grounds the sim with the real controller (sim_phase_transition_mode:
enforce -- illegal requests dropped/forced exactly as at deployment) and PENALIZES the
training reward for every illegal request, so the agent learns to avoid them through the
reward signal. Deployed unmasked, it keeps a low-but-nonzero violation rate -- the soft
point between naive (high) and shield (zero).

Two method-specific overrides on the base trainer, both soft_shield-exclusive (which is
why they live here, not on the base):

  * ``action_to_store`` -- buffer the REQUESTED action, not the executed one, so the
    penalty lands on the choice the agent actually made (Q(s, a_requested) is driven
    down). The transition is dynamically inconsistent on purpose -- the penalty teaches
    avoidance, it is not a faithful dynamics sample.
  * ``build_reward_transforms`` -- add a ViolationPenalty over the TRAINING reward (the
    reported metric stays raw). The penalty must out-weigh the gamma*maxQ(s') bootstrap
    to actually suppress the choice, so ``violation_penalty`` is a tuned hyperparameter.
"""

from common.registry import Registry
from trainer.actions.reward_transforms import ViolationPenalty
from trainer.actions.sim2real_actions_trainer import Sim2RealActionsTrainer


@Registry.register_trainer("sim2real_actions_soft_shield")
class Sim2RealActionsSoftShieldTrainer(Sim2RealActionsTrainer):
    def __init__(self, logger, gpu=0, cpu=False, name="sim2real_actions"):
        super().__init__(logger=logger, gpu=gpu, cpu=cpu, name=name)
        self.method = "soft_shield"

    def action_to_store(self, chosen_action, executed_action):
        # Learn on the action the agent REQUESTED, so the penalty attaches to its choice.
        return chosen_action

    def build_reward_transforms(self):
        # Read the penalty from config directly: the base __init__ calls this before a
        # subclass __init__ body runs, so we cannot rely on an instance attribute here.
        # Reward penalty per illegal request -- the soft constraint's strength.
        penalty = float(self.get_sim2real_config().get("violation_penalty", 0.0))
        return [ViolationPenalty(penalty)] if penalty else []
