"""Trainer for the PRLight action-gap mitigation method.

Standalone -- it does NOT subclass the Delayed-Q trainer. PRLight differs from
Delayed-Q in both the predictor (one-shot, neighbor-aware instead of a rolling
one-step model) and in the data it needs: a global decision-time snapshot for
neighbor lookup, plus per-decision executed actions for the m-apart training
pairs. Rather than thread those differences through inheritance, the trainer
owns its hooks outright; the small amount of shared boilerplate (horizon
resolution, action routing, episode reset) is duplicated on purpose so each
method reads top-to-bottom. Like Delayed-Q the base agents stay plain wrappers
-- only the mitigation hooks are overridden.
"""

from common.registry import Registry
from agent.action_delay import PRLightModel
from trainer.actions.sim2real_actions_trainer import Sim2RealActionsTrainer


@Registry.register_trainer("sim2real_actions_prlight")
class Sim2RealActionsPRLightTrainer(Sim2RealActionsTrainer):
    def __init__(self, logger, gpu=0, cpu=False, name="sim2real_actions"):
        super().__init__(logger=logger, gpu=gpu, cpu=cpu, name=name)

        sim2real_config = self.get_sim2real_config()
        # Compensation horizon m = the assumed (sim) delay in decisions. The
        # one-shot predictor is trained for this fixed horizon -- m is baked into
        # its input width. The real env may apply a different delay, and that
        # sim<->real mismatch is exactly the experimental variable.
        delay = self.sim_action_delay.delay
        self.m = delay // self.action_interval if self.action_interval else 0
        if self.m > 0 and self.action_interval and delay % self.action_interval != 0:
            self.logger.warning(
                "sim_action_delay (%s) is not a multiple of action_interval (%s); "
                "PRLight uses decision delay m=%s (remainder ignored).",
                delay,
                self.action_interval,
                self.m,
            )

        self.model = PRLightModel(
            self.world_sim,
            self.agents_sim,
            self.m,
            lr=sim2real_config.get("pred_learning_rate", 1e-3),
            train_iters=sim2real_config.get("pred_train_iters", 1),
            probing_radius=sim2real_config.get("probing_radius", 600.0),
            include_neighbor_actions=sim2real_config.get(
                "include_neighbor_actions", False
            ),
            buffer_cap=sim2real_config.get("pred_buffer_cap", 1000),
            logger=self.logger,
        )
        self.logger.info(
            "PRLight (one-shot neighbor-aware prediction): m=%s decision(s) of "
            "assumed delay (sim_delay=%s, real_delay=%s)",
            self.m,
            self.sim_action_delay.delay,
            self.real_action_delay.delay,
        )

    def select_action(self, ag, idx, ob, phase, test, valid_mask_fn=None):
        # Every agent (sim and real) acts on the predicted state ŝ_{t+m}; m=0
        # falls back to plain base action selection inside the model.
        # valid_mask_fn (phase-transition validity) is forwarded to get_action.
        if self.m > 0:
            return self.model.select_action(
                ag, idx, ob, phase, test, valid_mask_fn=valid_mask_fn
            )
        return super().select_action(
            ag, idx, ob, phase, test, valid_mask_fn=valid_mask_fn
        )

    def on_decision_start(self, obs, phases, test):
        # Hand the model the full per-intersection snapshot so select_action can
        # assemble neighbor features for each ego intersection.
        if self.m > 0:
            self.model.cache_decision_state(obs, phases)

    def store_transition(self, ag, idx, **kwargs):
        # Q-replay stays the baseline's (handled by super()); additionally feed
        # the model the per-decision executed actions it turns into m-apart
        # (state_t, actions [t, t+m)) -> state_{t+m} training pairs.
        super().store_transition(ag, idx, **kwargs)
        if self.m > 0:
            self.model.record_step(
                idx,
                kwargs["last_obs"],
                kwargs["last_phase"],
                kwargs["executed_action"],
            )

    def train_agents(self, agents):
        losses = super().train_agents(agents)
        # Train the one-shot predictor online, in parallel with the policy
        # (paper Alg. 1), on whatever agents are training (sim, zero-shot).
        if self.m > 0:
            self.model.train_predictor(agents)
        return losses

    def reset_episode_state(self, agents, init_phases):
        # Reseed the pending-action queues + drop any partial prediction window
        # at the start of every rollout (sim train, sim eval, real eval).
        if self.m > 0:
            self.model.reset_all(init_phases)
