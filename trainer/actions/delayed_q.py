"""Trainer for the Delayed-Q action-gap mitigation method.

Subclasses the baseline actions trainer and only overrides the mitigation hooks
(``select_action`` / ``store_transition`` / ``train_agents`` / ``reset_episode_state``)
to route the REAL agents through a :class:`DelayedQModel`. The SIM agents (no
delay) keep the baseline behavior. The base run loop -- including the existing
sim-step delay queue -- is reused unchanged.

This same trainer also drives the Oblivious-Q baseline: it builds whichever model
the configured method names (``sim2real.method``), and Oblivious-Q is just a
:class:`DelayedQModel` with the compensation horizon forced to 0 (env delay still
applied, no prediction). No trainer subclass is needed for it.
"""

import agent.action_delay  # noqa: F401 -- register the method models
from common.registry import Registry
from trainer.actions.sim2real_actions_trainer import Sim2RealActionsTrainer


_METHOD_LABELS = {
    "delayed_q": "Delayed-Q enabled",
    "oblivious_q": "Oblivious-Q baseline (prediction disabled, env delay still applied)",
}


@Registry.register_trainer("sim2real_actions_delayed_q")
class Sim2RealActionsDelayedQTrainer(Sim2RealActionsTrainer):
    def __init__(self, logger, gpu=0, cpu=False, name="sim2real_actions"):
        super().__init__(logger=logger, gpu=gpu, cpu=cpu, name=name)

        sim2real_config = self.get_sim2real_config()
        self.method = sim2real_config.get("method", "delayed_q")
        # The compensation horizon m is the delay the agent is DESIGNED for -- the
        # sim (assumed) delay -- NOT the real delay. The real env may apply a
        # different delay, and that sim<->real mismatch is exactly the experimental
        # variable. The env delay is applied by the run loop's action-delay queue,
        # independent of the model; m is what the *model* compensates for.
        delay = self.sim_action_delay.delay
        self.m = delay // self.action_interval if self.action_interval else 0
        if self.m > 0 and self.action_interval and delay % self.action_interval != 0:
            self.logger.warning(
                "sim_action_delay (%s) is not a multiple of action_interval (%s); "
                "decision delay m=%s (remainder ignored).",
                delay,
                self.action_interval,
                self.m,
            )

        # Build the model registered under the configured method and delegate the
        # mitigation hooks to it -- the trainer stays method-agnostic. Each model
        # owns its own policy: Delayed-Q rolls a forward model m steps ahead;
        # Oblivious-Q ignores m and acts on the current (delayed) observation.
        # (sim and real agents share intersection dims and idx mapping, so one set
        # of per-intersection models serves both rollouts, which never run at once.)
        model_cls = Registry.mapping["sim2real_model_mapping"][self.method]
        self.model = model_cls(
            self.agents_sim,
            self.m,
            lr=sim2real_config.get("pred_learning_rate", 1e-3),
            train_iters=sim2real_config.get("pred_train_iters", 1),
        )
        self.logger.info(
            "%s: m=%s decision(s) of assumed delay (sim_delay=%s, real_delay=%s)",
            _METHOD_LABELS.get(self.method, self.method),
            self.m,
            self.sim_action_delay.delay,
            self.real_action_delay.delay,
        )

    def select_action(self, ag, idx, ob, phase, test, valid_mask_fn=None):
        # The model decides how to act: predict ŝ_{t+m} and act on it (Delayed-Q),
        # or act on the current observation (Oblivious-Q). m=0 is handled inside
        # the model. valid_mask_fn (phase-transition validity) is forwarded to the
        # base agent's get_action.
        return self.model.select_action(
            ag, idx, ob, phase, test, valid_mask_fn=valid_mask_fn
        )

    # Q-replay is identical to the baseline: the environment's delay queue already
    # stores the executed (delayed) action with its execution-time state/reward,
    # which is exactly the shifted tuple (s_{t+m}, a_t, r_{t+m}, s_{t+m+1}). So we
    # do NOT override store_transition.

    def train_agents(self, agents):
        losses = super().train_agents(agents)
        # Train the model's predictor on whatever agents are training (sim, under
        # the zero-shot protocol). Delayed-Q's f models the delay-independent
        # one-step dynamics, so sim transitions are exactly the right training
        # data; Oblivious-Q's is a no-op.
        self.model.train_predictor(agents)
        return losses

    def reset_episode_state(self, agents, init_phases):
        # Reset the model's per-episode state at the start of every rollout (sim
        # train, sim eval, real eval); a no-op for models without one.
        self.model.reset_all(init_phases)
