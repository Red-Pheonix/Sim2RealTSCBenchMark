"""GAT / UGAT for the phase-transition action gap, REUSING the proven transition-gap
grounding infrastructure (agent/grounding/DecentralizedSim2RealTransitionModel: N_net
forward + Dec_Inverse_N_net inverse with evidential uncertainty + its pickle train cycle).

Two variants, selected by config:
  * gat  -- plain Grounded Action Transformation: ground every action.
  * ugat -- Uncertainty-gated GAT: ground only when the inverse model is confident
    (`uncertainty < running average`). The gate is the stability mechanism plain GAT lacks.
Both learn the forward model from the PROPOSED (requested) action, so it captures the
controller's validity transform (illegal request -> hold) directly.

Per episode (mirrors trainer/transitions/grounding_trainer.py):
  1. policy_training -- sim DQN; ground each action via the model once warmed up.
  2. sim_rollout     -- ungrounded sim rollout, collect (s,a,s') for the INVERSE (sim data).
  3. real_rollout    -- real env (real_action_transforms), collect (s, PROPOSED a, s') for
                        the FORWARD (real data); log REAL_TRAIN.
  4. train the forward/inverse models.

Outcome (documented negative result): grounding learns, but does not close the PT gap --
phase-blind, it can't see the dwell/phase that determine legality, so it never helps on
timing (flexible) and only noisily on structure (barrier); gating (ugat) keeps it stable,
plain gat can gridlock on barrier. Deploys plain (no runtime shield); real rewards unused.
"""

import numpy as np
import torch

from common.registry import Registry
from trainer.actions.sim2real_actions_trainer import Sim2RealActionsTrainer
import agent.grounding  # noqa: F401 -- register the decentralized sim2real model


@Registry.register_trainer("sim2real_actions_gat")
class Sim2RealActionsGATTrainer(Sim2RealActionsTrainer):
    def __init__(self, logger, gpu=0, cpu=False, name="sim2real_actions"):
        super().__init__(logger=logger, gpu=gpu, cpu=cpu, name=name)
        sc = self.get_sim2real_config()
        self.method = sc.get("method", "gat")
        self.gat_warmup = int(sc.get("gat_warmup", 5))
        # Forward-model action source: "proposed" (the requested action, so f_real learns
        # the controller's drop-illegal transform) vs "executed" (the dropped outcome).
        self.gat_forward_action = sc.get("gat_forward_action", "proposed")
        # The reference model reads these Registry params; the action runner doesn't set
        # them. `uncertainty` toggles the gate: True = ugat (gated), False = gat (always).
        cmd = Registry.mapping["command_mapping"]["setting"].param
        cmd.setdefault("gat_model", "decentralized")
        sc.setdefault("uncertainty", bool(sc.get("gat_uncertainty", True)))
        sc.setdefault("last_n_uncertainties", int(sc.get("gat_last_n_uncertainties", 2)))
        self.device = torch.device("cpu")
        self.transition_model = Registry.mapping["sim2real_model_mapping"]["decentralized"](
            logger=self.logger,
            device=self.device,
            world_sim=self.world_sim,
            agents_sim=self.agents_sim,
            world_real=self.world_real,
            agents_real=self.agents_real,
            dataset_dir=(
                "collected/gat_"
                f"{cmd['network']}_{cmd.get('real_setting', 'default')}_{cmd.get('prefix', '')}"
            ),
        )
        self._gat_ready = False

    # ------------------------------------------------------------------
    def train(self):
        if self.load_pretrained:
            pre = self.pretrained_model_dir()
            self.load_agents(self.agents_sim, pre)
            self.load_agents(self.agents_real, pre)
        for episode in range(self.episodes):
            grounded = self._gat_ready and episode >= self.gat_warmup
            sim_loss = self._policy_training(episode, grounded)
            self.save_agents(self.agents_sim, self.model_dir)
            self._sim_rollout()          # inverse (sim) data
            self._real_rollout(episode)  # forward (real) data + REAL_TRAIN
            self.transition_model.train_transition_models()
            self._gat_ready = True
            self.logger.info(
                "GAT episode:%s/%s grounded:%s sim_loss:%.4f", episode, self.episodes,
                grounded, sim_loss,
            )
        # Persist the forward/inverse grounding models (reproducibility: they are
        # part of the method) + a final e-tagged policy checkpoint for test().
        self.transition_model.save_models(self.episodes)
        self.save_agents(self.agents_sim, self.model_dir, e=self.episodes)

    # ------------------------------------------------------------------
    def _records(self, last_obs, actions, obs):
        """Grounding records in the reference shape: (agent_idx, state(1,ob_len),
        action(1,), next_state(1,ob_len)). The action-task obs is (ob_len,) while the
        model wants a leading agent-dim, and the action must be a length-1 array
        (prepare_forward_data iterates it), not a scalar."""
        acts = np.asarray(actions).reshape(-1)
        return [
            (
                idx,
                np.asarray(last_obs[idx], dtype=np.float32).reshape(1, -1),
                np.array([int(acts[idx])], dtype=np.int64),
                np.asarray(obs[idx], dtype=np.float32).reshape(1, -1),
            )
            for idx in range(len(self.agents_real))
        ]

    def _ground(self, last_obs, actions, stats):
        shaped = [np.asarray(o, dtype=np.float32).reshape(1, -1) for o in last_obs]
        grounded, _ = self.transition_model.ground_actions(shaped, actions, stats)
        return np.asarray(grounded).reshape(-1)

    # ------------------------------------------------------------------
    def _policy_training(self, episode, grounded):
        env, agents, metric = self.env_sim, self.agents_sim, self.metric_sim
        metric.clear()
        last_obs = env.reset()
        for ag in agents:
            ag.reset()
        i, loss = 0, []
        dones = [False] * len(agents)
        stats = self.transition_model.init_policy_stats()
        while i < self.steps:
            if i % self.action_interval == 0:
                last_phase = np.stack([ag.get_phase() for ag in agents])
                actions = np.stack([
                    ag.get_action(last_obs[idx], last_phase[idx], test=False)
                    for idx, ag in enumerate(agents)
                ])
                probs = [ag.get_action_prob(last_obs[idx], last_phase[idx])
                         for idx, ag in enumerate(agents)]
                if grounded:
                    actions = self._ground(last_obs, actions, stats)
                actions = np.asarray(actions).reshape(-1)
                rewards_list = []
                for _ in range(self.action_interval):
                    obs, rewards, dones, _ = env.step(actions.flatten())
                    i += 1
                    rewards_list.append(np.stack(rewards))
                rewards = np.mean(rewards_list, axis=0)
                metric.update(rewards)
                cur_phase = np.stack([ag.get_phase() for ag in agents])
                for idx, ag in enumerate(agents):
                    ag.remember(last_obs[idx], last_phase[idx], int(actions[idx]),
                                probs[idx], rewards[idx], obs[idx], cur_phase[idx],
                                dones[idx], f"{episode}_{i // self.action_interval}_{ag.id}")
                last_obs = obs
                self.total_decision_num_sim += 1
                if (self.total_decision_num_sim > self.learning_start
                        and self.total_decision_num_sim % self.update_model_rate
                        == self.update_model_rate - 1):
                    loss.append(np.mean([ag.train() for ag in agents]))
                if (self.total_decision_num_sim > self.learning_start
                        and self.total_decision_num_sim % self.update_target_rate
                        == self.update_target_rate - 1):
                    [ag.update_target_network() for ag in agents]
            if all(dones):
                break
        # Update the per-agent average uncertainty the grounding gate compares against.
        # Without this the average stays 0, so `uncertainty >= avg` is always true and
        # ugat grounding is ALWAYS skipped -- silently degenerating to naive.
        self.transition_model.finalize_policy_stats(episode, stats)
        return float(np.mean(loss)) if loss else 0.0

    # ------------------------------------------------------------------
    def _sim_rollout(self):
        """Ungrounded sim rollout -> collect (s,a,s') for the INVERSE model (sim data)."""
        env, agents = self.env_sim, self.agents_sim
        last_obs = env.reset()
        for ag in agents:
            ag.reset()
        records, i = [], 0
        dones = [False] * len(agents)
        while i < self.steps:
            if i % self.action_interval == 0:
                phase = np.stack([ag.get_phase() for ag in agents])
                actions = np.stack([ag.get_action(last_obs[idx], phase[idx], test=True)
                                    for idx, ag in enumerate(agents)])
                for _ in range(self.action_interval):
                    obs, _, dones, _ = env.step(actions.flatten())
                    i += 1
                records.extend(self._records(last_obs, actions, obs))
                last_obs = obs
            if all(dones):
                break
        self.transition_model._write_pickle(
            self.transition_model._dataset_file(forward=False, train=True), records)

    # ------------------------------------------------------------------
    def _real_rollout(self, episode):
        """Real env rollout (real_action_transforms applied) -> collect (s, fwd_action, s')
        for the FORWARD model (real data); log REAL_TRAIN (the rollout FEEDS the method,
        so it is real-budget spend, not a scoring eval)."""
        # Deploy the freshly-trained sim policy: load its weights into the real agents,
        # else real eval runs stale/initial weights (byte-identical every episode).
        self.load_agents(self.agents_real, self.model_dir)
        env, agents = self.env_real, self.agents_real
        transforms = self.real_action_transforms
        metric = self.metric_real
        metric.clear()
        last_obs = env.reset()
        for ag in agents:
            ag.reset()
        phase = np.stack([ag.get_phase() for ag in agents])
        for t in transforms:
            t.reset(agents, phase)
        records, i = [], 0
        dones = [False] * len(agents)
        while i < self.test_steps:
            if i % self.action_interval == 0:
                phase = np.stack([ag.get_phase() for ag in agents])
                # Mask-aware deployment: in shield mode make_valid_mask_fn hands the agent
                # the legal-action set (0 violations); in enforce mode it is all-permissive
                # (force-off only), so plain GAT is unchanged. This is what gives the
                # *_shield gat/ugat configs their guarantee.
                actions = np.stack([
                    self.select_action(
                        ag, idx, last_obs[idx], phase[idx], test=True,
                        valid_mask_fn=self.make_valid_mask_fn(transforms, idx),
                    )
                    for idx, ag in enumerate(agents)
                ])
                for t in transforms:
                    t.begin_interval(actions)
                rewards_list = []
                executed = actions
                for _ in range(self.action_interval):
                    executed = actions
                    for t in transforms:
                        executed = t.resolve_step(executed)
                    obs, rewards, dones, _ = env.step(executed.flatten())
                    i += 1
                    rewards_list.append(np.stack(rewards))
                metric.update(np.mean(rewards_list, axis=0))
                # Forward-model data: learn from the proposed (requested) or executed action.
                fwd_actions = (
                    np.asarray(executed).reshape(-1)
                    if self.gat_forward_action == "executed"
                    else actions
                )
                records.extend(self._records(last_obs, fwd_actions, obs))
                last_obs = obs
            if all(dones):
                break
        self.transition_model._write_pickle(
            self.transition_model._dataset_file(forward=True, train=True), records)
        self.log_metrics("REAL_TRAIN", episode, metric, 100, transforms)
