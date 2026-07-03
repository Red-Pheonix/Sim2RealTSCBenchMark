"""Phase-transition special case: the SHIELD method (and its unshielded PT baseline).

A separate special case (own trainer) because it needs the action-gap
`PhaseTransition` machinery and a per-network transition table, which only `tempe`
and `bullhead` ship. The reward-gap angle: when `R_real` weights a **safety** term
(controller-rule violations), a method can either try to *infer* that weight
(reward_inference / morl_grid / dynamic_reward_shaping) or simply *enforce* safety as a hard
constraint and never put it in the reward.
Shielding does the latter (Safe RL via Shielding, Alshiekh et al. AAAI 2018).

Two methods share this trainer:
  * `shield`    -- PhaseTransition in `shield` mode: the agent is masked to legal
    actions at decision time (sim train AND real eval), so violations -> ~0. Trains
    on the native proxy reward (efficiency); safety is handled by the constraint.
  * `pt_naive`  -- PhaseTransition in `enforce` mode: no decision-time mask, illegal
    switches are dropped at execution and COUNTED as violations -> the safety cost is
    incurred. The baseline the shield is compared against.

The `safety` feature-bank component is the per-decision violation flag from the PT
transform (0 elsewhere -- only this trainer measures it), so a `safety_heavy` setting
scores both methods on the same `R_real`.
"""

import os

import numpy as np
from tqdm import tqdm

from common.metrics import Metrics
from common.registry import Registry
from environment import TSCEnv
from trainer.actions.phase_transition import PhaseTransition
from trainer.rewards.base import Sim2RealRewardsTrainer, resolve_components
from trainer.rewards.feature_bank import FeatureBank
from trainer.rewards.reward_transforms import TrueReward


class _PhaseTransitionRewardTrainer(Sim2RealRewardsTrainer):
    """Shared base for the shield / pt_naive methods. `shield` toggles the mode."""

    SHIELD = False

    def __init__(self, logger, gpu=0, cpu=False, name="sim2real_rewards"):
        super().__init__(logger=logger, gpu=gpu, cpu=cpu, name=name)
        cfg = self.get_sim2real_config()
        self.method = "shield" if self.SHIELD else "pt_naive"
        variant = cfg.get("phase_transition")
        if not variant:
            raise ValueError(
                "phase-transition methods need `phase_transition: <variant>` in the "
                "setting (e.g. pt_cyclic). Only tempe/bullhead ship tables."
            )
        network = Registry.mapping["world_mapping"]["setting"].param.get("network")
        world_dir = Registry.mapping["world_mapping"]["setting"].param["dir"]
        pt_path = os.path.join(
            world_dir, "raw_data", network, "phase_transitions", variant + ".json"
        )
        if not os.path.exists(pt_path):
            raise FileNotFoundError(
                f"phase-transition table not found at {pt_path}. Shield only supports "
                f"networks that ship a table (tempe, bullhead)."
            )
        mode = "shield" if self.SHIELD else "enforce"
        # Make sure the safety component exists in the bank so R_real can score it.
        if "safety" not in self.feature_bank_real.components:
            comps = list(self.extra_components) + ["safety"]
            avail_sim, fns_sim = resolve_components(self.world_sim, comps)
            avail_real, fns_real = resolve_components(self.world_real, comps)
            self.feature_bank_sim = FeatureBank(
                self.world_sim, self.agents_sim, comps,
                available=avail_sim, info_fns=fns_sim,
            )
            self.feature_bank_real = FeatureBank(
                self.world_real, self.agents_real, comps,
                available=avail_real, info_fns=fns_real,
            )
            self.components = self.feature_bank_real.components
            w_star = self.feature_bank_real.weight_vector(self.true_reward_weights)
            # Rebuild (not mutate) the scorer: the component list grew, so w AND norm
            # must be re-derived over the new layout. Mutating only `.w` would leave
            # `.norm` at the old length -> shape mismatch at eval.
            self.true_reward = TrueReward(
                w_star, self.components, norm=self.component_norm
            )
        self._safety_idx = self.components.index("safety")
        # This trainer DOES compute safety (from PhaseTransition.last_violation), which
        # `resolve_components` can't know about (no info fn), so mark it available on both
        # banks -- otherwise it stays sim-unobservable and R_real wouldn't score it.
        for fb in (self.feature_bank_sim, self.feature_bank_real):
            fb.available[fb.components.index("safety")] = True
        self.pt_sim = PhaseTransition(self.agents_sim, self.action_interval, pt_path, mode=mode)
        self.pt_real = PhaseTransition(self.agents_real, self.action_interval, pt_path, mode=mode)

    def create_world(self):
        # Same-sim (sumo<->sumo) for the phase-transition special case: the PT tables
        # are sumo-NEMA (8-phase), and cityflow's tempe is a different 2-phase network,
        # so a cross-sim policy can't even transfer (action spaces differ). Using one
        # sumo world for both sides also correctly isolates the reward/safety gap from
        # the dynamics gap. libsumo is a global singleton, so sim and real SHARE the
        # single sumo instance (sequential train-then-eval is fine).
        interface = Registry.mapping["command_mapping"]["setting"].param["interface"]
        self.world_real = Registry.mapping["world_mapping"]["sumo"](
            self.sumo_path, **{"interface": interface}
        )
        self.world_sim = self.world_real

    def _mask_fn(self, pt, idx):
        return lambda phase: pt.valid_mask(idx, phase)

    def _feature_matrix_pt(self, feature_bank, agents, last_phase, cur_phase, pt):
        """Core φ from the bank, with the safety slot = this decision's violation flag."""
        phi = self._feature_matrix(feature_bank, agents, last_phase, cur_phase)
        viol = np.asarray(pt.last_violation, dtype=float)
        phi[:, self._safety_idx] = viol
        return phi

    # --- PT-wired rollouts (proxy training reward; mask/enforce via PhaseTransition) ---
    def run_train_episode(self, *, env, metric, agents, feature_bank, episode, desc):
        pt = self.pt_sim if env is self.env_sim else self.pt_real
        metric.clear()
        last_obs = env.reset()
        for ag in agents:
            ag.reset()
        init_phase = np.stack([ag.get_phase() for ag in agents])
        pt.reset(agents, init_phase)
        i, dones = 0, [False] * len(agents)
        episode_loss = []
        pbar = tqdm(total=int(self.steps / self.action_interval), desc=desc)
        while i < self.steps:
            if i % self.action_interval == 0:
                pbar.update()
                last_phase = np.stack([ag.get_phase() for ag in agents])
                actions = np.stack(
                    [
                        ag.get_action(
                            last_obs[idx], last_phase[idx], test=False,
                            valid_mask_fn=self._mask_fn(pt, idx),
                        )
                        for idx, ag in enumerate(agents)
                    ]
                )
                actions_prob = [
                    ag.get_action_prob(last_obs[idx], last_phase[idx])
                    for idx, ag in enumerate(agents)
                ]
                pt.begin_interval(actions)
                rewards_list = []
                for _ in range(self.action_interval):
                    executed = pt.resolve_step(actions)
                    obs, rewards, dones, _ = env.step(executed.flatten())
                    i += 1
                    rewards_list.append(np.stack(rewards))
                proxy_rewards = np.mean(rewards_list, axis=0)
                metric.update(proxy_rewards)
                cur_phase = np.stack([ag.get_phase() for ag in agents])
                for idx, ag in enumerate(agents):
                    ag.remember(
                        last_obs[idx], last_phase[idx], actions[idx], actions_prob[idx],
                        float(proxy_rewards[idx]), obs[idx], cur_phase[idx], dones[idx],
                        f"{episode}_{i // self.action_interval}_{ag.id}",
                    )
                self.total_decision_num_sim += 1
                last_obs = obs
            if (
                self.total_decision_num_sim > self.learning_start
                and self.total_decision_num_sim % self.update_model_rate
                == self.update_model_rate - 1
            ):
                episode_loss.append(np.stack([ag.train() for ag in agents]))
            if (
                self.total_decision_num_sim > self.learning_start
                and self.total_decision_num_sim % self.update_target_rate
                == self.update_target_rate - 1
            ):
                [ag.update_target_network() for ag in agents]
            if all(dones):
                break
        pbar.close()
        return (np.mean(np.array(episode_loss)) if episode_loss else 0), i

    def run_eval_episode(self, *, env, metric, agents, feature_bank, desc,
                         count_budget=True):
        if count_budget:
            self._real_rollouts += 1
        pt = self.pt_sim if env is self.env_sim else self.pt_real
        metric.clear()
        obs = env.reset()
        for ag in agents:
            ag.reset()
        init_phase = np.stack([ag.get_phase() for ag in agents])
        pt.reset(agents, init_phase)
        pt.reset_stats()
        i, dones = 0, [False] * len(agents)
        phi_sum = np.zeros(len(feature_bank.components))
        pbar = tqdm(total=int(self.test_steps / self.action_interval), desc=desc)
        while i < self.test_steps:
            if i % self.action_interval == 0:
                pbar.update()
                last_phase = np.stack([ag.get_phase() for ag in agents])
                actions = np.stack(
                    [
                        ag.get_action(
                            obs[idx], last_phase[idx], test=True,
                            valid_mask_fn=self._mask_fn(pt, idx),
                        )
                        for idx, ag in enumerate(agents)
                    ]
                )
                pt.begin_interval(actions)
                rewards_list = []
                for _ in range(self.action_interval):
                    executed = pt.resolve_step(actions)
                    obs, rewards, dones, _ = env.step(executed.flatten())
                    i += 1
                    rewards_list.append(np.stack(rewards))
                metric.update(np.mean(rewards_list, axis=0))
                cur_phase = np.stack([ag.get_phase() for ag in agents])
                phi_sum += self._feature_matrix_pt(
                    feature_bank, agents, last_phase, cur_phase, pt
                ).sum(axis=0)
            if all(dones):
                break
        pbar.close()
        v, f, dnum = pt.collect_stats()
        vr = v / dnum if dnum else 0.0
        self.logger.info("%s violation_rate=%.4f (viol=%s, decisions=%s)", desc, vr, v, dnum)
        decisions = max(int(self.test_steps / self.action_interval), 1)
        true_reward = float(self.true_reward.reward(phi_sum[None, :])[0]) / decisions
        breakdown = {k: v_ / decisions for k, v_ in self.true_reward.breakdown(phi_sum[None, :]).items()}
        return true_reward, breakdown


@Registry.register_trainer("sim2real_rewards_shield")
class Sim2RealRewardsShieldTrainer(_PhaseTransitionRewardTrainer):
    SHIELD = True


@Registry.register_trainer("sim2real_rewards_pt_naive")
class Sim2RealRewardsPTNaiveTrainer(_PhaseTransitionRewardTrainer):
    SHIELD = False
