import os
import numpy as np
import torch

from agent.utils import idx2onehot
from common.gat_utils import (
    load_and_split_forward_data,
    load_and_split_inverse_data,
    NN_predictor,
    UNCERTAINTY_predictor,
)
from common.registry import Registry
from .base import BaseSim2RealTransitionModel


@Registry.register_sim2real_model("decentralized")
class DecentralizedSim2RealTransitionModel(BaseSim2RealTransitionModel):
    def __init__(self, logger, device, world_sim, agents_sim, world_real, agents_real):
        super().__init__(logger, device, world_sim, agents_sim, world_real, agents_real)
        sim2real_params = Registry.mapping["sim2real_mapping"]["setting"].param
        self.uncertainty_setting = sim2real_params["uncertainty"]
        self.last_n_uncertainties = sim2real_params["last_n_uncertainties"]
        self.gat_path = os.path.join(
            Registry.mapping["logger_mapping"]["path"].path, "model"
        )
        self.last_two_uncertainties = {idx: [] for idx in range(len(self.agents_sim))}
        self.avg_agent_uncertainties = [0 for _ in range(len(self.agents_sim))]

        for agent in self.agents_real:
            forward_model = NN_predictor(
                self.logger,
                (1, agent.ob_generator.ob_length),
                (1, agent.action_space.n),
                agent.ob_generator.ob_length,
                self.device,
                self.gat_path,
                "collected/ereal_train_full.pkl",
            )
            inverse_model = UNCERTAINTY_predictor(
                self.logger,
                (1, agent.ob_generator.ob_length),
                0,
                0,
                (1, agent.ob_generator.ob_length),
                agent.action_space.n,
                self.device,
                self.gat_path,
                "collected/esim_train_full.pkl",
                backward=True,
                history=1,
                mode="dec",
            )
            self.forward_models.append(forward_model)
            self.inverse_models.append(inverse_model)
            self.action_dims.append(agent.action_space.n)

    def collect_sim_transition(self, last_obs, actions, obs):
        return [
            (idx, state, action, next_state)
            for idx, (state, action, next_state) in enumerate(zip(last_obs, actions, obs))
        ]

    collect_real_transition = collect_sim_transition

    def ground_actions(self, last_obs, actions, stats):
        grounded_actions = [9 for _ in range(len(self.agents_sim))]
        updated_actions = actions.copy()
        for idx, agent in enumerate(self.agents_sim):
            action_dim = agent.action_space.n
            individual_state = torch.from_numpy(last_obs[idx]).float().to(self.device).unsqueeze(0)
            individual_action = (
                torch.from_numpy(idx2onehot(np.array([updated_actions[idx]]), action_dim))
                .float()
                .to(self.device)
                .unsqueeze(0)
            )
            pred_next_state = self.forward_models[idx].model(
                individual_state, individual_action
            ).unsqueeze(0)
            grounded_action, uncertainty = self.inverse_models[idx].model(
                individual_state, pred_next_state
            )
            if self.uncertainty_setting:
                stats["agent_uncertainty_sums"][idx] += uncertainty.item()
                if uncertainty >= self.avg_agent_uncertainties[idx]:
                    continue
            updated_actions[idx] = (
                torch.argmax(grounded_action.view(1, action_dim), dim=1).cpu().item()
            )
            grounded_actions[idx] = updated_actions[idx]
            stats["grounded_action_count"] += 1
            stats["ga_by_agent"][idx] += 1
        return updated_actions, grounded_actions

    def finalize_policy_stats(self, episode, stats):
        for idx in range(len(self.agents_sim)):
            self.last_two_uncertainties[idx].append(
                stats["agent_uncertainty_sums"][idx] / 360
            )
            if len(self.last_two_uncertainties[idx]) > self.last_n_uncertainties:
                self.last_two_uncertainties[idx].pop(0)
            self.avg_agent_uncertainties[idx] = np.mean(
                self.last_two_uncertainties[idx]
            )
        self.logger.info(
            "Policy training episode: {}, grounded actions taken: {}, last two uncertainties: {}, avg agent uncertainties: {}, grounded actions by agent: {}".format(
                episode,
                stats["grounded_action_count"],
                self.last_two_uncertainties,
                self.avg_agent_uncertainties,
                stats["ga_by_agent"],
            )
        )

    def train_transition_models(self):
        load_and_split_forward_data(
            "collected/ereal_train.pkl",
            "collected/ereal_train_full",
            "collected/ereal_test_full",
            self.action_dims,
            0.2,
            42,
            "decentralized",
            len(self.agents_real),
        )
        load_and_split_inverse_data(
            "collected/esim_train.pkl",
            "collected/esim_train_full",
            "collected/esim_test_full",
            self.action_dims,
            0.2,
            42,
            "decentralized",
            len(self.agents_sim),
        )
        for idx in range(len(self.agents_sim)):
            self.forward_models[idx].train(100, "forward", idx, 5000, "decentralized")
            self.inverse_models[idx].train(100, "inverse", idx, 5000, "decentralized")
