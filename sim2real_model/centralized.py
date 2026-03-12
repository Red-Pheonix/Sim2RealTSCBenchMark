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


@Registry.register_sim2real_model("centralized")
class CentralizedSim2RealTransitionModel(BaseSim2RealTransitionModel):
    def __init__(self, logger, device, world_sim, agents_sim, world_real, agents_real):
        super().__init__(logger, device, world_sim, agents_sim, world_real, agents_real)
        sim2real_params = Registry.mapping["sim2real_mapping"]["setting"].param
        self.uncertainty_setting = sim2real_params["uncertainty"]
        self.last_n_uncertainties = sim2real_params["last_n_uncertainties"]
        self.gat_path = os.path.join(
            Registry.mapping["logger_mapping"]["path"].path, "model"
        )
        self.last_two_central_uncertainties = []
        self.mean_uncertainty = 0

        num_agents = len(self.agents_real)
        ob_length = self.agents_real[0].ob_generator.ob_length
        action_dim = self.agents_real[0].action_space.n
        self.forward_model = NN_predictor(
            self.logger,
            (num_agents, ob_length),
            (num_agents, action_dim),
            ob_length,
            self.device,
            self.gat_path,
            "collected/ereal_train_full.pkl",
            False,
            1,
            "central",
        )
        self.inverse_model = UNCERTAINTY_predictor(
            self.logger,
            (num_agents, ob_length),
            0,
            0,
            (num_agents, ob_length),
            action_dim,
            self.device,
            self.gat_path,
            "collected/esim_train_full.pkl",
            backward=True,
            history=1,
            mode="central",
        )
        self.forward_models = [self.forward_model]
        self.inverse_models = [self.inverse_model]
        self.action_dims = [action_dim for _ in range(num_agents)]
        self.action_dim = action_dim

    def ground_actions(self, last_obs, actions, stats):
        one_hot_actions = np.concatenate(
            [idx2onehot(np.array([action]), self.action_dim) for action in actions],
            axis=0,
        )
        state_tensor = (
            torch.tensor(np.array(last_obs)).squeeze(1).unsqueeze(0).float().to(self.device)
        )
        action_tensor = torch.tensor(one_hot_actions).unsqueeze(0).float().to(self.device)
        pred_next_state = self.forward_model.model(state_tensor, action_tensor)
        grounded_action, uncertainty = self.inverse_model.model(
            state_tensor, pred_next_state
        )

        if self.uncertainty_setting:
            stats["uncertainty_sum"] += uncertainty.item()
            if uncertainty >= self.mean_uncertainty:
                return actions, [9 for _ in range(len(self.agents_sim))]

        grounded_action = grounded_action.view(len(self.agents_sim), self.action_dim)
        actions = torch.argmax(grounded_action, dim=1).cpu().numpy()
        grounded_actions = actions.copy()
        stats["grounded_action_count"] += len(self.agents_sim)
        for idx in range(len(stats["ga_by_agent"])):
            stats["ga_by_agent"][idx] += 1
        return actions, grounded_actions

    def finalize_policy_stats(self, episode, stats):
        self.last_two_central_uncertainties.append(stats["uncertainty_sum"] / 360)
        if len(self.last_two_central_uncertainties) > self.last_n_uncertainties:
            self.last_two_central_uncertainties.pop(0)
        self.mean_uncertainty = np.mean(self.last_two_central_uncertainties)
        self.logger.info(
            "Policy training episode: {}, grounded actions taken: {}, last uncertainties: {}, avg uncertainty: {}, grounded actions by agent: {}".format(
                episode,
                stats["grounded_action_count"],
                self.last_two_central_uncertainties,
                self.mean_uncertainty,
                stats["ga_by_agent"],
            )
        )

    def train_transition_models(self):
        load_and_split_forward_data(
            "collected/ereal_train.pkl",
            "collected/ereal_train_full.pkl",
            "collected/ereal_test_full.pkl",
            self.action_dim,
            0.2,
            42,
            "centralized",
            len(self.agents_real),
        )
        load_and_split_inverse_data(
            "collected/esim_train.pkl",
            "collected/esim_train_full.pkl",
            "collected/esim_test_full.pkl",
            self.action_dim,
            0.2,
            42,
            "centralized",
            len(self.agents_sim),
        )
        self.forward_model.train(
            100, "forward", len(self.agents_real), 5000 * len(self.agents_real)
        )
        self.inverse_model.train(
            100, "inverse", len(self.agents_sim), 5000 * len(self.agents_real)
        )
