import os
import numpy as np
import torch

from agent.utils import idx2onehot
from common.gat_utils import (
    BaseNNPredictor,
    BaseUncertaintyPredictor,
    Central_Inverse_N_net,
    Central_N_net,
    PKLDataset,
)
from common.registry import Registry
from .base import BaseSim2RealTransitionModel, pad_and_concat


class CentralizedNNPredictor(BaseNNPredictor):
    def make_model(self):
        return Central_N_net(self.state_dim, self.action_dim, self.out_dim, self.backward).float()

    def get_train_dataset_path(self, agent_num=None):
        return "collected/ereal_train_full.pkl"

    def get_test_dataset_path(self, agent_num=None):
        return "collected/ereal_test_full.pkl"

    def compute_loss(self, y_pred, y_true):
        return self.criterion(y_pred, y_true)


class CentralizedUncertaintyPredictor(BaseUncertaintyPredictor):
    def make_model(self):
        return Central_Inverse_N_net(
            self.ind_state_dim, self.pred_state_dim, self.out_dim, self.backward
        ).float()

    def get_train_dataset_path(self, agent_num=None):
        return "collected/esim_train_full.pkl"

    def get_test_dataset_path(self, agent_num=None):
        return "collected/esim_test_full.pkl"

    def build_dataset(self, dataset_path):
        return PKLDataset(dataset_path, "central")

    def unpack_batch(self, data):
        state, pred_state, y_true = data
        return (
            (
                state.to(self.DEVICE, non_blocking=True),
                pred_state.to(self.DEVICE, non_blocking=True),
            ),
            y_true.to(self.DEVICE, non_blocking=True),
        )

    def compute_loss(self, y_pred, y_true):
        return self.criterion(y_pred.permute(0, 2, 1), y_true.squeeze(-1).long())


@Registry.register_sim2real_model("centralized")
class CentralizedSim2RealTransitionModel(BaseSim2RealTransitionModel):
    def __init__(self, logger, device, world_sim, agents_sim, world_real, agents_real):
        super().__init__(logger, device, world_sim, agents_sim, world_real, agents_real)
        sim2real_params = Registry.mapping["sim2real_mapping"]["setting"].param
        self.uncertainty_setting = sim2real_params["uncertainty"]
        self.last_n_uncertainties = sim2real_params["last_n_uncertainties"]
        self.setting = Registry.mapping["command_mapping"]["setting"].param.get("real_setting")
        self.gat_path = os.path.join(
            Registry.mapping["logger_mapping"]["path"].path,
            "model",
            Registry.mapping["command_mapping"]["setting"].param["gat_model"],
            self.setting,
        )
        self.last_two_central_uncertainties = []
        self.mean_uncertainty = 0

        num_agents = len(self.agents_real)
        # ob_length = self.agents_real[0].ob_generator.ob_length
        # action_dim = self.agents_real[0].action_space.n
        
        ob_length = max([ag.ob_generator.ob_length for ag in self.agents_real])
        action_dim = max([ag.action_space.n for ag in self.agents_real])
        
        self.forward_model = CentralizedNNPredictor(
            0,
            self.logger,
            (num_agents, ob_length),
            (num_agents, action_dim),
            ob_length,
            self.device,
            self.gat_path,
            "collected/ereal_train_full.pkl",
            False,
            1,
        )
        self.inverse_model = CentralizedUncertaintyPredictor(
            0,
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
        )
        self.forward_models = [self.forward_model]
        self.inverse_models = [self.inverse_model]
        self.action_dims = [action_dim for _ in range(num_agents)]
        self.action_dim = action_dim

    def collect_sim_transition(self, last_obs, actions, obs):
        transitions = []
        for idx in range(len(self.agents_real)):
            transitions.append(
                (
                    pad_and_concat(last_obs),
                    np.asarray(actions).reshape(-1, 1),
                    pad_and_concat(obs),
                    actions[idx],
                )
            )
        
        return transitions

    collect_real_transition = collect_sim_transition

    def ground_actions(self, last_obs, actions, stats):
        one_hot_actions = np.concatenate(
            [idx2onehot(np.array([action]), self.action_dim) for action in actions],
            axis=0,
        )
        state_tensor = (
            torch.tensor(np.array(pad_and_concat(last_obs))).squeeze(1).unsqueeze(0).float().to(self.device)
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

    def prepare_forward_data(self, test_size=0.2, random_seed=42):
        data = self._load_pickle("collected/ereal_train.pkl")
        records = []

        for record in data:
            states, actions, next_states = record[:3]
            one_hot_actions = np.concatenate(
                [idx2onehot(np.array([action]), self.action_dim) for action in actions],
                axis=0,
            )
            records.append((states, one_hot_actions, next_states))

        self._split_joint_records(
            records,
            "collected/ereal_train_full.pkl",
            "collected/ereal_test_full.pkl",
            test_size=test_size,
            random_seed=random_seed,
        )

    def prepare_inverse_data(self, test_size=0.2, random_seed=42):
        data = self._load_pickle("collected/esim_train.pkl")
        records = []

        for record in data:
            states, actions, next_states = record[:3]
            records.append((states, next_states, actions))

        self._split_joint_records(
            records,
            "collected/esim_train_full.pkl",
            "collected/esim_test_full.pkl",
            test_size=test_size,
            random_seed=random_seed,
        )

    def train_transition_models(self):
        self.prepare_forward_data()
        self.prepare_inverse_data()
        self.forward_model.train(
            100, "forward", len(self.agents_real), 5000 * len(self.agents_real)
        )
        self.inverse_model.train(
            100, "inverse", len(self.agents_sim), 5000 * len(self.agents_real)
        )

    def save_models(self, e):
        for model in self.forward_models:
            model.save_model(e)
        for model in self.inverse_models:
            model.save_model(e)

    def load_models(self):
        for model in self.forward_models:
            model.load_model()
        for model in self.inverse_models:
            model.load_model()
