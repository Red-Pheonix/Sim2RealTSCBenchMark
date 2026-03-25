import os
import numpy as np
import torch

from agent.utils import idx2onehot
from common.gat_utils import (
    BaseNNPredictor,
    BaseUncertaintyPredictor,
    Dec_Inverse_N_net,
    N_net,
    PKLDataset,
)
from common.registry import Registry
from .base import BaseSim2RealTransitionModel


class DecentralizedNNPredictor(BaseNNPredictor):
    def make_model(self):
        return N_net(self.state_dim, self.action_dim, self.out_dim, self.backward).float()

    def get_train_dataset_path(self, agent_num=None):
        if agent_num is None:
            raise ValueError("agent_num is required for decentralized forward training.")
        return f"collected/ereal_train_full_agent_{agent_num}.pkl"

    def get_test_dataset_path(self, agent_num=None):
        if agent_num is None:
            raise ValueError("agent_num is required for decentralized forward evaluation.")
        return f"collected/ereal_test_full_agent_{agent_num}.pkl"

    def compute_loss(self, y_pred, y_true):
        return self.criterion(y_pred.squeeze(1), y_true.squeeze(1))


class DecentralizedUncertaintyPredictor(BaseUncertaintyPredictor):
    def make_model(self):
        return Dec_Inverse_N_net(
            self.ind_state_dim, self.pred_state_dim, self.out_dim, self.backward
        ).float()

    def get_train_dataset_path(self, agent_num=None):
        if agent_num is None:
            raise ValueError("agent_num is required for decentralized inverse training.")
        return f"collected/esim_train_full_agent_{agent_num}.pkl"

    def get_test_dataset_path(self, agent_num=None):
        if agent_num is None:
            raise ValueError("agent_num is required for decentralized inverse evaluation.")
        return f"collected/esim_test_full_agent_{agent_num}.pkl"

    def build_dataset(self, dataset_path):
        return PKLDataset(dataset_path)

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
        return self.criterion(y_pred.squeeze(1), y_true.squeeze().long())


@Registry.register_sim2real_model("decentralized")
class DecentralizedSim2RealTransitionModel(BaseSim2RealTransitionModel):
    def __init__(self, logger, device, world_sim, agents_sim, world_real, agents_real):
        super().__init__(logger, device, world_sim, agents_sim, world_real, agents_real)
        sim2real_params = Registry.mapping["sim2real_mapping"]["setting"].param
        self.uncertainty_setting = sim2real_params["uncertainty"]
        self.last_n_uncertainties = sim2real_params["last_n_uncertainties"]
        self.setting = Registry.mapping["command_mapping"]["setting"].param.get("real_setting")
        self.gat_path = os.path.join(
            Registry.mapping["logger_mapping"]["path"].path, "model", sim2real_params["gattype"] , self.setting
        )
        self.last_two_uncertainties = {idx: [] for idx in range(len(self.agents_sim))}
        self.avg_agent_uncertainties = [0 for _ in range(len(self.agents_sim))]

        for i, agent in enumerate(self.agents_real):
            forward_model = DecentralizedNNPredictor(
                i,
                self.logger,
                (1, agent.ob_generator.ob_length),
                (1, agent.action_space.n),
                agent.ob_generator.ob_length,
                self.device,
                self.gat_path,
                "collected/ereal_train_full.pkl",
            )
            inverse_model = DecentralizedUncertaintyPredictor(
                i,
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

    def prepare_forward_data(self, test_size=0.2, random_seed=42):
        data = self._load_pickle("collected/ereal_train.pkl")
        agent_records = {agent_idx: [] for agent_idx in range(len(self.agents_real))}

        for record in data:
            agent_idx, states, actions, next_states = record
            action_dim = self.action_dims[agent_idx]
            one_hot_actions = torch.tensor(
                np.concatenate(
                    [idx2onehot(np.array([action]), action_dim) for action in actions],
                    axis=0,
                ),
                dtype=torch.float32,
            )
            agent_records[agent_idx].append(
                (
                    torch.tensor(states, dtype=torch.float32),
                    one_hot_actions,
                    torch.tensor(next_states, dtype=torch.float32),
                )
            )

        self._split_agent_records(
            agent_records,
            "collected/ereal_train_full",
            "collected/ereal_test_full",
            test_size=test_size,
            random_seed=random_seed,
        )

    def prepare_inverse_data(self, test_size=0.2, random_seed=42):
        data = self._load_pickle("collected/esim_train.pkl")
        agent_records = {agent_idx: [] for agent_idx in range(len(self.agents_sim))}

        for record in data:
            agent_idx, states, actions, next_states = record
            agent_records[agent_idx].append(
                (
                    torch.tensor(states, dtype=torch.float32),
                    torch.tensor(next_states, dtype=torch.float32),
                    torch.tensor(actions, dtype=torch.long),
                )
            )

        self._split_agent_records(
            agent_records,
            "collected/esim_train_full",
            "collected/esim_test_full",
            test_size=test_size,
            random_seed=random_seed,
        )

    def train_transition_models(self):
        self.prepare_forward_data()
        self.prepare_inverse_data()
        for idx in range(len(self.agents_sim)):
            self.forward_models[idx].train(100, "forward", idx, 5000, "decentralized")
            self.inverse_models[idx].train(100, "inverse", idx, 5000, "decentralized")
    
    def save_models(self, e):
        for model in self.forward_models:
            model.save_model(e=e)
        for model in self.inverse_models:
            model.save_model(e)
    
    def load_models(self):
        for model in self.forward_models:
            model.load_model()
        for model in self.inverse_models:
            model.load_model()
