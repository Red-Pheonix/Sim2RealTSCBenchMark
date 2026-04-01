import os
import json
import random
from pathlib import Path
import numpy as np
import torch

from agent.utils import idx2onehot
from common.gat_utils import (
    BaseNNPredictor,
    BaseUncertaintyPredictor,
    Inverse_N_Net_No_State,
    Inverse_N_net,
    N_net,
    N_net_noAction,
    N_net_noState,
    PKLDataset,
)
from common.registry import Registry
from .base import BaseSim2RealTransitionModel, pad_and_concat
from .decentralized import DecentralizedSim2RealTransitionModel


class JLGATNNPredictor(BaseNNPredictor):
    def make_model(self):
        return N_net(self.state_dim, self.action_dim, self.out_dim, self.backward).float()

    def get_train_dataset_path(self, agent_num=None):
        if agent_num is None:
            raise ValueError("agent_num is required for JL-GAT forward training.")
        return self._dataset_path(train=True, agent_num=agent_num)

    def get_test_dataset_path(self, agent_num=None):
        if agent_num is None:
            raise ValueError("agent_num is required for JL-GAT forward evaluation.")
        return self._dataset_path(train=False, agent_num=agent_num)

    def compute_loss(self, y_pred, y_true):
        return self.criterion(y_pred, y_true.squeeze(1))


class JLGATNoActionNNPredictor(JLGATNNPredictor):
    def make_model(self):
        return N_net_noAction(self.state_dim, self.action_dim, self.out_dim, self.backward).float()


class JLGATNoStateNNPredictor(JLGATNNPredictor):
    def make_model(self):
        return N_net_noState(self.state_dim, self.action_dim, self.out_dim, self.backward).float()


class JLGATUncertaintyPredictor(BaseUncertaintyPredictor):
    def make_model(self):
        return Inverse_N_net(
            self.ind_state_dim,
            self.n_state_dim,
            self.action_dim,
            self.pred_state_dim,
            self.out_dim,
            self.backward,
        ).float()

    def get_train_dataset_path(self, agent_num=None):
        if agent_num is None:
            raise ValueError("agent_num is required for JL-GAT inverse training.")
        return self._dataset_path(train=True, agent_num=agent_num)

    def get_test_dataset_path(self, agent_num=None):
        if agent_num is None:
            raise ValueError("agent_num is required for JL-GAT inverse evaluation.")
        return self._dataset_path(train=False, agent_num=agent_num)

    def build_dataset(self, dataset_path):
        return PKLDataset(dataset_path, "jlg")

    def unpack_batch(self, data):
        state, n_state, n_action, pred_state, y_true = data
        return (
            (
                state.to(self.DEVICE, non_blocking=True),
                n_state.to(self.DEVICE, non_blocking=True),
                n_action.to(self.DEVICE, non_blocking=True),
                pred_state.to(self.DEVICE, non_blocking=True),
            ),
            y_true.to(self.DEVICE, non_blocking=True),
        )

    def compute_loss(self, y_pred, y_true):
        return self.criterion(y_pred, y_true.squeeze().long())


class JLGATNoStateUncertaintyPredictor(JLGATUncertaintyPredictor):
    def make_model(self):
        return Inverse_N_Net_No_State(
            self.ind_state_dim,
            self.action_dim,
            self.pred_state_dim,
            self.out_dim,
            self.backward,
        ).float()


@Registry.register_sim2real_model("jlgat")
class JLGATSim2RealTransitionModel(DecentralizedSim2RealTransitionModel):
    def __init__(
        self,
        logger,
        device,
        world_sim,
        agents_sim,
        world_real,
        agents_real,
        dataset_dir="collected",
    ):
        BaseSim2RealTransitionModel.__init__(
            self,
            logger,
            device,
            world_sim,
            agents_sim,
            world_real,
            agents_real,
            dataset_dir=dataset_dir,
        )
        sim2real_params = Registry.mapping["sim2real_mapping"]["setting"].param
        self.uncertainty_setting = sim2real_params["uncertainty"]
        self.last_n_uncertainties = sim2real_params["last_n_uncertainties"]
        self.prob_grounding = sim2real_params.get("prob_grounding", 0)
        
        probing_radius = Registry.mapping["world_mapping"]["setting"].param.get("probing_radius")
        if probing_radius:
            self.probing_radius = probing_radius
        else:         
            self.probing_radius = sim2real_params.get("probing_radius")
        
        Registry.mapping["world_mapping"]["setting"].param.get("probing_radius")
        
        self.net = Registry.mapping["trainer_mapping"]["setting"].param["network"]
        self.setting = Registry.mapping["command_mapping"]["setting"].param.get("real_setting")
        self.gat_path = os.path.join(
            Registry.mapping["logger_mapping"]["path"].path,
            "model",
            Registry.mapping["command_mapping"]["setting"].param["gat_model"],
            self.setting,
        )
        self.last_two_uncertainties = {idx: [] for idx in range(len(self.agents_sim))}
        self.avg_agent_uncertainties = [0 for _ in range(len(self.agents_sim))]

        rank2inter = {rank: idx for idx, rank in self.world_sim.id2idx.items()}
        inter_positions = {
            self.world_sim.id2idx[inter["id"]]: inter["point"]
            for inter in self.world_sim.roadnet["intersections"]
            if inter["id"] in self.world_sim.intersection_ids
        }
        inter_positions = dict(sorted(inter_positions.items()))

        self.neighbour_infos = {}
        for inter_id, point1 in inter_positions.items():
            distances = [self.calc_dist(point1, point2) for point2 in inter_positions.values()]
            self.neighbour_infos[inter_id] = [
                i for i, dist in enumerate(distances) if dist < self.probing_radius
            ]

        world_param = Registry.mapping["world_mapping"]["setting"].param
        roadnet_path = Path(world_param["dir"]) / world_param["roadnetFile"]
        neighbors_file = roadnet_path.parent / f"{self.net}_neighbour_overrides.json"
        if os.path.exists(neighbors_file):
            with open(neighbors_file, "r") as file_obj:
                neighbor_overrides = json.load(file_obj)
            for inter, inter_neighbors in neighbor_overrides.items():
                self.neighbour_infos[self.world_sim.id2idx[inter]] = sorted(
                    [self.world_sim.id2idx[neighbour] for neighbour in inter_neighbors]
                )

        for rank, neighbors in self.neighbour_infos.items():
            n_neighbors = len(neighbors)
            if n_neighbors <= 1:
                raise Exception(
                    f"At least one neighbor required for JL-GAT for intersection: {rank2inter[rank]}. Try adjusting the probing radius."
                )
            print(f"JLGAT Neighbors: {self.neighbour_infos}")
            self.agents_real[rank].neighbors = n_neighbors

        action_length = max(
            [self.agents_real[inter].action_space.n for inter in inter_positions.keys()]
        )
        for idx, agent in enumerate(self.agents_real):
            ob_length = max(
                [
                    self.agents_real[inter].ob_generator.ob_length
                    for inter in self.neighbour_infos[idx]
                ]
            )
            forward_model = JLGATNNPredictor(
                idx,
                self.logger,
                (agent.neighbors, ob_length),
                (agent.neighbors, action_length),
                self.agents_real[idx].ob_generator.ob_length,
                self.device,
                self.gat_path,
                self.dataset_dir,
            )
            only_neighbors = [inter for inter in self.neighbour_infos[idx] if inter != idx]
            neighbor_ob_length = max(
                [
                    self.agents_real[inter].ob_generator.ob_length
                    for inter in only_neighbors
                ]
            )
            inverse_model = JLGATUncertaintyPredictor(
                idx,
                self.logger,
                (1, self.agents_real[idx].ob_generator.ob_length),
                (agent.neighbors - 1, neighbor_ob_length),
                (agent.neighbors - 1, action_length),
                (1, self.agents_real[idx].ob_generator.ob_length),
                self.agents_real[idx].action_space.n,
                self.device,
                self.gat_path,
                self.dataset_dir,
                backward=True,
            )
            self.forward_models.append(forward_model)
            self.inverse_models.append(inverse_model)
            self.action_dims.append(forward_model.action_dim[-1])

    def collect_sim_transition(self, last_obs, actions, obs):
        transitions = []
        for agent, neighbors in self.neighbour_infos.items():
            neighbor_idx = [i for i in neighbors if i != agent]
            transitions.append(
                (
                    agent,
                    last_obs[agent],
                    pad_and_concat([last_obs[i] for i in neighbor_idx]),
                    np.concatenate([actions[i] for i in neighbor_idx], axis=0).reshape(-1, 1),
                    obs[agent],
                    actions[agent],
                )
            )
        return transitions

    def collect_real_transition(self, last_obs, actions, obs):
        transitions = []
        for rank, neighbors in self.neighbour_infos.items():
            transitions.append(
                (
                    rank,
                    pad_and_concat([last_obs[i] for i in neighbors]),
                    np.concatenate([actions[i] for i in neighbors], axis=0).reshape(-1, 1),
                    obs[rank],
                )
            )
        return transitions

    def ground_actions(self, last_obs, actions, stats):
        grounded_actions = [9 for _ in range(len(self.agents_sim))]
        updated_actions = actions.copy()
        for idx, _agent in enumerate(self.agents_sim):
            relevant_indices = self.neighbour_infos.get(idx, [])
            relevant_states = pad_and_concat([last_obs[i] for i in relevant_indices])
            relevant_n_states = pad_and_concat(
                [last_obs[i] for i in relevant_indices if i != idx]
            )
            relevant_actions = pad_and_concat(
                [
                    idx2onehot(np.array([updated_actions[i]]), self.action_dims[i])
                    for i in relevant_indices
                ]
            )
            neighbor_actions = pad_and_concat(
                [
                    idx2onehot(np.array([updated_actions[i]]), self.action_dims[i])
                    for i in relevant_indices
                    if i != idx
                ]
            )
            ind_state = last_obs[idx]
            relevant_states = torch.from_numpy(relevant_states).float().to(self.device).unsqueeze(0)
            relevant_n_states = torch.from_numpy(relevant_n_states).float().to(self.device).unsqueeze(0)
            relevant_actions = torch.from_numpy(relevant_actions).float().to(self.device).unsqueeze(0)
            neighbor_actions = torch.from_numpy(neighbor_actions).float().to(self.device).unsqueeze(0)
            ind_state = torch.from_numpy(ind_state).float().to(self.device).unsqueeze(0)

            pred_next_state = self.forward_models[idx].model(
                relevant_states, relevant_actions
            ).unsqueeze(0)
            grounded_action, uncertainty = self.inverse_models[idx].model(
                ind_state,
                relevant_n_states,
                neighbor_actions,
                pred_next_state,
            )

            if self.uncertainty_setting:
                stats["agent_uncertainty_sums"][idx] += uncertainty.item()
                if uncertainty >= self.avg_agent_uncertainties[idx]:
                    continue
            elif self.prob_grounding not in (None, 0) and random.random() >= self.prob_grounding:
                continue

            updated_actions[idx] = torch.argmax(grounded_action, dim=1).cpu().item()
            grounded_actions[idx] = updated_actions[idx]
            stats["grounded_action_count"] += 1
            stats["ga_by_agent"][idx] += 1
        return updated_actions, grounded_actions

    def prepare_forward_data(self, test_size=0.2, random_seed=42):
        data = self._load_pickle(self._dataset_file(forward=True, train=True))
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
            self._dataset_prefix(forward=True, train=True),
            self._dataset_prefix(forward=True, train=False),
            test_size=test_size,
            random_seed=random_seed,
        )

    def prepare_inverse_data(self, test_size=0.2, random_seed=42):
        data = self._load_pickle(self._dataset_file(forward=False, train=True))
        agent_records = {agent_idx: [] for agent_idx in range(len(self.agents_sim))}

        for record in data:
            agent_idx, states, n_states, n_actions, next_states, ind_action = record
            action_dim = self.action_dims[agent_idx]
            one_hot_actions = torch.tensor(
                np.concatenate(
                    [idx2onehot(np.array([action]), action_dim) for action in n_actions],
                    axis=0,
                ),
                dtype=torch.float32,
            )
            agent_records[agent_idx].append(
                (
                    torch.tensor(states, dtype=torch.float32),
                    torch.tensor(n_states, dtype=torch.float32),
                    one_hot_actions,
                    torch.tensor(next_states, dtype=torch.float32),
                    torch.tensor(ind_action, dtype=torch.long),
                )
            )

        self._split_agent_records(
            agent_records,
            self._dataset_prefix(forward=False, train=True),
            self._dataset_prefix(forward=False, train=False),
            test_size=test_size,
            random_seed=random_seed,
        )

    def train_transition_models(self):
        self.prepare_forward_data()
        self.prepare_inverse_data()
        for idx in range(len(self.agents_sim)):
            self.forward_models[idx].train(100, "forward", idx, 5000, "jlgat")
            self.inverse_models[idx].train(100, "inverse", idx, 5000, "jlgat")

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
