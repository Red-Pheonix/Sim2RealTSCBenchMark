import os
import json
import random
from pathlib import Path
import numpy as np
import torch

from agent.utils import idx2onehot
from common.gat_utils import (
    NN_predictor,
    UNCERTAINTY_predictor,
)
from common.registry import Registry
from .base import BaseSim2RealTransitionModel, pad_and_concat
from .decentralized import DecentralizedSim2RealTransitionModel


@Registry.register_sim2real_model("jlgat")
class JLGATSim2RealTransitionModel(DecentralizedSim2RealTransitionModel):
    def __init__(self, logger, device, world_sim, agents_sim, world_real, agents_real):
        BaseSim2RealTransitionModel.__init__(
            self, logger, device, world_sim, agents_sim, world_real, agents_real
        )
        sim2real_params = Registry.mapping["sim2real_mapping"]["setting"].param
        self.uncertainty_setting = sim2real_params["uncertainty"]
        self.last_n_uncertainties = sim2real_params["last_n_uncertainties"]
        self.prob_grounding = sim2real_params.get("prob_grounding", 0)
        self.probing_radius = sim2real_params.get("probing_radius")
        self.net = Registry.mapping["trainer_mapping"]["setting"].param["network"]
        self.setting = Registry.mapping["command_mapping"]["setting"].param.get("real_setting")
        self.gat_path = os.path.join(
            Registry.mapping["logger_mapping"]["path"].path,
            "model",
            sim2real_params["gattype"],
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
            forward_model = NN_predictor(
                idx,
                self.logger,
                (agent.neighbors, ob_length),
                (agent.neighbors, action_length),
                self.agents_real[idx].ob_generator.ob_length,
                self.device,
                self.gat_path,
                "collected/ereal_train_full.pkl",
            )
            only_neighbors = [inter for inter in self.neighbour_infos[idx] if inter != idx]
            neighbor_ob_length = max(
                [
                    self.agents_real[inter].ob_generator.ob_length
                    for inter in only_neighbors
                ]
            )
            inverse_model = UNCERTAINTY_predictor(
                idx,
                self.logger,
                (1, self.agents_real[idx].ob_generator.ob_length),
                (agent.neighbors - 1, neighbor_ob_length),
                (agent.neighbors - 1, action_length),
                (1, self.agents_real[idx].ob_generator.ob_length),
                self.agents_real[idx].action_space.n,
                self.device,
                self.gat_path,
                "collected/esim_train_full.pkl",
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
            "collected/esim_train_full",
            "collected/esim_test_full",
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
