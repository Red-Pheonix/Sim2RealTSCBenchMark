import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def pad_and_concat(arrays, pad_value=0):
    max_width = max(a.shape[-1] for a in arrays)
    padded = [
        np.pad(
            a,
            pad_width=[(0, 0), (0, max_width - a.shape[-1])],
            mode="constant",
            constant_values=pad_value,
        )
        for a in arrays
    ]
    return np.concatenate(padded, axis=0)


class BaseSim2RealTransitionModel:
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
        self.logger = logger
        self.device = device
        self.world_sim = world_sim
        self.agents_sim = agents_sim
        self.world_real = world_real
        self.agents_real = agents_real

        self.forward_models = []
        self.inverse_models = []
        self.action_dims = []
        self.dataset_dir = Path(dataset_dir)

    def collect_sim_transition(self, last_obs, actions, obs):
        return [(last_obs, actions, obs)]

    def collect_real_transition(self, last_obs, actions, obs):
        return [(last_obs, actions, obs)]

    def init_policy_stats(self):
        return {
            "uncertainty_sum": 0.0,
            "agent_uncertainty_sums": [0.0 for _ in range(len(self.agents_sim))],
            "ga_by_agent": [0 for _ in range(len(self.agents_sim))],
            "grounded_action_count": 0,
        }

    def ground_actions(self, last_obs, actions, stats):
        return actions, [9 for _ in range(len(self.agents_sim))]

    def finalize_policy_stats(self, episode, stats):
        return

    def prepare_forward_data(self):
        return

    def prepare_inverse_data(self):
        return

    def train_transition_models(self):
        return
    
    def save_models(self, e):
        return

    def load_models(self):
        return

    @staticmethod
    def _load_pickle(file_path):
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    @staticmethod
    def _write_pickle(file_path, data):
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(data, file_obj)

    def _dataset_file(self, *, forward, train, full=False, agent_idx=None):
        prefix = "ereal" if forward else "esim"
        split = "train" if train else "test"
        name = f"{prefix}_{split}"
        if full:
            name += "_full"
        if agent_idx is not None:
            name += f"_agent_{agent_idx}"
        return (self.dataset_dir / f"{name}.pkl").as_posix()

    def _dataset_prefix(self, *, forward, train):
        prefix = "ereal" if forward else "esim"
        split = "train" if train else "test"
        return (self.dataset_dir / f"{prefix}_{split}_full").as_posix()

    def _split_joint_records(
        self, records, train_file, test_file, test_size=0.2, random_seed=42
    ):
        train_idx, test_idx = train_test_split(
            np.arange(len(records)), test_size=test_size, random_state=random_seed
        )
        train_data = [records[i] for i in train_idx]
        test_data = [records[i] for i in test_idx]
        self._write_pickle(train_file, train_data)
        self._write_pickle(test_file, test_data)

    def _split_agent_records(
        self, agent_records, train_prefix, test_prefix, test_size=0.2, random_seed=42
    ):
        for agent_idx, records in agent_records.items():
            train_idx, test_idx = train_test_split(
                np.arange(len(records)), test_size=test_size, random_state=random_seed
            )
            train_data = [records[i] for i in train_idx]
            test_data = [records[i] for i in test_idx]
            self._write_pickle(f"{train_prefix}_agent_{agent_idx}.pkl", train_data)
            self._write_pickle(f"{test_prefix}_agent_{agent_idx}.pkl", test_data)

    @staticmethod
    def calc_dist(p1, p2):
        return np.sqrt((p1["x"] - p2["x"]) ** 2 + (p1["y"] - p2["y"]) ** 2)
