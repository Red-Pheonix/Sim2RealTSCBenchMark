import numpy as np


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
    def __init__(self, logger, device, world_sim, agents_sim, world_real, agents_real):
        self.logger = logger
        self.device = device
        self.world_sim = world_sim
        self.agents_sim = agents_sim
        self.world_real = world_real
        self.agents_real = agents_real

        self.forward_models = []
        self.inverse_models = []
        self.action_dims = []

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

    def train_transition_models(self):
        return

    @staticmethod
    def calc_dist(p1, p2):
        return np.sqrt((p1["x"] - p2["x"]) ** 2 + (p1["y"] - p2["y"]) ** 2)
