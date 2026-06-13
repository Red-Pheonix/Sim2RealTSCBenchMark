"""PRLight-style one-shot, neighbor-aware prediction for the action gap
(Han, Zhao, Zhang & Wang, KDD 2023 -- "Mitigating Action Hysteresis in Traffic
Signal Control with Traffic Predictive Reinforcement Learning").

PRLight mitigates the countdown delay t_cd by predicting the traffic state t_cd
ahead with an Online Traffic Prediction Model (OTPM) over a local-view graph
(ego + surrounding intersections) and letting the control policy act on the
predicted state. Like Delayed-Q this is a wrapper: the base RL agents stay plain
and are simply handed the predicted state at action-selection time.

Differences from Delayed-Q (the point of the method):
- ONE-SHOT prediction: a single forward pass maps (state at t, in-flight
  actions) -> s_{t+m}; no m-step autoregressive rollout, so no compounding
  error. The flip side: m is baked into the input width, so changing the
  assumed delay requires retraining (Delayed-Q just rolls more/fewer steps).
- NEIGHBOR-AWARE: the predictor also consumes neighboring intersections' raw
  observations and current phases (PRLight's local-view graph), built with the
  JL-GAT methodology (probing-radius topology + pad-and-concat + MLP).

Simplifications vs the paper (user-directed / benchmark adaptation):
- lane-level digraph + fast attention + GCL  ->  intersection-level neighbors,
  pad_and_concat + plain MLP (JL-GAT style);
- clip-PPO on predicted state  ->  existing base agent's get_action on the
  predicted state (the paper's Fig. 7 ablation shows plain DQN works too);
- federated globally-averaged predictor  ->  per-intersection predictors;
- their phase-lock countdown admits a single in-flight switch, encoded in
  dynamic edge weights; our delay model keeps a rolling queue of m in-flight
  actions, so the ego queue is fed to the predictor as its generalization.
- Neighbor PENDING actions are excluded by default, mirroring the paper's
  argument (sec 3.3) that other intersections' control can be ignored within
  the short prediction horizon; `include_neighbor_actions=True` ablates this.

Training data: the paper trains OTPM online on sample pairs t_cd apart
(Alg. 1 line 12). We do the analog: a sliding window of per-decision records
yields (state_t + executed actions over [t, t+m)) -> state_{t+m} pairs, stored
in a small FIFO buffer. The base agents' one-step replay cannot supply these
m-apart pairs, hence the model's own buffer.
"""

import json
import os
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.optim as optim

from common.registry import Registry
from agent import utils


def calc_dist(p1, p2):
    """Euclidean distance between two intersection points {"x":.., "y":..}."""
    return np.sqrt((p1["x"] - p2["x"]) ** 2 + (p1["y"] - p2["y"]) ** 2)


def pad_and_concat(arrays, pad_value=0):
    """Right-pad each (1, w) array to the max width, then concat along axis 0
    -> (n, max_w). Lets neighbors with different obs lengths stack together."""
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


class OneShotPredictionModel(nn.Module):
    """MLP mapping the full decision-time feature vector straight to the ego
    observation m decisions ahead (PRLight Eq. 5 analog, OTPM)."""

    def __init__(self, input_dim, output_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x, train=True):
        if train:
            return self.net(x)
        with torch.no_grad():
            return self.net(x)


class MStepPairCollector:
    """Turns a stream of per-decision records into training pairs m decisions
    apart -- the data PRLight's one-shot predictor needs.

    The paper trains OTPM on sample pairs t_cd apart (Alg. 1 line 12). The base
    agents' one-step replay can't supply m-apart pairs, so we assemble them from
    a sliding window of the last ``m + 1`` decisions: the oldest decision (t)
    paired with the newest (t + m) gives one record, carrying the executed
    actions across the whole window. Records are feature-agnostic -- the model
    turns them into predictor inputs at train time -- so this object only knows
    about decisions, not topology.

    A "decision" is one full per-intersection snapshot. Because the trainer
    reports intersections one at a time (via ``push``), we stage them in
    ``_pending`` and only finalize a decision once all ``n`` have arrived.
    """

    def __init__(self, n, m, buffer_cap):
        self.n = int(n)
        self.m = int(m)
        # finished training records: (obs_t, phases_t, actions, obs_tm), where
        # each element is a per-intersection list (actions[i] is i's executed
        # actions over [t, t+m)). Capped FIFO (paper: buffer length 1000).
        self.records = deque(maxlen=int(buffer_cap))
        # rolling window of the last m+1 finalized decisions
        self.window = deque(maxlen=self.m + 1)
        # current decision being assembled, keyed by intersection idx
        self._pending = {}

    def push(self, idx, last_ob, last_phase, executed_action):
        """Add intersection ``idx``'s slice of the current decision. When all n
        have arrived the decision is appended to the window, and once the window
        spans m+1 decisions one training record is emitted."""
        self._pending[idx] = (
            np.asarray(last_ob, dtype=np.float32).reshape(1, -1),
            int(np.asarray(last_phase).reshape(-1)[0]),
            int(np.asarray(executed_action).flatten()[0]),
        )
        if len(self._pending) < self.n:
            return
        decision = (
            [self._pending[i][0] for i in range(self.n)],
            [self._pending[i][1] for i in range(self.n)],
            [self._pending[i][2] for i in range(self.n)],
        )
        self._pending = {}
        self.window.append(decision)
        if len(self.window) == self.m + 1:
            obs_t, phases_t, _ = self.window[0]
            obs_tm = self.window[-1][0]
            # executed actions over [t, t+m) per intersection -- equals the
            # in-flight queue at t under a constant delay of m decisions.
            actions = [
                [self.window[k][2][i] for k in range(self.m)] for i in range(self.n)
            ]
            self.records.append((obs_t, phases_t, actions, obs_tm))

    def sample(self, batch_size):
        return random.sample(self.records, batch_size)

    def reset(self):
        """Drop any partial cross-episode window (kept records survive)."""
        self.window.clear()
        self._pending = {}

    def __len__(self):
        return len(self.records)


@Registry.register_sim2real_model("prlight")
class PRLightModel:
    """PRLight mitigation over a set of (per-intersection) base agents.

    Holds, per intersection: a one-shot prediction model + optimizer, the
    pending in-flight action queue, and a FIFO buffer of m-apart training
    pairs. Q-replay is left to the base agent unchanged.
    """

    def __init__(
        self,
        world,
        agents,
        m,
        lr=1e-3,
        train_iters=1,
        probing_radius=600.0,
        include_neighbor_actions=False,
        buffer_cap=1000,
        hidden=128,
        logger=None,
    ):
        self.m = int(m)
        self.train_iters = int(train_iters)
        self.include_neighbor_actions = bool(include_neighbor_actions)
        self.criterion = nn.MSELoss()
        self.predictor_trained = False
        self.last_pred_loss = None
        self.logger = logger

        self.n = len(agents)
        # One-hot width shared by phases and actions across all intersections
        # (JL-GAT uses the max action dim the same way).
        self.action_len = max(ag.action_space.n for ag in agents)
        self.raw_lens = [ag.ob_generator.ob_length for ag in agents]

        # pending in-flight actions a_{t-m}..a_{t-1} per intersection (length m)
        self.queues = [deque(maxlen=max(self.m, 1)) for _ in range(self.n)]

        self.neighbour_infos = self._build_topology(world, probing_radius)
        # Per-ego pad width for neighbor observations (pad_and_concat pads per
        # call; fixing the width per ego keeps the input dim constant).
        self.neighbor_pad = [
            max((self.raw_lens[j] for j in self.neighbour_infos[i]), default=0)
            for i in range(self.n)
        ]

        self.models = []
        self.optimizers = []
        for i in range(self.n):
            n_neigh = len(self.neighbour_infos[i])
            # [ego raw | ego phase | ego m in-flight actions |
            #  per neighbor: padded raw + current phase (+ m actions if flag)]
            in_dim = (
                self.raw_lens[i]
                + self.action_len
                + self.m * self.action_len
                + n_neigh * (self.neighbor_pad[i] + self.action_len)
            )
            if self.include_neighbor_actions:
                in_dim += n_neigh * self.m * self.action_len
            model = OneShotPredictionModel(in_dim, self.raw_lens[i], hidden=hidden)
            self.models.append(model)
            self.optimizers.append(optim.Adam(model.parameters(), lr=lr))

        # Assembles the m-apart (state_t, executed actions [t,t+m)) -> state_{t+m}
        # training pairs the one-shot predictor needs (see MStepPairCollector).
        self.collector = MStepPairCollector(self.n, self.m, buffer_cap)

        # Global decision-time snapshot (set via the trainer's
        # on_decision_start hook) so select_action can see neighbors.
        self._cached_obs = None
        self._cached_phases = None

    # ------------------------------------------------------------------
    # neighbor topology (JL-GAT methodology)
    # ------------------------------------------------------------------

    def _build_topology(self, world, probing_radius):
        """Neighbors = intersections within `probing_radius` (Euclidean, from
        roadnet coordinates), excluding the ego; optional per-network override
        JSON, same convention as JL-GAT. Zero neighbors is allowed (single-
        intersection networks degenerate to an ego-only one-shot predictor)."""
        inter_positions = {
            world.id2idx[inter["id"]]: inter["point"]
            for inter in world.roadnet["intersections"]
            if inter["id"] in world.intersection_ids
        }
        inter_positions = dict(sorted(inter_positions.items()))

        neighbour_infos = {}
        for inter_id, point1 in inter_positions.items():
            neighbour_infos[inter_id] = sorted(
                i
                for i, point2 in inter_positions.items()
                if i != inter_id
                and calc_dist(point1, point2) < probing_radius
            )

        world_param = Registry.mapping["world_mapping"]["setting"].param
        net = Registry.mapping["command_mapping"]["setting"].param["network"]
        roadnet_path = Path(world_param["dir"]) / world_param["roadnetFile"]
        neighbors_file = roadnet_path.parent / f"{net}_neighbour_overrides.json"
        if os.path.exists(neighbors_file):
            with open(neighbors_file, "r") as file_obj:
                neighbor_overrides = json.load(file_obj)
            for inter, inter_neighbors in neighbor_overrides.items():
                neighbour_infos[world.id2idx[inter]] = sorted(
                    world.id2idx[neighbour]
                    for neighbour in inter_neighbors
                    if world.id2idx[neighbour] != world.id2idx[inter]
                )

        if self.logger is not None:
            self.logger.info(
                "PRLight neighbor topology (radius=%s): %s",
                probing_radius,
                {i: neighbour_infos[i] for i in sorted(neighbour_infos)},
            )
        return neighbour_infos

    # ------------------------------------------------------------------
    # episode lifecycle
    # ------------------------------------------------------------------

    def reset_all(self, init_phases):
        """Seed each pending queue with ``m`` copies of the current phase and
        drop any partial cross-episode prediction window."""
        if self.m == 0:
            return
        for idx in range(self.n):
            init = int(np.asarray(init_phases[idx]).flatten()[0])
            self.queues[idx] = deque([init] * self.m, maxlen=self.m)
        self.collector.reset()
        self._cached_obs = None
        self._cached_phases = None

    def cache_decision_state(self, obs, phases):
        """Store this decision's full per-intersection snapshot (from the
        trainer's on_decision_start hook) for neighbor feature lookup."""
        self._cached_obs = [
            np.asarray(obs[i], dtype=np.float32).reshape(1, -1) for i in range(self.n)
        ]
        self._cached_phases = [
            int(np.asarray(phases[i]).reshape(-1)[0]) for i in range(self.n)
        ]

    # ------------------------------------------------------------------
    # feature assembly
    # ------------------------------------------------------------------

    def _onehot(self, idx_value):
        return utils.idx2onehot(
            np.asarray(idx_value).reshape(-1).astype(int), self.action_len
        ).astype(np.float32)

    def _features(self, i, obs_all, phases_all, ego_actions, neighbor_actions=None):
        """Build the predictor input row (1, in_dim) for ego intersection i.

        obs_all/phases_all: per-intersection lists (decision-time snapshot).
        ego_actions: the m actions executing during [t, t+m) -- the in-flight
        queue at deploy time, the recorded executed actions at training time.
        neighbor_actions: same per neighbor (only with include_neighbor_actions).
        """
        parts = [np.asarray(obs_all[i], dtype=np.float32).reshape(1, -1)]
        parts.append(self._onehot(phases_all[i]))
        for a in ego_actions:
            parts.append(self._onehot(a))
        neighbors = self.neighbour_infos[i]
        if neighbors:
            padded = pad_and_concat(
                [
                    np.asarray(obs_all[j], dtype=np.float32).reshape(1, -1)
                    for j in neighbors
                ]
                # extra zero row fixes pad width to neighbor_pad[i] even when
                # this batch's widest neighbor is narrower
                + [np.zeros((1, self.neighbor_pad[i]), dtype=np.float32)]
            )[:-1]
            for k, j in enumerate(neighbors):
                parts.append(padded[k : k + 1])
                parts.append(self._onehot(phases_all[j]))
                if self.include_neighbor_actions:
                    for a in neighbor_actions[j]:
                        parts.append(self._onehot(a))
        return np.concatenate(parts, axis=1)

    # ------------------------------------------------------------------
    # action selection on the predicted m-steps-ahead state
    # ------------------------------------------------------------------

    def select_action(self, agent, idx, ob, phase, test):
        if self.m == 0:
            return agent.get_action(ob, phase, test=test)

        # One forward pass to s_{t+m} once trained; warmup falls back to the
        # current observation. The queue advances either way so it stays a
        # faithful record of in-flight actions.
        if self.predictor_trained and self._cached_obs is not None:
            neighbor_actions = (
                {j: list(self.queues[j]) for j in self.neighbour_infos[idx]}
                if self.include_neighbor_actions
                else None
            )
            inp = self._features(
                idx,
                self._cached_obs,
                self._cached_phases,
                list(self.queues[idx]),
                neighbor_actions,
            )
            nxt = self.models[idx](torch.tensor(inp, dtype=torch.float32), train=False)
            # Lane counts are non-negative; clamp keeps the prediction physical
            # and in-distribution for the base agent's Q-net.
            ob_in = np.clip(nxt.numpy(), 0.0, None).astype(np.float32)
            phase_in = np.array([int(self.queues[idx][-1])], dtype=np.int8)
        else:
            ob_in, phase_in = ob, phase

        action = agent.get_action(ob_in, phase_in, test=test)
        self.queues[idx].append(int(np.asarray(action).flatten()[0]))
        return action

    # ------------------------------------------------------------------
    # m-apart training pairs (fed from the trainer's store_transition hook)
    # ------------------------------------------------------------------

    def record_step(self, idx, last_ob, last_phase, executed_action):
        """Feed one intersection's slice of this decision to the pair collector
        (called from the trainer's store_transition hook)."""
        if self.m == 0:
            return
        self.collector.push(idx, last_ob, last_phase, executed_action)

    # ------------------------------------------------------------------
    # predictor training (online, in parallel with policy -- paper Alg. 1)
    # ------------------------------------------------------------------

    def train_predictor(self, agents):
        if self.m == 0:
            return
        for i, ag in enumerate(agents):
            if len(self.collector) < ag.batch_size:
                continue
            for _ in range(self.train_iters):
                # Build this intersection's predictor inputs from the sampled
                # m-apart records (features are model-specific, so they are
                # assembled here rather than stored in the collector).
                batch = self.collector.sample(ag.batch_size)
                xs, ys = [], []
                for obs_t, phases_t, actions, obs_tm in batch:
                    neighbor_actions = (
                        {j: actions[j] for j in self.neighbour_infos[i]}
                        if self.include_neighbor_actions
                        else None
                    )
                    xs.append(
                        self._features(i, obs_t, phases_t, actions[i], neighbor_actions)[0]
                    )
                    ys.append(obs_tm[i][0])
                x = torch.tensor(np.stack(xs), dtype=torch.float32)
                y = torch.tensor(np.stack(ys), dtype=torch.float32)
                pred = self.models[i](x, train=True)
                loss = self.criterion(pred, y)
                self.optimizers[i].zero_grad()
                loss.backward()
                self.optimizers[i].step()
                self.last_pred_loss = float(loss.detach().numpy())
            self.predictor_trained = True
