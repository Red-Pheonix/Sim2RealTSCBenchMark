"""Delayed-Q action-gap mitigation (Derman, Dalal & Mannor, ICLR 2021 --
"Acting in Delayed Environments with Non-Stationary Markov Policies").

Under an execution delay of ``m`` decisions, an action chosen now does not take
effect until ``m`` decisions later, so acting on the current observation is wrong
-- by execution time the ``m`` in-flight actions will have moved the world on.
Delayed-Q instead predicts the state ``m`` steps ahead (rolling a learned
*prediction model* over the pending in-flight actions) and lets the base agent
choose greedily on that predicted state.

Replay shift -- IMPORTANT: the paper keeps a length-m buffer to shift the chosen
action a_t into the tuple (s_{t+m}, a_t, r_{t+m}, s_{t+m+1}), because its pipeline
logs the action at *decision* time. In this benchmark the environment's delay
queue already releases the delayed action and the trainer stores it as
``executed_action`` together with the *execution-time* state/reward -- i.e. the
baseline's ``remember(s_D, executed_action_D, r_D, s_{D+1})`` with
``executed_action_D = a_{D-m}`` is ALREADY the shifted tuple. So no manual shift
is needed: the Q-function is trained exactly as the baseline, and the only change
Delayed-Q makes is acting on the predicted future state. The prediction model is
trained on those same (s, executed_action, s') one-step transitions.

This is implemented as a method *model* (registered under
``sim2real_model_mapping``) that wraps a set of base RL agents WITHOUT modifying
them, driven by the trainer's ``select_action`` / ``train_agents`` /
``reset_episode_state`` hooks.

TSC shortcut: the state is (lane vehicle counts, current phase) and the action
index *is* the next phase, so the predicted future phase is exactly the most
recent in-flight action -- the prediction model only has to learn the lane-count
dynamics.
"""

import random
from collections import deque

import numpy as np
import torch
from torch import nn
import torch.optim as optim

from common.registry import Registry
from agent import utils


class PredictionModel(nn.Module):
    """Small MLP modelling the one-step dynamics, independent of the base agent's
    Q-net feature representation.

    Input  = concat(raw observation, one-hot phase, one-hot action).
    Output = next raw observation (lane counts).
    """

    def __init__(self, input_dim, output_dim, hidden=64):
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


@Registry.register_sim2real_model("delayed_q")
class DelayedQModel:
    """Delayed-Q mitigation over a set of (per-intersection) base agents.

    Holds, per intersection: a prediction model + optimizer and the pending
    in-flight action queue. Q-replay is left to the base agent unchanged.
    """

    def __init__(self, agents, m, lr=1e-3, train_iters=1, hidden=64):
        self.m = int(m)
        self.train_iters = int(train_iters)
        self.criterion = nn.MSELoss()
        self.predictor_trained = False
        self.last_pred_loss = None

        self.n = len(agents)
        # pending in-flight actions a_{t-m}..a_{t-1} per intersection (length m)
        self.queues = [deque(maxlen=max(self.m, 1)) for _ in range(self.n)]
        # diagnostic: predictions awaiting their realization m decisions later,
        # and an EMA of the m-step rollout error (predicted vs actual obs, MSE).
        self.pred_log = [deque() for _ in range(self.n)]
        self.rollout_err_ema = None

        self.models = []
        self.optimizers = []
        for ag in agents:
            raw_len = ag.ob_generator.ob_length
            n_actions = ag.action_space.n
            # Predictor representation is fixed and independent of the base agent's
            # Q-net feature choice: input = [raw obs | one-hot(phase) | one-hot(action)],
            # output = next raw obs. (The agent's phase/one_hot flags only affect what
            # its Q-net consumes, handled inside agent.get_action.)
            in_dim = raw_len + 2 * n_actions
            out_dim = raw_len
            model = PredictionModel(in_dim, out_dim, hidden=hidden)
            self.models.append(model)
            self.optimizers.append(optim.Adam(model.parameters(), lr=lr))

    @staticmethod
    def _pred_input(raw_ob, phase, action, n_actions):
        """Build the prediction model's input row(s): [raw obs | one-hot(phase) |
        one-hot(action)]. Works for a single step (B=1) or a training batch.

        raw_ob: (B, raw_len) float; phase, action: length-B int index arrays.
        """
        return np.concatenate(
            [
                np.asarray(raw_ob, dtype=np.float32),
                utils.idx2onehot(np.asarray(phase).reshape(-1).astype(int), n_actions),
                utils.idx2onehot(np.asarray(action).reshape(-1).astype(int), n_actions),
            ],
            axis=1,
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # episode lifecycle
    # ------------------------------------------------------------------

    def reset_all(self, init_phases):
        """Seed each pending queue with ``m`` copies of the current phase."""
        if self.m == 0:
            return
        for idx in range(self.n):
            init = int(np.asarray(init_phases[idx]).flatten()[0])
            self.queues[idx] = deque([init] * self.m, maxlen=self.m)
            self.pred_log[idx].clear()

    # ------------------------------------------------------------------
    # action selection on the predicted m-steps-ahead state
    # ------------------------------------------------------------------

    def select_action(self, agent, idx, ob, phase, test, valid_mask_fn=None):
        kw = {"valid_mask_fn": valid_mask_fn} if valid_mask_fn is not None else {}
        if self.m == 0:
            return agent.get_action(ob, phase, test=test, **kw)

        # Predict the state m steps ahead once the model has data; during warmup
        # fall back to the current observation. Either way the pending-action
        # queue advances below so it stays a faithful record of in-flight actions.
        if self.predictor_trained:
            ob_in, phase_in = self._predict_ahead(agent, idx, ob, phase)
            self._track_rollout_error(idx, ob, ob_in)
        else:
            ob_in, phase_in = ob, phase

        action = agent.get_action(ob_in, phase_in, test=test, **kw)
        self.queues[idx].append(int(np.asarray(action).flatten()[0]))
        return action

    def _track_rollout_error(self, idx, ob, new_pred):
        """Diagnostic only (does not affect behavior). The prediction made ``m``
        decisions ago targeted "now", so compare it to the just-observed obs and
        fold the MSE into an EMA; then log ``new_pred`` to be checked ``m``
        decisions from now."""
        plog = self.pred_log[idx]
        if len(plog) >= self.m:
            past_pred = plog.popleft()
            err = float(np.mean((past_pred - np.asarray(ob, dtype=np.float32)) ** 2))
            self.rollout_err_ema = (
                err
                if self.rollout_err_ema is None
                else 0.99 * self.rollout_err_ema + 0.01 * err
            )
        plog.append(np.array(new_pred, copy=True))

    def _predict_ahead(self, agent, idx, ob, phase):
        """Roll the prediction model m steps over the pending action queue and
        return the predicted (raw observation, phase) at t+m."""
        n_actions = agent.action_space.n
        model = self.models[idx]

        raw = np.asarray(ob, dtype=np.float32)  # (1, raw_len)
        cur_phase = int(np.asarray(phase).reshape(-1)[0])
        for a in self.queues[idx]:
            inp = self._pred_input(raw, [cur_phase], [int(a)], n_actions)
            nxt = model(torch.tensor(inp, dtype=torch.float32), train=False)
            # Lane counts are non-negative; clamp to keep the autoregressive
            # rollout physical and in-distribution for the base agent's Q-net.
            raw = np.clip(nxt.numpy(), 0.0, None).astype(np.float32)
            cur_phase = int(a)  # action index == next phase (TSC)

        phase_hat = np.array([int(self.queues[idx][-1])], dtype=np.int8)
        return raw, phase_hat

    # ------------------------------------------------------------------
    # prediction-model training (on the base agent's one-step replay)
    # ------------------------------------------------------------------

    def train_predictor(self, agents):
        if self.m == 0:
            return
        for idx, ag in enumerate(agents):
            if len(ag.replay_buffer) < ag.batch_size:
                continue
            n_actions = ag.action_space.n
            for _ in range(self.train_iters):
                samples = random.sample(ag.replay_buffer, ag.batch_size)
                # Replay tuple: (key, (last_obs, last_phase, action, reward, obs,
                # cur_phase)). `action` is the EXECUTED action that drove
                # last_obs -> obs, i.e. a true one-step dynamics sample.
                raw_t = np.concatenate([s[1][0] for s in samples])  # (B, raw)
                phase_t = np.array([int(np.asarray(s[1][1]).flatten()[0]) for s in samples])
                act = np.array([int(np.asarray(s[1][2]).flatten()[0]) for s in samples])
                raw_tp = np.concatenate([s[1][4] for s in samples]).astype(np.float32)

                inp = self._pred_input(raw_t, phase_t, act, n_actions)
                pred = self.models[idx](torch.tensor(inp, dtype=torch.float32), train=True)
                loss = self.criterion(pred, torch.tensor(raw_tp, dtype=torch.float32))
                self.optimizers[idx].zero_grad()
                loss.backward()
                self.optimizers[idx].step()
                self.last_pred_loss = float(loss.detach().numpy())
            self.predictor_trained = True
