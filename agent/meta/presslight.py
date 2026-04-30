import random

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from agent import utils
from agent.presslight import PressLightAgent
from common.registry import Registry


@Registry.register_sim2real_model("presslight")
class MetaPressLightAgent(PressLightAgent):
    """
    Internal PressLight variant for observation-side MAML training.

    The base learner remains identical to standard PressLight; this subclass only
    adds meta-learning helpers used by the MAML observations trainer.
    """

    def __init__(self, world, rank):
        super().__init__(world, rank)

        try:
            import learn2learn as l2l
        except ImportError as exc:
            raise ImportError(
                "MetaPressLightAgent requires learn2learn in the active environment."
            ) from exc

        self.l2l = l2l
        sim2real_setting = self.get_sim2real_setting()
        self.maml_config = sim2real_setting.get("maml", {})
        self.inner_lr = self.maml_config.get("inner_lr", 1e-3)
        self.meta_lr = self.maml_config.get("meta_lr", self.learning_rate)
        self.inner_steps = self.maml_config.get("inner_steps", 1)
        self.support_batch_size = self.maml_config.get(
            "support_batch_size", self.batch_size
        )
        self.query_batch_size = self.maml_config.get("query_batch_size", self.batch_size)
        # self.first_order = self.maml_config.get("first_order", True)
        self.first_order = False
        self.meta_grad_clip = self.maml_config.get("meta_grad_clip", self.grad_clip)

        self.maml = self.l2l.algorithms.MAML(
            self.model,
            lr=self.inner_lr,
            first_order=self.first_order,
            allow_unused=True,
            allow_nograd=True,
        )
        self.meta_optimizer = torch.optim.Adam(self.maml.parameters(), lr=self.meta_lr)
        self.meta_learner = self.maml.clone()

    def get_sim2real_setting(self):
        setting = Registry.mapping.get("sim2real_mapping", {}).get("setting")
        if setting and hasattr(setting, "param"):
            return setting.param
        return {}

    def clear_replay_buffer(self):
        self.replay_buffer.clear()

    def sync_base_from(self, source_agent):
        self.model.load_state_dict(source_agent.model.state_dict())
        self.target_model.load_state_dict(source_agent.target_model.state_dict())
        self.epsilon = source_agent.epsilon

    def sample_replay_batch(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return None
        return random.sample(self.replay_buffer, batch_size)

    def compute_training_loss(self, batch_size, model):
        # self.train()
        self.meta_learner = self.model
        samples = self.sample_replay_batch(batch_size)
        if samples is None:
            return None

        batch_t, batch_tp, rewards, actions = self._batchwise(samples)

        out = self.target_model(batch_tp, train=False)
        target = rewards + self.gamma * torch.max(out, dim=1)[0]
        target_f = model(batch_t, train=False)
        for idx, action in enumerate(actions):
            target_f[idx][action] = target[idx]
        return self.criterion(model(batch_t, train=True), target_f)

    def compute_support_loss(self):
        learner = self.maml.clone()
        self.meta_learner = learner
        latest_loss = None

        for _ in range(self.inner_steps):
            support_loss = self.compute_training_loss(
                self.support_batch_size,
                learner,
            )
            if support_loss is None:
                break
            learner.adapt(support_loss)
            latest_loss = support_loss

        return latest_loss

    def compute_query_loss(self):
        learner = self.meta_learner
        return self.compute_training_loss(
            self.query_batch_size,
            learner,
        )

    def zero_meta_grad(self):
        self.meta_optimizer.zero_grad()

    def apply_meta_update(self):
        clip_grad_norm_(self.maml.parameters(), self.meta_grad_clip)
        self.meta_optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_meta_action(self, ob, phase, test=False):
        if not test:
            if np.random.rand() <= self.epsilon:
                return self.sample()
        if self.phase:
            if self.one_hot:
                feature = np.concatenate(
                    [ob, utils.idx2onehot(phase, self.action_space.n)], axis=1
                )
            else:
                feature = np.concatenate([ob, phase], axis=1)
        else:
            feature = ob
        observation = torch.tensor(feature, dtype=torch.float32)
        actions = self.meta_learner(observation, train=False)
        actions = actions.clone().detach().numpy()
        return np.argmax(actions, axis=1)
