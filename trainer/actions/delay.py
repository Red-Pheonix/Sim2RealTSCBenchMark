"""Execution-delay action transform.

Encapsulates the constant action-delay queue that used to live inline in the
actions trainer (``ConstantActionDelay`` + ``initialize/enqueue/resolve/advance``).
An action chosen at a decision is enqueued and only released ``delay`` env steps
later; until then the last released action is held. ``delay=0`` is the identity
(the proposed action executes immediately and holds for the action_interval).
"""

from collections import deque

import numpy as np

from .base import ActionTransform


class ActionDelay(ActionTransform):
    def __init__(self, delay=0):
        self.delay = int(delay)
        self.env_step_idx = 0
        self.queues = []
        self.last_actions = None

    def sample(self, num_agents):
        return np.full(num_agents, self.delay, dtype=int)

    def reset(self, agents, init_phases):
        self.env_step_idx = 0
        self.queues = [deque() for _ in agents]
        self.last_actions = np.array(init_phases, copy=True)

    def begin_interval(self, proposed_actions):
        delays = self.sample(len(proposed_actions))
        for idx, action in enumerate(np.asarray(proposed_actions)):
            sampled_delay = int(delays[idx])
            queue = self.queues[idx]
            # Chain the release off the last queued action so ordering is kept.
            if queue:
                release = queue[-1][0] + sampled_delay
            else:
                release = self.env_step_idx + sampled_delay
            queue.append((release, action))

    def resolve_step(self, executed_actions):
        out = np.array(self.last_actions, copy=True)
        for idx, queue in enumerate(self.queues):
            while queue and queue[0][0] <= self.env_step_idx:
                _, out[idx] = queue.popleft()
        self.last_actions = np.array(out, copy=True)
        self.env_step_idx += 1
        return out
