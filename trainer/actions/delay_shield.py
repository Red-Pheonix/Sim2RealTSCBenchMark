"""Delay-aware shield (action gap: delay + phase-transition).

A plain action shield masks the agent to the legal set *at decision time*. Under an
actuation/comms delay the action chosen at ``t`` does not reach the controller until
``t+delta``, by which point the controller's clock has advanced and that legal set is
stale -- so the decision-time mask leaks violations at roughly the naive rate (see the
delay-20 matrix: shield's violation_rate jumps from 0 at delay 0 to ~naive at delay 20).

The delay-aware shield moves the masking to EXECUTION time. The agent cannot re-decide
at ``t+delta`` (it only ever saw ``obs(t)``), but its *preference ordering* over actions
-- the raw Q-vector it produced at ``t`` -- can be carried through the same delay queue
as the action. When the action lands, we know the controller's true legal set, so we
pick the highest-Q action among the legal ones and commit it for the interval. The
executed action is legal by construction (0 violations) while still honoring the agent's
committed preference; the only "illegal-ish" moves left are force-offs (past max-green
the controller must leave, so we force to the agent's preferred legal successor).

Owns BOTH the delay queue and the legality model, so it replaces the
``[ActionDelay, PhaseTransition]`` pair on the side that uses it. Subclasses
PhaseTransition to reuse its table/legality (`_legal_set`, `_forceoff_target`) and the
safety counters the trainer reads.
"""

from collections import deque

import numpy as np
import torch

from .phase_transition import PhaseTransition


class DelayAwareShield(PhaseTransition):
    def __init__(self, agents, action_interval, json_path, delay):
        super().__init__(agents, action_interval, json_path, mode="enforce")
        self.delay = int(delay)
        self.env_step_idx = 0
        self.queues = [deque() for _ in self.agents]
        self.last_q = [None] * len(self.agents)

    # Decision time: the agent is unaware of the constraint (it picks by its raw Q and
    # we re-rank at execution), so expose no mask -- all actions allowed.
    def valid_mask(self, idx, phase):
        return torch.ones(1, self.agents[idx].action_space.n, dtype=torch.bool)

    def reset(self, agents, init_phases):
        super().reset(agents, init_phases)
        self.env_step_idx = 0
        self.queues = [deque() for _ in self.agents]
        self.last_q = [None] * len(self.agents)

    def begin_interval(self, proposed_actions):
        # Enqueue each agent's committed preference (its decision-time Q-vector) with
        # the same delay the action would have. Chain the release off the last queued
        # item so ordering holds, mirroring ActionDelay.
        for idx, ag in enumerate(self.agents):
            q = ag.last_q_values
            q = None if q is None else np.asarray(q, dtype=float).reshape(-1)
            if self.queues[idx]:
                release = self.queues[idx][-1][0] + self.delay
            else:
                release = self.env_step_idx + self.delay
            self.queues[idx].append((release, q))
        self.hold.begin()

    def resolve_step(self, executed_actions):
        # Release any preferences whose delay has elapsed (latest wins, held until the
        # next release), exactly like the action-delay queue.
        for idx, q in enumerate(self.queues):
            while q and q[0][0] <= self.env_step_idx:
                _, self.last_q[idx] = q.popleft()
        if self.hold.take_boundary():
            for idx, ag in enumerate(self.agents):
                cur = int(np.asarray(ag.get_phase()).reshape(-1)[0])
                n = self.allowed_mask[idx].shape[0]
                if not (0 <= cur < n):
                    continue  # transitional read; keep the held phase
                legal, forced, _ = self._legal_set(idx, cur)
                qv = self.last_q[idx]
                if qv is None or qv.shape[0] != n:
                    # warmup (no preference has landed yet): hold, or force-off if the
                    # hold itself is past max-green
                    target = cur if bool(legal[cur]) else self._forceoff_target(idx, cur)
                else:
                    # highest-Q action among the EXECUTION-time legal set
                    masked = np.where(legal.numpy(), qv, -np.inf)
                    target = int(np.argmax(masked))
                # executed action is legal by construction -> never an avoidable
                # violation; only force-offs (past max-green) are tallied.
                self.safety.record(idx, violated=False, forced=forced)
                self.hold.commit(idx, target)
        self.env_step_idx += 1
        return self.hold.apply(executed_actions)
