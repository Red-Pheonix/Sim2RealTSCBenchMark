"""Per-decision commit-and-hold for execution-time action transforms.

``resolve_step`` runs on EVERY env step, but the controller commits a phase once per
decision -- at the release boundary -- and holds it for the rest of the interval, so the
signal gets one consistent command (per-step rewrites issued conflicting mid-yellow
commands and crashed SUMO). This object owns that mechanism:

  * the boundary latch -- ``begin()`` marks a new decision; ``take_boundary()`` is True
    exactly once, on the first ``resolve_step`` after it, where the owner decides what
    to commit;
  * the held action -- ``commit(idx, action)`` stores the committed phase and
    ``apply()`` writes it over every step's output until the next decision.

It deliberately does NOT decide WHAT to commit -- that is the transform's job (legality
for PhaseTransition, highest-Q-legal for DelayAwareShield). This is the execution-time
plumbing that makes a per-step resolve_step behave as one-decision-per-interval, which
is what validating at execution (post-delay) requires. With delay 0 the boundary is the
decision step itself, so it is a no-op passthrough.
"""

import numpy as np


class IntervalHold:
    """Holds one committed action per agent across an action interval."""

    def __init__(self, n_agents):
        self.n_agents = n_agents
        self._boundary_pending = False
        self._held = np.zeros(n_agents, dtype=int)

    def begin(self):
        """Mark that a new decision has arrived (called from begin_interval). The next
        resolve_step is its release boundary."""
        self._boundary_pending = True

    def take_boundary(self):
        """True exactly once per decision -- on the first resolve_step after begin().
        Clears the latch so the rest of the interval's steps just hold."""
        if self._boundary_pending:
            self._boundary_pending = False
            return True
        return False

    def commit(self, idx, action):
        """Commit agent ``idx``'s phase for this interval (called at the boundary)."""
        self._held[idx] = int(action)

    def apply(self, executed_actions):
        """Return a copy of ``executed_actions`` with each agent's first entry
        overwritten by its committed phase -- the one consistent command per step."""
        out = np.array(executed_actions, copy=True)
        out.reshape(self.n_agents, -1)[:, 0] = self._held
        return out

    def reset(self):
        self._boundary_pending = False
        self._held = np.zeros(self.n_agents, dtype=int)
