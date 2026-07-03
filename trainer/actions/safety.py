"""Safety accounting for phase-transition transforms.

A phase-transition transform decides legality (what's a violation vs a controller
force-off); this object just KEEPS THE BOOKS. Splitting it out means the legality
logic and the bookkeeping compose independently, and the two transforms that share it
(PhaseTransition and its DelayAwareShield subclass) record through one code path
instead of each poking the same four counters by hand.

Two distinct quantities, tracked apart (see PhaseTransition for the full rationale):
  *violations* -- the agent's own AVOIDABLE illegal switches (a legal action existed,
    it picked an illegal one). The naive agent racks these up; a shielded agent has
    zero -- that rate is what justifies the shield.
  *force_offs* -- the controller forcing a leave at max-green, when NO legal action
    exists. The controller's doing, not an avoidable agent violation.
"""

import numpy as np


def collect_violations(action_transforms, n_agents):
    """Per-agent illegal-request flags (0/1 float) for the current decision, OR-ed
    across the side's phase-transition transforms' ``last_violation``. Zeros when the
    side has no phase-transition transform. This reads the action side's own published
    result (it does not re-detect legality), so reward-side mitigations can consume it
    without the trainer having to assemble it."""
    violated = np.zeros(n_agents, dtype=float)
    for transform in action_transforms or []:
        last_violation = getattr(transform, "last_violation", None)
        if last_violation is not None:
            violated = np.maximum(violated, np.asarray(last_violation, dtype=float))
    return violated


class SafetyStats:
    """Per-episode safety counters for one phase-transition transform.

    ``last_violation`` is the per-agent flag from the most recent decision (read by
    soft_shield to penalize the exact agent-decisions that violated). The cumulative
    counters are summed + reset by the trainer at each log point via collect()/reset().
    """

    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.violations = 0
        self.force_offs = 0
        self.decisions = 0
        self.last_violation = np.zeros(n_agents, dtype=bool)
        # Per-agent cumulative counters (for per-intersection BRF logging on PT runs).
        # The scalar totals above are just these summed; kept separate to avoid
        # touching the hot aggregate path.
        self.agent_violations = np.zeros(n_agents, dtype=np.int64)
        self.agent_force_offs = np.zeros(n_agents, dtype=np.int64)
        self.agent_decisions = np.zeros(n_agents, dtype=np.int64)

    def record(self, idx, violated, forced):
        """Tally one agent-decision. ``violated`` and ``forced`` are mutually
        exclusive by construction (a force-off is not an avoidable violation), so a
        decision increments at most one of the two counters."""
        self.decisions += 1
        self.last_violation[idx] = violated
        self.violations += int(violated)
        self.force_offs += int(forced)
        self.agent_decisions[idx] += 1
        self.agent_violations[idx] += int(violated)
        self.agent_force_offs[idx] += int(forced)

    def collect(self):
        """(violations, force_offs, decisions) accumulated since the last reset."""
        return self.violations, self.force_offs, self.decisions

    def collect_per_agent(self):
        """Per-agent (violations, force_offs, decisions) arrays since the last reset."""
        return (
            self.agent_violations.copy(),
            self.agent_force_offs.copy(),
            self.agent_decisions.copy(),
        )

    def reset(self):
        self.violations = 0
        self.force_offs = 0
        self.decisions = 0
        self.agent_violations[:] = 0
        self.agent_force_offs[:] = 0
        self.agent_decisions[:] = 0
