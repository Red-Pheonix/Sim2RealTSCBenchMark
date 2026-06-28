"""Reward-shaping transforms applied to the TRAINING signal only.

These mirror the action-transform pipeline (delay / phase_transition / delay_shield):
the trainer holds a per-side list and folds the training reward through it, while the
reported metric keeps the RAW reward. Action transforms change WHAT the agent does;
reward transforms change WHAT IT LEARNS FROM.

A reward transform maps ``(rewards, action_transforms) -> rewards``. It is handed the
action pipeline as the decision context and reads whatever published signal it needs
from it (e.g. the violation flags) -- it does NOT re-detect anything (legality is the
phase-transition transform's job; the delay methods own the timing). The trainer only
knows this protocol, not what any particular transform pulls from the context.
"""

import numpy as np

from .safety import collect_violations


class ViolationPenalty:
    """soft_shield: ``rewards -> rewards - penalty * violations``.

    Reads the per-agent illegal-request flags the action pipeline published this
    decision (phase_transition's ``last_violation``, via ``collect_violations``) and
    subtracts a flat penalty for each -- a soft constraint (penalize the choice) vs the
    shield's hard mask (prevent it).
    """

    def __init__(self, penalty):
        self.penalty = float(penalty)

    def __call__(self, rewards, action_transforms):
        rewards = np.asarray(rewards, dtype=float)
        violations = collect_violations(action_transforms, len(rewards))
        return rewards - self.penalty * violations
