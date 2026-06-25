"""Pluggable action-pipeline transforms (the sim2real action *gaps*).

Mirrors the observation-transformer pattern: the trainer builds two lists
(``sim_action_transforms`` / ``real_action_transforms``) and applies them in the
action pipeline. A transform is *what the (sim or real) world does to the agent's
action* -- e.g. apply an execution delay, or ignore/mask transitions that violate
the controller's phase-transition rules. The mitigation methods
(delayed_q / prlight / ...) are a separate wrapper layer that sits on top.

All hooks default to no-op / identity, so a concrete transform overrides only the
stage it cares about:

- ``valid_mask`` (decision time)  -- validity transforms restrict the action set;
  set ``provides_mask = True`` so the trainer collects the mask.
- ``begin_interval`` / ``resolve_step`` (execution)  -- delay-like transforms that
  decide which action actually reaches ``env.step`` over the action_interval.
- ``reset`` (episode start), ``train`` (optional).
"""


class ActionTransform:
    """Base class for action transforms. Override only the relevant hooks."""

    # True if this transform contributes a per-decision validity mask.
    provides_mask = False

    def reset(self, agents, init_phases):
        """Called at the start of every rollout (sim train, sim eval, real eval)."""

    def valid_mask(self, idx, phase):
        """Return a (1, n_actions) bool mask of valid actions for intersection
        ``idx`` given its current ``phase`` (read any world state, e.g.
        current_phase_time, off the held agents). Return ``None`` for no constraint."""
        return None

    def begin_interval(self, proposed_actions):
        """Called once per decision with the proposed per-intersection actions."""

    def resolve_step(self, executed_actions):
        """Called once per env step; return the actions that actually reach
        ``env.step`` this step. Default: identity."""
        return executed_actions

    def train(self, agents):
        """Optional per-train-step hook."""
