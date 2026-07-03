"""Phase-transition (action-validity) transform.

Loads a per-node phase-transition table (JSON, under
``raw_data/<network>/phase_transitions/``) and applies the controller's transition
rules. Two modes:

- ``mode="enforce"`` (default) -- models the REAL controller: the agent is NOT told
  the dwell-window rules. At each decision it may request any phase; if the requested
  transition from the current phase is illegal (disallowed, or its min/max dwell-time
  window is not met) the request is silently dropped and the intersection HOLDS the
  current phase. The agent thinks it switched but the signal didn't move -- that
  mismatch is the sim2real phase-transition gap (naive baseline). The one rule the
  controller still imposes is force-off: past a phase's max-green, holding is illegal,
  so the agent is masked to its non-current phases and forced to leave (it picks the
  highest-Q successor). This keeps the gap's controller realistic instead of letting
  the agent dwell on a phase forever.
- ``mode="shield"`` -- the shield mitigation: hand the agent a per-decision validity
  mask via ``valid_mask`` so it only ever picks legal actions, AND keep the enforce
  backstop (drop-illegal at execution) so the controller's spec cannot be violated
  even if an action slips through (non-DQN agent, edge case). In the normal case the
  mask makes the backstop a no-op -- shield is the safe superset of pure masking.

JSON format (per node id) -- phase indices are 1-based and already in the agent's
ACTION order (pre-reordered by scripts/reorder_phase_transitions_json.py):
    {
      "nema_phase_combinations": {"1": "1+6", ...},   # action idx -> combo (info only)
      "transitions": {"1": {"2": {"allowed": 1, "min_time_to_transition": ..,
                                  "max_time_to_transition": ..}}, ...}
    }

The trainer builds one of these over the sim agents and one over the real agents,
which is what lets the validity model differ sim-vs-real / be non-stationary.
"""

import json

import numpy as np
import torch

from .base import ActionTransform
from .interval_hold import IntervalHold
from .safety import SafetyStats


class PhaseTransition(ActionTransform):
    def __init__(self, agents, action_interval, json_path=None, mode="enforce", table=None):
        if mode not in ("enforce", "shield"):
            raise ValueError(
                f"PhaseTransition mode must be 'enforce' or 'shield', got {mode!r}"
            )
        if json_path is None and table is None:
            raise ValueError("PhaseTransition needs either json_path or an in-memory table.")
        self.agents = list(agents)
        self.action_interval = action_interval
        self.mode = mode
        # Both modes hand the agent a mask via valid_mask. Shield exposes the full
        # legal set (the agent only ever picks legal actions). Enforce exposes ONLY
        # the force-off constraint (past max-green the agent cannot hold) and is
        # otherwise all-permissive, so the agent stays unaware of the dwell windows
        # -- that unawareness, plus resolve_step dropping illegal switches to hold,
        # is the gap. resolve_step is the execution backstop behind the mask in both
        # modes (it judges the released action at execution time), so the controller's
        # spec holds even for a non-masking agent, and even under action delay.
        self.provides_mask = True

        # The transition table is either loaded from disk (json_path) or supplied in
        # memory (table) -- the latter lets DR hand it a procedurally generated constraint.
        if table is None:
            with open(json_path, encoding="utf-8") as f:
                table = json.load(f)

        self.allowed_mask = []
        self.min_time = []
        self.max_time = []
        for ag in self.agents:
            num_phases = ag.action_space.n
            inter_id = str(int(ag.phase_generator.I.id))

            allowed = torch.zeros(num_phases, num_phases, dtype=torch.bool)
            min_time = torch.zeros(num_phases, num_phases)
            max_time = torch.full((num_phases, num_phases), float("inf"))

            node = table.get(inter_id)
            if node is not None:
                for fj, tos in node["transitions"].items():
                    af = int(fj) - 1
                    if not 0 <= af < num_phases:
                        continue
                    for tj, info in tos.items():
                        at = int(tj) - 1
                        if not 0 <= at < num_phases:
                            continue
                        allowed[af, at] = bool(int(info["allowed"]))
                        min_time[af, at] = float(info["min_time_to_transition"])
                        max_time[af, at] = float(info["max_time_to_transition"])

            # Hold (phase -> same phase) is a normal windowed action, not an always-on
            # escape hatch: legal from dwell 0 up to the phase's max-green, then it
            # force-offs. max-green = the latest deadline among the phase's real
            # (allowed, off-diagonal) outgoing transitions -- past it every transition
            # window has also closed, so the controller must leave. A phase with no
            # outgoing transition in the table keeps an infinite hold (nothing to
            # force off to).
            offdiag_allowed = allowed.clone()
            offdiag_allowed.fill_diagonal_(False)
            cand = torch.where(
                offdiag_allowed, max_time, torch.full_like(max_time, float("-inf"))
            )
            max_green = cand.max(dim=1).values  # (num_phases,); -inf if no outgoing
            for p in range(num_phases):
                allowed[p, p] = True
                min_time[p, p] = 0.0
                max_time[p, p] = (
                    float(max_green[p]) if torch.isfinite(max_green[p]) else float("inf")
                )

            self.allowed_mask.append(allowed)
            self.min_time.append(min_time)
            self.max_time.append(max_time)

        # Safety accounting (per-episode; read+reset by the trainer). The transform
        # decides legality; SafetyStats keeps the books -- avoidable *violations* vs
        # controller *force_offs*, plus the per-agent last_violation flag soft_shield
        # reads. See trainer/actions/safety.py for the violation/force-off rationale.
        self.safety = SafetyStats(len(self.agents))
        # Validation happens at EXECUTION (resolve_step), not decision time: a delayed
        # action is judged against the controller's clock at the moment it is released,
        # which is why a decision-time shield goes stale under delay. resolve_step runs
        # every env step; IntervalHold makes that per-step path behave as one decision
        # per interval -- it flags the release boundary (where this transform validates
        # and counts once) and holds the committed phase across the interval's steps.
        self.hold = IntervalHold(len(self.agents))

    # ------------------------------------------------------------------
    # core legality: the per-decision legal-action set, shared by the
    # execution backstop and the agent-facing mask so they cannot disagree
    # ------------------------------------------------------------------
    def _legal_set(self, idx, cur):
        """Return ``(legal, forced, strict)`` for current phase ``cur`` (int).

        ``strict`` is the (num_phases,) bool of actions legal under the real window
        rules: each transition within its [min, max] dwell window, plus holding (the
        diagonal) up to the phase's max-green. ``forced`` is True when ``strict`` is
        empty (dwell past max-green) -- the controller must leave. ``legal`` is the
        set the agent is actually offered: ``strict`` normally, or every non-current
        phase when forced (so it is never empty). A force-off leave is therefore in
        ``legal`` but NOT in ``strict`` -- it is still an illegal (expired) move.
        """
        t = self.agents[idx].phase_generator.I.current_phase_time
        allowed = self.allowed_mask[idx][cur]
        min_ok = t >= self.min_time[idx][cur]
        max_ok = t + self.action_interval <= self.max_time[idx][cur]
        strict = allowed & min_ok & max_ok
        forced = not bool(strict.any())
        if forced:
            # Force-off: dwell is past max-green, the controller MUST leave. Relax
            # only the TIMING -- the agent may go to any normally-ALLOWED successor
            # (for cyclic that is the single next phase; for flexible, the allowed
            # set), never an arbitrary jump that would break the controller's
            # transition graph. Fall back to any non-current only if the table lists
            # no successor at all (degenerate node).
            legal = self.allowed_mask[idx][cur].clone()
            legal[cur] = False
            if not bool(legal.any()):
                legal = torch.ones_like(strict)
                legal[cur] = False
        else:
            legal = strict
        return legal, forced, strict

    def _forceoff_target(self, idx, cur):
        """Deterministic force-off successor for the execution backstop -- used only
        when an unmasked/edge action would otherwise hold illegally past max-green (a
        masking agent never requests this). Picks the allowed successor with the
        latest deadline; falls back to the next phase index if the table lists none."""
        n = self.allowed_mask[idx].shape[0]
        allowed = self.allowed_mask[idx][cur].clone()
        allowed[cur] = False
        cand = torch.where(
            allowed,
            self.max_time[idx][cur],
            torch.full_like(self.max_time[idx][cur], float("-inf")),
        )
        if bool(torch.isfinite(cand).any()):
            return int(torch.argmax(cand))
        return (cur + 1) % n

    # ------------------------------------------------------------------
    # execution backstop (both modes): resolve an illegal action at the moment
    # it is RELEASED by the upstream delay, against the controller's live clock
    # ------------------------------------------------------------------
    def reset(self, agents, init_phases):
        self.hold.reset()

    def begin_interval(self, proposed_actions):
        # Validation is deferred to resolve_step (execution time). Here we only mark
        # that a new decision has begun so resolve_step tallies one decision's stats on
        # its first call of the interval. The action is enqueued raw by the upstream
        # ActionDelay; the controller judges it when it is released, not now -- so under
        # delay the decision-time mask the agent used may no longer be legal.
        self.hold.begin()

    def resolve_step(self, executed_actions):
        # The controller commits one phase per decision. At the interval's first step
        # (the release boundary) we judge the action the upstream ActionDelay just
        # released against the controller's CURRENT clock (post-delay) and commit the
        # corrected phase; IntervalHold then holds that committed phase for every step
        # of the interval, so the signal gets one consistent command (no conflicting
        # mid-yellow rewrites). Judging at release -- not at the agent's decision -- is
        # what makes a decision-time shield go stale under delay: the legal set has
        # moved on, so the residual violations tallied here are exactly that gap.
        flat = np.asarray(executed_actions).reshape(len(self.agents), -1)
        if self.hold.take_boundary():
            for idx, ag in enumerate(self.agents):
                cur = int(np.asarray(ag.get_phase()).reshape(-1)[0])
                req = int(flat[idx, 0])
                n = self.allowed_mask[idx].shape[0]
                if not (0 <= cur < n) or not (0 <= req < n):
                    # transitional/out-of-range read at the boundary: pass through
                    self.hold.commit(idx, req)
                    continue
                legal, forced, strict = self._legal_set(idx, cur)
                # A violation is an AVOIDABLE illegal switch: out-of-window or
                # disallowed while a legal action still existed. Force-off leaves (past
                # max-green, when NO legal action exists) are tracked separately.
                violated = (not forced) and (not bool(strict[req]))
                self.safety.record(idx, violated, forced)
                if forced:
                    target = req if bool(legal[req]) else self._forceoff_target(idx, cur)
                elif not bool(legal[req]):
                    target = cur  # illegal switch dropped to a hold (the gap)
                else:
                    target = req
                self.hold.commit(idx, target)
        return self.hold.apply(executed_actions)

    @property
    def last_violation(self):
        """Per-agent illegal-request flag from the most recent decision (read by the
        soft_shield reward transform). Lives on the SafetyStats; exposed here so the
        trainer can read it off the transform directly."""
        return self.safety.last_violation

    def collect_stats(self):
        """(violations, force_offs, decisions) accumulated since the last reset."""
        return self.safety.collect()

    def collect_per_agent_stats(self):
        """Per-agent (violations, force_offs, decisions) arrays since the last reset,
        indexed to match ``self.agents`` (== the trainer's per-side agent order)."""
        return self.safety.collect_per_agent()

    def reset_stats(self):
        self.safety.reset()

    # ------------------------------------------------------------------
    # the legal-action mask handed to the agent at decision time
    # ------------------------------------------------------------------
    def valid_mask(self, idx, phase):
        cur = int(np.asarray(phase).reshape(-1)[0])
        legal, forced, _ = self._legal_set(idx, cur)
        if self.mode == "shield":
            # full constraint: the agent only ever picks legal actions
            return legal.unsqueeze(0)
        # enforce/gap: keep the agent unaware of the normal dwell windows (it may
        # request anything; illegal switches are dropped to hold by begin_interval).
        # The one thing the real controller still imposes is force-off -- past
        # max-green it cannot hold -- so expose only that.
        if forced:
            return legal.unsqueeze(0)
        return torch.ones_like(legal).unsqueeze(0)
