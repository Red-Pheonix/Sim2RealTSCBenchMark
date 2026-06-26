"""Phase-transition (action-validity) transform.

Loads a per-node phase-transition table (JSON, under
``raw_data/<network>/phase_transitions/``) and applies the controller's transition
rules. Two modes:

- ``mode="enforce"`` (default) -- models the REAL controller: the agent is NOT told
  the rules (no policy mask). At each decision it may request any phase; if the
  requested transition from the current phase is illegal (disallowed, or its
  min/max dwell-time window is not met) the request is silently dropped and the
  intersection HOLDS the current phase. The agent thinks it switched but the signal
  didn't move -- that mismatch is the sim2real phase-transition gap (naive baseline).
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


class PhaseTransition(ActionTransform):
    def __init__(self, agents, action_interval, json_path, mode="enforce"):
        if mode not in ("enforce", "shield"):
            raise ValueError(
                f"PhaseTransition mode must be 'enforce' or 'shield', got {mode!r}"
            )
        self.agents = list(agents)
        self.action_interval = action_interval
        self.mode = mode
        # The shield contributes a policy mask (agent only ever picks legal actions);
        # the gap (enforce) mode leaves the agent unmasked. Both modes drop illegal
        # transitions at execution -- for shield that is just a backstop behind the
        # mask, for enforce it is the gap itself.
        self.provides_mask = mode == "shield"

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

            # phase -> same phase is always allowed (holding is always legal)
            allowed |= torch.eye(num_phases, dtype=torch.bool)

            self.allowed_mask.append(allowed)
            self.min_time.append(min_time)
            self.max_time.append(max_time)

    # ------------------------------------------------------------------
    # drop illegal requested transitions at execution time (both modes:
    # the gap for enforce, the backstop behind the mask for shield)
    # ------------------------------------------------------------------
    def begin_interval(self, proposed_actions):
        # Rewrite illegal requests to "hold current phase" IN PLACE, before any
        # downstream transform (e.g. ActionDelay) enqueues the action. The agent is
        # unaware -- it requested a switch that the real controller silently ignores.
        for idx, ag in enumerate(self.agents):
            cur = int(np.asarray(ag.get_phase()).reshape(-1)[0])
            req = int(np.asarray(proposed_actions[idx]).reshape(-1)[0])
            if req != cur and not self._transition_ok(idx, cur, req):
                proposed_actions[idx] = cur

    def _transition_ok(self, idx, cur, nxt):
        t = self.agents[idx].phase_generator.I.current_phase_time
        if not bool(self.allowed_mask[idx][cur, nxt]):
            return False
        if t < float(self.min_time[idx][cur, nxt]):
            return False
        if t + self.action_interval > float(self.max_time[idx][cur, nxt]):
            return False
        return True

    # ------------------------------------------------------------------
    # shield mode: hand the agent the legal-action set
    # ------------------------------------------------------------------
    def valid_mask(self, idx, phase):
        current_phase_time = self.agents[idx].phase_generator.I.current_phase_time

        allowed = self.allowed_mask[idx][phase]
        min_ok = current_phase_time >= self.min_time[idx][phase]
        max_ok = current_phase_time + self.action_interval <= self.max_time[idx][phase]
        valid_mask = allowed & (min_ok & max_ok)

        # special case where no valid actions are available -> keep current phase
        if (~valid_mask).all().item():
            valid_mask[0][phase] = True

        return valid_mask
