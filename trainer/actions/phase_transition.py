"""Phase-transition (action-validity) transform.

Loads a per-node phase-transition table (JSON, under
``raw_data/<network>/phase_transitions/``) and exposes, per intersection, the
legal-action mask given the current phase and its dwell time.

JSON format (per node id) -- phase indices are 1-based and already in the agent's
ACTION order (the SUMO tlLogic green-phase order; the raw canonical-NEMA tables are
pre-reordered into this order by scripts/reorder_phase_transitions_json.py):
    {
      "nema_phase_combinations": {"1": "1+6", "2": "2+6", ...},   # action idx -> combo (info only)
      "transitions": {                                            # from -> to -> info
        "1": {"2": {"allowed": 1, "min_time_to_transition": .., "max_time_to_transition": ..}},
        ...
      }
    }

``valid_mask(idx, phase)`` indexes the allowed / min / max tensors directly with the
action index. Owning this here (trainer builds one over sim agents and one over real
agents) is what lets the validity model differ sim-vs-real / be non-stationary.
"""

import json

import torch

from .base import ActionTransform


class PhaseTransition(ActionTransform):
    provides_mask = True

    def __init__(self, agents, action_interval, json_path):
        self.agents = list(agents)
        self.action_interval = action_interval

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

            # phase -> same phase is always allowed
            allowed |= torch.eye(num_phases, dtype=torch.bool)

            self.allowed_mask.append(allowed)
            self.min_time.append(min_time)
            self.max_time.append(max_time)

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
