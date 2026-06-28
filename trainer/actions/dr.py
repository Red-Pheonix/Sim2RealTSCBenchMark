"""Trainer for Domain Randomization (DR) over the phase-transition constraint.

DR trains the agent across RANDOMLY GENERATED constraints -- each sim episode a fresh
valid phase-transition table is synthesized by taking the PERMISSIVE base (flexible, all
movements allowed) and randomly disallowing a fraction of its transitions, keeping the
graph strongly connected (see random_mask.py). The agent never trains on the deployment
table (barrier/cyclic); that target is just one member of the randomized family, so DR
tests genuine robustness to an UNSEEN constraint rather than oracle access to it.

The shield is a toggle: the per-episode table is applied in the SAME mode the side deploys
in (real_phase_transition_mode) -- `shield` (masked: sees each episode's legal set, 0
violations) vs `enforce` (unmasked: experiences the drops, nonzero violations). Reuses the
base run loop; sim_train swaps sim_action_transforms to that episode's random constraint.
"""

import json
import os

import numpy as np

from common.registry import Registry
from trainer.actions.phase_transition import PhaseTransition
from trainer.actions.random_mask import random_constrained_table
from trainer.actions.sim2real_actions_trainer import Sim2RealActionsTrainer


@Registry.register_trainer("sim2real_actions_dr")
class Sim2RealActionsDRTrainer(Sim2RealActionsTrainer):
    def __init__(self, logger, gpu=0, cpu=False, name="sim2real_actions"):
        super().__init__(logger=logger, gpu=gpu, cpu=cpu, name=name)
        self.method = "dr"
        cfg = self.get_sim2real_config()
        # Permissive base table (flexible): full transition graph + real timings. DR drops
        # edges from THIS; it does not read the deployment (barrier/cyclic) table.
        base_name = cfg.get("dr_base_table", "pt_flexible")
        world_param = Registry.mapping["world_mapping"]["setting"].param
        base_path = os.path.join(
            world_param["dir"], self.resolve_phase_transition_file(base_name)
        )
        with open(base_path, encoding="utf-8") as f:
            self.dr_base_table = json.load(f)
        # Per-episode drop fraction, sampled uniformly from [lo, hi] for diversity (a
        # single value if both ends are equal). barrier ~0.43, cyclic ~0.86 of the edges.
        drop = cfg.get("dr_drop_range", [0.2, 0.45])
        self.dr_drop_range = (float(drop[0]), float(drop[1]))
        # Apply the random table in the deployment mode, so the shield toggles both sides.
        self.pool_mode = cfg.get("real_phase_transition_mode", "enforce")
        self.dr_rng = np.random.default_rng(0)

    def sim_train(self, episode):
        # Synthesize this episode's constraint: drop a random fraction of flexible's edges.
        drop_rate = self.dr_rng.uniform(*self.dr_drop_range)
        table = random_constrained_table(self.dr_base_table, drop_rate, self.dr_rng)
        self.sim_action_transforms = [
            self.sim_action_delay,
            PhaseTransition(
                self.agents_sim, self.action_interval, mode=self.pool_mode, table=table
            ),
        ]
        return super().sim_train(episode)
