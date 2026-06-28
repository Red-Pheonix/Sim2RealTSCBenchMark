"""Action-side sim2real gap: trainer, gap transforms, and mitigation method trainers.

- Transforms (the gap appliers applied in the action pipeline): ``ActionTransform``
  base, ``ActionDelay`` (execution delay), ``PhaseTransition`` (action-validity mask).
- ``Sim2RealActionsTrainer``: the base action trainer (naive / direct-transfer).
- Method trainers: ``Sim2RealActionsDelayedQTrainer`` (also serves oblivious-q) and
  ``Sim2RealActionsPRLightTrainer``.

The delay/prlight *models* remain RL agents under ``agent/action_delay``.
"""

from .base import ActionTransform
from .delay import ActionDelay
from .phase_transition import PhaseTransition
from .sim2real_actions_trainer import Sim2RealActionsTrainer
from .delayed_q import Sim2RealActionsDelayedQTrainer
from .prlight import Sim2RealActionsPRLightTrainer
from .soft_shield import Sim2RealActionsSoftShieldTrainer
from .dr import Sim2RealActionsDRTrainer
from .gat_trainer import Sim2RealActionsGATTrainer

__all__ = [
    "ActionTransform",
    "ActionDelay",
    "PhaseTransition",
    "Sim2RealActionsTrainer",
    "Sim2RealActionsDelayedQTrainer",
    "Sim2RealActionsPRLightTrainer",
    "Sim2RealActionsSoftShieldTrainer",
    "Sim2RealActionsDRTrainer",
    "Sim2RealActionsGATTrainer",
]
