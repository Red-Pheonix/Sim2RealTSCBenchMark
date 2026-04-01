from common.registry import Registry
from trainer.tsc_trainer import TSCTrainer


@Registry.register_trainer("sim2real_observations")
class Sim2RealObservationsTrainer(TSCTrainer):
    """
    Trainer scaffold for sim-to-real observation work.

    This currently reuses the TSC training loop so the new task can be
    registered end-to-end while observation-specific behavior is added later.
    """

    def __init__(self, logger, gpu=0, cpu=False, name="sim2real_observations"):
        super().__init__(logger=logger, gpu=gpu, cpu=cpu, name=name)
