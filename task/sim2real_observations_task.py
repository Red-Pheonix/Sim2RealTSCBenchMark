from .task import BaseTask
from common.registry import Registry


@Registry.register_task("sim2real_observations")
class Sim2RealObservationsTask(BaseTask):
    """
    Task entrypoint for sim-to-real observation experiments.
    Selects the concrete trainer implementation from the configured method.
    """

    METHOD_TO_TRAINER = {
        "domain_randomization": "sim2real_observations_domain_randomization",
        "maml": "sim2real_observations_maml",
    }

    def __init__(
        self,
        logger,
        method=None,
        gpu=0,
        cpu=False,
        name="sim2real_observations",
    ):
        method = method or self.resolve_method()
        trainer_name = self.METHOD_TO_TRAINER.get(method)
        if trainer_name is None:
            raise ValueError(
                f"Unsupported sim2real_observations method: {method}. "
                f"Expected one of {sorted(self.METHOD_TO_TRAINER)}."
            )

        trainer = Registry.mapping["trainer_mapping"][trainer_name](
            logger, gpu=gpu, cpu=cpu, name=name
        )
        super().__init__(trainer)
        self.method = method

    def resolve_method(self):
        sim2real_setting = Registry.mapping.get("sim2real_mapping", {}).get("setting")
        if not sim2real_setting or not hasattr(sim2real_setting, "param"):
            return "domain_randomization"

        sim2real_config = sim2real_setting.param
        return sim2real_config.get("method", "domain_randomization")

    def run(self):
        try:
            if Registry.mapping["model_mapping"]["setting"].param["train_model"]:
                self.trainer.train()
            if Registry.mapping["model_mapping"]["setting"].param["test_model"]:
                self.trainer.test()
        except RuntimeError as e:
            self._process_error(e)
            raise e
