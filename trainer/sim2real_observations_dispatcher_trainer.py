from common.registry import Registry


@Registry.register_trainer("sim2real_observations")
class Sim2RealObservationsDispatcherTrainer:
    """
    Umbrella trainer for observation sim2real methods.
    Dispatches to a concrete implementation based on the observation model method.
    """

    METHOD_TO_TRAINER = {
        "standard": "sim2real_observations_standard",
        "maml": "sim2real_observations_maml",
    }

    def __init__(self, logger, gpu=0, cpu=False, name="sim2real_observations"):
        sim2real_setting = Registry.mapping.get("sim2real_mapping", {}).get("setting")
        method = "standard"
        if sim2real_setting and hasattr(sim2real_setting, "param"):
            obs_model_config = sim2real_setting.param.get("obs_model_config", {})
            method = obs_model_config.get(
                "method", sim2real_setting.param.get("method", "standard")
            )

        trainer_name = self.METHOD_TO_TRAINER.get(method)
        if trainer_name is None:
            raise ValueError(
                f"Unsupported sim2real_observations method: {method}. "
                f"Expected one of {sorted(self.METHOD_TO_TRAINER)}."
            )

        self.method = method
        self.impl = Registry.mapping["trainer_mapping"][trainer_name](
            logger, gpu=gpu, cpu=cpu, name=name
        )

    def train(self):
        return self.impl.train()

    def test(self):
        return self.impl.test()

    def __getattr__(self, item):
        return getattr(self.impl, item)
