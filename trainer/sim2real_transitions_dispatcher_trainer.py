from common.registry import Registry


@Registry.register_trainer("sim2real_transitions")
class Sim2RealTransitionsDispatcherTrainer:
    """
    Umbrella trainer for transition sim2real methods.
    Dispatches to a concrete implementation based on sim2real.method.
    """

    METHOD_TO_TRAINER = {
        "grounding": "sim2real_transitions_grounding",
        "domain_randomization": "sim2real_transitions_domain_randomization",
        "domain_adaptation": "sim2real_transitions_domain_adaptation",
    }

    def __init__(self, logger, gpu=0, cpu=False, name="sim2real_transitions"):
        sim2real_setting = Registry.mapping.get("sim2real_mapping", {}).get("setting")
        method = sim2real_setting.param.get("method", "grounding")

        trainer_name = self.METHOD_TO_TRAINER.get(method)
        if trainer_name is None:
            raise ValueError(
                f"Unsupported sim2real_transitions method: {method}. "
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
