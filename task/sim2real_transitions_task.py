import logging
from .task import BaseTask
from common.registry import Registry



@Registry.register_task("sim2real_transitions")
class Sim2RealTransitionsTask(BaseTask):
    """
    Task entrypoint for transition sim2real experiments.
    Selects the concrete trainer implementation from the provided method.
    """

    METHOD_TO_TRAINER = {
        "grounding": "sim2real_transitions_grounding",
        "domain_randomization": "sim2real_transitions_domain_randomization",
        "domain_adaptation": "sim2real_transitions_domain_adaptation",
    }

    def __init__(
        self,
        logger,
        method,
        gpu=0,
        cpu=False,
        name="sim2real_transitions",
    ):
        trainer_name = self.METHOD_TO_TRAINER.get(method)
        if trainer_name is None:
            raise ValueError(
                f"Unsupported sim2real_transitions method: {method}. "
                f"Expected one of {sorted(self.METHOD_TO_TRAINER)}."
            )

        trainer = Registry.mapping["trainer_mapping"][trainer_name](
            logger, gpu=gpu, cpu=cpu, name=name
        )
        super().__init__(trainer)
        self.method = method

    def run(self):
        """
        run
        Run the whole task, including training and testing.

        :param: None
        :return: None
        """
        try:
            if Registry.mapping['model_mapping']['setting'].param['train_model']:
                print("-----conducting training--------")
                self.trainer.train()

            if Registry.mapping['model_mapping']['setting'].param['test_model']:
                print("-----conducting testing--------")
                self.trainer.test()
        except RuntimeError as e:
            self._process_error(e)
            raise e
