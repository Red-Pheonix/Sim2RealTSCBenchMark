from .task import BaseTask
from common.registry import Registry


@Registry.register_task("sim2real_actions")
class Sim2RealActionsTask(BaseTask):
    """
    Task scaffold for sim-to-real action experiments.
    """

    def run(self):
        try:
            if Registry.mapping["model_mapping"]["setting"].param["train_model"]:
                self.trainer.train()
            if Registry.mapping["model_mapping"]["setting"].param["test_model"]:
                self.trainer.test()
        except RuntimeError as e:
            self._process_error(e)
            raise e
