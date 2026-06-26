from .task import BaseTask
from common.registry import Registry


@Registry.register_task("sim2real_actions")
class Sim2RealActionsTask(BaseTask):
    """
    Task entrypoint for sim-to-real action experiments.
    Selects the concrete trainer implementation from the configured mitigation
    method (``--act_model`` / ``sim2real.method``).
    """

    METHOD_TO_TRAINER = {
        "naive": "sim2real_actions",  # direct transfer: delay-free sim, no mitigation
        # Action shielding (phase-transition gap): reuses the base trainer; the legal
        # -action mask flows through the existing valid_mask -> get_action path, so no
        # mitigation model/subclass is needed. The shield mode is set in shield.yml.
        "shield": "sim2real_actions",
        # Oblivious-Q reuses the Delayed-Q trainer, which forces the compensation
        # horizon to 0 for it: trained under delay, ignores it (no model rolled).
        "oblivious_q": "sim2real_actions_delayed_q",
        "delayed_q": "sim2real_actions_delayed_q",
        "prlight": "sim2real_actions_prlight",  # one-shot neighbor-aware prediction
    }

    def __init__(
        self,
        logger,
        method=None,
        gpu=0,
        cpu=False,
        name="sim2real_actions",
    ):
        method = method or self.resolve_method()
        trainer_name = self.METHOD_TO_TRAINER.get(method)
        if trainer_name is None:
            raise ValueError(
                f"Unsupported sim2real_actions method: {method}. "
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
            return "naive"
        return sim2real_setting.param.get("method", "naive")

    def run(self):
        try:
            if Registry.mapping["model_mapping"]["setting"].param["train_model"]:
                self.trainer.train()
            if Registry.mapping["model_mapping"]["setting"].param["test_model"]:
                self.trainer.test()
        except RuntimeError as e:
            self._process_error(e)
            raise e
