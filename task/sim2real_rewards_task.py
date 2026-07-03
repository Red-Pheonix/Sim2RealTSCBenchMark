from .task import BaseTask
from common.registry import Registry


@Registry.register_task("sim2real_rewards")
class Sim2RealRewardsTask(BaseTask):
    """
    Task entrypoint for sim-to-real reward experiments.
    Selects the concrete trainer implementation from the configured method.
    """

    METHOD_TO_TRAINER = {
        "naive": "sim2real_rewards",
        # Direct transfer proper: pretrained sim policy, 0 training episodes, one real
        # eval. Same naive trainer; sim2real.direct_transfer forces sim_episodes=0.
        "direct_transfer": "sim2real_rewards",
        "reward_shaping": "sim2real_rewards_shaping",
        "reward_random": "sim2real_rewards_random",
        "reward_inference": "sim2real_rewards_reward_inference",
        "shield": "sim2real_rewards_shield",
        "pt_naive": "sim2real_rewards_pt_naive",
        "morl_grid": "sim2real_rewards_morl_grid",
        "dynamic_reward_shaping": "sim2real_rewards_dynamic_reward_shaping",
    }

    def __init__(self, logger, method=None, gpu=0, cpu=False, name="sim2real_rewards"):
        method = method or self.resolve_method()
        trainer_name = self.METHOD_TO_TRAINER.get(method)
        if trainer_name is None:
            raise ValueError(
                f"Unsupported sim2real_rewards method: {method}. "
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
