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
        "naive": "sim2real_actions",  # baseline: delay-free sim training, no mitigation
        # Direct transfer proper: pretrained sim policy, 0 training episodes, one real
        # eval (the raw action-gap number). Same base trainer, config-only.
        "direct_transfer": "sim2real_actions",
        # Action shielding (phase-transition gap): reuses the base trainer; the legal
        # -action mask flows through the existing valid_mask -> get_action path, so no
        # mitigation model/subclass is needed. The shield mode is set in shield.yml.
        "shield": "sim2real_actions",
        # Train-time shield: mask the agent to the legal set DURING sim training
        # (vs `shield`, which masks only at deployment). Pure config on the base trainer.
        "train_shield": "sim2real_actions",
        # Soft shield: ground the sim with the real controller and penalize illegal
        # requests in the reward -- a soft constraint vs shield's hard mask. Its own
        # trainer (stores the requested action + adds the violation-penalty reward
        # transform), since both are soft_shield-exclusive.
        "soft_shield": "sim2real_actions_soft_shield",
        # Oblivious-Q reuses the Delayed-Q trainer, which forces the compensation
        # horizon to 0 for it: trained under delay, ignores it (no model rolled).
        # Domain randomization: train across a randomized family of phase-transition
        # tables (masked) so the policy is robust to the legal-action structure.
        "dr": "sim2real_actions_dr",
        # GAT / UGAT (Grounded Action Transformation): KEPT AS NEGATIVE RESULTS -- grounding
        # learns but does not close the PT gap. Both reuse the proven grounding infra; gat
        # grounds always, ugat gates on inverse-model uncertainty. Same trainer, the
        # `uncertainty` flag (gat.yml / ugat.yml) selects the variant.
        "gat": "sim2real_actions_gat",
        "ugat": "sim2real_actions_gat",
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
