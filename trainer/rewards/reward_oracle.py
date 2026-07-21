"""reward_oracle: known-w skyline reference (NOT a mitigation method).

Trains DIRECTLY in the real simulator on the true hidden objective: the training
reward is `LinearReward(w*)` over the REAL feature bank -- the exact integrand the
`TrueReward` scorer evaluates -- so the policy optimizes precisely what it is scored
on. Every component (fairness / emission / ssm_conflicts / ...) is measurable in the
real (sumo) world, so nothing is inert.

Purpose: turn the roster's best-of-known reference into a true regret bracket.
  * DT vs oracle           = measurable regret of deploying the proxy-trained policy;
  * method vs oracle       = how much of the KNOWABLE reward the method recovers,
                             separating "cannot know w" (method failure) from
                             "signal control has no lever on this objective"
                             (environment property -- if the oracle also fails to move
                             emission/SSM, the objective itself is unoptimizable here).

Protocol: follows the PRETRAINING convention, not the method budget -- from scratch,
`oracle_episodes` (default 200, the tsc pretraining count) training episodes in real.
The real-interaction budget cap does not apply to a reference run; the DTL still
logs every training rollout honestly as REAL_TRAIN (invariant: REAL_TRAIN row count
== real rollouts used), with the naive-family REAL_TEST scoring curve every
`real_eval_interval` episodes on top (budget-exempt, as for every method).
"""

from common.registry import Registry
from trainer.rewards.base import Sim2RealRewardsTrainer
from trainer.rewards.reward_transforms import LinearReward


@Registry.register_trainer("sim2real_rewards_reward_oracle")
class Sim2RealRewardsOracleTrainer(Sim2RealRewardsTrainer):
    def __init__(self, logger, gpu=0, cpu=False, name="sim2real_rewards"):
        super().__init__(logger=logger, gpu=gpu, cpu=cpu, name=name)
        self.method = "reward_oracle"
        cfg = self.get_sim2real_config()
        self.oracle_episodes = int(cfg.get("oracle_episodes", 200))
        self.oracle_from_scratch = bool(cfg.get("oracle_from_scratch", True))
        # Train on the true objective, on the REAL side. The base class built the
        # transform over the SIM bank (where hidden components are absent); rebuild
        # it over the real bank with the true weights.
        w_star = self.feature_bank_real.weight_vector(self.true_reward_weights)
        self.reward_transform = LinearReward(
            w_star, self.feature_bank_real.components, norm=self.component_norm
        )
        # Reference-run bookkeeping: the whole training run is real interaction. The
        # method budget cap is not applicable; align the cap with the plan so
        # _log_real_budget reports spend without a spurious EXCEEDED warning.
        self.real_episodes = self.oracle_episodes

    def train(self):
        if not self.oracle_from_scratch and self.load_pretrained:
            self.load_agents(self.agents_real, self.pretrained_model_dir())
        for episode in range(self.oracle_episodes):
            self.on_episode_start(episode)
            self._real_rollouts += 1
            mean_loss, steps_run = self.run_train_episode(
                env=self.env_real,
                metric=self.metric_real,
                agents=self.agents_real,
                feature_bank=self.feature_bank_real,
                episode=episode,
                desc=f"ORACLE REAL_TRAIN Epoch {episode}",
            )
            self.log_metrics(
                "REAL_TRAIN", episode, self.metric_real, mean_loss,
                train_reward=self._last_train_reward,
            )
            self.logger.info("real step:%s/%s", steps_run, self.steps)
            self.save_agents(self.agents_real, self.model_dir)
            if episode % self.save_rate == 0:
                self.save_agents(self.agents_real, self.model_dir, e=episode)
            self.logger.info(
                "episode:%s/%s, oracle_loss:%s", episode, self.oracle_episodes, mean_loss
            )
            if self.test_when_train and self._should_run_real_eval(episode):
                self.train_test(episode)
        self.save_agents(self.agents_real, self.model_dir)
        # Tag the final weights under the tag test() loads by convention.
        self.save_agents(self.agents_real, self.model_dir, e=self.sim_episodes)
        self._save_method_state({
            "method": self.method,
            "oracle_episodes": self.oracle_episodes,
            "from_scratch": self.oracle_from_scratch,
            "true_reward_weights": dict(self.true_reward_weights),
        })
        self._log_real_budget()

    def test(self, drop_load=False):
        # agents_real already hold the final trained weights; no reload needed.
        return super().test(drop_load=True)
