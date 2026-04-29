import numpy as np
from tqdm import tqdm

from common.registry import Registry
from .base import BaseObservationTrainer


@Registry.register_trainer("sim2real_observations_domain_randomization")
class ObservationDomainRandomizationTrainer(BaseObservationTrainer):
    """
    Trainer for observation domain randomization experiments.
    """

    def run_train_episode(
        self,
        *,
        env,
        metric,
        world,
        agents,
        episode,
        total_decision_num,
        desc,
    ):
        metric.clear()
        for agent in agents:
            agent.reset()
            if env is self.env_sim and self.domain_randomization_enabled:
                self.configure_observation_generator(
                    agent, self.current_sim_observation_config
                )
            elif env is self.env_real:
                self.configure_observation_generator(agent, self.real_observation_config)

        self.reset_observation_transforms(world)
        last_obs = env.reset()

        episode_loss = []
        flush = 0
        i = 0
        dones = [False] * len(agents)

        pbar = tqdm(total=int(self.steps / self.action_interval), desc=desc)

        while i < self.steps:
            if i % self.action_interval == 0:
                pbar.update()
                last_phase = np.stack([ag.get_phase() for ag in agents])

                actions = []
                for idx, ag in enumerate(agents):
                    actions.append(
                        ag.get_action(last_obs[idx], last_phase[idx], test=False)
                    )
                actions = np.stack(actions)

                actions_prob = []
                for idx, ag in enumerate(agents):
                    actions_prob.append(
                        ag.get_action_prob(last_obs[idx], last_phase[idx])
                    )

                rewards_list = []
                for _ in range(self.action_interval):
                    obs, rewards, dones, _ = env.step(actions.flatten())
                    i += 1
                    rewards_list.append(np.stack(rewards))

                rewards = np.mean(rewards_list, axis=0)
                metric.update(rewards)

                cur_phase = np.stack([ag.get_phase() for ag in agents])
                for idx, ag in enumerate(agents):
                    ag.remember(
                        last_obs[idx],
                        last_phase[idx],
                        actions[idx],
                        actions_prob[idx],
                        rewards[idx],
                        obs[idx],
                        cur_phase[idx],
                        dones[idx],
                        f"{episode}_{i // self.action_interval}_{ag.id}",
                    )

                flush += 1
                if flush == self.buffer_size - 1:
                    flush = 0

                total_decision_num += 1
                last_obs = obs

            if (
                total_decision_num > self.learning_start
                and total_decision_num % self.update_model_rate
                == self.update_model_rate - 1
            ):
                cur_loss_q = np.stack([ag.train() for ag in agents])
                episode_loss.append(cur_loss_q)

            if (
                total_decision_num > self.learning_start
                and total_decision_num % self.update_target_rate
                == self.update_target_rate - 1
            ):
                [ag.update_target_network() for ag in agents]

            if all(dones):
                break

        pbar.close()

        mean_loss = np.mean(np.array(episode_loss)) if episode_loss else 0
        return total_decision_num, mean_loss, i

    def sim_train(self, episode):
        if self.domain_randomization_enabled:
            self.current_sim_observation_config = self.build_domain_randomization_config()
            self.sim_observation_transforms = self.build_observation_transforms(
                self.current_sim_observation_config
            )
            self.world_sim.observation_transforms = self.sim_observation_transforms
            print(
                f"Episode {episode} sampled sim observation config:\n"
                f"{self.current_sim_observation_config}"
            )
            for ag in self.agents_sim:
                self.configure_observation_generator(
                    ag, self.current_sim_observation_config
                )
        self.set_replay(
            self.env_sim,
            f"sim_episode_{episode}.txt",
            episode % self.save_rate == 0,
        )
        self.total_decision_num_sim, mean_loss, steps_run = self.run_train_episode(
            env=self.env_sim,
            metric=self.metric_sim,
            world=self.world_sim,
            agents=self.agents_sim,
            episode=episode,
            total_decision_num=self.total_decision_num_sim,
            desc=f"Sim Training Epoch {episode}",
        )
        self.log_metrics("SIM_TRAIN", episode, self.metric_sim, mean_loss)
        self.logger.info("sim step:%s/%s", steps_run, self.steps)
        return mean_loss

    def train(self):
        if self.load_pretrained:
            pretrained_dir = self.pretrained_model_dir()
            self.load_agents(self.agents_sim, pretrained_dir)

        for episode in range(self.episodes):
            sim_loss = self.sim_train(episode)
            self.save_agents(self.agents_sim, self.model_dir)

            if episode % self.save_rate == 0:
                self.save_agents(self.agents_sim, self.model_dir, e=episode)

            self.logger.info(
                "episode:%s/%s, sim_loss:%s",
                episode,
                self.episodes,
                sim_loss,
            )

        self.save_agents(self.agents_sim, self.model_dir)
        self.load_agents(self.agents_real, self.model_dir)
        self.run_eval_episode(
            env=self.env_real,
            metric=self.metric_real,
            world=self.world_real,
            agents=self.agents_real,
            desc="Post-Train Real Eval",
        )
        self.log_metrics("TRAIN_REAL", self.episodes, self.metric_real, 100)
