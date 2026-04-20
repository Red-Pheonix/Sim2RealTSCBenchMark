import torch
import numpy as np
from tqdm import tqdm

from common.registry import Registry
from trainer.sim2real_observations_trainer import Sim2RealObservationsTrainer
from utils.logger import load_config


@Registry.register_trainer("sim2real_observations_maml")
class Sim2RealObservationsMAMLTrainer(Sim2RealObservationsTrainer):
    """
    MAML trainer for observation-based sim2real experiments.

    The trainer owns support/query rollout orchestration while the
    configured agent owns model-specific meta-learning logic.
    """

    def __init__(self, logger, gpu=0, cpu=False, name="sim2real_observations"):
        super().__init__(logger=logger, gpu=gpu, cpu=cpu, name=name)

        self.maml_config = self.sim2real_config["obs_model_config"]["maml"]
        self.rollout_test = self.maml_config.get("rollout_test", True)
        self.target_update_interval = self.maml_config.get("target_update_interval", 1)
        self.meta_update_interval = self.maml_config.get("meta_update_interval", 10)
        self.sample_tasks_from_dr = self.maml_config.get("sample_tasks_from_dr", False)
        if self.sample_tasks_from_dr:
            dr_config, _ = load_config(
                "configs/sim2real_observations/domain_randomization.yml"
            )
            self.domain_randomization_config = dr_config.get("sim2real", {}).get(
                "domain_randomization", {}
            )
            self.domain_randomization_enabled = self.domain_randomization_config.get(
                "enabled", False
            )
            self.randomized_observation_parameters = (
                self.build_randomized_observation_parameters()
            )
            self.sim_observation_rng = np.random.default_rng(
                self.domain_randomization_config.get("seed", 0)
            )
        else:
            self.domain_randomization_enabled = False
        self.sync_real_agents_from_sim()

    def resolve_meta_agent_class(self):
        agent_name = Registry.mapping["command_mapping"]["setting"].param["agent"]
        meta_model_mapping = Registry.mapping["sim2real_model_mapping"]
        if agent_name not in meta_model_mapping:
            raise TypeError(
                f"Agent '{agent_name}' is not registered for observation-side MAML. "
                "Register a sim2real model variant with "
                "@Registry.register_sim2real_model(agent_name)."
            )
        return agent_name, meta_model_mapping[agent_name]

    def sync_real_agents_from_sim(self):
        for sim_agent, real_agent in zip(self.agents_sim, self.agents_real):
            real_agent.sync_base_from(sim_agent)

    def create_agent_world(self, world):
        agents = []
        _, agent_cls = self.resolve_meta_agent_class()
        agent = agent_cls(world, 0)
        num_agent = int(len(world.intersections) / agent.sub_agents)
        agents.append(agent)
        for idx in range(1, num_agent):
            agents.append(agent_cls(world, idx))

        if Registry.mapping["model_mapping"]["setting"].param["name"] == "magd":
            for ag in agents:
                ag.link_agents(agents)
        return agents

    def clear_agent_buffers(self, agents):
        for agent in agents:
            agent.clear_replay_buffer()

    def mean_loss(self, losses):
        valid_losses = [loss for loss in losses if not np.isnan(loss)]
        if not valid_losses:
            return np.nan
        return float(np.mean(valid_losses))

    def sample_sim_task(self, episode):
        if self.sample_tasks_from_dr:
            self.current_sim_observation_config = self.build_domain_randomization_config()
            self.logger.info(
                "maml task episode:%s sampled sim observation config:%s",
                episode,
                self.current_sim_observation_config,
            )
        else:
            self.current_sim_observation_config = self.sim_observation_config
        self.sim_observation_transforms = self.build_observation_transforms(
            self.current_sim_observation_config
        )
        self.world_sim.observation_transforms = self.sim_observation_transforms
        for ag in self.agents_sim:
            self.configure_observation_generator(ag, self.current_sim_observation_config)

    def rollout_episode(
        self,
        *,
        env,
        metric,
        world,
        agents,
        episode,
        desc,
        rollout_label,
        use_meta_policy=False,
    ):
        metric.clear()
        for agent in agents:
            agent.reset()
            if env is self.env_real:
                self.configure_observation_generator(agent, self.real_observation_config)

        self.reset_observation_transforms(world)
        last_obs = env.reset()
        steps_run = 0
        dones = [False] * len(agents)
        pbar = tqdm(total=int(self.steps / self.action_interval), desc=desc)

        while steps_run < self.steps:
            pbar.update()
            last_phase = np.stack([agent.get_phase() for agent in agents])
            actions = []
            action_probs = []
            for idx, agent in enumerate(agents):
                if use_meta_policy:
                    action = agent.get_meta_action(
                        last_obs[idx], last_phase[idx], test=self.rollout_test
                    )
                else:
                    action = agent.get_action(
                        last_obs[idx], last_phase[idx], test=self.rollout_test
                    )
                actions.append(action)
                action_probs.append(
                    agent.get_action_prob(last_obs[idx], last_phase[idx])
                )
            actions = np.stack(actions)

            rewards_list = []
            for _ in range(self.action_interval):
                obs, rewards, dones, _ = env.step(actions.flatten())
                steps_run += 1
                rewards_list.append(np.stack(rewards))
                if steps_run >= self.steps or all(dones):
                    break
            rewards = np.mean(rewards_list, axis=0)
            metric.update(rewards)

            cur_phase = np.stack([agent.get_phase() for agent in agents])
            for idx, agent in enumerate(agents):
                agent.remember(
                    last_obs[idx],
                    last_phase[idx],
                    actions[idx],
                    action_probs[idx],
                    rewards[idx],
                    obs[idx],
                    cur_phase[idx],
                    dones[idx],
                    f"{rollout_label}_{episode}_{steps_run // self.action_interval}_{agent.id}",
                )
            last_obs = obs

            if all(dones):
                break

        pbar.close()
        return steps_run

    def train(self):
        
        meta_loss_value = 0
        support_losses = torch.zeros(len(self.agents_sim))
        query_losses = torch.zeros(len(self.agents_sim))
        
        if self.load_pretrained:
            pretrained_dir = self.pretrained_model_dir()
            self.load_agents(self.agents_sim, pretrained_dir)
            self.sync_real_agents_from_sim()

        for episode in range(self.episodes):
            self.sample_sim_task(episode)
            self.clear_agent_buffers(self.agents_sim)
            self.clear_agent_buffers(self.agents_real)

            self.set_replay(
                self.env_sim,
                f"maml_support_episode_{episode}.txt",
                episode % self.save_rate == 0,
            )
            sim_steps = self.rollout_episode(
                env=self.env_sim,
                metric=self.metric_sim,
                world=self.world_sim,
                agents=self.agents_sim,
                episode=episode,
                desc=f"MAML Support Epoch {episode}",
                rollout_label="support",
                use_meta_policy=True,
            )
            sim_steps = self.rollout_episode(
                env=self.env_sim,
                metric=self.metric_sim,
                world=self.world_sim,
                agents=self.agents_sim,
                episode=episode,
                desc=f"MAML Support Epoch {episode}",
                rollout_label="support",
                use_meta_policy=True,
            )

            # reset support loss for every task
            support_losses = torch.zeros(len(self.agents_sim))            
            for i, sim_agent in enumerate(self.agents_sim):
                # support_loss = sim_agent.compute_support_loss()
                support_loss = torch.zeros(1)
                support_losses[i] = support_loss
                
            support_loss = support_losses.mean().detach().cpu().item()
            self.log_metrics("MAML_SUPPORT_SIM", episode, self.metric_sim, support_loss)
            self.logger.info("maml support step:%s/%s", sim_steps, self.steps)

            # for setting up query
            self.clear_agent_buffers(self.agents_sim)

            self.set_replay(
                self.env_sim,
                f"maml_query_episode_{episode}.txt",
                episode % self.save_rate == 0,
            )
            # this is still part of our per task work
            query_steps = self.rollout_episode(
                env=self.env_sim,
                metric=self.metric_sim,
                world=self.world_sim,
                agents=self.agents_sim,
                episode=episode,
                desc=f"MAML Query Epoch {episode}",
                rollout_label="query",
                use_meta_policy=True,
            )

            for i, sim_agent in enumerate(self.agents_sim):
                query_loss = sim_agent.compute_query_loss()
                # query_losses[i] += query_loss


            if (
                (episode) % self.meta_update_interval == 0
                or episode == self.episodes - 1
            ):
                for i, sim_agent in enumerate(self.agents_sim):
                    # query_losses[i].backward()
                    pass
                    sim_agent.apply_meta_update()
                    sim_agent.zero_meta_grad()
                
                # reset meta losss
                query_losses = torch.zeros(len(self.agents_sim))

                update_target = (
                    ((episode + 1) // self.meta_update_interval)
                    % self.target_update_interval
                    == 0
                )
                if update_target:
                    for sim_agent in self.agents_sim:
                        sim_agent.update_target_network()
                
                meta_loss_value = query_losses.mean().detach().cpu().item()

                self.log_metrics("MAML_QUERY_SIM", episode, self.metric_sim, meta_loss_value)
                self.logger.info("maml query step:%s/%s", query_steps, self.steps)

                self.sync_real_agents_from_sim()

                self.save_agents(self.agents_sim, self.model_dir)
                self.save_agents(self.agents_real, self.model_dir)
            
            if episode % self.save_rate == 0:
                self.save_agents(self.agents_sim, self.model_dir, e=episode)
                self.save_agents(self.agents_real, self.model_dir, e=episode)

            self.logger.info(
                "episode:%s/%s, support_loss:%s, meta_loss:%s",
                episode,
                self.episodes,
                support_loss,
                meta_loss_value,
            )

            if self.test_when_train:
                self.train_test(episode)

        self.sync_real_agents_from_sim()
        self.save_agents(self.agents_sim, self.model_dir, e=self.episodes)
        self.save_agents(self.agents_real, self.model_dir, e=self.episodes)
        self.save_agents(self.agents_sim, self.model_dir)
        self.save_agents(self.agents_real, self.model_dir)
        self.run_eval_episode(
            env=self.env_real,
            metric=self.metric_real,
            world=self.world_real,
            agents=self.agents_real,
            desc="Post-Train Real Eval",
        )
        self.log_metrics("TRAIN_REAL", self.episodes, self.metric_real, 100)
