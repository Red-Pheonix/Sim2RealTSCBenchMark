import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common.metrics import Metrics
from common.registry import Registry
from environment import TSCEnv
from .base import TransitionTrainer
from .distributions import NormalDistribution, UniformDistribution


@Registry.register_trainer("sim2real_transitions_domain_randomization")
class TransitionDomainRandomizationTrainer(TransitionTrainer):
    """
    Train entirely in randomized sim transitions and evaluate in the real environment.
    """

    def __init__(
        self,
        logger,
        gpu=0,
        cpu=False,
        name="sim2real_transitions_domain_randomization",
    ):
        super().__init__(logger=logger, gpu=gpu, cpu=cpu, name=name)

        self.episodes = self.trainer_args["episodes"]
        self.real_eval_interval = self.trainer_args.get("real_eval_interval", 0)

        self.method = self.sim2real_args.get("method", "domain_randomization")
        self.domain_randomization_config = self.sim2real_args.get(
            "domain_randomization", {}
        )
        self.domain_randomization_enabled = self.domain_randomization_config.get(
            "enabled", False
        )
        self.domain_randomization_rng = np.random.default_rng(
            self.domain_randomization_config.get("seed", self.seed)
        )

        self.exp_name = self.build_exp_name(self.method)
        self.model_dir = self.build_model_dir(self.exp_name)
        self.log_file = self.build_log_file()

        self.base_cityflow_config = self.load_cityflow_config(self.cityflow_path)
        self.base_cityflow_flow = self.load_cityflow_flow(self.base_cityflow_config)
        self.domain_randomization_dir = Path(
            self.base_cityflow_config.get("dir", "data")
        ) / self.domain_randomization_config.get(
            "output_dir", "output_data/sim2real_transitions/domain_randomization"
        )
        self.domain_randomization_temp_dir = (
            self.domain_randomization_dir / self.network_name / self.real_setting
        )
        self.generated_temp_files = []

        self.world_sim = None
        self.world_real = None
        self.agents_sim = None
        self.agents_real = None
        self.metric_sim = None
        self.metric_real = None
        self.env_sim = None
        self.env_real = None
        self.total_decision_num_sim = 0

        self.create()

        self.world = self.world_real
        self.agents = self.agents_real
        self.metric = self.metric_real
        self.env = self.env_real

        if self.domain_randomization_enabled:
            self.cleanup_stale_domain_randomization_files()

    def initial_pretrained_model_dir(self):
        return os.path.join(
            "pretrained",
            "tsc",
            self.agent_name,
            self.network_name,
        )

    def transition_pretrained_model_dir(self):
        return os.path.join(
            "pretrained",
            "sim2real_transitions",
            self.agent_name,
            self.network_name,
            self.method,
            self.real_setting,
        )

    def sample_randomized_vehicle_parameters(self):
        sampled_parameters = {}
        for param_name, param_config in self.domain_randomization_config.get(
            "parameters", {}
        ).items():
            distribution = self.build_parameter_distribution(param_config)
            sampled_value = distribution.sample(self.domain_randomization_rng)

            clip_min = param_config.get("clip_min")
            clip_max = param_config.get("clip_max")
            if clip_min is not None:
                sampled_value = max(clip_min, sampled_value)
            if clip_max is not None:
                sampled_value = min(clip_max, sampled_value)
            if param_config.get("round", False):
                sampled_value = np.round(sampled_value)

            sampled_parameters[param_name] = float(sampled_value)
        return sampled_parameters

    def build_parameter_distribution(self, param_config):
        distribution_name = param_config.get("distribution", "uniform")
        distribution_config = param_config.get(
            f"{distribution_name}_config", param_config
        )

        if distribution_name == "uniform":
            return UniformDistribution(distribution_config)
        if distribution_name == "normal":
            return NormalDistribution(distribution_config)

        raise ValueError(
            "Unsupported domain randomization distribution: "
            f"{distribution_name}"
        )

    def cleanup_temp_files(self, paths=None):
        paths_to_remove = paths if paths is not None else self.generated_temp_files
        for path in paths_to_remove:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass
        if paths is None:
            self.generated_temp_files = []

    def cleanup_stale_domain_randomization_files(self):
        stale_files = []
        if self.domain_randomization_temp_dir.exists():
            patterns = ["temp_*.cfg", "temp_flow_*.json", "temp.cfg", "temp_flow.json"]
            for pattern in patterns:
                stale_files.extend(self.domain_randomization_temp_dir.glob(pattern))

        # Clean up any older legacy temp files from the shared root as well.
        if self.domain_randomization_dir.exists():
            for pattern in ["temp.cfg", "temp_flow.json"]:
                stale_files.extend(self.domain_randomization_dir.glob(pattern))

        self.cleanup_temp_files(sorted(set(stale_files)))

    def write_episode_randomized_cityflow_config(self, episode):
        sampled_parameters = self.sample_randomized_vehicle_parameters()
        randomized_flow = json.loads(json.dumps(self.base_cityflow_flow))
        for record in randomized_flow:
            vehicle_config = record.setdefault("vehicle", {})
            for param_name, sampled_value in sampled_parameters.items():
                vehicle_config[param_name] = sampled_value

        self.domain_randomization_temp_dir.mkdir(parents=True, exist_ok=True)
        flow_filename = f"temp_flow_{episode}.json"
        config_filename = f"temp_{episode}.cfg"
        flow_path = self.domain_randomization_temp_dir / flow_filename
        config_path = self.domain_randomization_temp_dir / config_filename

        with open(flow_path, "w", encoding="utf-8") as file_obj:
            json.dump(randomized_flow, file_obj)

        cityflow_config = dict(self.base_cityflow_config)
        cityflow_config["flowFile"] = (
            Path(self.domain_randomization_config.get(
                "output_dir", "output_data/sim2real_transitions/domain_randomization"
            ))
            / self.network_name
            / self.real_setting
            / flow_filename
        ).as_posix()
        with open(config_path, "w", encoding="utf-8") as file_obj:
            json.dump(cityflow_config, file_obj, indent=2)

        self.generated_temp_files.extend([flow_path, config_path])
        return config_path.as_posix(), sampled_parameters

    def reload_sim_env_for_episode(self, episode):
        if self.domain_randomization_enabled:
            config_path, sampled_parameters = self.write_episode_randomized_cityflow_config(
                episode
            )
            self.logger.info(
                "Episode %s sim randomization params: %s",
                episode,
                sampled_parameters,
            )
        else:
            config_path = self.cityflow_path

        thread_num = Registry.mapping["command_mapping"]["setting"].param["thread_num"]
        self.world_sim = Registry.mapping["world_mapping"]["cityflow"](
            config_path,
            thread_num,
        )

        # Keep the same agent objects so replay buffers and optimizer state persist
        # across episode-level simulator reloads.
        for agent in self.agents_sim:
            agent.world = self.world_sim
            agent.reset()
        self.metric_sim = Metrics(
            self.metric_sim.lane_metric_List,
            self.metric_sim.world_metrics,
            self.world_sim,
            self.agents_sim,
        )
        self.env_sim = TSCEnv(self.world_sim, self.agents_sim, self.metric_sim)

    def run_train_episode(
        self,
        *,
        env,
        metric,
        agents,
        episode,
        total_decision_num,
        desc,
    ):
        metric.clear()
        last_obs = env.reset()
        for agent in agents:
            agent.reset()

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
                    if total_decision_num > self.learning_start:
                        actions.append(
                            ag.get_action(last_obs[idx], last_phase[idx], test=False)
                        )
                    else:
                        actions.append(ag.sample())
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

    def run_eval_episode(self, *, env, metric, agents, desc):
        metric.clear()
        obs = env.reset()
        for agent in agents:
            agent.reset()

        i = 0
        dones = [False] * len(agents)
        pbar = tqdm(total=int(self.test_steps / self.action_interval), desc=desc)

        while i < self.test_steps:
            if i % self.action_interval == 0:
                pbar.update()
                phases = np.stack([ag.get_phase() for ag in agents])
                actions = []
                for idx, ag in enumerate(agents):
                    actions.append(ag.get_action(obs[idx], phases[idx], test=True))
                actions = np.stack(actions)

                rewards_list = []
                for _ in range(self.action_interval):
                    obs, rewards, dones, _ = env.step(actions.flatten())
                    i += 1
                    rewards_list.append(np.stack(rewards))

                rewards = np.mean(rewards_list, axis=0)
                metric.update(rewards)

            if all(dones):
                break

        pbar.close()

    def log_metrics(self, mode, step, metric, loss):
        self.logger.info(
            "%s step:%s, travel time:%s, q_loss:%s, rewards:%s, queue:%s, delay:%s, throughput:%s",
            mode,
            step,
            metric.real_average_travel_time(),
            loss,
            metric.rewards(),
            metric.queue(),
            metric.delay(),
            int(metric.throughput()),
        )
        self.writeLog(
            mode,
            step,
            metric.real_average_travel_time(),
            loss,
            metric.rewards(),
            metric.queue(),
            metric.delay(),
            metric.throughput(),
        )

    def sim_train(self, episode):
        self.reload_sim_env_for_episode(episode)
        self.set_replay(
            self.env_sim,
            f"sim_episode_{episode}.txt",
            episode % self.save_rate == 0,
        )
        self.total_decision_num_sim, mean_loss, steps_run = self.run_train_episode(
            env=self.env_sim,
            metric=self.metric_sim,
            agents=self.agents_sim,
            episode=episode,
            total_decision_num=self.total_decision_num_sim,
            desc=f"Sim Training Epoch {episode}",
        )
        self.log_metrics("SIM_TRAIN", episode, self.metric_sim, mean_loss)
        return mean_loss

    def should_run_real_eval(self, episode):
        if self.real_eval_interval > 0:
            return episode > 0 and episode % self.real_eval_interval == 0
        return self.test_when_train

    def train(self):
        if self.load_pretrained:
            self.load_agents(self.agents_sim, self.initial_pretrained_model_dir())

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

            if self.should_run_real_eval(episode):
                self.train_test(episode)

        self.save_agents(self.agents_sim, self.model_dir, e=self.episodes)
        self.save_agents(self.agents_sim, self.model_dir)

        # Finish DR training by applying the learned policy once in the real env.
        self.load_agents(self.agents_real, self.model_dir)
        # self.load_agents(self.agents_real, self.pretrained_model_dir())
        self.run_eval_episode(
            env=self.env_real,
            metric=self.metric_real,
            agents=self.agents_real,
            desc="Final Real Run After Training",
        )
        self.log_metrics("TRAIN_REAL", self.episodes, self.metric_real, 100)

    def train_test(self, episode):
        self.load_agents(self.agents_real, self.model_dir)
        self.run_eval_episode(
            env=self.env_real,
            metric=self.metric_real,
            agents=self.agents_real,
            desc=f"Real Eval Epoch {episode}",
        )
        self.log_metrics("TEST_REAL", episode, self.metric_real, 100)
        return self.metric_real.real_average_travel_time()

    def test(self, drop_load=False):
        if not drop_load:
            self.load_agents(self.agents_real, self.transition_pretrained_model_dir())
        self.run_eval_episode(
            env=self.env_real,
            metric=self.metric_real,
            agents=self.agents_real,
            desc="Final Real Test",
        )
        self.log_metrics("TEST_REAL", 0, self.metric_real, 100)
        return self.metric_real

    def writeLog(
        self,
        mode,
        step,
        travel_time,
        loss,
        cur_rwd,
        cur_queue,
        cur_delay,
        cur_throughput,
    ):
        res = (
            self.exp_name
            + "\t"
            + mode
            + "\t"
            + str(step)
            + "\t"
            + "%.1f" % travel_time
            + "\t"
            + "%.1f" % loss
            + "\t"
            + "%.2f" % cur_rwd
            + "\t"
            + "%.2f" % cur_queue
            + "\t"
            + "%.2f" % cur_delay
            + "\t"
            + "%d" % cur_throughput
        )
        log_handle = open(self.log_file, "a")
        log_handle.write(res + "\n")
        log_handle.close()
