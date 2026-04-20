import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common.metrics import Metrics
from common.registry import Registry
from environment import TSCEnv
from trainer.base_trainer import BaseTrainer


class DomainRandomizationDistribution:
    def sample(self, rng):
        raise NotImplementedError


class UniformDomainRandomizationDistribution(DomainRandomizationDistribution):
    def __init__(self, config):
        self.low = config["low"]
        self.high = config["high"]

    def sample(self, rng):
        return rng.uniform(self.low, self.high)


class NormalDomainRandomizationDistribution(DomainRandomizationDistribution):
    def __init__(self, config):
        self.mean = config.get("mean", 0.0)
        self.std = config.get("std", 1.0)

    def sample(self, rng):
        return rng.normal(self.mean, self.std)


@Registry.register_trainer("sim2real_transitions_domain_adaptation")
class Sim2RealTransitionsDomainAdaptationTrainer(BaseTrainer):
    """
    Run a two-stage sim loop for transition domain adaptation:
    sim evaluation episodes first, then sim training episodes, followed by
    real-environment evaluation.
    """

    def __init__(
        self,
        logger,
        gpu=0,
        cpu=False,
        name="sim2real_transitions_domain_adaptation",
    ):
        super().__init__(logger=logger, gpu=gpu, cpu=cpu, name=name)

        cmd_args = Registry.mapping["command_mapping"]["setting"].param
        trainer_args = Registry.mapping["trainer_mapping"]["setting"].param
        logger_args = Registry.mapping["logger_mapping"]["setting"].param
        sim2real_args = Registry.mapping["sim2real_mapping"]["setting"].param

        self.cityflow_path = os.path.join(
            "configs/sim", "cityflow", cmd_args["network"] + ".cfg"
        )
        self.sumo_path = os.path.join(
            "configs/sim", "sumo", cmd_args["network"] + ".cfg"
        )
        self.steps = trainer_args["steps"]
        self.test_steps = trainer_args["test_steps"]
        self.buffer_size = trainer_args["buffer_size"]
        self.action_interval = trainer_args["action_interval"]
        self.save_rate = logger_args["save_rate"]
        self.learning_start = trainer_args["learning_start"]
        self.update_model_rate = trainer_args["update_model_rate"]
        self.update_target_rate = trainer_args["update_target_rate"]
        self.test_when_train = trainer_args["test_when_train"]
        self.real_eval_interval = trainer_args.get("real_eval_interval", 0)
        self.yellow_time = trainer_args["yellow_length"]
        self.load_pretrained = sim2real_args.get("load_pretrained", False)
        self.eval_episodes = sim2real_args.get("eval_episodes", 100)
        self.train_episodes = sim2real_args.get("train_episodes", 100)

        self.method = sim2real_args.get("method", "domain_adaptation")
        self.domain_randomization_config = sim2real_args.get(
            "domain_randomization", {}
        )
        self.domain_randomization_enabled = self.domain_randomization_config.get(
            "enabled", False
        )
        self.domain_randomization_rng = np.random.default_rng(
            self.domain_randomization_config.get("seed", self.seed)
        )

        self.exp_name = (
            f'{cmd_args["network"]}_{cmd_args["real_setting"]}_{cmd_args["agent"]}_{self.method}'
        )
        self.model_dir = os.path.join(
            Registry.mapping["logger_mapping"]["path"].path,
            logger_args["model_dir"],
            self.exp_name,
        )
        base_log_name = os.path.basename(self.logger.handlers[-1].baseFilename).rstrip(
            "_BRF.log"
        )
        self.log_file = os.path.join(
            Registry.mapping["logger_mapping"]["path"].path,
            logger_args["log_dir"],
            base_log_name + "_DTL.log",
        )

        self.base_cityflow_config = self.load_cityflow_config(self.cityflow_path)
        self.base_cityflow_flow = self.load_cityflow_flow(self.base_cityflow_config)
        self.domain_randomization_dir = Path(
            self.base_cityflow_config.get("dir", "data")
        ) / self.domain_randomization_config.get(
            "output_dir", "output_data/sim2real_transitions/domain_randomization"
        )

        self.world_sim = None
        self.world_real = None
        self.agents_sim = None
        self.agents_real = None
        self.metric_sim = None
        self.metric_real = None
        self.env_sim = None
        self.env_real = None
        self.total_decision_num_sim = 0
        self.eval_data = []

        self.create()

        self.world = self.world_real
        self.agents = self.agents_real
        self.metric = self.metric_real
        self.env = self.env_real

    def load_cityflow_config(self, cityflow_path):
        with open(cityflow_path, "r", encoding="utf-8") as file_obj:
            return json.load(file_obj)

    def load_cityflow_flow(self, cityflow_config):
        flow_path = Path(cityflow_config.get("dir", "data")) / cityflow_config["flowFile"]
        with open(flow_path, "r", encoding="utf-8") as file_obj:
            return json.load(file_obj)

    def create_world(self):
        thread_num = Registry.mapping["command_mapping"]["setting"].param["thread_num"]
        self.world_sim = Registry.mapping["world_mapping"]["cityflow"](
            self.cityflow_path,
            thread_num,
        )

        sumo_add = Registry.mapping["command_mapping"]["setting"].param.get(
            "real_setting"
        )
        if sumo_add != "default":
            sumo_add = sumo_add + ".add.xml"
        else:
            sumo_add = None

        self.world_real = Registry.mapping["world_mapping"]["sumo"](
            self.sumo_path,
            interface=Registry.mapping["command_mapping"]["setting"].param["interface"],
            sumo_add=sumo_add,
        )

    def create_agent_world(self, world):
        agents = []
        agent = Registry.mapping["model_mapping"][
            Registry.mapping["command_mapping"]["setting"].param["agent"]
        ](world, 0)
        num_agent = int(len(world.intersections) / agent.sub_agents)
        agents.append(agent)
        for i in range(1, num_agent):
            agents.append(
                Registry.mapping["model_mapping"][
                    Registry.mapping["command_mapping"]["setting"].param["agent"]
                ](world, i)
            )
        return agents

    def create_agents(self):
        self.agents_sim = self.create_agent_world(self.world_sim)
        self.agents_real = self.create_agent_world(self.world_real)

    def create_metrics(self):
        if Registry.mapping["command_mapping"]["setting"].param["delay_type"] == "apx":
            lane_metrics = ["rewards", "queue", "delay"]
            world_metrics = ["real avg travel time", "throughput"]
        else:
            lane_metrics = ["rewards", "queue"]
            world_metrics = ["delay", "real avg travel time", "throughput"]
        self.metric_sim = Metrics(
            lane_metrics, world_metrics, self.world_sim, self.agents_sim
        )
        self.metric_real = Metrics(
            lane_metrics, world_metrics, self.world_real, self.agents_real
        )

    def create_env(self):
        self.env_sim = TSCEnv(self.world_sim, self.agents_sim, self.metric_sim)
        self.env_real = TSCEnv(self.world_real, self.agents_real, self.metric_real)

    def pretrained_model_dir(self):
        return os.path.join(
            "pretrained",
            "tsc",
            Registry.mapping["command_mapping"]["setting"].param["agent"],
            Registry.mapping["command_mapping"]["setting"].param["network"],
        )

    def load_agents(self, agents, model_dir, e=None):
        for ag in agents:
            ag.load_model(model_dir, e)

    def save_agents(self, agents, model_dir, e=None):
        for ag in agents:
            ag.save_model(model_dir, e)

    def set_replay(self, env, suffix, enabled):
        if not self.save_replay or env is not self.env_sim:
            return
        env.eng.set_save_replay(enabled)
        if enabled:
            env.eng.set_replay_file(os.path.join(self.replay_file_dir, suffix))

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
            return UniformDomainRandomizationDistribution(distribution_config)
        if distribution_name == "normal":
            return NormalDomainRandomizationDistribution(distribution_config)

        raise ValueError(
            "Unsupported domain randomization distribution: "
            f"{distribution_name}"
        )

    def write_episode_randomized_cityflow_config(self, episode):
        sampled_parameters = self.sample_randomized_vehicle_parameters()
        randomized_flow = json.loads(json.dumps(self.base_cityflow_flow))
        for record in randomized_flow:
            vehicle_config = record.setdefault("vehicle", {})
            for param_name, sampled_value in sampled_parameters.items():
                vehicle_config[param_name] = sampled_value

        self.domain_randomization_dir.mkdir(parents=True, exist_ok=True)
        flow_filename = f"temp_flow.json"
        config_filename = f"temp.cfg"
        flow_path = self.domain_randomization_dir / flow_filename
        config_path = self.domain_randomization_dir / config_filename

        with open(flow_path, "w", encoding="utf-8") as file_obj:
            json.dump(randomized_flow, file_obj)

        cityflow_config = dict(self.base_cityflow_config)
        cityflow_config["flowFile"] = (
            Path(self.domain_randomization_config.get(
                "output_dir", "output_data/sim2real_transitions/domain_randomization"
            ))
            / flow_filename
        ).as_posix()
        with open(config_path, "w", encoding="utf-8") as file_obj:
            json.dump(cityflow_config, file_obj, indent=2)

        return config_path.as_posix(), sampled_parameters

    def reload_sim_env_for_episode(self, episode):
        sampled_parameters = {}
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
        return sampled_parameters

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

    def run_eval_episode(
        self,
        *,
        env,
        metric,
        agents,
        desc,
        max_steps=None,
        collect_trajectory=False,
    ):
        metric.clear()
        obs = env.reset()
        for agent in agents:
            agent.reset()

        if max_steps is None:
            max_steps = self.test_steps

        trajectory = []
        i = 0
        dones = [False] * len(agents)
        pbar = tqdm(total=int(max_steps / self.action_interval), desc=desc)

        while i < max_steps:
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
                if collect_trajectory:
                    trajectory.append((obs, actions))

            if all(dones):
                break

        pbar.close()
        if collect_trajectory:
            return trajectory
        return None

    def compute_summary_metric(self, trajectory):
        # adapted from here: https://github.com/rafaelpossas/bayes_sim/src/data/franka_data_generator.py:line 75
        if not trajectory:
            return np.zeros((0,))

        joint_states = [
            np.concatenate(
                [np.asarray(agent_state).reshape(-1) for agent_state in state], axis=0
            )
            for state, _ in trajectory
        ]
        joint_actions = [
            np.concatenate(
                [np.asarray(agent_action).reshape(-1) for agent_action in action],
                axis=0,
            )
            for _, action in trajectory
        ]
        

        state_dim = joint_states[0].shape[0]
        action_dim = joint_actions[0].shape[0]
        n_steps = len(joint_states)

        if n_steps <= 1:
            return np.zeros((state_dim * action_dim + 2 * state_dim,))

        # state_terms = np.stack(joint_states[1:], axis=0)
        state_terms = np.stack(joint_states[1:], axis=0) - np.stack(
            joint_states[:-1], axis=0
        )
        action_terms = np.stack(joint_actions[:-1], axis=0)
        sample = np.zeros((state_dim, action_dim))

        for i in range(state_dim):
            for j in range(action_dim):
                sample[i, j] = (
                    np.dot(state_terms[:, i], action_terms[:, j]) / (n_steps - 1)
                )

        sample = sample.reshape(-1)
        sample = np.append(sample, np.mean(state_terms, axis=0))
        sample = np.append(sample, np.std(state_terms.astype(np.float64), axis=0))

        return sample

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

    def sim_eval(self, episode):
        sampled_parameters = self.reload_sim_env_for_episode(episode)
        self.set_replay(
            self.env_sim,
            f"sim_eval_episode_{episode}.txt",
            episode % self.save_rate == 0,
        )
        trajectory = self.run_eval_episode(
            env=self.env_sim,
            metric=self.metric_sim,
            agents=self.agents_sim,
            desc=f"Sim Eval Epoch {episode}",
            max_steps=self.steps,
            collect_trajectory=True,
        )
        summary_metric = self.compute_summary_metric(trajectory)
        self.eval_data.append(
            {
                "episode": episode,
                "sampled_parameters": sampled_parameters,
                "summary_metric": summary_metric,
            }
        )
        self.log_metrics("SIM_EVAL", episode, self.metric_sim, 0)

    def should_run_real_eval(self, episode):
        if self.real_eval_interval > 0:
            return episode > 0 and episode % self.real_eval_interval == 0
        return self.test_when_train

    def train(self):
        if self.load_pretrained:
            self.load_agents(self.agents_sim, self.pretrained_model_dir())

        for episode in range(self.eval_episodes):
            self.sim_eval(episode)
            self.logger.info(
                "sim_eval_episode:%s/%s",
                episode,
                self.eval_episodes,
            )

        for episode in range(self.train_episodes):
            sim_loss = self.sim_train(episode)
            self.save_agents(self.agents_sim, self.model_dir)
            if episode % self.save_rate == 0:
                self.save_agents(self.agents_sim, self.model_dir, e=episode)
            self.logger.info(
                "sim_train_episode:%s/%s, sim_loss:%s",
                episode,
                self.train_episodes,
                sim_loss,
            )

            if self.should_run_real_eval(episode):
                self.train_test(episode)

        self.save_agents(self.agents_sim, self.model_dir, e=self.train_episodes)
        self.save_agents(self.agents_sim, self.model_dir)

        self.load_agents(self.agents_real, self.model_dir)
        self.run_eval_episode(
            env=self.env_real,
            metric=self.metric_real,
            agents=self.agents_real,
            desc="Final Real Run After Training",
        )
        self.log_metrics("TRAIN_REAL", self.train_episodes, self.metric_real, 100)

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
            self.load_agents(self.agents_real, self.model_dir, e=self.train_episodes)
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
