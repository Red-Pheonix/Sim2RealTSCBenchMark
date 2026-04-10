import os
import copy

import numpy as np
from tqdm import tqdm

from common.metrics import Metrics
from common.registry import Registry
from environment import TSCEnv
from generator import LaneVehicleGenerator
from trainer.base_trainer import BaseTrainer


class ObservationNoiseGenerator:
    def sample(self, rng, base_value):
        raise NotImplementedError


class GaussianNoiseGenerator(ObservationNoiseGenerator):
    def __init__(self, config):
        self.mean = config.get("mean", 0.0)
        self.std = config.get("std", 0.0)
        self.relative = config.get("relative", False)

    def sample(self, rng, base_value):
        sampled_noise = rng.normal(self.mean, self.std)
        if self.relative:
            sampled_noise *= abs(base_value)
        return sampled_noise


class PoissonNoiseGenerator(ObservationNoiseGenerator):
    def __init__(self, config):
        self.poisson_lambda = config.get("lambda")
        self.rate = config.get("rate", 1.0)
        self.centered = config.get("centered", True)
        self.relative = config.get("relative", False)

    def sample(self, rng, base_value):
        if self.poisson_lambda is None:
            lam = max(abs(base_value) * self.rate, 0.0)
        else:
            lam = max(self.poisson_lambda, 0.0)
            if self.relative:
                lam *= abs(base_value)

        sampled_noise = rng.poisson(lam)
        if self.centered:
            sampled_noise -= lam
        return sampled_noise


class DomainRandomizationDistribution:
    def sample(self, rng):
        raise NotImplementedError


class UniformDomainRandomizationDistribution(DomainRandomizationDistribution):
    def __init__(self, config):
        self.low = config["low"]
        self.high = config["high"]

    def sample(self, rng):
        return rng.uniform(self.low, self.high)


class RandomizedObservationParameter:
    @classmethod
    def from_config(cls, name, config):
        distribution_name = config["distribution"]
        distribution_config = config.get(
            f"{distribution_name}_config", config
        )

        if distribution_name == "uniform":
            distribution = UniformDomainRandomizationDistribution(distribution_config)
        else:
            raise ValueError(
                "Unsupported observation domain randomization distribution: "
                f"{distribution_name}"
            )

        return cls(name, config["path"], distribution)

    def __init__(self, name, path, distribution):
        self.name = name
        self.path = path
        self.distribution = distribution

    def sample(self, rng):
        return self.distribution.sample(rng)

    def apply(self, config, value):
        target = config
        for key in self.path[:-1]:
            target = target[key]
        target[self.path[-1]] = value


@Registry.register_trainer("sim2real_observations")
class Sim2RealObservationsTrainer(BaseTrainer):
    """
    Trainer for observation-based sim2real experiments with separate sim and real rollouts.
    """

    def __init__(self, logger, gpu=0, cpu=False, name="sim2real_observations"):
        super().__init__(logger=logger, gpu=gpu, cpu=cpu, name=name)

        cmd_args = Registry.mapping["command_mapping"]["setting"].param
        trainer_args = Registry.mapping["trainer_mapping"]["setting"].param
        logger_args = Registry.mapping["logger_mapping"]["setting"].param

        self.cityflow_path = os.path.join(
            "configs/sim", "cityflow", cmd_args["network"] + ".cfg"
        )
        self.sumo_path = os.path.join(
            "configs/sim", "sumo", cmd_args["network"] + ".cfg"
        )
        self.episodes = trainer_args["episodes"]
        self.steps = trainer_args["steps"]
        self.test_steps = trainer_args["test_steps"]
        self.buffer_size = trainer_args["buffer_size"]
        self.action_interval = trainer_args["action_interval"]
        self.real_train_interval = trainer_args["real_train_interval"]
        self.save_rate = logger_args["save_rate"]
        self.learning_start = trainer_args["learning_start"]
        self.update_model_rate = trainer_args["update_model_rate"]
        self.update_target_rate = trainer_args["update_target_rate"]
        self.test_when_train = trainer_args["test_when_train"]
        self.yellow_time = trainer_args["yellow_length"]
        sim2real_setting = Registry.mapping.get("sim2real_mapping", {}).get("setting")
        self.sim2real_config = (
            sim2real_setting.param
            if sim2real_setting and hasattr(sim2real_setting, "param")
            else {}
        )
        self.sim_observation_config = self.sim2real_config.get("sim_config", {})
        self.real_observation_config = self.sim2real_config.get("real_config", {})
        self.obs_model_config = self.sim2real_config.get("obs_model_config", {})
        self.obs_model_name = cmd_args.get("obs_model", "default")
        self.load_pretrained = self.sim2real_config.get("load_pretrained", False)
        self.domain_randomization_config = self.obs_model_config.get(
            "domain_randomization", {}
        )
        self.domain_randomization_enabled = (
            self.obs_model_name == "domain_randomization"
            and self.domain_randomization_config.get("enabled", False)
        )
        self.randomized_observation_parameters = (
            self.build_randomized_observation_parameters()
        )
        self.sim_observation_rng = np.random.default_rng(
            self.domain_randomization_config.get(
                "seed", 0
            )
        )
        self.current_sim_observation_config = self.build_domain_randomization_config()
        self.sim_observation_transforms = self.build_observation_transforms(
            self.current_sim_observation_config
        )
        self.real_observation_transforms = self.build_observation_transforms(
            self.real_observation_config
        )

        self.exp_name = (
            f'{cmd_args["network"]}_{cmd_args["real_setting"]}_{cmd_args["agent"]}_{cmd_args.get("obs_model", "default")}'
        )
        self.model_dir = os.path.join(
            Registry.mapping["logger_mapping"]["path"].path,
            Registry.mapping["logger_mapping"]["setting"].param["model_dir"],
            self.exp_name,
        )

        self.dataset = Registry.mapping["dataset_mapping"][cmd_args["dataset"]](
            os.path.join(
                Registry.mapping["logger_mapping"]["path"].path,
                logger_args["data_dir"],
            )
        )
        self.dataset.initiate(ep=self.episodes, step=self.steps, interval=self.action_interval)

        base_log_name = os.path.basename(self.logger.handlers[-1].baseFilename).rstrip(
            "_BRF.log"
        )
        self.log_file = os.path.join(
            Registry.mapping["logger_mapping"]["path"].path,
            logger_args["log_dir"],
            base_log_name + "_DTL.log",
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
        self.total_decision_num_real = 0

        self.create()

        # Preserve the base trainer's single-world attributes for compatibility.
        self.world = self.world_real
        self.agents = self.agents_real
        self.metric = self.metric_real
        self.env = self.env_real

    def build_domain_randomization_config(self):
        sim_config = copy.deepcopy(self.sim_observation_config)

        if not self.domain_randomization_enabled:
            return sim_config

        sim_config["detection_zone"]["enabled"] = True
        sim_config["noise"]["enabled"] = True
        sim_config["disable_sensor"]["enabled"] = True
        sim_config["disable_sensor"]["intersection_level"]["enabled"] = True

        for parameter in self.randomized_observation_parameters:
            sampled_value = parameter.sample(self.sim_observation_rng)
            parameter.apply(sim_config, sampled_value)
        return sim_config

    def build_randomized_observation_parameters(self):
        randomized_parameters = []
        for parameter_name, parameter_config in self.domain_randomization_config.get(
            "parameters", {}
        ).items():
            randomized_parameters.append(
                RandomizedObservationParameter.from_config(
                    parameter_name, parameter_config
                )
            )
        return randomized_parameters

    def build_observation_transforms(self, observation_config):
        base_seed = observation_config["transform_seed"]
        transforms = []

        noise_config = observation_config["noise"]
        if noise_config["enabled"]:
            transforms.append(self.make_noise_transform(noise_config, base_seed))

        disable_sensor_config = observation_config["disable_sensor"]
        if disable_sensor_config["enabled"]:
            transforms.append(
                self.make_disable_sensor_transform(disable_sensor_config, base_seed)
            )

        return transforms

    def reset_observation_transforms(self, world):
        transforms = getattr(world, "observation_transforms", [])
        for transform in transforms:
            reset_fn = getattr(transform, "reset", None)
            if callable(reset_fn):
                reset_fn()

    def make_noise_transform(self, noise_config, base_seed):
        rng = np.random.default_rng(noise_config.get("seed", base_seed))
        feature_names = set(noise_config["features"])
        noise_generator = self.build_noise_generator(noise_config)

        def transform(fn_name, values, intersection=None, lanes=None, meta=None):
            if feature_names and fn_name not in feature_names:
                return values

            transformed = dict(values)
            probability = noise_config["probability"]
            bias = noise_config["bias"]
            scale = noise_config["scale"]
            clip_min = noise_config.get("clip_min")
            clip_max = noise_config.get("clip_max")
            round_values = noise_config["round"]
            cast_int = noise_config["cast_int"]

            for lane_group in lanes or []:
                for lane_id in lane_group:
                    if lane_id not in transformed or rng.random() > probability:
                        continue

                    base_value = transformed[lane_id]
                    sampled_noise = noise_generator.sample(rng, base_value)

                    noisy_value = (base_value + sampled_noise + bias) * scale

                    if clip_min is not None:
                        noisy_value = max(clip_min, noisy_value)
                    if clip_max is not None:
                        noisy_value = min(clip_max, noisy_value)
                    if round_values:
                        noisy_value = np.round(noisy_value)
                    if cast_int:
                        noisy_value = int(noisy_value)

                    transformed[lane_id] = noisy_value

            return transformed

        return transform

    def build_noise_generator(self, noise_config):
        distribution = noise_config["distribution"]
        distribution_config = noise_config[f"{distribution}_config"]

        if distribution == "gaussian":
            return GaussianNoiseGenerator(distribution_config)
        if distribution == "poisson":
            return PoissonNoiseGenerator(distribution_config)

        raise ValueError(f"Unsupported noise distribution: {distribution}")

    def make_disable_sensor_transform(self, disable_sensor_config, base_seed):
        rng = np.random.default_rng(disable_sensor_config.get("seed", base_seed))
        feature_names = set(disable_sensor_config["features"])
        fill_value = disable_sensor_config["fill_value"]
        mask_mode = disable_sensor_config["mask_mode"]
        lane_level_config = disable_sensor_config["lane_level"]
        intersection_level_config = disable_sensor_config["intersection_level"]
        apply_lane_mask = lane_level_config["enabled"]
        apply_intersection_mask = intersection_level_config["enabled"]
        lane_probability = lane_level_config["probability"]
        intersection_probability = intersection_level_config["probability"]
        mask_state = {}
        intersection_mask_state = {}

        def lane_is_masked(lane_id):
            if mask_mode in {"run", "episode"}:
                if lane_id not in mask_state:
                    mask_state[lane_id] = rng.random() < lane_probability
                return mask_state[lane_id]
            return rng.random() < lane_probability

        def intersection_is_masked(intersection):
            intersection_id = intersection.id

            if mask_mode in {"run", "episode"}:
                if intersection_id not in intersection_mask_state:
                    intersection_mask_state[intersection_id] = (
                        rng.random() < intersection_probability
                    )
                return intersection_mask_state[intersection_id]

            return rng.random() < intersection_probability

        def transform(fn_name, values, intersection=None, lanes=None, meta=None):
            if feature_names and fn_name not in feature_names:
                return values

            transformed = dict(values)
            if apply_intersection_mask and intersection_is_masked(intersection):
                for lane_group in lanes or []:
                    for lane_id in lane_group:
                        if lane_id in transformed:
                            transformed[lane_id] = fill_value
                return transformed

            if not apply_lane_mask:
                return transformed

            for lane_group in lanes or []:
                for lane_id in lane_group:
                    if lane_id in transformed and lane_is_masked(lane_id):
                        transformed[lane_id] = fill_value
            return transformed

        def reset():
            if mask_mode != "episode":
                return
            mask_state.clear()
            intersection_mask_state.clear()

        transform.reset = reset
        return transform

    def build_world_kwargs(self, observation_transforms=None, include_interface=True):
        world_kwargs = {}
        if include_interface:
            world_kwargs["interface"] = Registry.mapping["command_mapping"]["setting"].param[
                "interface"
            ]
        if observation_transforms is not None:
            world_kwargs["observation_transforms"] = observation_transforms
        return world_kwargs

    def create_world(self):
        thread_num = Registry.mapping["command_mapping"]["setting"].param["thread_num"]
        detection_zone_config = self.real_observation_config["detection_zone"]
        detection_zone_m = 0.0
        if detection_zone_config["enabled"]:
            detection_zone_m = float(detection_zone_config["distance_m"])

        self.world_sim = Registry.mapping["world_mapping"]["cityflow"](
            self.cityflow_path,
            thread_num,
            **self.build_world_kwargs(
                observation_transforms=(
                    self.sim_observation_transforms
                    if self.domain_randomization_enabled
                    else []
                ),
                include_interface=False,
            ),
        )

        self.world_real = Registry.mapping["world_mapping"]["cityflow"](
            self.cityflow_path,
            thread_num,
            **self.build_world_kwargs(
                observation_transforms=self.real_observation_transforms,
                include_interface=False,
            ),
            detection_zone_m=detection_zone_m,
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

        if Registry.mapping["model_mapping"]["setting"].param["name"] == "magd":
            for ag in agents:
                ag.link_agents(agents)

        return agents

    def pretrained_model_dir(self):
        return os.path.join(
            "pretrained",
            "tsc",
            Registry.mapping["command_mapping"]["setting"].param["agent"],
            Registry.mapping["command_mapping"]["setting"].param["network"],
        )

    def create_agents(self):
        self.agents_sim = self.create_agent_world(self.world_sim)
        self.agents_real = self.create_agent_world(self.world_real)
        if self.domain_randomization_enabled:
            for ag in self.agents_sim:
                self.configure_observation_generator(
                    ag, self.current_sim_observation_config
                )
        for ag in self.agents_real:
            self.configure_observation_generator(ag, self.real_observation_config)

        if Registry.mapping["model_mapping"]["setting"].param["load_model"]:
            self.load_agents(self.agents_sim, self.model_dir)
            self.load_agents(self.agents_real, self.model_dir)

    def configure_observation_generator(self, agent, observation_config):
        detection_zone_config = observation_config["detection_zone"]

        ob_generator = getattr(agent, "ob_generator", None)
        if not isinstance(ob_generator, LaneVehicleGenerator):
            return

        detection_zone_m = 0.0
        if detection_zone_config["enabled"]:
            detection_zone_m = float(detection_zone_config["distance_m"])

        agent.world.detection_zone_m = detection_zone_m

        agent.ob_generator = LaneVehicleGenerator(
            agent.world,
            ob_generator.I,
            ob_generator.fns,
            in_only=ob_generator.in_only,
            average=ob_generator.average,
            negative=ob_generator.negative,
            detection_zone_m=detection_zone_m,
        )

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

                # if total_decision_num > self.learning_start:
                # 
                actions = []
                for idx, ag in enumerate(agents):
                    actions.append(ag.get_action(last_obs[idx], last_phase[idx], test=True))
                actions = np.stack(actions)
                # else:
                    # actions = np.stack([ag.sample() for ag in agents])

                actions_prob = []
                for idx, ag in enumerate(agents):
                    actions_prob.append(ag.get_action_prob(last_obs[idx], last_phase[idx]))

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

    def run_eval_episode(self, *, env, metric, world, agents, desc):
        metric.clear()
        for agent in agents:
            if env is self.env_sim and self.domain_randomization_enabled:
                self.configure_observation_generator(
                    agent, self.current_sim_observation_config
                )
            elif env is self.env_real:
                self.configure_observation_generator(agent, self.real_observation_config)
            agent.reset()
        self.reset_observation_transforms(world)
        obs = env.reset()

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

    def real_train(self, episode):
        self.load_agents(self.agents_real, self.model_dir)
        self.set_replay(
            self.env_real,
            f"real_episode_{episode}.txt",
            episode % self.save_rate == 0,
        )
        self.total_decision_num_real, mean_loss, steps_run = self.run_train_episode(
            env=self.env_real,
            metric=self.metric_real,
            world=self.world_real,
            agents=self.agents_real,
            episode=episode,
            total_decision_num=self.total_decision_num_real,
            desc=f"Real Training Epoch {episode}",
        )
        self.log_metrics("REAL_TRAIN", episode, self.metric_real, mean_loss)
        self.logger.info("real step:%s/%s", steps_run, self.steps)
        return mean_loss

    def train(self):
        
        if self.load_pretrained:
            pretrained_dir = self.pretrained_model_dir()
            self.load_agents(self.agents_sim, pretrained_dir)
            self.load_agents(self.agents_real, pretrained_dir)

        for episode in range(self.episodes):
            sim_loss = self.sim_train(episode)
            self.save_agents(self.agents_sim, self.model_dir)

            if self.real_train_interval > 0 and episode % self.real_train_interval == 0:
                real_loss = self.real_train(episode)
                self.save_agents(self.agents_real, self.model_dir)

                if episode % self.save_rate == 0:
                    self.save_agents(self.agents_real, self.model_dir, e=episode)

                self.logger.info(
                    "episode:%s/%s, sim_loss:%s, real_loss:%s",
                    episode,
                    self.episodes,
                    sim_loss,
                    real_loss,
                )

            else:
                if episode % self.save_rate == 0:
                    self.save_agents(self.agents_sim, self.model_dir, e=episode)

                self.logger.info(
                    "episode:%s/%s, sim_loss:%s, real_loss:skipped",
                    episode,
                    self.episodes,
                    sim_loss,
                )

            if self.test_when_train:
                self.train_test(episode)

        self.save_agents(self.agents_real, self.model_dir, e=self.episodes)
        self.save_agents(self.agents_real, self.model_dir)

    def train_test(self, episode):
        self.load_agents(self.agents_sim, self.model_dir)
        self.run_eval_episode(
            env=self.env_sim,
            metric=self.metric_sim,
            world=self.world_sim,
            agents=self.agents_sim,
            desc=f"Sim Eval Epoch {episode}",
        )
        self.log_metrics("TEST_SIM", episode, self.metric_sim, 100)

        self.load_agents(self.agents_real, self.model_dir)
        self.run_eval_episode(
            env=self.env_real,
            metric=self.metric_real,
            world=self.world_real,
            agents=self.agents_real,
            desc=f"Real Eval Epoch {episode}",
        )
        self.log_metrics("TEST_REAL", episode, self.metric_real, 100)
        return self.metric_real.real_average_travel_time()

    def test(self, drop_load=False):
        if not drop_load:
            self.load_agents(self.agents_sim, self.model_dir, e=self.episodes)
        self.set_replay(self.env_sim, "final_sim.txt", True)
        self.run_eval_episode(
            env=self.env_sim,
            metric=self.metric_sim,
            world=self.world_sim,
            agents=self.agents_sim,
            desc="Final Sim Test",
        )
        self.log_metrics("FINAL_TEST_SIM", 0, self.metric_sim, 100)

        if not drop_load:
            self.load_agents(self.agents_real, self.model_dir, e=self.episodes)
        self.set_replay(self.env_real, "final_real.txt", True)
        self.run_eval_episode(
            env=self.env_real,
            metric=self.metric_real,
            world=self.world_real,
            agents=self.agents_real,
            desc="Final Real Test",
        )
        self.log_metrics("FINAL_TEST_REAL", 0, self.metric_real, 100)
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
