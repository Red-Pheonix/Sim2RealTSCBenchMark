import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from common.metrics import Metrics
from common.registry import Registry
from environment import TSCEnv
from .base import TransitionTrainer
from .distributions import NormalDistribution, UniformDistribution


@Registry.register_trainer("sim2real_transitions_domain_adaptation")
class TransitionDomainAdaptationTrainer(TransitionTrainer):
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

        self.episodes = self.trainer_args["episodes"]
        self.real_eval_interval = self.trainer_args.get("real_eval_interval", 0)
        self.sim_rollouts = self.sim2real_args.get(
            "sim_rollouts", self.sim2real_args.get("eval_episodes", 100)
        )
        self.training_iterations = self.sim2real_args.get(
            "training_iterations", self.sim2real_args.get("train_episodes", 100)
        )
        self.total_training_iterations = self.episodes * self.training_iterations

        self.method = self.sim2real_args.get("method", "domain_adaptation")
        self.domain_randomization_config = self.sim2real_args.get(
            "domain_randomization", {}
        )
        self.domain_randomization_enabled = self.domain_randomization_config.get(
            "enabled", False
        )
        self.domain_randomization_rng = np.random.default_rng(
            self.domain_randomization_config.get("seed", self.seed)
        )
        self.inference_config = self.sim2real_args.get(
            "inference",
            self.sim2real_args.get("sbi", {}),
        )
        self.posterior_estimator_name = self.inference_config.get(
            "density_estimator", "nsf"
        )
        self.min_posterior_samples = self.inference_config.get(
            "min_posterior_samples", 5
        )
        self.posterior_sample_count = 0
        self.prior_distribution = None
        self.posterior_inference = None
        self.posterior = None

        self.exp_name = self.build_exp_name(self.method)
        self.model_dir = self.build_model_dir(self.exp_name)
        self.log_file = self.build_log_file()

        self.base_cityflow_config = self.load_cityflow_config(self.cityflow_path)
        self.base_cityflow_flow = self.load_cityflow_flow(self.base_cityflow_config)
        self.domain_randomization_dir = Path(
            self.base_cityflow_config.get("dir", "data")
        ) / self.domain_randomization_config.get(
            "output_dir", "output_data/sim2real_transitions/domain_adaptation"
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

        self.initialize_posterior_inference()
        self.create()

        self.world = self.world_real
        self.agents = self.agents_real
        self.metric = self.metric_real
        self.env = self.env_real

    def initialize_posterior_inference(self):
        try:
            from sbi.inference import NPE
        except ImportError as exc:
            raise ImportError(
                "Posterior inference requires the `sbi` package, but it is not "
                "installed. Install it with `python -m pip install sbi`."
            ) from exc

        self.prior_distribution = self.build_prior_distribution()
        self.posterior_inference = NPE(
            prior=self.prior_distribution,
            density_estimator=self.posterior_estimator_name,
        )
        self.posterior = None

    def build_prior_distribution(self):
        parameter_means = []
        parameter_stds = []

        for param_name, param_config in self.domain_randomization_config.get(
            "parameters", {}
        ).items():
            distribution_name = param_config.get("distribution", "uniform")
            if distribution_name != "normal":
                raise ValueError(
                    "Domain adaptation prior expects normal-configured "
                    f"parameters, but {param_name} uses {distribution_name}."
                )

            normal_config = param_config.get("normal_config", {})
            parameter_means.append(normal_config.get("mean", 0.0))
            parameter_stds.append(normal_config.get("std", 1.0))

        if not parameter_means:
            raise ValueError(
                "Cannot build prior distribution because no domain randomization "
                "parameters were configured."
            )

        return torch.distributions.Independent(
            torch.distributions.Normal(
                loc=torch.tensor(parameter_means, dtype=torch.float32),
                scale=torch.tensor(parameter_stds, dtype=torch.float32),
            ),
            1,
        )

    def pretrained_model_dir(self):
        return os.path.join(
            "pretrained",
            "tsc",
            self.agent_name,
            self.network_name,
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

    def get_parameter_names(self):
        return list(self.domain_randomization_config.get("parameters", {}).keys())

    def apply_parameter_constraints(self, sampled_parameters):
        constrained_parameters = {}
        for param_name in self.get_parameter_names():
            param_config = self.domain_randomization_config["parameters"][param_name]
            sampled_value = float(sampled_parameters[param_name])

            clip_min = param_config.get("clip_min")
            clip_max = param_config.get("clip_max")
            if clip_min is not None:
                sampled_value = max(clip_min, sampled_value)
            if clip_max is not None:
                sampled_value = min(clip_max, sampled_value)
            if param_config.get("round", False):
                sampled_value = np.round(sampled_value)

            constrained_parameters[param_name] = float(sampled_value)

        return constrained_parameters

    def train_posterior(self, theta, obs):
        if theta.size == 0 or obs.size == 0:
            self.logger.warning(
                "Skipping posterior training because theta or summary metrics "
                "are empty."
            )
            return

        theta = torch.as_tensor(theta, dtype=torch.float32)
        obs = torch.as_tensor(obs, dtype=torch.float32)
        self.posterior_sample_count += theta.shape[0]

        self.posterior_inference.append_simulations(theta, obs)
        if self.posterior_sample_count >= self.min_posterior_samples:
            density_estimator = self.posterior_inference.train(
                resume_training=self.posterior is not None
            )
            self.posterior = self.posterior_inference.build_posterior(
                density_estimator
            )
        
        self.logger.info(
            "Trained posterior estimator with theta shape %s and summary shape %s",
            tuple(theta.shape),
            tuple(obs.shape),
        )

    def sample_posterior_parameters(self, summary_metric):
        if self.posterior is None:
            theta_sample = self.prior_distribution.sample().cpu().numpy()
        else:
            x_obs = torch.as_tensor(summary_metric, dtype=torch.float32)
            theta_sample = self.posterior.sample((1,), x=x_obs).squeeze(0).cpu().numpy()
            
        sampled_parameters = {
            param_name: float(theta_sample[idx])
            for idx, param_name in enumerate(self.get_parameter_names())
        }
        return self.apply_parameter_constraints(sampled_parameters)

    def write_episode_randomized_cityflow_config(self, episode, sampled_parameters=None):
        if sampled_parameters is None:
            sampled_parameters = self.sample_randomized_vehicle_parameters()
        else:
            sampled_parameters = self.apply_parameter_constraints(sampled_parameters)
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
                "output_dir", "output_data/sim2real_transitions/domain_adaptation"
            ))
            / flow_filename
        ).as_posix()
        with open(config_path, "w", encoding="utf-8") as file_obj:
            json.dump(cityflow_config, file_obj, indent=2)

        return config_path.as_posix(), sampled_parameters

    def reload_sim_env_for_episode(self, episode, sampled_parameters=None):
        applied_parameters = {}
        if self.domain_randomization_enabled or sampled_parameters is not None:
            config_path, sampled_parameters = self.write_episode_randomized_cityflow_config(
                episode, sampled_parameters=sampled_parameters
            )
            self.logger.info(
                "Episode %s sim randomization params: %s",
                episode,
                sampled_parameters,
            )
            applied_parameters = sampled_parameters
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
        return applied_parameters

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

    def sim_train(self, episode, real_summary_metric):
        sampled_parameters = self.sample_posterior_parameters(real_summary_metric)
        self.reload_sim_env_for_episode(episode, sampled_parameters=sampled_parameters)
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
        self.logger.info(
            "Posterior-sampled train parameters for episode %s: %s",
            episode,
            sampled_parameters,
        )
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
        self.log_metrics("SIM_EVAL", episode, self.metric_sim, 0)
        return {
            "episode": episode,
            "sampled_parameters": sampled_parameters,
            "summary_metric": summary_metric,
        }

    def real_eval(self, episode=0):
        trajectory = self.run_eval_episode(
            env=self.env_real,
            metric=self.metric_real,
            agents=self.agents_real,
            desc=f"Real Eval Epoch {episode}",
            max_steps=self.steps,
            collect_trajectory=True,
        )
        summary_metric = self.compute_summary_metric(trajectory)
        self.log_metrics("REAL_EVAL", episode, self.metric_real, 0)
        return {
            "episode": episode,
            "summary_metric": summary_metric,
        }

    def should_run_real_eval(self, episode):
        if self.real_eval_interval > 0:
            return episode > 0 and episode % self.real_eval_interval == 0
        return self.test_when_train

    def collect_sim_rollout_data(self, start_episode):
        rollout_data = []
        for rollout_idx in range(self.sim_rollouts):
            episode = start_episode + rollout_idx
            rollout_data.append(self.sim_eval(episode))
            self.logger.info(
                "sim_rollout:%s/%s",
                rollout_idx + 1,
                self.sim_rollouts,
            )
        return rollout_data

    def collect_real_summary_metric(self, episode):
        real_eval_data = self.real_eval(episode)
        real_summary_metric = real_eval_data["summary_metric"]
        self.logger.info(
            "Collected real summary metric with shape %s from rollout %s",
            tuple(real_summary_metric.shape),
            episode,
        )
        return real_summary_metric

    def train(self):
        if self.load_pretrained:
            self.load_agents(self.agents_sim, self.pretrained_model_dir())

        sim_rollout_episode = 0
        training_iteration = 0
        real_summary_metric = None
        # save model weigghts
        self.save_agents(self.agents_sim, self.model_dir)
        for episode in range(self.episodes):
            self.logger.info(
                "domain_adaptation_episode:%s/%s",
                episode,
                self.episodes,
            )

            # Sim rollout: collect randomized sim trajectories for posterior training.
            rollout_data = self.collect_sim_rollout_data(sim_rollout_episode)
            sim_rollout_episode += self.sim_rollouts

            theta = np.stack(
                [
                    np.array(
                        [
                            item["sampled_parameters"][param_name]
                            for param_name in self.get_parameter_names()
                        ]
                    )
                    for item in rollout_data
                ]
            )
            sim_summary_metrics = np.stack(
                [item["summary_metric"] for item in rollout_data]
            )
            self.train_posterior(theta, sim_summary_metrics)
            
            # Real rollout: collect real-world observations for posterior conditioning.
            self.load_agents(self.agents_real, self.model_dir)
            real_summary_metric = self.collect_real_summary_metric(episode)

            # Policy training: train in sim domains sampled from the inferred posterior.
            for _ in range(self.training_iterations):
                sim_loss = self.sim_train(training_iteration, real_summary_metric)
                self.save_agents(self.agents_sim, self.model_dir)
                if training_iteration % self.save_rate == 0:
                    self.save_agents(
                        self.agents_sim,
                        self.model_dir,
                        e=training_iteration,
                    )
                self.logger.info(
                    "training_iteration:%s/%s, sim_loss:%s",
                    training_iteration,
                    self.total_training_iterations,
                    sim_loss,
                )

                if self.should_run_real_eval(training_iteration):
                    self.train_test(training_iteration)
                training_iteration += 1

        self.save_agents(
            self.agents_sim,
            self.model_dir,
            e=self.total_training_iterations,
        )
        self.save_agents(self.agents_sim, self.model_dir)

        self.load_agents(self.agents_real, self.model_dir)
        self.run_eval_episode(
            env=self.env_real,
            metric=self.metric_real,
            agents=self.agents_real,
            desc="Final Real Run After Training",
        )
        self.log_metrics(
            "REAL_EVAL_FINAL",
            self.total_training_iterations,
            self.metric_real,
            100,
        )

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
            self.load_agents(
                self.agents_real,
                self.model_dir,
                e=self.total_training_iterations,
            )
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
        with open(self.log_file, "a") as f:
            f.write(res + "\n")
