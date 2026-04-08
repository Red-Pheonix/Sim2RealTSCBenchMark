import os

import numpy as np
from tqdm import tqdm

from common.metrics import Metrics
from common.registry import Registry
from environment import TSCEnv
from trainer.base_trainer import BaseTrainer


@Registry.register_trainer("sim2real_rewards")
class Sim2RealRewardsTrainer(BaseTrainer):
    """
    Trainer for reward-based sim2real experiments with CityFlow sim rollouts and SUMO real rollouts.
    """

    def __init__(self, logger, gpu=0, cpu=False, name="sim2real_rewards"):
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
        self.save_rate = logger_args["save_rate"]
        self.learning_start = trainer_args["learning_start"]
        self.update_model_rate = trainer_args["update_model_rate"]
        self.update_target_rate = trainer_args["update_target_rate"]
        self.test_when_train = trainer_args["test_when_train"]
        self.yellow_time = trainer_args["yellow_length"]
        sim2real_config = self.get_sim2real_config()
        self.load_pretrained = sim2real_config.get("load_pretrained", False)
        self.detection_zone_m = sim2real_config.get("detection_zone_m", 0.0)

        self.exp_name = (
            f'{cmd_args["network"]}_{cmd_args["real_setting"]}_{cmd_args["agent"]}'
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
        self.dataset.initiate(
            ep=self.episodes, step=self.steps, interval=self.action_interval
        )

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

        self.world = self.world_real
        self.agents = self.agents_real
        self.metric = self.metric_real
        self.env = self.env_real

    def get_sim2real_config(self):
        sim2real_setting = Registry.mapping.get("sim2real_mapping", {}).get("setting")
        if sim2real_setting and hasattr(sim2real_setting, "param"):
            return sim2real_setting.param
        return {}

    def _build_world_kwargs(self):
        return {
            "interface": Registry.mapping["command_mapping"]["setting"].param[
                "interface"
            ]
        }

    def create_world(self):
        thread_num = Registry.mapping["command_mapping"]["setting"].param["thread_num"]
        interface = Registry.mapping["command_mapping"]["setting"].param["interface"]

        self.world_sim = Registry.mapping["world_mapping"]["cityflow"](
            self.cityflow_path,
            thread_num,
            detection_zone_m=self.detection_zone_m,
        )

        self.world_real = Registry.mapping["world_mapping"]["sumo"](
            self.sumo_path,
            **{
                "interface": interface,
                "detection_zone_m": self.detection_zone_m,
            },
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
        for ag in self.agents_real:
            self.configure_real_reward_generator(ag)

        if Registry.mapping["model_mapping"]["setting"].param["load_model"]:
            self.load_agents(self.agents_sim, self.model_dir)
            self.load_agents(self.agents_real, self.model_dir)

    def configure_real_reward_generator(self, agent):
        reward_generator = getattr(agent, "reward_generator", None)
        if reward_generator is not None:
            self.replace_reward_generator_fns(reward_generator)

    def replace_reward_generator_fns(self, reward_generator):
        if hasattr(reward_generator, "fns"):
            fn_map = {
                "pressure": "detected_pressure",
                "lane_waiting_count": "detected_lane_waiting_count",
                "lane_count": "detected_lane_count",
            }
            reward_generator.fns = [fn_map.get(fn, fn) for fn in reward_generator.fns]
            reward_generator.world.subscribe(reward_generator.fns)
            return

        if isinstance(reward_generator, list):
            for item in reward_generator:
                if isinstance(item, tuple) and len(item) > 1:
                    self.replace_reward_generator_fns(item[1])
                else:
                    self.replace_reward_generator_fns(item)

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
        return

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
            if env is self.env_real:
                self.configure_real_reward_generator(agent)

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
                actions = []
                for idx, ag in enumerate(agents):
                    actions.append(
                        ag.get_action(last_obs[idx], last_phase[idx], test=False)
                    )
                actions = np.stack(actions)
                # else:
                    # actions = np.stack([ag.sample() for ag in agents])

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
            if env is self.env_real:
                self.configure_real_reward_generator(agent)

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
        self.total_decision_num_sim, mean_loss, steps_run = self.run_train_episode(
            env=self.env_sim,
            metric=self.metric_sim,
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
        self.total_decision_num_real, mean_loss, steps_run = self.run_train_episode(
            env=self.env_real,
            metric=self.metric_real,
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
            # self.load_agents(self.agents_real, pretrained_dir)

        for episode in range(self.episodes):
            sim_loss = self.sim_train(episode)
            # sim_loss = self.sim_train(episode)
            # sim_loss = self.sim_train(episode)
            # sim_loss = self.sim_train(episode)
            # sim_loss = self.sim_train(episode)
            self.save_agents(self.agents_sim, self.model_dir)

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

            if self.test_when_train:
                self.train_test(episode)

        self.save_agents(self.agents_real, self.model_dir, e=self.episodes)
        self.save_agents(self.agents_real, self.model_dir)

    def train_test(self, episode):
        self.load_agents(self.agents_sim, self.model_dir)
        self.run_eval_episode(
            env=self.env_sim,
            metric=self.metric_sim,
            agents=self.agents_sim,
            desc=f"Sim Eval Epoch {episode}",
        )
        self.log_metrics("TEST_SIM", episode, self.metric_sim, 100)

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
            self.load_agents(self.agents_sim, self.model_dir, e=self.episodes)
        self.run_eval_episode(
            env=self.env_sim,
            metric=self.metric_sim,
            agents=self.agents_sim,
            desc="Final Sim Test",
        )
        self.log_metrics("FINAL_TEST_SIM", 0, self.metric_sim, 100)

        if not drop_load:
            self.load_agents(self.agents_real, self.model_dir, e=self.episodes)
        self.run_eval_episode(
            env=self.env_real,
            metric=self.metric_real,
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
