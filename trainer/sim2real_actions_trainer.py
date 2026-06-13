import os
from collections import deque

import numpy as np
from tqdm import tqdm

from common.metrics import Metrics
from common.registry import Registry
from environment import TSCEnv
from trainer.base_trainer import BaseTrainer


class ConstantActionDelay:
    def __init__(self, delay):
        self.delay = int(delay)

    def sample(self, num_agents):
        return np.full(num_agents, self.delay, dtype=int)


@Registry.register_trainer("sim2real_actions")
class Sim2RealActionsTrainer(BaseTrainer):
    """
    Trainer for action-based sim2real experiments with separate sim and real rollouts.
    """

    def __init__(self, logger, gpu=0, cpu=False, name="sim2real_actions"):
        super().__init__(logger=logger, gpu=gpu, cpu=cpu, name=name)

        cmd_args = Registry.mapping["command_mapping"]["setting"].param
        trainer_args = Registry.mapping["trainer_mapping"]["setting"].param
        logger_args = Registry.mapping["logger_mapping"]["setting"].param

        # Sim and real may use different simulators (cross-simulator transfer,
        # e.g. sim cityflow -> real sumo). real_world defaults to the sim world.
        self.sim_world_name = cmd_args["world"]
        self.real_world_name = cmd_args.get("real_world") or cmd_args["world"]
        self.path = os.path.join(
            "configs/sim", self.sim_world_name, cmd_args["network"] + ".cfg"
        )
        self.real_path = os.path.join(
            "configs/sim", self.real_world_name, cmd_args["network"] + ".cfg"
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
        # Periodic in-training real eval, mirroring the observation trainers:
        # every `real_eval_interval` episodes run train_test (sim + real eval).
        # 0 disables it.
        self.real_eval_interval = trainer_args.get("real_eval_interval", 0)
        # Whether to also run the sim-delay eval (TEST_SIM) in train_test/test.
        # The experiment only needs real-delay transfer, so this can be turned off.
        self.eval_sim = trainer_args.get("eval_sim", True)
        self.yellow_time = trainer_args["yellow_length"]
        sim2real_config = self.get_sim2real_config()
        self.load_pretrained = sim2real_config.get("load_pretrained", False)
        # Two independent execution delays:
        #   sim_action_delay  -- the delay the agent is trained under (and, for a
        #                        mitigation method, the delay it compensates for).
        #   real_action_delay -- the delay the deployment ("real") env actually
        #                        applies. The sim->real gap is the mismatch between
        #                        these two. Default sim delay 0 = naive-baseline
        #                        behavior (train delay-free, get hit by real delay).
        self.sim_action_delay = ConstantActionDelay(
            sim2real_config.get("sim_action_delay", 0)
        )
        self.real_action_delay = ConstantActionDelay(
            sim2real_config.get("action_delay", 0)
        )

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
        world_mapping = Registry.mapping["world_mapping"]
        thread_num = Registry.mapping["command_mapping"]["setting"].param["thread_num"]

        # Both cityflow and sumo World classes take (config_path, thread/placeholder,
        # **kwargs) and read the `interface` kwarg, so the call is identical for
        # either simulator -- only the class and config path differ.
        self.world_sim = world_mapping[self.sim_world_name](
            self.path,
            thread_num,
            **self._build_world_kwargs(),
        )
        self.world_real = world_mapping[self.real_world_name](
            self.real_path,
            thread_num,
            **self._build_world_kwargs(),
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

        if Registry.mapping["model_mapping"]["setting"].param["load_model"]:
            self.load_agents(self.agents_sim, self.model_dir)
            self.load_agents(self.agents_real, self.model_dir)

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

    def initialize_action_delay_state(self, agents, initial_actions):
        return {
            "env_step_idx": 0,
            "queues": [deque() for _ in agents],
            "last_actions": np.array(initial_actions, copy=True),
        }

    def enqueue_delayed_actions(self, proposed_actions, delay_state, action_delay):
        action_delays = action_delay.sample(len(proposed_actions))
        for idx, action in enumerate(np.asarray(proposed_actions)):
            sampled_delay = int(action_delays[idx])
            queue = delay_state["queues"][idx]
            if queue:
                # If there are already actions in the queue, the new action's 
                # release time should be based on the last action in the 
                # queue to ensure proper ordering.
                candidate_release = queue[-1][0] + sampled_delay
            else:
                candidate_release = delay_state["env_step_idx"] + sampled_delay
            queue.append((candidate_release, action))

    def resolve_delayed_actions(self, delay_state):
        executed_actions = np.array(delay_state["last_actions"], copy=True)

        for idx, queue in enumerate(delay_state["queues"]):
            while queue and queue[0][0] <= delay_state["env_step_idx"]:
                _, executed_actions[idx] = queue.popleft()

        delay_state["last_actions"] = np.array(executed_actions, copy=True)
        return executed_actions

    def advance_delay_state(self, delay_state):
        delay_state["env_step_idx"] += 1

    def load_agents(self, agents, model_dir, e=None):
        for ag in agents:
            ag.load_model(model_dir, e)

    def save_agents(self, agents, model_dir, e=None):
        for ag in agents:
            ag.save_model(model_dir, e)

    def set_replay(self, env, suffix, enabled):
        # Replay saving is a cityflow-only engine feature. Pick the world that
        # owns this env so the sumo (real) env is skipped under cross-sim setups.
        world_name = self.real_world_name if env is self.env_real else self.sim_world_name
        if world_name != "cityflow":
            return
        if not self.save_replay:
            return
        env.eng.set_save_replay(enabled)
        if enabled:
            env.eng.set_replay_file(os.path.join(self.replay_file_dir, suffix))

    # ------------------------------------------------------------------
    # Mitigation hooks. Defaults preserve the naive-baseline behavior;
    # action-gap method trainers (e.g. Delayed-Q) override these to route
    # decisions/transitions through their method model. ``idx`` identifies the
    # intersection so methods can keep per-intersection state.
    # ------------------------------------------------------------------

    def on_decision_start(self, obs, phases, test):
        """Called once per decision, before the per-agent select_action loop,
        with the FULL per-intersection obs/phase arrays. Default no-op; methods
        whose models need a global view (e.g. PRLight's neighbor-aware
        predictor) override this to cache the snapshot."""

    def select_action(self, ag, idx, ob, phase, test):
        return ag.get_action(ob, phase, test=test)

    def store_transition(
        self,
        ag,
        idx,
        *,
        last_obs,
        last_phase,
        chosen_action,
        executed_action,
        actions_prob,
        reward,
        obs,
        cur_phase,
        done,
        key,
    ):
        ag.remember(
            last_obs,
            last_phase,
            executed_action,
            actions_prob,
            reward,
            obs,
            cur_phase,
            done,
            key,
        )

    def train_agents(self, agents):
        return np.stack([ag.train() for ag in agents])

    def reset_episode_state(self, agents, init_phases):
        pass

    def run_train_episode(
        self,
        *,
        env,
        metric,
        agents,
        episode,
        total_decision_num,
        desc,
        action_delay,
    ):
        metric.clear()
        last_obs = env.reset()
        for agent in agents:
            agent.reset()

        episode_loss = []
        flush = 0
        i = 0
        dones = [False] * len(agents)
        last_phase = np.stack([ag.get_phase() for ag in agents])
        delay_state = self.initialize_action_delay_state(agents, last_phase)
        self.reset_episode_state(agents, last_phase)

        pbar = tqdm(total=int(self.steps / self.action_interval), desc=desc)

        while i < self.steps:
            if i % self.action_interval == 0:
                pbar.update()
                last_phase = np.stack([ag.get_phase() for ag in agents])

                self.on_decision_start(last_obs, last_phase, test=False)
                actions = []
                for idx, ag in enumerate(agents):
                    actions.append(
                        self.select_action(
                            ag, idx, last_obs[idx], last_phase[idx], test=False
                        )
                    )
                proposed_actions = np.stack(actions)
                self.enqueue_delayed_actions(
                    proposed_actions, delay_state, action_delay
                )

                actions_prob = []
                for idx, ag in enumerate(agents):
                    actions_prob.append(
                        ag.get_action_prob(last_obs[idx], last_phase[idx])
                    )

                rewards_list = []
                executed_actions = np.array(delay_state["last_actions"], copy=True)
                for _ in range(self.action_interval):
                    executed_actions = self.resolve_delayed_actions(delay_state)
                    obs, rewards, dones, _ = env.step(executed_actions.flatten())
                    i += 1
                    rewards_list.append(np.stack(rewards))
                    self.advance_delay_state(delay_state)

                rewards = np.mean(rewards_list, axis=0)
                metric.update(rewards)

                cur_phase = np.stack([ag.get_phase() for ag in agents])
                for idx, ag in enumerate(agents):
                    self.store_transition(
                        ag,
                        idx,
                        last_obs=last_obs[idx],
                        last_phase=last_phase[idx],
                        chosen_action=proposed_actions[idx],
                        executed_action=executed_actions[idx],
                        actions_prob=actions_prob[idx],
                        reward=rewards[idx],
                        obs=obs[idx],
                        cur_phase=cur_phase[idx],
                        done=dones[idx],
                        key=f"{episode}_{i // self.action_interval}_{ag.id}",
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
                cur_loss_q = self.train_agents(agents)
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
        phases = np.stack([ag.get_phase() for ag in agents])
        delay_state = self.initialize_action_delay_state(agents, phases)
        self.reset_episode_state(agents, phases)
        pbar = tqdm(total=int(self.test_steps / self.action_interval), desc=desc)

        while i < self.test_steps:
            if i % self.action_interval == 0:
                pbar.update()
                phases = np.stack([ag.get_phase() for ag in agents])
                self.on_decision_start(obs, phases, test=True)
                actions = []
                for idx, ag in enumerate(agents):
                    actions.append(
                        self.select_action(ag, idx, obs[idx], phases[idx], test=True)
                    )
                proposed_actions = np.stack(actions)
                self.enqueue_delayed_actions(
                    proposed_actions, delay_state, self.eval_action_delay
                )

                rewards_list = []
                for _ in range(self.action_interval):
                    actions = self.resolve_delayed_actions(delay_state)
                    obs, rewards, dones, _ = env.step(actions.flatten())
                    i += 1
                    rewards_list.append(np.stack(rewards))
                    self.advance_delay_state(delay_state)

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
            action_delay=self.sim_action_delay,
        )
        self.log_metrics("SIM_TRAIN", episode, self.metric_sim, mean_loss)
        self.logger.info("sim step:%s/%s", steps_run, self.steps)
        return mean_loss

    def train(self):
        # Zero-shot sim2real: train only in sim, evaluate in real (like the
        # observation/transition trainers). The real env is never trained on --
        # the action gap (execution delay) is handled at real-eval time, and any
        # mitigation method's auxiliary models are delay-independent so they too
        # learn from sim data.
        if self.load_pretrained:
            pretrained_dir = self.pretrained_model_dir()
            self.load_agents(self.agents_sim, pretrained_dir)
            self.load_agents(self.agents_real, pretrained_dir)

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

            if self.test_when_train or self.should_run_real_eval(episode):
                self.train_test(episode)

        self.save_agents(self.agents_sim, self.model_dir, e=self.episodes)
        self.save_agents(self.agents_sim, self.model_dir)

    def should_run_real_eval(self, episode):
        # Mirror the observation trainers: fire every `real_eval_interval`
        # episodes (skipping episode 0 so there's at least one trained epoch).
        return (
            self.real_eval_interval > 0
            and episode > 0
            and episode % self.real_eval_interval == 0
        )

    def train_test(self, episode):
        # Sim eval is optional (the experiment only cares about real-delay
        # transfer); `eval_sim: false` in the config skips it.
        if self.eval_sim:
            self.load_agents(self.agents_sim, self.model_dir)
            self.eval_action_delay = self.sim_action_delay
            self.run_eval_episode(
                env=self.env_sim,
                metric=self.metric_sim,
                agents=self.agents_sim,
                desc=f"Sim Eval Epoch {episode}",
            )
            self.log_metrics("TEST_SIM", episode, self.metric_sim, 100)

        self.load_agents(self.agents_real, self.model_dir)
        self.eval_action_delay = self.real_action_delay
        self.run_eval_episode(
            env=self.env_real,
            metric=self.metric_real,
            agents=self.agents_real,
            desc=f"Real Eval Epoch {episode}",
        )
        self.log_metrics("TEST_REAL", episode, self.metric_real, 100)
        return self.metric_real.real_average_travel_time()

    def test(self, drop_load=False):
        if self.eval_sim:
            if not drop_load:
                self.load_agents(self.agents_sim, self.model_dir, e=self.episodes)
            self.set_replay(self.env_sim, "final_sim.txt", True)
            self.eval_action_delay = self.sim_action_delay
            self.run_eval_episode(
                env=self.env_sim,
                metric=self.metric_sim,
                agents=self.agents_sim,
                desc="Final Sim Test",
            )
            self.log_metrics("FINAL_TEST_SIM", 0, self.metric_sim, 100)

        if not drop_load:
            self.load_agents(self.agents_real, self.model_dir, e=self.episodes)
        self.set_replay(self.env_real, "final_real.txt", True)
        self.eval_action_delay = self.real_action_delay
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
