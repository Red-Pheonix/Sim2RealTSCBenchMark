import os
import pickle as pkl
from pathlib import Path
import numpy as np
from common.metrics import Metrics
from environment import TSCEnv
from common.registry import Registry
from trainer.base_trainer import BaseTrainer
import torch
import torch.optim as optim
import sim2real_model


@Registry.register_trainer("sim2real_transitions_grounding")
class Sim2RealTransitionsTrainer(BaseTrainer):
    """
    Register TSCTrainer for traffic signal control tasks.
    """

    def __init__(self, logger, gpu=0, cpu=False, name="sim2real"):
        super().__init__(logger=logger, gpu=gpu, cpu=cpu, name=name)
        
        network_name = Registry.mapping['command_mapping']['setting'].param['network']
        self.cityflow_path = os.path.join('configs/sim', "cityflow", network_name + '.cfg')
        self.sumo_path = os.path.join('configs/sim', "sumo", network_name + '.cfg')
    
        self.episodes = Registry.mapping["trainer_mapping"]["setting"].param["episodes"]
        self.training_iterations = Registry.mapping["sim2real_mapping"]["setting"].param[
            "training_iterations"
        ]
        self.steps = Registry.mapping["trainer_mapping"]["setting"].param["steps"]
        self.test_steps = Registry.mapping["trainer_mapping"]["setting"].param[
            "test_steps"
        ]
        self.buffer_size = Registry.mapping["trainer_mapping"]["setting"].param[
            "buffer_size"
        ]
        self.action_interval = Registry.mapping["trainer_mapping"]["setting"].param[
            "action_interval"
        ]
        self.save_rate = Registry.mapping["logger_mapping"]["setting"].param[
            "save_rate"
        ]
        self.learning_start = Registry.mapping["trainer_mapping"]["setting"].param[
            "learning_start"
        ]
        self.update_model_rate = Registry.mapping["trainer_mapping"]["setting"].param[
            "update_model_rate"
        ]
        self.update_target_rate = Registry.mapping["trainer_mapping"]["setting"].param[
            "update_target_rate"
        ]
        self.test_when_train = Registry.mapping["trainer_mapping"]["setting"].param[
            "test_when_train"
        ]

        self.gat = Registry.mapping["sim2real_mapping"]["setting"].param["gat"]
        self.gattype = Registry.mapping["sim2real_mapping"]["setting"].param["gattype"]
        self.uncertainty_setting = Registry.mapping["sim2real_mapping"]["setting"].param[
            "uncertainty"
        ]
        self.delayedgat = Registry.mapping["sim2real_mapping"]["setting"].param[
            "delayedgat"
        ]
        self.ground_original = Registry.mapping["sim2real_mapping"]["setting"].param[
            "ground_original"
        ]
        self.last_n_uncertainties = Registry.mapping["sim2real_mapping"][
            "setting"
        ].param["last_n_uncertainties"]

        self.net = Registry.mapping["trainer_mapping"]["setting"].param["network"]
        self.load_pretrained = Registry.mapping["sim2real_mapping"]["setting"].param.get(
            "load_pretrained"
        )
        
        cmd_args = Registry.mapping['command_mapping']['setting'].param
        self.exp_name = f'{cmd_args["network"]}_{cmd_args["real_setting"]}_{cmd_args["agent"]}_{cmd_args["gat_model"]}'
        self.dataset_dir = Path("collected") / self.exp_name
        
        # replay file is only valid in cityflow now.
        # TODO: support SUMO and Openengine later

        # TODO: support other dataset in the future
        self.create()
        self.dataset = Registry.mapping["dataset_mapping"][
            Registry.mapping["command_mapping"]["setting"].param["dataset"]
        ](
            os.path.join(
                Registry.mapping["logger_mapping"]["path"].path,
                Registry.mapping["logger_mapping"]["setting"].param["data_dir"],
            )
        )
        self.dataset.initiate(
            ep=self.episodes, step=self.steps, interval=self.action_interval
        )
        self.yellow_time = Registry.mapping["trainer_mapping"]["setting"].param[
            "yellow_length"
        ]
        # consists of path of output dir + log_dir + file handlers name
        self.log_file = os.path.join(
            Registry.mapping["logger_mapping"]["path"].path,
            Registry.mapping["logger_mapping"]["setting"].param["log_dir"],
            os.path.basename(self.logger.handlers[-1].baseFilename)
            .rstrip("_BRF.log")
            .rstrip("_ACT.log")
            + "_DTL.log",
        )

        self.action_log_file = os.path.join(
            Registry.mapping["logger_mapping"]["path"].path,
            Registry.mapping["logger_mapping"]["setting"].param["log_dir"],
            os.path.basename(self.logger.handlers[-1].baseFilename)
            .rstrip("_BRF.log")
            .rstrip("_DTL.log")
            + "_ACT.log",
        )

        # Check if the folder exists
        if self.dataset_dir.exists():
            for file_path in self.dataset_dir.iterdir():
                if file_path.is_file():
                    file_path.unlink()
                    print(f"{file_path} has been deleted")
            print(f"All files in the '{self.dataset_dir}' folder have been deleted.")
        else:
            print(f"The folder '{self.dataset_dir}' does not exist.")

        self.total_decision_num = 0

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transition_model = Registry.mapping["sim2real_model_mapping"][
            self.gattype
        ](
            logger=self.logger,
            device=self.device,
            world_sim=self.world_sim,
            agents_sim=self.agents_sim,
            world_real=self.world_real,
            agents_real=self.agents_real,
            dataset_dir=self.dataset_dir,
        )

    def _dataset_file(self, *, forward, train):
        prefix = "ereal" if forward else "esim"
        split = "train" if train else "test"
        return self.dataset_dir / f"{prefix}_{split}.pkl"

    def create_world(self):
        """
        create_world
        Create world, currently support CityFlow World, SUMO World and Citypb World.

        :param: None
        :return: None
        """
        # traffic setting is in the world mapping
        self.world_sim = Registry.mapping["world_mapping"]["cityflow"](
            self.cityflow_path,
            Registry.mapping["command_mapping"]["setting"].param["thread_num"],
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

    def create_metrics(self):
        """
        create_metrics
        Create metrics to evaluate model performance, currently support reward, queue length, delay(approximate or real) and throughput.

        :param: None
        :return: None
        """
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

    def create_agents(self):
        """
        create_agents
        Create agents for traffic signal control tasks.

        :param: None
        :return: None
        """

        self.agents_sim = []
        self.agents_real = []

        agent_sim = Registry.mapping["model_mapping"][
            Registry.mapping["command_mapping"]["setting"].param["agent"]
        ](self.world_sim, 0)

        num_agent = int(len(self.world_sim.intersections) / agent_sim.sub_agents)

        print(
            f"Total number of agents: {num_agent}, Total number of sub agents: {agent_sim.sub_agents}"
        )
        self.agents_sim.append(
            agent_sim
        )  # initialized N agents for traffic light control

        for i in range(1, num_agent):
            self.agents_sim.append(
                Registry.mapping["model_mapping"][
                    Registry.mapping["command_mapping"]["setting"].param["agent"]
                ](self.world_sim, i)
            )

        agent_real = Registry.mapping["model_mapping"][
            Registry.mapping["command_mapping"]["setting"].param["agent"]
        ](self.world_real, 0)

        num_agent = int(len(self.world_real.intersections) / agent_real.sub_agents)
        self.agents_real.append(
            agent_real
        )  # initialized N agents for traffic light control
        for i in range(1, num_agent):
            self.agents_real.append(
                Registry.mapping["model_mapping"][
                    Registry.mapping["command_mapping"]["setting"].param["agent"]
                ](self.world_real, i)
            )

    def create_env(self):
        """
        create_env
        Create simulation environment for communication with agents.

        :param: None
        :return: None
        """
        # TODO: finalized list or non list
        self.env_sim = TSCEnv(self.world_sim, self.agents_sim, self.metric_sim)
        self.env_real = TSCEnv(self.world_real, self.agents_real, self.metric_real)

    def train(self):
        """
        Main training flow
        """
        
        if self.load_pretrained:
                for ag in self.agents_sim:
                    model_dir = os.path.join(
                        "pretrained",
                        "tsc",
                        Registry.mapping["command_mapping"]["setting"].param["agent"],
                        Registry.mapping["command_mapping"]["setting"].param["network"],
                    )
                    ag.load_model(model_dir)
                    ag.optimizer = optim.RMSprop(
                        ag.model.parameters(),
                        lr=ag.learning_rate,
                        alpha=0.9,
                        centered=False,
                        eps=1e-7,
                    )
                
                # self.transition_model.load_models()
        
        if self.delayedgat == True:
            # Run for a set number of episodes
            for e in range(self.episodes):

                # Sim rollout + collect data
                self.sim_rollout(e)

                # Real rollout + collect data
                self.train_test(e)

                # Update GAT models
                self.gat_training(e)

                # Delay GAT training until episode 150
                if e < 200:
                    # Run regular policy training for some number of iterations
                    self.policy_training(e, is_gat=False)
                else:
                    # Run GAT policy training for some number of iterations
                    self.policy_training(e, is_gat=True)

        else:

            # Run for a set number of episodes
            for e in range(self.episodes):

                # Sim rollout + collect data
                self.sim_rollout(e)

                # Real rollout + collect data
                self.train_test(e)

                # Update GAT models
                self.gat_training(e)

                # Run policy training for some number of iterations
                self.policy_training(e, is_gat=self.gat)

    def policy_training(self, episode, is_gat=False):
        """
        Train the agent(s) without saving data or collecting it, and log the iteration number.

        :param episode: int, the current episode number for logging
        :return: None
        """
        flush = 0

        for e in range(self.training_iterations):
            policy_stats = self.transition_model.init_policy_stats()
            self.metric_sim.clear()
            last_obs = self.env_sim.reset()
            for a in self.agents_sim:
                a.reset()
            if (
                Registry.mapping["command_mapping"]["setting"].param["world"]
                == "cityflow"
            ):
                self.env_sim.eng.set_save_replay(False)

            episode_loss = []
            i = 0
            while i < self.steps:
                if i % self.action_interval == 0:
                    last_phase = np.stack([ag.get_phase() for ag in self.agents_sim])

                    if self.total_decision_num > self.learning_start:
                        actions = []
                        for idx, ag in enumerate(self.agents_sim):
                            actions.append(
                                ag.get_action(
                                    last_obs[idx], last_phase[idx], test=False
                                )
                            )
                        actions = np.stack(actions)
                    else:
                        actions = np.stack([ag.sample() for ag in self.agents_sim])

                    actions_prob = [
                        ag.get_action_prob(last_obs[idx], last_phase[idx])
                        for idx, ag in enumerate(self.agents_sim)
                    ]

                    original_actions = actions.copy()
                    grounded_actions = [9 for i in range(len(self.agents_sim))]

                    if is_gat:
                        actions, grounded_actions = self.transition_model.ground_actions(
                            last_obs, actions, policy_stats
                        )

                    actions = actions.flatten()

                    rewards_list = []
                    for _ in range(self.action_interval):
                        obs, rewards, dones, _ = self.env_sim.step(actions)
                        i += 1
                        rewards_list.append(np.stack(rewards))
                    rewards = np.mean(rewards_list, axis=0)
                    self.metric_sim.update(rewards)

                    cur_phase = np.stack([ag.get_phase() for ag in self.agents_sim])
                    for idx, ag in enumerate(self.agents_sim):
                        ag.remember(
                            last_obs[idx],
                            last_phase[idx],
                            actions[idx],
                            actions_prob[idx],
                            rewards[idx],
                            obs[idx],
                            cur_phase[idx],
                            dones[idx],
                            f"{e}_{i // self.action_interval}_{ag.id}",
                        )
                    flush += 1
                    if flush == self.buffer_size - 1:
                        flush = 0

                    self.total_decision_num += 1
                    last_obs = obs

                    if (
                        self.total_decision_num > self.learning_start
                        and self.total_decision_num % self.update_model_rate
                        == self.update_model_rate - 1
                    ):
                        cur_loss_q = np.stack([ag.train() for ag in self.agents_sim])
                        episode_loss.append(cur_loss_q)
                    if (
                        self.total_decision_num > self.learning_start
                        and self.total_decision_num % self.update_target_rate
                        == self.update_target_rate - 1
                    ):
                        [ag.update_target_network() for ag in self.agents_sim]

                    # Ensure actions are in the correct format (list of numbers)
                    # original_actions = original_actions.flatten().tolist()
                    # grounded_actions_fixed = [item.item() if isinstance(item, np.ndarray) else item for item in grounded_actions]
                    # actions = actions.tolist()

                    # self.writeActionLog(episode, i, 3600, original_actions, grounded_actions_fixed, actions)

                    if all(dones):
                        break

            if len(episode_loss) > 0:
                mean_loss = np.mean(np.array(episode_loss))
            else:
                mean_loss = 0

            self.transition_model.finalize_policy_stats(episode, policy_stats)

            self.writeLog(
                "TRAIN",
                e,
                self.metric_sim.real_average_travel_time(),
                mean_loss,
                self.metric_sim.rewards(),
                self.metric_sim.queue(),
                self.metric_sim.delay(),
                self.metric_sim.throughput(),
            )
            self.logger.info(
                "Policy training episode: {}, iteration {}/{}, policy training avg travel time:{}, q_loss:{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format(
                    episode,
                    e,
                    self.training_iterations,
                    self.metric_sim.real_average_travel_time(),
                    mean_loss,
                    self.metric_sim.rewards(),
                    self.metric_sim.queue(),
                    self.metric_sim.delay(),
                    int(self.metric_sim.throughput()),
                )
            )

            # if e % self.save_rate == 0:
            #     [ag.save_model(e=e) for ag in self.agents_sim]

    def gat_training(self, e):
        if self.gat == True:
            self.transition_model.train_transition_models()
            self.transition_model.save_models(e)

    def sim_rollout(self, e):
        """
        single_rollout
        Perform a single rollout in the simulated environment and save data.

        :param: None
        :return: None
        """

        file_path = self._dataset_file(forward=False, train=True)

        # Initialize metrics and reset the environment
        self.metric_sim.clear()
        last_obs = self.env_sim.reset()
        state_action_next_state = []

        # Reset agents
        for a in self.agents_sim:
            a.reset()

        if Registry.mapping["command_mapping"]["setting"].param["world"] == "cityflow":
            if self.save_replay:
                self.env_sim.eng.set_save_replay(True)
                self.env_sim.eng.set_replay_file(
                    os.path.join(self.replay_file_dir, "single_rollout_replay.txt")
                )
            else:
                self.env_sim.eng.set_save_replay(False)

        i = 0
        while i < self.steps:
            if i % self.action_interval == 0:
                last_phase = np.stack([ag.get_phase() for ag in self.agents_sim])

                # Get agent actions
                actions = []
                for idx, ag in enumerate(self.agents_sim):
                    actions.append(
                        ag.get_action(last_obs[idx], last_phase[idx], test=True)
                    )
                actions = np.stack(actions)

                # Perform actions for the specified interval and collect data
                rewards_list = []
                for _ in range(self.action_interval):
                    obs, rewards, dones, _ = self.env_sim.step(actions.flatten())
                    i += 1
                    rewards_list.append(np.stack(rewards))

                rewards = np.mean(rewards_list, axis=0)
                self.metric_sim.update(rewards)

                state_action_next_state.extend(
                    self.transition_model.collect_sim_transition(last_obs, actions, obs)
                )

                last_obs = obs

            if all(dones):
                break

        if e % self.save_rate == 0:
            model_dir = os.path.join(Registry.mapping["logger_mapping"]["path"].path, 
                                    "model",
                                    Registry.mapping["command_mapping"]["setting"].param["gat_model"], 
                                    Registry.mapping["command_mapping"]["setting"].param.get("real_setting")
            )
            [ag.save_model(model_dir=model_dir, e=e) for ag in self.agents_sim]

        # Save the collected rollout data
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            pkl.dump(state_action_next_state, f)

            self.writeLog(
                "Sim Rollout",
                e,
                self.metric_sim.real_average_travel_time(),
                100,
                self.metric_sim.rewards(),
                self.metric_sim.queue(),
                self.metric_sim.delay(),
                self.metric_sim.throughput(),
            )
            
            self.logger.info(
                "Sim Rollout episode:{}/{}, sim avg travel time:{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format(
                    e,
                    self.episodes,
                    self.metric_sim.real_average_travel_time(),
                    self.metric_sim.rewards(),
                    self.metric_sim.queue(),
                    self.metric_sim.delay(),
                    int(self.metric_sim.throughput()),
                )
            )
        # for j in range(len(self.world_sim.intersections)):
        # self.logger.debug("intersection:{}, individual reward:{}, individual queue:{}, individual delay:{}, individual throughput:{}".format(j, self.metric_sim.lane_rewards()[j], self.metric_sim.queue()[j], self.metric_sim.delay()[j], int(self.metric_sim.throughput()[j])))

    def train_test(self, e):
        """
        train_test
        Evaluate model performance after each episode training process.

        :param e: number of episode
        :return self.metric.real_average_travel_time: travel time of vehicles
        """
        last_obs = self.env_real.reset()
        self.metric_real.clear()
        state_action_next_state = []
        
        model_dir = os.path.join(Registry.mapping["logger_mapping"]["path"].path, 
                            "model",
                            Registry.mapping["command_mapping"]["setting"].param["gat_model"], 
                            Registry.mapping["command_mapping"]["setting"].param.get("real_setting")
        )

        for a in self.agents_real:
            a.load_model(model_dir, e)
            a.reset()

        for i in range(self.test_steps):
            if i % self.action_interval == 0:
                phases = np.stack([ag.get_phase() for ag in self.agents_real])
                actions = []
                for idx, ag in enumerate(self.agents_real):
                    actions.append(ag.get_action(last_obs[idx], phases[idx], test=True))
                actions = np.stack(actions)
                rewards_list = []
                for _ in range(self.action_interval):
                    obs, rewards, dones, _ = self.env_real.step(
                        actions.flatten()
                    )  # make sure action is [intersection]
                    i += 1
                    rewards_list.append(np.stack(rewards))
                rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                self.metric_real.update(rewards)

                state_action_next_state.extend(
                    self.transition_model.collect_real_transition(last_obs, actions, obs)
                )

                last_obs = obs

            if all(dones):
                break

        self.logger.info(
            "Real rollout step:{}/{}, travel time:{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format(
                e,
                self.episodes,
                self.metric_real.real_average_travel_time(),
                self.metric_real.rewards(),
                self.metric_real.queue(),
                self.metric_real.delay(),
                int(self.metric_real.throughput()),
            )
        )
        self.writeLog(
            "Real rollout",
            e,
            self.metric_real.real_average_travel_time(),
            100,
            self.metric_real.rewards(),
            self.metric_real.queue(),
            self.metric_real.delay(),
            self.metric_real.throughput(),
        )

        # Save the data for the current episode
        file_path = self._dataset_file(forward=True, train=True)

        # Save new data directly to the file
        with open(file_path, "wb") as f:
            pkl.dump(state_action_next_state, f)

        return self.metric_real.real_average_travel_time()

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
        """
        writeLog
        Write log for record and debug.

        :param mode: "TRAIN" or "TEST"
        :param step: current step in simulation
        :param travel_time: current travel time
        :param loss: current loss
        :param cur_rwd: current reward
        :param cur_queue: current queue length
        :param cur_delay: current delay
        :param cur_throughput: current throughput
        :return: None
        """
        
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

    def writeActionLog(
        self,
        episode_num,
        step,
        total_steps,
        orig_actions,
        grounded_actions,
        actions_taken,
    ):
        """
        writeActionLog
        Write log for record and debug, including episode information and actions taken by both the original and grounded agents in the specified format.

        :param episode_num: current episode number
        :param total_episodes: total number of episodes
        :param agent_actions: actions taken by the original agent
        :param grounded_actions: actions taken by the grounded agent
        :return: None
        """
        res = f"Policy training episode:{episode_num}, step:{step}/{total_steps}, original actions:{orig_actions}, grounded actions:{grounded_actions}, actions taken: {actions_taken}"

        log_handle = open(self.action_log_file, "a")
        log_handle.write(res + "\n")
        log_handle.close()

    def test(self):        
        # sim environment
        last_obs = self.env_sim.reset()
        self.metric_sim.clear()
        
        model_dir = os.path.join(
                "pretrained",
                Registry.mapping["command_mapping"]["setting"].param["task"],
                Registry.mapping["command_mapping"]["setting"].param["agent"],
                Registry.mapping["command_mapping"]["setting"].param["network"],
                Registry.mapping["command_mapping"]["setting"].param["gat_model"], 
                Registry.mapping["command_mapping"]["setting"].param.get("real_setting")
        )
        
        for a in self.agents_sim:
            a.load_model(model_dir)
            a.reset()
        
        for i in range(self.test_steps):
            if i % self.action_interval == 0:
                phases = np.stack([ag.get_phase() for ag in self.agents_sim])
                actions = []
                for idx, ag in enumerate(self.agents_sim):
                    actions.append(ag.get_action(last_obs[idx], phases[idx], test=True))
                actions = np.stack(actions)
                rewards_list = []
                for _ in range(self.action_interval):
                    obs, rewards, dones, _ = self.env_sim.step(
                        actions.flatten()
                    )  # make sure action is [intersection]
                    i += 1
                    rewards_list.append(np.stack(rewards))
                rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                self.metric_sim.update(rewards)

                last_obs = obs

            if all(dones):
                break
        
        self.logger.info(
            "Test - Sim rollout: travel time:{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format(
                self.metric_sim.real_average_travel_time(),
                self.metric_sim.rewards(),
                self.metric_sim.queue(),
                self.metric_sim.delay(),
                int(self.metric_sim.throughput()),
            )
        )
        self.writeLog(
            "Test_Sim_rollout",
            1,
            self.metric_sim.real_average_travel_time(),
            1,
            self.metric_sim.rewards(),
            self.metric_sim.queue(),
            self.metric_sim.delay(),
            self.metric_sim.throughput(),
        )
        
        # real environment
        last_obs = self.env_real.reset()
        self.metric_real.clear()
        
        for a in self.agents_real:
            a.load_model(model_dir)
            a.reset()
            
        for i in range(self.test_steps):
            if i % self.action_interval == 0:
                phases = np.stack([ag.get_phase() for ag in self.agents_real])
                actions = []
                for idx, ag in enumerate(self.agents_real):
                    actions.append(ag.get_action(last_obs[idx], phases[idx], test=True))
                actions = np.stack(actions)
                rewards_list = []
                for _ in range(self.action_interval):
                    obs, rewards, dones, _ = self.env_real.step(
                        actions.flatten()
                    )  # make sure action is [intersection]
                    i += 1
                    rewards_list.append(np.stack(rewards))
                rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                self.metric_real.update(rewards)

                last_obs = obs

            if all(dones):
                break
        
        self.logger.info(
            "Test - Real rollout step: travel time:{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format(
                self.metric_real.real_average_travel_time(),
                self.metric_real.rewards(),
                self.metric_real.queue(),
                self.metric_real.delay(),
                int(self.metric_real.throughput()),
            )
        )
        self.writeLog(
            "Test_Real_rollout",
            1,
            self.metric_real.real_average_travel_time(),
            1,
            self.metric_real.rewards(),
            self.metric_real.queue(),
            self.metric_real.delay(),
            self.metric_real.throughput(),
        )
