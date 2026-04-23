import json
import os
from pathlib import Path

from common.metrics import Metrics
from common.registry import Registry
from environment import TSCEnv
from trainer.base_trainer import BaseTrainer


class TransitionTrainer(BaseTrainer):
    """
    Lightweight shared base for transition trainers.
    Keeps common setup/helpers together while leaving method-specific train
    loops in the concrete trainers.
    """

    def __init__(self, logger, gpu=0, cpu=False, name="sim2real_transitions"):
        super().__init__(logger=logger, gpu=gpu, cpu=cpu, name=name)

        self.cmd_args = Registry.mapping["command_mapping"]["setting"].param
        self.trainer_args = Registry.mapping["trainer_mapping"]["setting"].param
        self.logger_args = Registry.mapping["logger_mapping"]["setting"].param
        self.sim2real_args = Registry.mapping["sim2real_mapping"]["setting"].param

        self.network_name = self.cmd_args["network"]
        self.real_setting = self.cmd_args["real_setting"]
        self.agent_name = self.cmd_args["agent"]

        self.cityflow_path = os.path.join(
            "configs/sim", "cityflow", self.network_name + ".cfg"
        )
        self.sumo_path = os.path.join(
            "configs/sim", "sumo", self.network_name + ".cfg"
        )

        self.steps = self.trainer_args["steps"]
        self.test_steps = self.trainer_args["test_steps"]
        self.buffer_size = self.trainer_args["buffer_size"]
        self.action_interval = self.trainer_args["action_interval"]
        self.save_rate = self.logger_args["save_rate"]
        self.learning_start = self.trainer_args["learning_start"]
        self.update_model_rate = self.trainer_args["update_model_rate"]
        self.update_target_rate = self.trainer_args["update_target_rate"]
        self.test_when_train = self.trainer_args["test_when_train"]
        self.yellow_time = self.trainer_args["yellow_length"]
        self.load_pretrained = self.sim2real_args.get("load_pretrained", False)

    def build_exp_name(self, suffix):
        return (
            f"{self.network_name}_{self.real_setting}_{self.agent_name}_{suffix}"
        )

    def build_model_dir(self, exp_name):
        return os.path.join(
            Registry.mapping["logger_mapping"]["path"].path,
            self.logger_args["model_dir"],
            exp_name,
        )

    def build_log_file(self):
        base_log_name = os.path.basename(self.logger.handlers[-1].baseFilename).rstrip(
            "_BRF.log"
        )
        return os.path.join(
            Registry.mapping["logger_mapping"]["path"].path,
            self.logger_args["log_dir"],
            base_log_name + "_DTL.log",
        )

    def load_cityflow_config(self, cityflow_path):
        with open(cityflow_path, "r", encoding="utf-8") as file_obj:
            return json.load(file_obj)

    def load_cityflow_flow(self, cityflow_config):
        flow_path = Path(cityflow_config.get("dir", "data")) / cityflow_config["flowFile"]
        with open(flow_path, "r", encoding="utf-8") as file_obj:
            return json.load(file_obj)

    def create_world(self):
        thread_num = self.cmd_args["thread_num"]
        self.world_sim = Registry.mapping["world_mapping"]["cityflow"](
            self.cityflow_path,
            thread_num,
        )

        sumo_add = self.cmd_args.get("real_setting")
        if sumo_add != "default":
            sumo_add = sumo_add + ".add.xml"
        else:
            sumo_add = None

        self.world_real = Registry.mapping["world_mapping"]["sumo"](
            self.sumo_path,
            interface=self.cmd_args["interface"],
            sumo_add=sumo_add,
        )

    def create_agent_world(self, world):
        agents = []
        agent = Registry.mapping["model_mapping"][self.agent_name](world, 0)
        num_agent = int(len(world.intersections) / agent.sub_agents)
        agents.append(agent)
        for i in range(1, num_agent):
            agents.append(Registry.mapping["model_mapping"][self.agent_name](world, i))
        return agents

    def create_agents(self):
        self.agents_sim = self.create_agent_world(self.world_sim)
        self.agents_real = self.create_agent_world(self.world_real)

    def create_metrics(self):
        if self.cmd_args["delay_type"] == "apx":
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
