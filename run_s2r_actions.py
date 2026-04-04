import os
import time
import argparse
import logging

import task
import trainer
import agent
import dataset
from common.registry import Registry
from common import interface
from utils.logger import build_config, setup_logging


parser = argparse.ArgumentParser(description="Run Sim2Real Actions Experiment")
parser.add_argument("--thread_num", type=int, default=4, help="number of threads")
parser.add_argument("--ngpu", type=str, default="-1", help="gpu to be used")
parser.add_argument(
    "--prefix", type=str, default="test", help="the number of prefix in this running process"
)
parser.add_argument("--seed", type=int, default=None, help="seed for pytorch backend")
parser.add_argument("--debug", type=bool, default=True)
parser.add_argument(
    "--interface",
    type=str,
    default="libsumo",
    choices=["libsumo", "traci"],
    help="interface type",
)
parser.add_argument(
    "--delay_type",
    type=str,
    default="apx",
    choices=["apx", "real"],
    help="method of calculating delay",
)
parser.add_argument(
    "--real_setting",
    type=str,
    default="default",
    help="action setting file name under configs/sim2real_actions/settings",
)

parser.add_argument(
    "-t", "--task", type=str, default="sim2real_actions", help="task type to run"
)
parser.add_argument(
    "-a", "--agent", type=str, default="dqn", help="agent type of agents in RL environment"
)
parser.add_argument(
    "-w",
    "--world",
    type=str,
    default="cityflow",
    choices=["cityflow", "sumo"],
    help="simulator type",
)
parser.add_argument("-n", "--network", type=str, default="cityflow1x1", help="network name")
parser.add_argument(
    "-d", "--dataset", type=str, default="onfly", help="type of dataset in training process"
)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.ngpu

logging_level = logging.INFO
if args.debug:
    logging_level = logging.DEBUG


class Runner:
    def __init__(self, pArgs):
        self.config, self.duplicate_config = build_config(pArgs)
        self.config_registry()

    def config_registry(self):
        self.config["command"]["network"] = args.network
        self.config["command"]["real_setting"] = args.real_setting

        interface.Command_Setting_Interface(self.config)
        interface.Logger_param_Interface(self.config)
        interface.World_param_Interface(self.config)

        interface.Logger_path_Interface(self.config)
        if not os.path.exists(Registry.mapping["logger_mapping"]["path"].path):
            os.makedirs(Registry.mapping["logger_mapping"]["path"].path)
        interface.Trainer_param_Interface(self.config)
        interface.ModelAgent_param_Interface(self.config)
        interface.Sim2Real_param_Interface(self.config)

    def run(self):
        logger = setup_logging(logging_level)

        Registry.mapping["trainer_mapping"]["setting"].param["network"] = self.config[
            "command"
        ]["network"]

        self.trainer = Registry.mapping["trainer_mapping"][
            Registry.mapping["command_mapping"]["setting"].param["task"]
        ](logger)
        self.task = Registry.mapping["task_mapping"][
            Registry.mapping["command_mapping"]["setting"].param["task"]
        ](self.trainer)
        start_time = time.time()
        self.task.run()
        logger.info(f"Total time taken: {time.time() - start_time}")


if __name__ == "__main__":
    test = Runner(args)
    test.run()
