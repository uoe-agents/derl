from collections import defaultdict
import glob
from hashlib import sha256
import json
import logging
import os

from omegaconf import OmegaConf

import numpy as np
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(
    level=logging.INFO,
    format="(%(process)d) [%(levelname).1s] - (%(asctime)s) - %(name)s >> %(message)s",
    datefmt="%m/%d %H:%M:%S",
)

def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, "*.monitor.csv"))
        for f in files:
            os.remove(f)

class NumpyQueue:
    """
    Class to hold queues computed as fixed-length numpy arrasy for windows-averages
    """
    def __init__(self, length):
        self.length = length
        self.queues = defaultdict(lambda: np.zeros((length)))

    def __len__(self):
        return self.length
    
    def add(self, key, value):
        self.queues[key][:-1] = self.queues[key][1:]
        self.queues[key][-1] = value
    
    def mean(self, key):
        return self.queues[key].mean()


class Logger:
    def __init__(self, cfg={}, tensorboard_dir=None, step_average=10):
        self.config = OmegaConf.to_container(cfg)

        self.log = logging.getLogger(__name__)

        # average over that many steps for timestep metrics
        self.steps = defaultdict(int)
        self.step_queue = NumpyQueue(step_average)

        hash_keys = []
        for k in self.config:
            if k.startswith('algorithm') or k.startswith('env') or k.startswith('curiosity') or k.startswith('exploitation_algorithm'):
                hash_keys.append(k)
            elif "train_extrinsic_intrinsic_rewards" in k or "train_intrinsic_extrinsic_rewards" in k:
                hash_keys.append(k)

        self.config_hash = sha256(
            json.dumps(
                {k: v for k, v in self.config.items() if k in hash_keys},
                sort_keys=True,
            ).encode("utf8")
        ).hexdigest()[-10:]

        self.summary_statistics = {}

        if tensorboard_dir:
            cleanup_log_dir(tensorboard_dir)
            self.tensorboard_logger = SummaryWriter(tensorboard_dir)
        else:
            self.tensorboard_logger = None

    def info(self, message):
        self.log.info(message)

    def warning(self, message):
        self.log.warning(message)
    
    def __log_metric(self, key, value, timestep, custom_step_name="timestep", commit=True):
        if self.tensorboard_logger:
            self.tensorboard_logger.add_scalar(key, value, timestep)
    
    def __log_global_maximum(self, key, value, timestep, commit=False):
        if key in self.summary_statistics:
            if self.summary_statistics[key] < value:
                self.summary_statistics[key] = value
                self.__log_metric(f"{key}", value, timestep, "episode", commit)
        else:
            self.summary_statistics[key] = value
            self.__log_metric(f"{key}", value, timestep, "episode", commit)

    def __log_global_minimum(self, key, value, timestep, commit=False):
        if key in self.summary_statistics:
            if self.summary_statistics[key] > value:
                self.summary_statistics[key] = value
                self.__log_metric(f"{key}", value, timestep, "episode", commit)
        else:
            self.summary_statistics[key] = value
            self.__log_metric(f"{key}", value, timestep, "episode", commit)

    def __log_step_metric(self, key, value, timestep, commit=False):
        self.steps[key] += 1
        self.step_queue.add(key, value)
        if self.steps[key] % len(self.step_queue) == 0:
            self.__log_metric(key, self.step_queue.mean(key), timestep, "timestep", commit)

    def __log_episode_metric(self, key, value, timestep, commit=False):
        self.__log_metric(key, value, timestep, "episode", commit)

    def log_epsilon(self, epsilon, timestep):
        self.__log_step_metric("Train/epsilon", epsilon, timestep)

    def log_intrinsic_reward(self, int_dict, timestep):
        if self.tensorboard_logger:
            for key, value in int_dict.items():
                self.tensorboard_logger.add_scalar(f"intrinsic_reward_{key}", value, timestep)

    def log_extrinsic_reward(self, reward, timestep):
        self.__log_step_metric("Train/extrinsic_reward", reward, timestep)

    def log_losses(self, loss_dict, timestep, custom_step_name="timestep"):
        if self.tensorboard_logger:
            for key, value in loss_dict.items():
                self.tensorboard_logger.add_scalar(key, value, timestep)

    def log_update(self, completed_steps, completed_updates):
        self.__log_step_metric("Train/completed_updates", completed_updates, completed_steps, commit=True)
    
    def log_episode(self, completed_episodes, info, print_log=False):
        for k, v in info.items():
            if k == "episode":
                for episode_key, episode_v in v.items():
                    if episode_key == "r":
                        self.__log_episode_metric("Train/episode_return", episode_v, completed_episodes)
                        self.__log_global_maximum("Train/max_episode_return", episode_v, completed_episodes)
                        episode_return = episode_v
                    elif episode_key == "l":
                        self.__log_episode_metric("Train/episode_length", episode_v, completed_episodes)
                        self.__log_global_maximum("Train/max_episode_length", episode_v, completed_episodes)
                        self.__log_global_minimum("Train/min_episode_length", episode_v, completed_episodes)
                        episode_length = episode_v
                    elif episode_key == "t":
                        self.__log_episode_metric("Train/training_duration", episode_v, completed_episodes)
                        training_duration = episode_v

        if print_log:
            self.info(
                f"Completed episode {completed_episodes}: Steps = {episode_length} / Return = {episode_return:.3f} / Total duration = {training_duration}s"
            )

    def log_evaluation(self, completed_episodes, info, num_episodes):
        for k, v in info.items():
            if k == "r":
                self.__log_episode_metric("Eval/episode_return", v, completed_episodes)
                self.__log_global_maximum("Eval/max_episode_return", v, completed_episodes)
                episode_return = v
            elif k == "l":
                self.__log_episode_metric("Eval/episode_length", v, completed_episodes)
                self.__log_global_maximum("Eval/max_episode_length", v, completed_episodes)
                self.__log_global_minimum("Eval/min_episode_length", v, completed_episodes)
                episode_length = v

        self.info(
            f"Completed evaluation {completed_episodes} with {num_episodes} episodes: Steps = {episode_length} / Return = {episode_return:.3f}"
        )

class TensorboardLogger(Logger):
    def __init__(self, cfg, tensorboard_dir):
        super(TensorboardLogger, self).__init__(cfg, tensorboard_dir)
