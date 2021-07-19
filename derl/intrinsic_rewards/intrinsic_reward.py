from abc import ABC, abstractmethod

from gym.spaces.utils import flatdim

from derl.utils.utils import flatten_dict as flatten


class IntrinsicReward(ABC):
    """
    Abstract class for intrinsic rewards as exploration bonuses
    """

    def __init__(self, observation_space, action_space, parallel_envs, cfg):
        """
        Initialise parameters for intrinsic reward
        :param observation_space: observation space of environment
        :param action space: action space of environment
        :param parallel_envs: number of parallel environments
        :param config: configuration for intrinsic reward
        """
        self.observation_space = observation_space
        self.action_space = action_space

        self.obs_size = flatdim(observation_space)
        self.action_size = flatdim(action_space)

        self.parallel_envs = parallel_envs

        # set all values from config as attributes
        for k, v in flatten(cfg).items():
            setattr(IntrinsicReward, k, v)

    @abstractmethod
    def compute_intrinsic_reward(self, state, action, next_state, train=False):
        """
        Compute intrinsic reward for given input
        :param state: (batch of) current state(s)
        :param action: (batch of) applied action(s)
        :param next_state: (batch of) next/reached state(s)
        :param train: flag if model should be trained
        :return: dict of 'intrinsic reward' and losses
        """
        raise NotImplementedError
   
    def episode_reset(self, environment_idx):
        """
        Indicate termination of episode/ start of new episode

        :param environment_idx: index of environment for which new episode started
        """
        pass
